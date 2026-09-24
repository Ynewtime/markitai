"""An interrupted batch has to leave something `--resume` can read.

State is written as a base ``.state.json`` plus a ``.state.jsonl`` sidecar
that later saves append their deltas to, and ``load_state`` gives up the
moment the base file is missing. The CLI batch path called ``init_state``
without writing that base — only ``BatchProcessor.process_batch``, the
library entry point, did — so a run interrupted before it finished left a
sidecar nothing could replay. ``--resume`` then started from zero and paid
for every LLM call a second time, while reporting nothing unusual.

The check is the timing, not the file: by the time the first document is
converted the base state must already be on disk, because that is the
earliest moment an interrupt can arrive.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from markitai.llm import LLMProcessor

import pytest

from markitai.config import MarkitaiConfig


def _state_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / ".markitai" / "states").glob("*.state.json"))


@pytest.mark.asyncio
async def test_base_state_exists_before_the_first_document_is_converted(
    tmp_path: Path,
) -> None:
    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    for index in range(3):
        (input_dir / f"doc{index}.txt").write_text(f"content {index}")
    output_dir = tmp_path / "out"

    seen_at_first_file: list[list[Path]] = []
    from markitai.cli.processors import batch as batch_module

    original = batch_module.create_process_file

    def _recording_factory(
        cfg: MarkitaiConfig,
        input_dir: Path,
        output_dir: Path,
        shared_processor: LLMProcessor | None,
    ):
        process_file = original(cfg, input_dir, output_dir, shared_processor)

        async def _wrapped(path: Path):
            if not seen_at_first_file:
                seen_at_first_file.append(_state_files(output_dir))
            return await process_file(path)

        return _wrapped

    with patch.object(batch_module, "create_process_file", _recording_factory):
        await process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            cfg=MarkitaiConfig(),
            resume=False,
            dry_run=False,
            quiet=True,
        )

    assert seen_at_first_file, "no document was processed; the test proves nothing"
    assert seen_at_first_file[0], (
        "the batch started converting with no base state file on disk — an "
        "interrupt here leaves only a .jsonl sidecar, which load_state() "
        "cannot replay, so --resume restarts from zero"
    )


def _load_state(output_dir: Path) -> dict:
    import json

    (state_file,) = _state_files(output_dir)
    return json.loads(state_file.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_ctrl_c_keeps_items_finished_since_the_last_throttled_save(
    tmp_path: Path,
) -> None:
    """Saves are throttled (10s by default); an interrupt must still flush.

    The gather used to be wrapped in a finally that only stopped the live
    display, so a CancelledError skipped compact_state(): items completed
    inside the last interval were lost and --resume redid (and re-paid
    for) them.
    """
    import asyncio

    from markitai.batch import ProcessResult
    from markitai.cli.processors import batch as batch_module
    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("a")
    (input_dir / "b.txt").write_text("b")
    output_dir = tmp_path / "out"
    b_started = asyncio.Event()

    def _factory(cfg, input_dir, output_dir, shared_processor):
        async def _process(path: Path) -> ProcessResult:
            if path.name == "a.txt":
                return ProcessResult(success=True, output_path=str(output_dir / "a"))
            b_started.set()
            await asyncio.Event().wait()  # an LLM call that never returns
            raise AssertionError("unreachable")

        return _process

    cfg = MarkitaiConfig()
    cfg.batch.concurrency = 1
    cfg.batch.state_flush_interval_seconds = 3600
    with patch.object(batch_module, "create_process_file", _factory):
        run = asyncio.create_task(
            process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                cfg=cfg,
                resume=False,
                dry_run=False,
                quiet=True,
            )
        )
        await b_started.wait()
        await asyncio.sleep(0.05)
        # What asyncio.run does on Ctrl-C: cancel the main task
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    documents = _load_state(output_dir)["documents"]
    assert documents["a.txt"]["status"] == "completed"
    assert documents["b.txt"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_items_waiting_for_a_slot_stay_pending(tmp_path: Path) -> None:
    """With -j 1 only the file holding the slot is in_progress.

    The worker pool is sized for URLs too (url_concurrency workers), so
    files used to be marked in_progress before waiting on the file
    semaphore: the state showed several at once, and after an interrupt
    --resume treated the never-started ones as failed (redone in overwrite
    mode) instead of pending.
    """
    import asyncio

    from markitai.cli.processors import batch as batch_module
    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (input_dir / name).write_text(name)
    output_dir = tmp_path / "out"
    started = asyncio.Event()

    def _factory(cfg, input_dir, output_dir, shared_processor):
        async def _process(path: Path):
            started.set()
            await asyncio.Event().wait()  # the first file never finishes
            raise AssertionError("unreachable")

        return _process

    cfg = MarkitaiConfig()
    cfg.batch.concurrency = 1
    cfg.batch.url_concurrency = 3  # three workers compete for one file slot
    cfg.batch.state_flush_interval_seconds = 3600
    with patch.object(batch_module, "create_process_file", _factory):
        run = asyncio.create_task(
            process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                cfg=cfg,
                resume=False,
                dry_run=False,
                quiet=True,
            )
        )
        await started.wait()
        await asyncio.sleep(0.05)  # the other workers are parked on the slot
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    statuses = sorted(
        d["status"] for d in _load_state(output_dir)["documents"].values()
    )
    assert statuses == ["in_progress", "pending", "pending"]


@pytest.mark.asyncio
async def test_library_process_batch_saves_state_on_cancellation(
    tmp_path: Path,
) -> None:
    import asyncio

    from markitai.batch import BatchProcessor, ProcessResult
    from markitai.config import BatchConfig

    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for f in files:
        f.write_text("x")
    b_started = asyncio.Event()

    async def _process(path: Path) -> ProcessResult:
        if path.name == "a.txt":
            return ProcessResult(success=True, output_path="a.txt.md")
        b_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    processor = BatchProcessor(
        BatchConfig(concurrency=1, state_flush_interval_seconds=3600),
        tmp_path / "out",
    )
    run = asyncio.create_task(processor.process_batch(files, _process))
    await b_started.wait()
    await asyncio.sleep(0.05)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    documents = _load_state(tmp_path / "out")["documents"]
    # Stored absolute: the relative value only meant something in this cwd
    assert documents["a.txt"] == {
        "status": "completed",
        "output": os.path.abspath("a.txt.md"),
    }
    assert documents["b.txt"]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_resume_overwrites_the_items_own_output_instead_of_v2(
    tmp_path: Path,
) -> None:
    """A re-queued item redoes its work over the output it had claimed.

    Resumed failed/interrupted items took the default rename path, found
    their own earlier base .md, and wrote x.v2.md — a duplicate the state
    then pointed at.
    """
    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha v1")
    (input_dir / "b.txt").write_text("bravo v1")
    output_dir = tmp_path / "out"

    await process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        cfg=MarkitaiConfig(),
        resume=False,
        dry_run=False,
        quiet=True,
    )
    assert (output_dir / "b.txt.md").is_file()

    # Simulate b.txt having been interrupted mid-run (its output written,
    # its state still in progress with the claimed target recorded)
    import json

    (state_file,) = _state_files(output_dir)
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["documents"]["b.txt"]["status"] == "completed"
    data["documents"]["b.txt"] = {
        "status": "in_progress",
        "target": str(output_dir / "b.txt.md"),
    }
    state_file.write_text(json.dumps(data), encoding="utf-8")
    (input_dir / "b.txt").write_text("bravo v2")
    (input_dir / "a.txt").write_text("alpha changed but completed")

    await process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        cfg=MarkitaiConfig(),
        resume=True,
        dry_run=False,
        quiet=True,
    )

    assert sorted(p.name for p in output_dir.glob("*.md")) == ["a.txt.md", "b.txt.md"]
    assert "bravo v2" in (output_dir / "b.txt.md").read_text(encoding="utf-8")
    assert "alpha v1" in (output_dir / "a.txt.md").read_text(encoding="utf-8")
    documents = _load_state(output_dir)["documents"]
    assert Path(documents["b.txt"]["output"]).name == "b.txt.md"


@pytest.mark.asyncio
async def test_resume_without_recorded_target_keeps_the_users_on_conflict(
    tmp_path: Path,
) -> None:
    """A re-queued item with no recorded target is not forced to overwrite.

    Without a target nothing says the file at the default name is this
    batch's own output (an older state, or a failure before the name was
    claimed): it may be someone else's file, so the item follows the
    user's on_conflict (rename by default) instead of clobbering it.
    """
    import json

    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("alpha v2")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "a.txt.md").write_text("fallback from the failed run")

    # A state as an older markitai wrote it: failed, no target recorded
    from markitai.batch import BatchProcessor

    cfg = MarkitaiConfig()
    probe = BatchProcessor(
        cfg.batch,
        output_dir,
        input_path=input_dir,
        task_options={
            "llm": False,
            "ocr": False,
            "screenshot": False,
            "alt": False,
            "desc": False,
            "scan_max_depth": cfg.batch.scan_max_depth,
        },
    )
    probe.state_file.parent.mkdir(parents=True)
    probe.state_file.write_text(
        json.dumps(
            {
                "version": "1.0",
                "options": {
                    "input_dir": str(input_dir.resolve()),
                    "output_dir": str(output_dir.resolve()),
                },
                "documents": {"a.txt": {"status": "failed", "error": "boom"}},
                "urls": {},
            }
        ),
        encoding="utf-8",
    )

    await process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        cfg=MarkitaiConfig(),
        resume=True,
        dry_run=False,
        quiet=True,
    )

    assert sorted(p.name for p in output_dir.glob("*.md")) == [
        "a.txt.md",
        "a.txt.v2.md",
    ]
    assert (output_dir / "a.txt.md").read_text(
        encoding="utf-8"
    ) == "fallback from the failed run"
    assert "alpha v2" in (output_dir / "a.txt.v2.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_resume_from_another_cwd_writes_into_the_original_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`markitai in -o out` interrupted, then `markitai ../in -o ../out --resume`.

    The state hash uses absolute paths, so the second run finds the state;
    the recorded target was cwd-relative, so the interrupted item was
    redone into ``other/out/`` instead of over its own output.
    """
    import asyncio

    from markitai.cli.processors import batch as batch_module
    from markitai.cli.processors.batch import process_batch
    from markitai.utils.output import resolve_item_output_path

    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.txt").write_text("alpha")
    (tmp_path / "in" / "b.txt").write_text("bravo")
    (tmp_path / "other").mkdir()
    monkeypatch.chdir(tmp_path)
    original = batch_module.create_process_file
    b_claimed = asyncio.Event()

    def _interrupting_factory(cfg, input_dir, output_dir, shared_processor):
        process_file = original(cfg, input_dir, output_dir, shared_processor)

        async def _process(path: Path):
            if path.name != "b.txt":
                return await process_file(path)
            # Claim the output name (recording the target), then hang
            resolve_item_output_path(output_dir / "b.txt.md", "rename")
            b_claimed.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        return _process

    cfg = MarkitaiConfig()
    cfg.batch.concurrency = 1
    with patch.object(batch_module, "create_process_file", _interrupting_factory):
        run = asyncio.create_task(
            process_batch(
                input_dir=Path("in"),
                output_dir=Path("out"),
                cfg=cfg,
                resume=False,
                dry_run=False,
                quiet=True,
            )
        )
        await b_claimed.wait()
        await asyncio.sleep(0.05)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    documents = _load_state(tmp_path / "out")["documents"]
    assert documents["b.txt"]["target"] == str(tmp_path / "out" / "b.txt.md")

    monkeypatch.chdir(tmp_path / "other")
    await process_batch(
        input_dir=Path("../in"),
        output_dir=Path("../out"),
        cfg=MarkitaiConfig(),
        resume=True,
        dry_run=False,
        quiet=True,
    )

    assert not (tmp_path / "other" / "out").exists()
    assert sorted(p.name for p in (tmp_path / "out").glob("*.md")) == [
        "a.txt.md",
        "b.txt.md",
    ]
    assert "bravo" in (tmp_path / "out" / "b.txt.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# URL-list batches (`markitai list.urls -o out`) share the state machinery
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, url: str) -> None:
        self.url = url
        self.output_name = None


@pytest.fixture
def url_fakes(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fake fetch + URL document LLM; ``behavior`` steers the LLM per URL."""
    import asyncio
    from unittest.mock import MagicMock

    from markitai.fetch_types import FetchResult

    calls: dict = {"fetch": [], "llm": [], "behavior": {}}

    async def _fetch(url, *args, **kwargs):
        calls["fetch"].append(url)
        return FetchResult(
            content=f"# Page\n\nBody of {url}\n", strategy_used="static", url=url
        )

    async def _document_llm(markdown, url, cfg, output_file, *args, **kwargs):
        calls["llm"].append(url)
        behavior = calls["behavior"].get(url)
        if behavior == "fail":
            raise RuntimeError("provider exploded")
        if behavior == "hang":
            calls["hanging"].set()
            await asyncio.Event().wait()
        output_file.with_suffix(".llm.md").write_text(f"LLM for {url}\n")
        return "", 0.0, {"m": {"requests": 1}}

    monkeypatch.setattr("markitai.fetch.fetch_url", _fetch)
    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_document_llm", _document_llm
    )
    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor",
        MagicMock(return_value=MagicMock()),
    )
    calls["hanging"] = asyncio.Event()
    return calls


def _llm_cfg() -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.cache.enabled = False
    cfg.batch.state_flush_interval_seconds = 3600
    return cfg


ONE = "https://example.com/one"
TWO = "https://example.com/two"


async def _run_url_list(tmp_path: Path, *, resume: bool) -> None:
    from markitai.cli.processors.url import process_url_batch

    source = tmp_path / "list.urls"
    source.write_text(f"{ONE}\n{TWO}\n")
    await process_url_batch(
        [_Entry(ONE), _Entry(TWO)],
        tmp_path / "out",
        _llm_cfg(),
        dry_run=False,
        verbose=False,
        quiet=True,
        resume=resume,
        source_file=source,
    )


@pytest.mark.asyncio
async def test_url_list_resume_redoes_only_failed_urls_in_place(
    tmp_path: Path, url_fakes: dict
) -> None:
    """--resume used to be ignored for URL lists: every URL was fetched and
    sent to the LLM again, and the outputs came back as .v2/.v3 copies."""
    url_fakes["behavior"][TWO] = "fail"
    with pytest.raises(SystemExit):  # partial failure exit code
        await _run_url_list(tmp_path, resume=False)
    out = tmp_path / "out"
    urls = _load_state(out)["urls"]
    assert urls[ONE]["status"] == "completed"
    assert urls[TWO]["status"] == "failed"
    assert urls[TWO]["target"].endswith("two.md")
    assert (out / "two.md").is_file()  # base .md written as LLM fallback

    url_fakes["behavior"].clear()
    url_fakes["fetch"].clear()
    url_fakes["llm"].clear()
    await _run_url_list(tmp_path, resume=True)

    assert url_fakes["fetch"] == [TWO]
    assert url_fakes["llm"] == [TWO]
    assert not list(out.glob("*.v2*"))
    assert (out / "two.llm.md").read_text() == f"LLM for {TWO}\n"
    urls = _load_state(out)["urls"]
    assert urls[TWO]["status"] == "completed"
    assert Path(urls[TWO]["output"]).name == "two.llm.md"


@pytest.mark.asyncio
async def test_url_list_interrupt_saves_state_for_resume(
    tmp_path: Path, url_fakes: dict
) -> None:
    import asyncio

    url_fakes["behavior"][TWO] = "hang"
    run = asyncio.create_task(_run_url_list(tmp_path, resume=False))
    await url_fakes["hanging"].wait()
    for _ in range(100):  # until /one's result is recorded
        if (tmp_path / "out" / "one.llm.md").exists():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    urls = _load_state(tmp_path / "out")["urls"]
    assert urls[ONE]["status"] == "completed"
    assert urls[TWO]["status"] == "in_progress"
    assert urls[TWO]["target"].endswith("two.md")

    url_fakes["behavior"].clear()
    url_fakes["llm"].clear()
    await _run_url_list(tmp_path, resume=True)
    assert url_fakes["llm"] == [TWO]
    assert sorted(p.name for p in (tmp_path / "out").glob("*.md")) == [
        "one.llm.md",
        "two.llm.md",
    ]


class _NamedEntry:
    def __init__(self, url: str, output_name: str | None) -> None:
        self.url = url
        self.output_name = output_name


@pytest.mark.asyncio
async def test_url_list_same_url_under_two_names_resumes_each_line(
    tmp_path: Path, url_fakes: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two lines, one URL, two output names: two resumable items.

    Both lines used to share one state keyed by the URL, so an interrupt
    after the first finished let --resume report the list done while the
    second output was never written.
    """
    import asyncio

    from markitai.cli.processors.url import process_url_batch

    hang = {"second": True}
    llm_calls: list[str] = []

    async def _document_llm(markdown, url, cfg, output_file, *args, **kwargs):
        llm_calls.append(output_file.stem)
        if output_file.stem == "second" and hang["second"]:
            url_fakes["hanging"].set()
            await asyncio.Event().wait()
        output_file.with_suffix(".llm.md").write_text(f"LLM {output_file.stem}\n")
        return "", 0.0, {"m": {"requests": 1}}

    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_document_llm", _document_llm
    )
    source = tmp_path / "list.urls"
    source.write_text(f"{ONE} first\n{ONE} second\n")
    out = tmp_path / "out"

    async def _run(resume: bool) -> None:
        await process_url_batch(
            [_NamedEntry(ONE, "first"), _NamedEntry(ONE, "second")],
            out,
            _llm_cfg(),
            dry_run=False,
            verbose=False,
            quiet=True,
            resume=resume,
            source_file=source,
            concurrency=1,
        )

    run = asyncio.create_task(_run(resume=False))
    await url_fakes["hanging"].wait()
    await asyncio.sleep(0.05)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    urls = _load_state(out)["urls"]
    assert urls[f"{ONE} first"]["status"] == "completed"
    assert urls[f"{ONE} first"]["url"] == ONE
    assert urls[f"{ONE} second"]["status"] == "in_progress"

    hang["second"] = False
    llm_calls.clear()
    await _run(resume=True)

    assert llm_calls == ["second"]
    assert (out / "second.llm.md").read_text() == "LLM second\n"
    assert sorted(p.name for p in out.glob("*.md")) == [
        "first.llm.md",
        "second.llm.md",
    ]
    assert {u["status"] for u in _load_state(out)["urls"].values()} == {"completed"}


@pytest.mark.asyncio
async def test_url_list_exact_duplicate_line_is_processed_once(
    tmp_path: Path, url_fakes: dict
) -> None:
    """The same URL with the same (or no) name twice is one item, not a .v2."""
    from markitai.cli.processors.url import process_url_batch

    source = tmp_path / "list.urls"
    source.write_text(f"{ONE}\n{ONE}\n")
    await process_url_batch(
        [_Entry(ONE), _Entry(ONE)],
        tmp_path / "out",
        _llm_cfg(),
        dry_run=False,
        verbose=False,
        quiet=True,
        source_file=source,
    )

    assert url_fakes["llm"] == [ONE]
    assert sorted(p.name for p in (tmp_path / "out").glob("*.md")) == ["one.llm.md"]


@pytest.mark.asyncio
async def test_url_list_resume_adopts_a_legacy_bare_url_state(
    tmp_path: Path, url_fakes: dict
) -> None:
    """A state an older markitai keyed by the bare URL still resumes."""
    import json

    from markitai.cli.processors.url import process_url_batch

    source = tmp_path / "list.urls"
    source.write_text(f"{ONE} first\n{TWO}\n")
    entries = [_NamedEntry(ONE, "first"), _NamedEntry(TWO, None)]
    out = tmp_path / "out"

    async def _run(resume: bool) -> None:
        await process_url_batch(
            entries,
            out,
            _llm_cfg(),
            dry_run=False,
            verbose=False,
            quiet=True,
            resume=resume,
            source_file=source,
        )

    await _run(resume=False)
    # Rewrite the state the way it used to be keyed: by the URL alone
    (state_file,) = _state_files(out)
    data = json.loads(state_file.read_text(encoding="utf-8"))
    entry = data["urls"].pop(f"{ONE} first")
    entry.pop("url")
    data["urls"][ONE] = entry
    state_file.write_text(json.dumps(data), encoding="utf-8")
    url_fakes["llm"].clear()

    await _run(resume=True)

    assert url_fakes["llm"] == []  # both were done; nothing is redone
    assert set(_load_state(out)["urls"]) == {f"{ONE} first", TWO}
