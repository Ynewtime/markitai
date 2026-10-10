"""A resumed batch reports what the whole batch cost, not just its last run.

The state file kept only each item's status and output, so after Ctrl-C and
``--resume`` the report listed the items the first run finished with
``cost_usd: 0.0``, no ``llm_usage`` and null timestamps, and the summary's
cost and request count covered only the resumed run. The state now keeps
each completed item's cost, usage and timestamps; the report and the summary
show the whole batch's spend with this run's part, and a state written
before (no usage recorded) still resumes, with the earlier spend reported as
unknown instead of zero.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from markitai.batch import (
    BatchProcessor,
    BatchState,
    FileState,
    FileStatus,
    ProcessResult,
    UrlState,
)
from markitai.config import MarkitaiConfig
from markitai.runs.json_output import build_envelope
from markitai.runs.types import Outcome


def _usage(requests: int, cost: float, **extra: Any) -> dict[str, dict[str, Any]]:
    return {
        "gpt-4o-mini": {
            "requests": requests,
            "input_tokens": 1000 * requests,
            "output_tokens": 200 * requests,
            "cost_usd": cost,
            **extra,
        }
    }


def _processor(tmp_path: Path, *, llm: bool = True) -> BatchProcessor:
    cfg = MarkitaiConfig()
    processor = BatchProcessor(
        cfg.batch,
        tmp_path / "out",
        input_path=tmp_path / "in",
        task_options={"llm": llm},
    )
    input_dir = str((tmp_path / "in").resolve())
    output_dir = str((tmp_path / "out").resolve())
    processor.state = BatchState(
        input_dir=input_dir,
        output_dir=output_dir,
        options={"llm": llm, "input_dir": input_dir, "output_dir": output_dir},
    )
    return processor


def _key(tmp_path: Path, name: str) -> str:
    return str((tmp_path / "in" / name).resolve())


def _completed_file(tmp_path: Path, name: str, cost: float) -> FileState:
    return FileState(
        path=_key(tmp_path, name),
        status=FileStatus.COMPLETED,
        output=str(tmp_path / "out" / f"{name}.llm.md"),
        started_at="2026-10-10T10:00:00+08:00",
        completed_at="2026-10-10T10:00:04+08:00",
        duration=4.0,
        images=2,
        screenshots=1,
        cost_usd=cost,
        llm_usage=_usage(1, cost, cached_input_tokens=0),
    )


# ---------------------------------------------------------------------------
# State file
# ---------------------------------------------------------------------------


def test_state_keeps_completed_items_cost_usage_and_timestamps(
    tmp_path: Path,
) -> None:
    processor = _processor(tmp_path)
    assert processor.state is not None
    done = _completed_file(tmp_path, "a.pdf", 0.03)
    done.skip_reason = None
    processor.state.files[done.path] = done
    url = UrlState(
        url="https://example.com/",
        source_file=str(tmp_path / "in" / "two.urls"),
        status=FileStatus.COMPLETED,
        output=str(tmp_path / "out" / "example.com.llm.md"),
        fetch_strategy="static",
        started_at="2026-10-10T10:00:01+08:00",
        completed_at="2026-10-10T10:00:02+08:00",
        duration=1.0,
        cost_usd=0.01,
        llm_usage=_usage(1, 0.01),
        cache_hit=False,
    )
    processor.state.urls[url.url] = url
    pending = FileState(path=_key(tmp_path, "b.pdf"))
    processor.state.files[pending.path] = pending
    processor.save_state(force=True)

    resumed = processor.load_state()

    assert resumed is not None
    a = resumed.files[_key(tmp_path, "a.pdf")]
    assert (a.cost_usd, a.llm_usage) == (0.03, done.llm_usage)
    assert (a.started_at, a.completed_at, a.duration) == (
        done.started_at,
        done.completed_at,
        4.0,
    )
    assert (a.images, a.screenshots) == (2, 1)
    assert a.earlier_run and not a.usage_unknown
    u = resumed.urls["https://example.com/"]
    assert (u.cost_usd, u.llm_usage, u.fetch_strategy) == (
        0.01,
        url.llm_usage,
        "static",
    )
    assert u.earlier_run and not u.usage_unknown
    b = resumed.files[_key(tmp_path, "b.pdf")]
    assert not b.earlier_run and b.cost_usd == 0.0


def test_sidecar_entries_keep_the_accounting_too(tmp_path: Path) -> None:
    """Items finished since the last full save live only in the .jsonl."""
    processor = _processor(tmp_path)
    assert processor.state is not None
    key = _key(tmp_path, "a.pdf")
    processor.state.files[key] = FileState(path=key)
    processor.save_state(force=True)  # base state: a.pdf pending

    processor.state.files[key] = _completed_file(tmp_path, "a.pdf", 0.03)
    processor._append_incremental({key})  # what a throttled save appends

    resumed = processor.load_state()
    assert resumed is not None
    a = resumed.files[key]
    assert a.status == FileStatus.COMPLETED
    assert (a.cost_usd, a.duration, a.started_at) == (
        0.03,
        4.0,
        "2026-10-10T10:00:00+08:00",
    )
    assert a.earlier_run and not a.usage_unknown


def _legacy_state(processor: BatchProcessor, *, llm: bool) -> None:
    """A state as markitai wrote it before: status and output only."""
    assert processor.state is not None
    processor.state_file.parent.mkdir(parents=True, exist_ok=True)
    processor.state_file.write_text(
        json.dumps(
            {
                "version": "1.0",
                "options": {**processor.state.options, "llm": llm},
                "documents": {
                    "a.pdf": {"status": "completed", "output": "/x/a.pdf.llm.md"},
                    "b.pdf": {"status": "in_progress", "target": "/x/b.pdf.md"},
                },
                "urls": {},
            }
        ),
        encoding="utf-8",
    )


def test_a_state_without_usage_resumes_with_the_earlier_cost_unknown(
    tmp_path: Path,
) -> None:
    processor = _processor(tmp_path)
    _legacy_state(processor, llm=True)

    resumed = processor.load_state()

    assert resumed is not None
    a = resumed.files[_key(tmp_path, "a.pdf")]
    assert a.status == FileStatus.COMPLETED and a.earlier_run and a.usage_unknown
    assert resumed.files[_key(tmp_path, "b.pdf")].status == FileStatus.FAILED
    processor.state = resumed
    report = processor.generate_report()
    assert report["documents"]["a.pdf"]["cost_usd"] is None
    assert report["documents"]["a.pdf"]["llm_usage"] is None
    assert report["llm_usage"]["usage_unknown_items"] == 1

    # Saved again, it stays unknown for the next --resume (never 0.0)
    processor.save_state(force=True)
    saved = json.loads(processor.state_file.read_text(encoding="utf-8"))
    assert "cost_usd" not in saved["documents"]["a.pdf"]
    again = processor.load_state()
    assert again is not None and again.files[_key(tmp_path, "a.pdf")].usage_unknown


def test_a_state_without_usage_of_a_batch_without_llm_cost_nothing(
    tmp_path: Path,
) -> None:
    processor = _processor(tmp_path, llm=False)
    _legacy_state(processor, llm=False)

    resumed = processor.load_state()

    assert resumed is not None
    assert not resumed.files[_key(tmp_path, "a.pdf")].usage_unknown
    processor.state = resumed
    assert "usage_unknown_items" not in processor.generate_report()["llm_usage"]


# ---------------------------------------------------------------------------
# Report and summary
# ---------------------------------------------------------------------------


def _resumed_processor(tmp_path: Path) -> BatchProcessor:
    """a.pdf ($0.03) done by an earlier run, b.pdf ($0.05) by this one."""
    processor = _processor(tmp_path)
    assert processor.state is not None
    earlier = _completed_file(tmp_path, "a.pdf", 0.03)
    earlier.earlier_run = True
    processor.state.files[earlier.path] = earlier
    current = _completed_file(tmp_path, "b.pdf", 0.05)
    processor.state.files[current.path] = current
    return processor


def test_report_covers_the_whole_batch_with_this_runs_part(tmp_path: Path) -> None:
    report = _resumed_processor(tmp_path).generate_report()

    usage = report["llm_usage"]
    assert usage["requests"] == 2
    assert usage["cost_usd"] == pytest.approx(0.08)
    assert usage["models"]["gpt-4o-mini"]["requests"] == 2
    assert usage["this_run"]["requests"] == 1
    assert usage["this_run"]["cost_usd"] == pytest.approx(0.05)
    assert "usage_unknown_items" not in usage
    a, b = report["documents"]["a.pdf"], report["documents"]["b.pdf"]
    assert a["earlier_run"] is True and a["cost_usd"] == 0.03
    assert a["started_at"] == "2026-10-10T10:00:00+08:00"
    assert "earlier_run" not in b


def test_report_of_a_batch_that_was_not_resumed_is_unchanged(tmp_path: Path) -> None:
    processor = _processor(tmp_path)
    assert processor.state is not None
    done = _completed_file(tmp_path, "a.pdf", 0.03)
    processor.state.files[done.path] = done

    usage = processor.generate_report()["llm_usage"]

    assert set(usage) == {
        "models",
        "requests",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    }


@pytest.mark.parametrize(
    ("earlier", "current", "expected"),
    [
        # Not resumed: as before
        (None, {"cost": 0.12}, ", $0.120"),
        (None, {"cost": 0.0}, ""),
        # Resumed: the whole batch, with this run's part
        ({"cost": 0.08}, {"cost": 0.12}, ", $0.200 total, $0.120 this run"),
        ({"cost": 0.08}, {"cost": 0.0}, ", $0.080 total, $0.000 this run"),
        ({"cost": 0.0}, {"cost": 0.0}, ""),
        # A state without usage, or an unpriced model earlier: said plainly
        (
            {"cost": 0.0, "unknown": True},
            {"cost": 0.12},
            ", $0.120 this run, cost of earlier runs unknown",
        ),
        (
            {"cost": 0.08, "unpriced": True},
            {"cost": 0.12},
            ", $0.120 this run, cost of earlier runs unknown",
        ),
        # An unpriced model in this run: no amount is right
        ({"cost": 0.08}, {"cost": 0.12, "unpriced": True}, ", cost unknown"),
    ],
)
def test_summary_cost_text(
    tmp_path: Path,
    earlier: dict[str, Any] | None,
    current: dict[str, Any],
    expected: str,
) -> None:
    processor = _processor(tmp_path)
    assert processor.state is not None

    def add(name: str, spec: dict[str, Any], *, earlier_run: bool) -> None:
        item = _completed_file(tmp_path, name, spec["cost"])
        if spec.get("unpriced"):
            item.llm_usage["gpt-4o-mini"]["unpriced_requests"] = 1
        if spec.get("unknown"):
            item.cost_usd, item.llm_usage, item.usage_unknown = 0.0, {}, True
        item.earlier_run = earlier_run
        processor.state.files[item.path] = item  # type: ignore[union-attr]

    if earlier is not None:
        add("a.pdf", earlier, earlier_run=True)
    add("b.pdf", current, earlier_run=False)

    assert processor._cost_text() == expected


# ---------------------------------------------------------------------------
# --json envelope
# ---------------------------------------------------------------------------


def test_json_lists_earlier_items_and_totals_the_whole_batch() -> None:
    envelope = build_envelope(
        [
            Outcome(
                kind="file",
                source="a.pdf",
                status="completed",
                cost_usd=0.03,
                llm_usage=_usage(1, 0.03),
                earlier_run=True,
            ),
            Outcome(
                kind="url",
                source="https://example.com/",
                status="completed",
                earlier_run=True,
                usage_unknown=True,
            ),
            Outcome(
                kind="file",
                source="b.pdf",
                status="completed",
                cost_usd=0.05,
                llm_usage=_usage(1, 0.05),
            ),
        ]
    )

    a, url, b = envelope["items"]
    assert (a["earlier_run"], a["cost_usd"]) == (True, 0.03)
    assert (url["earlier_run"], url["cost_usd"], url["llm_usage"]) == (
        True,
        None,
        None,
    )
    assert (b["earlier_run"], b["cost_usd"]) == (False, 0.05)
    assert envelope["totals"]["cost_usd"] == pytest.approx(0.08)
    assert envelope["totals"]["this_run_cost_usd"] == pytest.approx(0.05)
    assert envelope["ok"] is True


# ---------------------------------------------------------------------------
# The CLI batch, interrupted and resumed
# ---------------------------------------------------------------------------


def _report(output_dir: Path) -> dict[str, Any]:
    """The newest report (each run of a URL list writes its own)."""
    reports = (output_dir / ".markitai" / "reports").glob("*.json")
    newest = max(reports, key=lambda path: path.stat().st_mtime_ns)
    return json.loads(newest.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_resumed_directory_batch_reports_the_whole_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.cli.processors import batch as batch_module
    from markitai.cli.processors.batch import process_batch

    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "a.txt").write_text("a")
    (input_dir / "b.txt").write_text("b")
    output_dir = tmp_path / "out"
    b_started = asyncio.Event()
    hang_b = {"value": True}
    costs = {"a.txt": 0.03, "b.txt": 0.05}

    def _factory(cfg, input_dir, output_dir, shared_processor):
        async def _process(path: Path) -> ProcessResult:
            if path.name == "b.txt" and hang_b["value"]:
                b_started.set()
                await asyncio.Event().wait()  # an LLM call cut by Ctrl-C
            output = output_dir / f"{path.name}.llm.md"
            output.write_text(path.name)
            cost = costs[path.name]
            return ProcessResult(
                success=True,
                output_path=str(output),
                cost_usd=cost,
                llm_usage=_usage(1, cost),
            )

        return _process

    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.batch.concurrency = 1
    cfg.batch.state_flush_interval_seconds = 3600
    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        batch_module, "create_llm_processor", MagicMock(return_value=MagicMock())
    )
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
        run.cancel()  # what asyncio.run does on Ctrl-C
        with pytest.raises(asyncio.CancelledError):
            await run

        hang_b["value"] = False
        stderr = io.StringIO()
        monkeypatch.setattr(
            "markitai.batch.get_stderr_console",
            lambda: Console(file=stderr, width=200, force_terminal=False),
        )
        history: list[Outcome] = []
        earlier: list[Outcome] = []
        await process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            cfg=cfg,
            resume=True,
            dry_run=False,
            history=history,
            earlier=earlier,
        )

    report = _report(output_dir)
    usage = report["llm_usage"]
    assert usage["requests"] == 2
    assert usage["cost_usd"] == pytest.approx(0.08)
    assert usage["this_run"]["cost_usd"] == pytest.approx(0.05)
    a = report["documents"]["a.txt"]
    assert a["earlier_run"] is True
    assert a["cost_usd"] == pytest.approx(0.03)
    assert a["llm_usage"]["gpt-4o-mini"]["requests"] == 1
    assert a["started_at"] and a["completed_at"] and a["duration"]
    assert re.search(
        r"Done: 2 files \(\d+:\d\d, \$0\.080 total, \$0\.050 this run\)",
        stderr.getvalue(),
    ), stderr.getvalue()
    # --json lists the earlier item; history records only this run's
    assert [(o.source, o.earlier_run, o.cost_usd) for o in earlier] == [
        ("a.txt", True, 0.03)
    ]
    assert [o.source for o in history] == ["b.txt"]


@pytest.mark.asyncio
async def test_resumed_url_list_reports_the_whole_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.cli.processors.url import process_url_batch
    from markitai.fetch_types import FetchResult

    one, two = "https://example.com/one", "https://example.com/two"
    behavior: dict[str, str] = {two: "fail"}
    costs = {one: 0.03, two: 0.05}

    async def _fetch(url, *args, **kwargs):
        return FetchResult(
            content=f"# Page\n\nBody of {url}\n", strategy_used="static", url=url
        )

    async def _document_llm(markdown, url, cfg, output_file, *args, **kwargs):
        if behavior.get(url) == "fail":
            raise RuntimeError("provider exploded")
        output_file.with_suffix(".llm.md").write_text(f"LLM for {url}\n")
        return "", costs[url], _usage(1, costs[url])

    monkeypatch.setattr("markitai.fetch.fetch_url", _fetch)
    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_document_llm", _document_llm
    )
    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor",
        MagicMock(return_value=MagicMock()),
    )

    class _Entry:
        def __init__(self, url: str) -> None:
            self.url, self.output_name = url, None

    source = tmp_path / "list.urls"
    source.write_text(f"{one}\n{two}\n")
    out = tmp_path / "out"
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.llm.on_failure = "fail"
    cfg.cache.enabled = False

    async def _run(resume: bool, earlier: list[Outcome] | None = None) -> None:
        await process_url_batch(
            [_Entry(one), _Entry(two)],
            out,
            cfg,
            dry_run=False,
            verbose=False,
            quiet=True,
            resume=resume,
            source_file=source,
            earlier=earlier,
        )

    with pytest.raises(SystemExit):  # /two failed: partial failure
        await _run(resume=False)
    behavior.clear()
    earlier: list[Outcome] = []
    await _run(resume=True, earlier=earlier)

    report = _report(out)
    usage = report["llm_usage"]
    assert usage["cost_usd"] == pytest.approx(0.08)
    assert usage["requests"] == 2
    assert usage["this_run"]["cost_usd"] == pytest.approx(0.05)
    (group,) = report["url_sources"].values()
    urls = group["urls"]
    assert urls[one]["earlier_run"] is True
    assert "earlier_run" not in urls[two]
    assert [(o.source, o.cost_usd, o.earlier_run) for o in earlier] == [
        (one, 0.03, True)
    ]
