"""The batch URL worker (``create_url_processor``) that both URL batches share.

Covers the directory batch (``.urls`` files found in a directory) keeping the
same semantics as the URL-list batch and the single-URL path:

- ``--screenshot-only`` is honored (it used to be ignored in directories:
  a .md from the text layer was written, and ``--llm`` read the text).
- An LLM failure in the CLI branches (screenshot-only, vision, alt/desc)
  fails the URL *and* leaves the base .md as the fallback output.
- Relative images resolve against the post-redirect URL.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from markitai.config import MarkitaiConfig
from markitai.fetch_types import FetchResult

URL = "https://example.com/docs"
FINAL_URL = "https://example.com/docs/"


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Fake fetch returning a page with a screenshot; knobs in the dict."""
    shot = tmp_path / "out" / ".markitai" / "screenshots" / "docs.full.jpg"
    shot.parent.mkdir(parents=True)
    shot.write_bytes(b"jpeg")
    state: dict[str, Any] = {"screenshot": shot, "static_content": None}

    async def _fetch(url: str, *args: Any, **kwargs: Any) -> FetchResult:
        return FetchResult(
            content="# Docs\n\n![diagram](img/d.png)\n\nText layer.\n",
            strategy_used="playwright",
            url=url,
            final_url=FINAL_URL,
            screenshot_path=state["screenshot"],
            static_content=state["static_content"],
        )

    monkeypatch.setattr("markitai.fetch.fetch_url", _fetch)
    # Import the processors before patching: they bind create_llm_processor
    # at import time, and a first import under the patch would keep the
    # MagicMock for every later test in the session
    for module in ("batch", "llm", "url"):
        importlib.import_module(f"markitai.cli.processors.{module}")

    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor",
        MagicMock(return_value=MagicMock()),
    )
    return state


def _cfg(*, llm: bool, screenshot_only: bool = False) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.llm.enabled = llm
    cfg.screenshot.enabled = True
    cfg.screenshot.screenshot_only = screenshot_only
    return cfg


async def _run_directory_worker(cfg: MarkitaiConfig, out: Path) -> Any:
    """Build the worker exactly as the directory batch does (defaults)."""
    from markitai.cli.processors.batch import create_url_processor
    from markitai.fetch_types import FetchStrategy

    process_url = create_url_processor(
        cfg=cfg,
        output_dir=out,
        fetch_strategy=FetchStrategy.PLAYWRIGHT,
        explicit_fetch_strategy=True,
        renderer=MagicMock(),
    )
    result, _extra = await process_url(URL)
    return result


@pytest.mark.asyncio
async def test_directory_batch_screenshot_only_writes_no_markdown(
    tmp_path: Path, fetched: dict[str, Any]
) -> None:
    out = tmp_path / "out"
    result = await _run_directory_worker(_cfg(llm=False, screenshot_only=True), out)

    assert result.success
    assert result.output_path == str(fetched["screenshot"])
    assert not list(out.glob("*.md"))


@pytest.mark.asyncio
async def test_directory_batch_screenshot_only_llm_reads_the_screenshot(
    tmp_path: Path, fetched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def _screenshot_llm(screenshot_path, url, cfg, output_file, *a, **kw):
        calls.append("screenshot")
        output_file.with_suffix(".llm.md").write_text("from the screenshot\n")
        return 0.25, {"m": {"requests": 1, "cost_usd": 0.25}}, None

    async def _document_llm(*args: Any, **kwargs: Any) -> Any:
        calls.append("text")
        raise AssertionError("--screenshot-only must not send the text layer")

    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_screenshot_only_llm", _screenshot_llm
    )
    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_document_llm", _document_llm
    )
    out = tmp_path / "out"
    result = await _run_directory_worker(_cfg(llm=True, screenshot_only=True), out)

    assert result.success, result.error
    assert calls == ["screenshot"]
    assert result.cost_usd == 0.25
    assert Path(result.output_path).read_text() == "from the screenshot\n"


@pytest.mark.asyncio
async def test_screenshot_only_llm_failure_fails_the_url_with_a_base_md(
    tmp_path: Path, fetched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _screenshot_llm(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("vision provider exploded")

    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_screenshot_only_llm", _screenshot_llm
    )
    out = tmp_path / "out"
    result = await _run_directory_worker(_cfg(llm=True, screenshot_only=True), out)

    assert not result.success
    assert "LLM processing failed" in (result.error or "")
    assert "vision provider exploded" in (result.error or "")
    base = (out / "docs.md").read_text()
    # The fallback references the screenshot the page is read from
    assert "screenshots/docs.full.jpg" in base
    assert not (out / "docs.llm.md").exists()


@pytest.mark.asyncio
async def test_vision_llm_failure_fails_the_url_with_a_base_md(
    tmp_path: Path, fetched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched["static_content"] = "static text"  # multi-source: vision branch

    async def _vision(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("vision and fallback both failed")

    monkeypatch.setattr("markitai.cli.processors.url.process_url_with_vision", _vision)
    out = tmp_path / "out"
    result = await _run_directory_worker(_cfg(llm=True), out)

    assert not result.success
    assert "LLM processing failed" in (result.error or "")
    assert "Text layer." in (out / "docs.md").read_text()


@pytest.mark.asyncio
async def test_image_analysis_branch_failure_writes_base_md_and_images_use_final_url(
    tmp_path: Path, fetched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    image = out / ".markitai" / "assets" / "docs.0001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    base_urls: list[str] = []

    async def _download(markdown: str, output_dir: Path, base_url: str, **kw: Any):
        base_urls.append(base_url)
        result = MagicMock()
        result.updated_markdown = markdown.replace(
            "img/d.png", ".markitai/assets/docs.0001.png"
        )
        result.downloaded_paths = [image]
        result.failed_urls = []
        return result

    async def _with_images(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("image analysis exploded")

    monkeypatch.setattr("markitai.image.download_url_images", _download)
    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_llm_with_images", _with_images
    )
    cfg = _cfg(llm=True)
    cfg.screenshot.enabled = False
    fetched["screenshot"] = None
    cfg.image.alt_enabled = True

    result = await _run_directory_worker(cfg, out)

    # Relative image paths resolve against where the page ended up
    assert base_urls == [FINAL_URL]
    assert not result.success
    assert "image analysis exploded" in (result.error or "")
    assert "Text layer." in (out / "docs.md").read_text()


@pytest.mark.asyncio
async def test_pure_llm_screenshot_only_reads_the_text_layer_like_a_single_url(
    tmp_path: Path, fetched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--llm --pure sends the text layer, so a missing capture is no failure.

    The single-URL path already read it this way; the batch failed the URL
    with "--screenshot-only: Screenshot not captured".
    """
    fetched["screenshot"] = None
    sent: list[str] = []

    async def _process_with_llm(markdown, _source, _cfg, output_file, *a, **kw):
        sent.append(markdown)
        output_file.with_suffix(".llm.md").write_text("# Cleaned\n")
        return "# Cleaned\n", 0.0, {}

    monkeypatch.setattr(
        "markitai.cli.processors.llm.process_with_llm", _process_with_llm
    )
    cfg = _cfg(llm=True, screenshot_only=True)
    cfg.llm.pure = True
    out = tmp_path / "out"
    result = await _run_directory_worker(cfg, out)

    assert result.success, result.error
    assert len(sent) == 1 and "Text layer." in sent[0]


async def test_renamed_output_names_the_downloaded_images_after_itself(
    tmp_path: Path, fetched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Images are named after the claimed output, not the default name.

    The images were downloaded (as ``docs.0001.png``) before the output
    name was claimed, so when ``docs.md`` already belonged to another file
    and the page went to ``docs.v2.md``, its images still overwrote the
    other page's ``docs.*`` assets.
    """
    from markitai.utils.output import OutputNameReservations, output_claim_scope

    out = tmp_path / "out"
    (out / "docs.md").write_text("someone else's page")
    image = out / ".markitai" / "assets" / "docs.v2.0001.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    source_names: list[str] = []

    async def _download(markdown: str, output_dir: Path, base_url: str, **kw: Any):
        source_names.append(kw["source_name"])
        result = MagicMock()
        result.updated_markdown = markdown
        result.downloaded_paths = [image]
        result.failed_urls = []
        return result

    async def _with_images(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("image analysis exploded")

    monkeypatch.setattr("markitai.image.download_url_images", _download)
    monkeypatch.setattr(
        "markitai.cli.processors.url.run_url_llm_with_images", _with_images
    )
    cfg = _cfg(llm=True)
    cfg.screenshot.enabled = False
    fetched["screenshot"] = None
    cfg.image.alt_enabled = True

    with output_claim_scope(OutputNameReservations()):
        result = await _run_directory_worker(cfg, out)

    assert source_names == ["docs.v2"]
    assert not result.success  # the faked image analysis failed
    assert (out / "docs.md").read_text() == "someone else's page"
    assert "Text layer." in (out / "docs.v2.md").read_text()


@pytest.mark.asyncio
async def test_directory_batch_reports_a_screenshot_that_was_not_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A .urls entry in a directory batch keeps its screenshot warning.

    The worker put it in ``extra_info["warnings"]``, but the directory batch
    dropped it: nothing on stderr, and ``--json`` showed ``warnings: []``,
    while the same URL in a ``.urls`` list run warned on both.
    """
    from unittest.mock import AsyncMock

    from markitai.cli import ui
    from markitai.cli.processors.batch import process_batch
    from markitai.runs import Outcome
    from markitai.runs.json_output import build_envelope

    async def _fetch(url: str, *args: Any, **kwargs: Any) -> FetchResult:
        return FetchResult(
            content="# Docs\n\nText layer.\n",
            strategy_used="static",
            url=url,
            metadata={"screenshot_error": "Playwright is not installed"},
        )

    monkeypatch.setattr("markitai.fetch.fetch_url", _fetch)
    monkeypatch.setattr(
        "markitai.fetch._get_playwright_renderer", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "markitai.cli.processors.validators.check_playwright_for_urls",
        MagicMock(),
    )
    printed: list[str] = []

    def _warning(text: str, **_kwargs: Any) -> None:
        printed.append(text)

    monkeypatch.setattr(ui, "warning", _warning)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "list.urls").write_text(f"{URL}\n")
    cfg = _cfg(llm=False)
    history: list[Outcome] = []

    await process_batch(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        cfg=cfg,
        resume=False,
        dry_run=False,
        quiet=False,
        history=history,
    )

    expected = "Screenshot not captured: Playwright is not installed"
    assert any(expected in line for line in printed), printed
    (outcome,) = history
    assert outcome.status == "completed"
    assert outcome.warnings == [expected]
    (item,) = build_envelope(history)["items"]
    assert item["warnings"] == [expected]


@pytest.mark.asyncio
async def test_directory_batch_reports_a_file_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file's non-fatal warning (a failed image analysis) reaches --json.

    ``ProcessResult.warnings`` was dropped by the directory batch, as the
    URL warnings above were.
    """
    from markitai.batch import ProcessResult
    from markitai.cli import ui
    from markitai.cli.processors.batch import process_batch
    from markitai.runs import Outcome
    from markitai.runs.json_output import build_envelope

    warning = "chart.png: image analysis failed; kept the original alt text"

    def _create_process_file(cfg, input_dir, output_dir, shared_processor):
        async def _process(file_path: Path) -> ProcessResult:
            out = output_dir / f"{file_path.name}.md"
            out.write_text("# Note\n")
            return ProcessResult(success=True, output_path=str(out), warnings=[warning])

        return _process

    monkeypatch.setattr(
        "markitai.cli.processors.batch.create_process_file", _create_process_file
    )
    printed: list[str] = []
    monkeypatch.setattr(ui, "warning", lambda text, **_kw: printed.append(text))

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "note.txt").write_text("hello")
    (tmp_path / "out").mkdir()
    history: list[Outcome] = []

    await process_batch(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        cfg=_cfg(llm=False),
        resume=False,
        dry_run=False,
        quiet=False,
        history=history,
    )

    assert any(warning in line for line in printed), printed
    (outcome,) = history
    assert outcome.status == "completed"
    assert outcome.warnings == [warning]
    (item,) = build_envelope(history)["items"]
    assert item["warnings"] == [warning]
