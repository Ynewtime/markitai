"""URL items follow the file pipeline when LLM work fails.

Regressions found in the 1.2.0 pre-release review:
- A URL whose LLM enhancement fell back under ``on_conflict=overwrite``
  kept the previous run's ``.llm.md`` next to the new base ``.md`` (the
  shared cascade and both CLI URL branches); the file pipeline removes it.
- In the CLI URL branches a failed image analysis (alt/desc) counted as a
  failed enhancement: it discarded, or cancelled, the page's own LLM result
  and, under ``llm.on_failure = "fail"``, failed the URL. The file pipeline
  and the cascade keep the enhancement and warn that the alt text was kept.
- The LLM fallback warning was printed twice: once as the user notice and
  again as the item's warning (single URL; the batch summary).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import MarkitaiConfig
from markitai.fetch_types import FetchResult
from markitai.workflow.helpers import image_analysis_error_warning
from markitai.workflow.llm_failure import is_llm_fallback_warning
from markitai.workflow.single import ImageAnalysisResult


def _cfg(
    on_failure: str = "fallback", on_conflict: str = "overwrite"
) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.cache.enabled = False
    cfg.llm.on_failure = on_failure  # type: ignore[assignment]
    cfg.output.on_conflict = on_conflict  # type: ignore[assignment]
    return cfg


class TestCascadeLeavesNoStaleLlmMd:
    URL = "https://example.com/page"

    async def _cascade(self, tmp_path: Path, cfg: MarkitaiConfig) -> Any:
        from markitai.workflow.url import convert_url_cascade

        async def _stage(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("provider timed out")

        return await convert_url_cascade(
            self.URL,
            cfg,
            tmp_path,
            processor=MagicMock(),
            fetch_result=FetchResult(
                content="# Page\n\nBody.\n", strategy_used="static", url=self.URL
            ),
            llm_stage=_stage,
        )

    @pytest.mark.asyncio
    async def test_overwrite_removes_the_previous_runs_llm_md(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "page.llm.md").write_text("OLD ENHANCED", encoding="utf-8")

        result = await self._cascade(tmp_path, _cfg())

        assert result.output_path == tmp_path / "page.md"
        assert not (tmp_path / "page.llm.md").exists()

    @pytest.mark.asyncio
    async def test_fail_removes_it_too(self, tmp_path: Path) -> None:
        from markitai.utils.errors import ConversionError

        (tmp_path / "page.llm.md").write_text("OLD ENHANCED", encoding="utf-8")

        with pytest.raises(ConversionError):
            await self._cascade(tmp_path, _cfg("fail"))
        assert not (tmp_path / "page.llm.md").exists()

    @pytest.mark.asyncio
    async def test_rename_keeps_the_other_outputs_llm_md(self, tmp_path: Path) -> None:
        (tmp_path / "page.md").write_text("OTHER BASE", encoding="utf-8")
        (tmp_path / "page.llm.md").write_text("OTHER ENHANCED", encoding="utf-8")

        result = await self._cascade(tmp_path, _cfg(on_conflict="rename"))

        assert result.output_path is not None
        assert result.output_path.name != "page.md"
        assert (tmp_path / "page.llm.md").read_text(encoding="utf-8") == (
            "OTHER ENHANCED"
        )


class TestBatchUrlCliBranch:
    """The CLI's own URL branch (here: --screenshot-only) in a batch."""

    URL = "https://example.com/docs"

    async def _run(
        self, tmp_path: Path, cfg: MarkitaiConfig, screenshot_llm: Any
    ) -> Any:
        from markitai.cli.processors.batch import create_url_processor
        from markitai.fetch_types import FetchStrategy

        shot = tmp_path / "out" / ".markitai" / "screenshots" / "docs.full.jpg"
        shot.parent.mkdir(parents=True, exist_ok=True)
        shot.write_bytes(b"jpeg")

        async def _fetch(url: str, *_a: Any, **_kw: Any) -> FetchResult:
            return FetchResult(
                content="# Docs\n\nText layer.\n",
                strategy_used="playwright",
                url=url,
                screenshot_path=shot,
            )

        cfg.screenshot.enabled = True
        cfg.screenshot.screenshot_only = True
        with (
            patch("markitai.fetch.fetch_url", _fetch),
            patch(
                "markitai.cli.processors.url.run_url_screenshot_only_llm",
                screenshot_llm,
            ),
            patch(
                "markitai.workflow.helpers.create_llm_processor",
                MagicMock(return_value=MagicMock()),
            ),
        ):
            process_url = create_url_processor(
                cfg=cfg,
                output_dir=tmp_path / "out",
                fetch_strategy=FetchStrategy.PLAYWRIGHT,
                explicit_fetch_strategy=True,
                renderer=MagicMock(),
            )
            return await process_url(self.URL)

    @pytest.mark.asyncio
    async def test_fallback_removes_the_previous_runs_llm_md(
        self, tmp_path: Path
    ) -> None:
        stale = tmp_path / "out" / "docs.llm.md"
        stale.parent.mkdir(parents=True)
        stale.write_text("OLD ENHANCED", encoding="utf-8")

        async def _failing(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("vision provider exploded")

        result, _extra = await self._run(tmp_path, _cfg(), _failing)

        assert result.success, result.error
        assert result.output_path.endswith("docs.md")
        assert not stale.exists()

    @pytest.mark.asyncio
    async def test_image_warnings_reach_the_item(self, tmp_path: Path) -> None:
        warning = image_analysis_error_warning("image model 500")

        async def _enhanced(
            _shot: Path, _url: str, _cfg: Any, output: Path, *_a: Any, **_kw: Any
        ) -> Any:
            output.with_suffix(".llm.md").write_text("enhanced", encoding="utf-8")
            return (
                0.0,
                {},
                ImageAnalysisResult(
                    source_file=self.URL, assets=[], warnings=[warning]
                ),
            )

        result, extra = await self._run(tmp_path, _cfg("fail"), _enhanced)

        assert result.success, result.error
        assert result.output_path.endswith("docs.llm.md")
        assert warning in extra["warnings"]


class TestImageAnalysisFailureIsAWarning:
    URL = "https://example.com/page"

    @pytest.mark.asyncio
    async def test_the_parallel_document_task_is_kept(self, tmp_path: Path) -> None:
        from markitai.cli.processors.url import run_url_llm_with_images

        cfg = _cfg()
        cfg.image.alt_enabled = True
        output = tmp_path / "page.md"

        async def _doc_task() -> tuple[str, float, dict[str, dict[str, Any]]]:
            await asyncio.sleep(0.01)  # the image task fails first
            output.with_suffix(".llm.md").write_text("enhanced", encoding="utf-8")
            return "enhanced", 0.25, {"model": {"requests": 1}}

        with patch(
            "markitai.cli.processors.llm.analyze_images_with_llm",
            AsyncMock(side_effect=RuntimeError("image model 500")),
        ):
            cost, usage, analysis = await run_url_llm_with_images(
                _doc_task,
                downloaded_images=[tmp_path / "a.png"],
                image_context="",
                output_file=output,
                cfg=cfg,
                url=self.URL,
            )

        assert cost == 0.25
        assert usage == {"model": {"requests": 1}}
        assert output.with_suffix(".llm.md").read_text(encoding="utf-8") == "enhanced"
        assert analysis is not None and analysis.assets == []
        assert len(analysis.warnings) == 1
        assert "image model 500" in analysis.warnings[0]
        assert "original alt text was kept" in analysis.warnings[0]


class TestSingleUrlCli:
    URL = "https://example.com/page"

    async def _run(
        self,
        tmp_path: Path,
        cfg: MarkitaiConfig,
        *,
        document: Any,
        images: Any,
    ) -> MagicMock:
        from markitai.cli.processors.url import process_url

        cfg.image.alt_enabled = True
        fetched = FetchResult(
            content="# Page\n\nBody.\n\n![](https://example.com/a.png)\n",
            strategy_used="static",
            url=self.URL,
        )
        download = MagicMock()
        download.updated_markdown = "# Page\n\nBody.\n\n![](.markitai/assets/a.png)\n"
        download.downloaded_paths = [tmp_path / ".markitai" / "assets" / "a.png"]
        download.failed_urls = []

        warn = MagicMock()
        with (
            patch("markitai.fetch.fetch_url", AsyncMock(return_value=fetched)),
            patch(
                "markitai.image.download_url_images",
                AsyncMock(return_value=download),
            ),
            patch(
                "markitai.cli.processors.llm.process_with_llm",
                AsyncMock(side_effect=document),
            ),
            patch(
                "markitai.cli.processors.llm.analyze_images_with_llm",
                AsyncMock(side_effect=images),
            ),
            patch("markitai.cli.processors.url.ui.warning", warn),
        ):
            await process_url(
                url=self.URL,
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )
        return warn

    @staticmethod
    def _printed(warn: MagicMock) -> list[str]:
        return [str(call.args[0]) for call in warn.call_args_list]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_failure", ["fallback", "fail"])
    async def test_failed_image_analysis_keeps_the_enhancement(
        self, tmp_path: Path, on_failure: str
    ) -> None:
        async def _document(
            _md: str, _url: str, _cfg: Any, output: Path, **_kw: Any
        ) -> Any:
            output.with_suffix(".llm.md").write_text("enhanced", encoding="utf-8")
            return "enhanced", 0.0, {}

        warn = await self._run(
            tmp_path,
            _cfg(on_failure),
            document=_document,
            images=RuntimeError("image model 500"),
        )

        assert (tmp_path / "page.llm.md").read_text(encoding="utf-8") == "enhanced"
        printed = self._printed(warn)
        assert any("image model 500" in w and "alt text" in w for w in printed)
        assert not any(is_llm_fallback_warning(w) for w in printed)

    @pytest.mark.asyncio
    async def test_fallback_removes_the_previous_runs_llm_md_and_warns_once(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "page.llm.md").write_text("OLD ENHANCED", encoding="utf-8")

        async def _images(*_a: Any, **_kw: Any) -> Any:
            await asyncio.Future()  # cancelled when the document fails

        warn = await self._run(
            tmp_path,
            _cfg(),
            document=RuntimeError("invalid api key"),
            images=_images,
        )

        assert (tmp_path / "page.md").is_file()
        assert not (tmp_path / "page.llm.md").exists()
        # The fallback warning went out as a user notice; not printed again
        assert not any(is_llm_fallback_warning(w) for w in self._printed(warn))


def test_is_llm_fallback_warning_matches_only_fallback_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from markitai.workflow import llm_failure

    monkeypatch.setattr(llm_failure, "user_notice", lambda *_a: None)
    warning = llm_failure.llm_fallback_warning("a.pdf", "LLM processing failed: boom")

    assert is_llm_fallback_warning(warning)
    assert not is_llm_fallback_warning(image_analysis_error_warning("boom"))
    assert not is_llm_fallback_warning("Screenshot not captured: timeout")


def test_batch_summary_skips_notices_an_item_already_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from markitai.batch import BatchProcessor, BatchState
    from markitai.config import BatchConfig

    processor = BatchProcessor(BatchConfig(), tmp_path)
    processor.state = BatchState(
        started_at="2026-09-27T10:00:00Z",
        input_dir=str(tmp_path),
        output_dir=str(tmp_path),
    )
    printed = "LLM enhancement failed (invalid api key); kept the unenhanced output"
    processor.announced_warnings.add(printed)
    processor._notice_collector.notices.extend(
        [
            f"a.pdf: {printed}",
            "b.pdf: pages 2-3 look scanned; re-run with --ocr",
        ]
    )

    processor.print_summary()

    err = capsys.readouterr().err
    assert "a.pdf: LLM enhancement failed" not in err
    assert "b.pdf: pages 2-3 look scanned" in err
