"""Tests for workflow.url cascade's vision / screenshot-only auto branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from markitai.config import MarkitaiConfig
from markitai.fetch_types import FetchResult
from markitai.workflow.url import (
    convert_url_cascade,
    uses_screenshot_only,
    uses_vision_enhancement,
)


def _cfg() -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.cache.enabled = False
    return cfg


def _fetch_result(tmp_path: Path, *, multi_source: bool = True) -> FetchResult:
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"png")
    return FetchResult(
        content="# Page\n\nbody text",
        strategy_used="auto",
        title="Page",
        url="https://example.com/x",
        screenshot_path=shot,
        screenshot_tiles=[shot],
        static_content="# Page static" if multi_source else None,
    )


def _processor() -> MagicMock:
    proc = MagicMock()
    proc.enhance_url_with_vision = AsyncMock(
        return_value=("# Clean", "---\ntitle: Page")
    )
    proc.extract_from_screenshot = AsyncMock(
        return_value=("# Tile", "---\ntitle: Page")
    )
    proc.process_document = AsyncMock(return_value=("# Std", "---\ntitle: Page"))
    proc.clean_document_pure = AsyncMock(return_value="# Pure")
    proc.format_llm_output.side_effect = lambda md, fm: f"{fm}\n---\n\n{md}"
    proc.get_context_cost.return_value = 0.001
    proc.get_context_usage.return_value = {"m": {"requests": 1}}
    return proc


class TestBranchPredicates:
    def test_vision_requires_screenshot_and_multi_source(self, tmp_path: Path) -> None:
        cfg = _cfg()
        assert uses_vision_enhancement(cfg, _fetch_result(tmp_path)) is True
        assert (
            uses_vision_enhancement(cfg, _fetch_result(tmp_path, multi_source=False))
            is False
        )
        no_shot = _fetch_result(tmp_path)
        no_shot.screenshot_path = None
        assert uses_vision_enhancement(cfg, no_shot) is False
        cfg.llm.pure = True
        assert uses_vision_enhancement(cfg, _fetch_result(tmp_path)) is False

    def test_screenshot_only_flag(self, tmp_path: Path) -> None:
        cfg = _cfg()
        assert uses_screenshot_only(cfg, _fetch_result(tmp_path)) is False
        cfg.screenshot.screenshot_only = True
        assert uses_screenshot_only(cfg, _fetch_result(tmp_path)) is True
        cfg.llm.enabled = False
        assert uses_screenshot_only(cfg, _fetch_result(tmp_path)) is False


class TestCascadeBranches:
    async def test_vision_branch_writes_llm_md_with_screenshot_comment(
        self, tmp_path: Path
    ) -> None:
        cfg = _cfg()
        proc = _processor()
        result = await convert_url_cascade(
            "https://example.com/x",
            cfg,
            tmp_path / "out",
            fetch_result=_fetch_result(tmp_path),
            processor=proc,
            llm_error_policy="fallback",
        )
        proc_target = result.llm_output_path
        assert proc_target is not None and proc_target.exists()
        text = proc_target.read_text(encoding="utf-8")
        assert "# Clean" in text
        assert "Screenshot for reference" in text

    async def test_vision_failure_falls_back_to_document_stage(
        self, tmp_path: Path
    ) -> None:
        cfg = _cfg()
        proc = _processor()
        proc.enhance_url_with_vision.side_effect = RuntimeError("vision down")
        result = await convert_url_cascade(
            "https://example.com/x",
            cfg,
            tmp_path / "out",
            fetch_result=_fetch_result(tmp_path),
            processor=proc,
        )
        assert result.llm_error is None  # fallback succeeded
        assert result.llm_output_path is not None
        assert "# Std" in result.llm_output_path.read_text(encoding="utf-8")
        proc.process_document.assert_awaited_once()

    async def test_screenshot_only_reads_tiles(self, tmp_path: Path) -> None:
        cfg = _cfg()
        cfg.screenshot.screenshot_only = True
        cfg.llm.keep_base = True  # the screenshot-reference base needs it
        proc = _processor()
        result = await convert_url_cascade(
            "https://example.com/x",
            cfg,
            tmp_path / "out",
            fetch_result=_fetch_result(tmp_path),
            processor=proc,
        )
        assert result.llm_output_path is not None
        text = result.llm_output_path.read_text(encoding="utf-8")
        assert "# Tile" in text
        proc.extract_from_screenshot.assert_awaited()
        # screenshot-only base references the screenshot, not the text layer
        assert result.output_path is not None
        base = result.output_path.read_text(encoding="utf-8")
        assert "Screenshot" in base and ".markitai/screenshots/shot.png" in base
