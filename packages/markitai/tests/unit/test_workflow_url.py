"""Tests for workflow.url cascade's vision / screenshot-only auto branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

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


class TestCascadeImageAnalysis:
    async def test_screenshot_only_honours_the_vision_page_cap(
        self, tmp_path: Path
    ) -> None:
        """A long page tiles into many vision requests; the page cap bounds them."""
        cfg = _cfg()
        cfg.screenshot.screenshot_only = True
        cfg.llm.keep_base = True
        cfg.llm.max_vision_pages_per_document = 2
        fetch_result = _fetch_result(tmp_path)
        tiles = []
        for i in range(5):
            tile = tmp_path / f"tile{i}.png"
            tile.write_bytes(b"png")
            tiles.append(tile)
        fetch_result.screenshot_tiles = tiles
        proc = _processor()
        result = await convert_url_cascade(
            "https://example.com/x",
            cfg,
            tmp_path / "out",
            fetch_result=fetch_result,
            processor=proc,
        )
        assert proc.extract_from_screenshot.await_count == 2
        assert result.llm_output_path is not None
        text = result.llm_output_path.read_text(encoding="utf-8")
        assert "<!-- Tile 2 -->" in text and "<!-- Tile 3 -->" not in text

    async def test_alt_and_desc_run_after_llm(self, tmp_path: Path) -> None:
        from markitai.llm.types import ImageAnalysis

        cfg = _cfg()
        cfg.image.alt_enabled = True
        cfg.image.desc_enabled = True
        proc = _processor()
        img = tmp_path / "out" / ".markitai" / "assets" / "pic.jpg"
        img.parent.mkdir(parents=True)
        img.write_bytes(b"jpg")
        # 让 fetch 的 markdown 里引用这张图（模拟已下载）
        fetch = _fetch_result(tmp_path, multi_source=False)
        fetch.content = "# Page\n\n![old alt](.markitai/assets/pic.jpg)"
        analysis = ImageAnalysis(
            caption="A nice picture", description="d", llm_usage={}, extracted_text=""
        )
        proc.analyze_images_batch = AsyncMock(return_value=[analysis])
        # 标准 document stage 保留图片引用，供 alt 更新
        proc.process_document = AsyncMock(
            return_value=(
                "# Std\n\n![old alt](.markitai/assets/pic.jpg)",
                "---\ntitle: Page",
            )
        )
        with pytest.MonkeyPatch.context() as mp:
            # patch markitai.image.download_url_images so cascade 不真下载
            mp.setattr(
                "markitai.image.download_url_images",
                AsyncMock(
                    return_value=MagicMock(
                        downloaded_paths=[img], updated_markdown=fetch.content
                    )
                ),
                raising=False,
            )
            result = await convert_url_cascade(
                "https://example.com/x",
                cfg,
                tmp_path / "out",
                fetch_result=fetch,
                processor=proc,
            )
        assert result.llm_output_path is not None
        text = result.llm_output_path.read_text(encoding="utf-8")
        assert "A nice picture" in text  # alt updated in .llm.md
        # images.json written for desc
        images_json = tmp_path / "out" / ".markitai" / "assets" / "images.json"
        assert images_json.exists()
        assert "A nice picture" in images_json.read_text(encoding="utf-8")
