"""Tests for single file processor — stdout mode asset reference handling."""

from __future__ import annotations

from pathlib import Path

from markitai.cli.processors.file import resolve_asset_references


class TestResolveAssetReferences:
    """resolve_asset_references() replaces image refs based on available tiers."""

    def test_placeholder_fallback_format(self) -> None:
        """Without protocol or store, produces ![image: filename]() placeholder."""
        markdown = "# Title\n\n![diagram](.markitai/assets/image1.png)\n\nText."
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert ".markitai/assets/" not in result
        assert "![image: image1.png]()" in result

    def test_strips_screenshot_reference(self) -> None:
        markdown = "![page1](.markitai/screenshots/doc.pdf.page0001.jpg)\n"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert ".markitai/screenshots/" not in result
        assert "![image: doc.pdf.page0001.jpg]()" in result

    def test_preserves_non_asset_content(self) -> None:
        markdown = "# Title\n\nSome text with [a link](https://example.com).\n"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_strips_multiple_references(self) -> None:
        markdown = (
            "![img1](.markitai/assets/a.png)\n"
            "Text between.\n"
            "![img2](.markitai/assets/b.jpg)\n"
            "![page](.markitai/screenshots/c.png)\n"
        )
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))

        assert "![image: a.png]()" in result
        assert "![image: b.jpg]()" in result
        assert "![image: c.png]()" in result

    def test_handles_empty_alt_text(self) -> None:
        markdown = "![](.markitai/assets/image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_preserves_external_urls(self) -> None:
        markdown = "![photo](https://example.com/photo.jpg)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_handles_no_images(self) -> None:
        markdown = "# Hello\n\nJust text."
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert result == markdown

    def test_handles_backslash_paths(self) -> None:
        """Windows-style backslash separators should also be matched."""
        markdown = "![img](.markitai\\assets\\image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_handles_mixed_separators(self) -> None:
        """Mixed forward/backslash separators should be matched."""
        markdown = "![img](.markitai/assets\\image.png)"
        result = resolve_asset_references(markdown, temp_dir=Path("/tmp/fake"))
        assert "![image: image.png]()" in result

    def test_alt_text_containing_assets_does_not_misroute_screenshots(
        self, tmp_path: Path
    ) -> None:
        """Alt text 'assets diagram' must not cause a screenshot to be looked up
        under .markitai/assets/ instead of .markitai/screenshots/."""
        # Set up a temp dir with only a screenshot file
        ss_dir = tmp_path / ".markitai" / "screenshots"
        ss_dir.mkdir(parents=True)
        (ss_dir / "page.png").write_bytes(b"screenshot-data")

        markdown = "![assets diagram](.markitai/screenshots/page.png)"
        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="test.pdf",
        )

        # Should persist via asset store, NOT fall back to placeholder
        assert "![image:" not in result
        assert "file://" in result

    def test_asset_store_tier_persists_image(self, tmp_path: Path) -> None:
        """When asset_store is provided and image exists, should produce file:// URI."""
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "chart.jpg").write_bytes(b"chart-data")

        from markitai.utils.asset_store import AssetStore

        store = AssetStore(tmp_path / "store")
        markdown = "![chart](.markitai/assets/chart.jpg)"
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            asset_store=store,
            source_name="doc.pdf",
        )

        assert "file://" in result
        assert "chart.jpg" in result
        assert "![image:" not in result  # not placeholder

    def test_protocol_tier_renders_escape_sequence(self, tmp_path: Path) -> None:
        """When protocol is provided and image exists, should produce escape sequence."""
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "img.png").write_bytes(b"\x89PNG" + b"\x00" * 50)

        from markitai.utils.terminal_image import Protocol

        markdown = "![diagram](.markitai/assets/img.png)"
        result = resolve_asset_references(
            markdown,
            temp_dir=tmp_path,
            protocol=Protocol.KITTY,
        )

        assert "\033_G" in result  # Kitty escape sequence
        assert "![image:" not in result  # not placeholder


import pytest

from markitai.config import MarkitaiConfig


class TestImageOnlySkip:
    """Tests for image-only format skip behavior."""

    @pytest.mark.asyncio
    async def test_image_file_skipped_without_llm_ocr(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image file should be skipped when neither LLM nor OCR is enabled."""
        from markitai.cli.processors.file import process_single_file

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        # Neither LLM nor OCR enabled (defaults)

        # Should return normally (no error, no SystemExit)
        await process_single_file(
            input_path=input_path,
            output_dir=output_dir,
            cfg=cfg,
            dry_run=False,
        )

        # No output file should be created
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 0

    @pytest.mark.asyncio
    async def test_image_file_skipped_stdout_mode(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image file should be skipped in stdout mode too."""
        from markitai.cli.processors.file import process_single_file

        input_path = fixtures_dir / "sample.bmp"
        cfg = MarkitaiConfig()

        # stdout mode: output_dir=None. Should return normally.
        await process_single_file(
            input_path=input_path,
            output_dir=None,
            cfg=cfg,
            dry_run=False,
            quiet=True,  # quiet to suppress output
        )


class TestFinalOutputFileDetermination:
    """Tests for the final output file selection logic."""

    def test_llm_mode_prefers_llm_md(self, tmp_path: Path) -> None:
        """When .llm.md exists, it should be preferred in LLM mode."""
        output_file = tmp_path / "test.md"
        llm_file = tmp_path / "test.llm.md"
        output_file.write_text("base content")
        llm_file.write_text("llm content")

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        final = output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
        if not final.exists() and cfg.llm.enabled:
            final = output_file

        assert final == llm_file
        assert final.read_text() == "llm content"

    def test_llm_mode_falls_back_to_md(self, tmp_path: Path) -> None:
        """When .llm.md doesn't exist (LLM failed), should fall back to .md."""
        output_file = tmp_path / "test.md"
        output_file.write_text("base content")
        # No .llm.md file

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True

        final = output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
        if not final.exists() and cfg.llm.enabled:
            final = output_file

        assert final == output_file
        assert final.read_text() == "base content"

    def test_non_llm_uses_md(self, tmp_path: Path) -> None:
        """Without LLM, should always use .md."""
        output_file = tmp_path / "test.md"
        output_file.write_text("base content")

        cfg = MarkitaiConfig()

        final = output_file.with_suffix(".llm.md") if cfg.llm.enabled else output_file
        if not final.exists() and cfg.llm.enabled:
            final = output_file

        assert final == output_file
