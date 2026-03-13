"""Tests for single file processor — stdout mode asset reference handling."""

from __future__ import annotations

from markitai.cli.processors.file import strip_asset_references


class TestStripAssetReferences:
    """In stdout mode, markdown content is printed to stdout and the temp
    directory is deleted. Any .markitai/assets/ or .markitai/screenshots/
    references become broken. strip_asset_references replaces them with
    descriptive text placeholders.
    """

    def test_strips_asset_image_reference(self) -> None:
        """Image references to .markitai/assets/ should be replaced."""
        markdown = "# Title\n\n![diagram](.markitai/assets/image1.png)\n\nText."
        result = strip_asset_references(markdown)

        assert ".markitai/assets/" not in result
        assert "[image: image1.png]" in result

    def test_strips_screenshot_reference(self) -> None:
        """Image references to .markitai/screenshots/ should be replaced."""
        markdown = "![page1](.markitai/screenshots/doc.pdf.page0001.jpg)\n"
        result = strip_asset_references(markdown)

        assert ".markitai/screenshots/" not in result
        assert "[image: doc.pdf.page0001.jpg]" in result

    def test_preserves_non_asset_content(self) -> None:
        """Regular markdown content without asset refs should be unchanged."""
        markdown = "# Title\n\nSome text with [a link](https://example.com).\n"
        result = strip_asset_references(markdown)

        assert result == markdown

    def test_strips_multiple_references(self) -> None:
        """Multiple asset references in the same document should all be stripped."""
        markdown = (
            "![img1](.markitai/assets/a.png)\n"
            "Text between.\n"
            "![img2](.markitai/assets/b.jpg)\n"
            "![page](.markitai/screenshots/c.png)\n"
        )
        result = strip_asset_references(markdown)

        assert ".markitai/assets/" not in result
        assert ".markitai/screenshots/" not in result
        assert "[image: a.png]" in result
        assert "[image: b.jpg]" in result
        assert "[image: c.png]" in result

    def test_handles_empty_alt_text(self) -> None:
        """Empty alt text should still produce a meaningful placeholder."""
        markdown = "![](.markitai/assets/image.png)"
        result = strip_asset_references(markdown)

        assert ".markitai/assets/" not in result
        assert "[image: image.png]" in result

    def test_preserves_external_urls(self) -> None:
        """External image URLs should not be affected."""
        markdown = "![photo](https://example.com/photo.jpg)"
        result = strip_asset_references(markdown)

        assert result == markdown

    def test_handles_no_images(self) -> None:
        """Pure text markdown should pass through unchanged."""
        markdown = "# Hello\n\nJust text."
        result = strip_asset_references(markdown)

        assert result == markdown

    def test_preserves_alt_text_in_placeholder(self) -> None:
        """When alt text is present, it should be included in the placeholder."""
        markdown = "![Architecture diagram](.markitai/assets/arch.png)"
        result = strip_asset_references(markdown)

        assert "[image: arch.png]" in result


from pathlib import Path

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
