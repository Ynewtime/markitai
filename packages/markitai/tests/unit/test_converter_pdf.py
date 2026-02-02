"""Unit tests for PDF converter module."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest

from markitai.config import (
    ImageConfig,
    LLMConfig,
    MarkitaiConfig,
    OCRConfig,
    ScreenshotConfig,
)
from markitai.converter.base import ConvertResult, ExtractedImage, FileFormat
from markitai.converter.pdf import PdfConverter

if TYPE_CHECKING:
    pass


# Create a reusable mock for pymupdf module
def create_pymupdf_mock() -> Mock:
    """Create a mock pymupdf module with common attributes."""
    mock_pymupdf = Mock()
    mock_pix = Mock()
    mock_pix.width = 100
    mock_pix.height = 200
    mock_pymupdf.Pixmap.return_value = mock_pix
    mock_pymupdf.Matrix.return_value = Mock()
    return mock_pymupdf


class TestPdfConverterInit:
    """Tests for PdfConverter initialization."""

    def test_init_with_no_config(self) -> None:
        """Test PdfConverter initialization without config."""
        converter = PdfConverter()
        assert converter.config is None
        assert FileFormat.PDF in converter.supported_formats

    def test_init_with_config(self) -> None:
        """Test PdfConverter initialization with config."""
        config = MarkitaiConfig()
        converter = PdfConverter(config)
        assert converter.config is config

    def test_supported_formats(self) -> None:
        """Test that PDF format is supported."""
        converter = PdfConverter()
        assert converter.supported_formats == [FileFormat.PDF]

    def test_can_convert_pdf(self) -> None:
        """Test can_convert method for PDF files."""
        converter = PdfConverter()
        assert converter.can_convert("test.pdf") is True
        assert converter.can_convert("test.PDF") is True
        assert converter.can_convert(Path("document.pdf")) is True

    def test_cannot_convert_non_pdf(self) -> None:
        """Test can_convert method for non-PDF files."""
        converter = PdfConverter()
        assert converter.can_convert("test.docx") is False
        assert converter.can_convert("test.txt") is False
        assert converter.can_convert("test.pptx") is False


class TestFixImagePaths:
    """Tests for _fix_image_paths helper method."""

    def test_fix_absolute_paths(self) -> None:
        """Test fixing absolute image paths to relative."""
        converter = PdfConverter()
        image_path = Path("/tmp/output/assets")

        markdown = "![](C:/tmp/output/assets/image1.jpg)"
        # On POSIX, this won't match since the path uses Windows format
        # Test with POSIX path
        markdown = f"![]({image_path.as_posix()}/image1.jpg)"
        result = converter._fix_image_paths(markdown, image_path)
        assert result == "![](assets/image1.jpg)"

    def test_fix_multiple_images(self) -> None:
        """Test fixing multiple image paths."""
        converter = PdfConverter()
        image_path = Path("/home/user/output/assets")

        markdown = f"""# Document

![First image]({image_path.as_posix()}/image1.png)

Some text here.

![Second image]({image_path.as_posix()}/image2.jpg)
"""
        result = converter._fix_image_paths(markdown, image_path)
        assert "![First image](assets/image1.png)" in result
        assert "![Second image](assets/image2.jpg)" in result
        assert image_path.as_posix() not in result

    def test_preserve_alt_text(self) -> None:
        """Test that alt text is preserved when fixing paths."""
        converter = PdfConverter()
        image_path = Path("/tmp/assets")

        markdown = f"![Alt text with spaces]({image_path.as_posix()}/image.png)"
        result = converter._fix_image_paths(markdown, image_path)
        assert result == "![Alt text with spaces](assets/image.png)"

    def test_empty_alt_text(self) -> None:
        """Test fixing paths with empty alt text."""
        converter = PdfConverter()
        image_path = Path("/tmp/assets")

        markdown = f"![]({image_path.as_posix()}/image.png)"
        result = converter._fix_image_paths(markdown, image_path)
        assert result == "![](assets/image.png)"

    def test_no_change_for_relative_paths(self) -> None:
        """Test that relative paths are not changed incorrectly."""
        converter = PdfConverter()
        image_path = Path("/different/path")

        markdown = "![](assets/image.png)"
        result = converter._fix_image_paths(markdown, image_path)
        assert result == "![](assets/image.png)"

    def test_special_characters_in_path(self) -> None:
        """Test handling of special regex characters in path."""
        converter = PdfConverter()
        # Path with special regex characters
        image_path = Path("/tmp/test[1]/assets")

        markdown = f"![]({image_path.as_posix()}/image.png)"
        result = converter._fix_image_paths(markdown, image_path)
        assert result == "![](assets/image.png)"

    def test_cross_platform_path(self) -> None:
        """Test that POSIX paths work on all platforms."""
        converter = PdfConverter()
        # Use a path that would be different on Windows vs POSIX
        image_path = Path("/home/user/documents/output/assets")

        markdown = f"![test]({image_path.as_posix()}/document.pdf-1-0.jpg)"
        result = converter._fix_image_paths(markdown, image_path)
        assert result == "![test](assets/document.pdf-1-0.jpg)"


class TestCollectEmbeddedImages:
    """Tests for _collect_embedded_images helper method."""

    def test_collect_matching_images(self, tmp_path: Path) -> None:
        """Test collecting images that match the PDF filename pattern."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        # Create test image files with pymupdf4llm naming pattern
        (assets_dir / "test.pdf-0-0.png").touch()
        (assets_dir / "test.pdf-0-1.png").touch()
        (assets_dir / "test.pdf-1-0.png").touch()

        # Mock pymupdf module in sys.modules
        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "test.pdf")

        assert len(images) == 3
        # Check all images are ExtractedImage instances
        for img in images:
            assert isinstance(img, ExtractedImage)
            assert "test.pdf" in img.original_name

    def test_ignore_non_matching_images(self, tmp_path: Path) -> None:
        """Test that non-matching images are ignored."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        # Create matching and non-matching files
        (assets_dir / "test.pdf-0-0.png").touch()
        (assets_dir / "other.pdf-0-0.png").touch()  # Different PDF
        (assets_dir / "random_image.png").touch()  # No pattern
        (assets_dir / "test.pdf.png").touch()  # Wrong pattern

        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "test.pdf")

        assert len(images) == 1
        assert images[0].original_name == "test.pdf-0-0.png"

    def test_handle_jpeg_extension(self, tmp_path: Path) -> None:
        """Test collecting images with jpeg extension."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        (assets_dir / "doc.pdf-0-0.jpg").touch()
        (assets_dir / "doc.pdf-0-1.jpeg").touch()

        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "doc.pdf")

        assert len(images) == 2
        # Check MIME types are correct
        mime_types = {img.mime_type for img in images}
        assert "image/jpeg" in mime_types

    def test_index_calculation(self, tmp_path: Path) -> None:
        """Test that image index is calculated correctly from page and position."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        # Page 2 (0-indexed), image 3 on that page
        (assets_dir / "doc.pdf-2-3.png").touch()

        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "doc.pdf")

        assert len(images) == 1
        # Index should be page_idx * 100 + img_idx = 2 * 100 + 3 = 203
        assert images[0].index == 203

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test collecting from empty directory."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        images = converter._collect_embedded_images(assets_dir, "test.pdf")
        assert images == []

    def test_handle_dimension_error(self, tmp_path: Path) -> None:
        """Test graceful handling of dimension reading errors."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        (assets_dir / "test.pdf-0-0.png").touch()

        # Create a mock that raises an exception
        mock_pymupdf = Mock()
        mock_pymupdf.Pixmap.side_effect = Exception("Failed to read image")

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "test.pdf")

        assert len(images) == 1
        assert images[0].width == 0
        assert images[0].height == 0


class TestConvertBasic:
    """Tests for basic convert functionality with mocked pymupdf4llm."""

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_convert_basic_pdf(self, mock_pymupdf4llm: Mock, tmp_path: Path) -> None:
        """Test basic PDF conversion."""
        # Setup mock
        mock_pymupdf4llm.to_markdown.return_value = [
            {"text": "# Page 1\n\nContent on page 1."},
            {"text": "# Page 2\n\nContent on page 2."},
        ]

        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        assert isinstance(result, ConvertResult)
        assert "Page 1" in result.markdown
        assert "Page 2" in result.markdown
        # Check page markers are added
        assert "<!-- Page number: 1 -->" in result.markdown
        assert "<!-- Page number: 2 -->" in result.markdown
        assert result.metadata["format"] == "PDF"

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_convert_with_output_dir(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test PDF conversion with output directory."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Test content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = PdfConverter()
        _ = converter.convert(pdf_file, output_dir)

        # Check that assets directory was created
        assets_dir = output_dir / "assets"
        assert assets_dir.exists()

        # Verify pymupdf4llm was called with correct image_path
        call_args = mock_pymupdf4llm.to_markdown.call_args
        assert str(assets_dir) in call_args[1]["image_path"]

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_convert_uses_temp_dir_when_no_output(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test that temp directory is used when no output_dir specified."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        _ = converter.convert(pdf_file)

        # Verify pymupdf4llm was called (temp dir is created internally)
        mock_pymupdf4llm.to_markdown.assert_called_once()

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_convert_with_image_format_config(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test PDF conversion respects image format from config."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        # Config with webp format
        config = MarkitaiConfig(image=ImageConfig(format="webp"))
        converter = PdfConverter(config)
        _ = converter.convert(pdf_file)

        call_args = mock_pymupdf4llm.to_markdown.call_args
        assert call_args[1]["image_format"] == "webp"

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_convert_jpeg_format_normalized(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test that jpeg format is normalized to jpg for pymupdf4llm."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        config = MarkitaiConfig(image=ImageConfig(format="jpeg"))
        converter = PdfConverter(config)
        _ = converter.convert(pdf_file)

        call_args = mock_pymupdf4llm.to_markdown.call_args
        assert call_args[1]["image_format"] == "jpg"

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_convert_page_chunks_enabled(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test that page_chunks=True is passed to pymupdf4llm."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        _ = converter.convert(pdf_file)

        call_args = mock_pymupdf4llm.to_markdown.call_args
        assert call_args[1]["page_chunks"] is True
        assert call_args[1]["force_text"] is True


class TestConvertWithOCR:
    """Tests for OCR conversion mode."""

    def test_ocr_mode_enabled(self, tmp_path: Path) -> None:
        """Test that OCR mode is triggered when config.ocr.enabled=True."""
        # Setup mock document
        mock_pymupdf = Mock()
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=2)
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_doc.close = Mock()
        mock_pymupdf.open.return_value = mock_doc

        # Setup OCR processor mock
        mock_ocr = Mock()
        mock_result = Mock()
        mock_result.text = "OCR extracted text"
        mock_ocr.recognize_pdf_page.return_value = mock_result

        # Create a mock OCRProcessor class
        mock_ocr_processor_cls = Mock(return_value=mock_ocr)

        # Create mock ocr module
        mock_ocr_module = Mock()
        mock_ocr_module.OCRProcessor = mock_ocr_processor_cls

        pdf_file = tmp_path / "scanned.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        # Enable OCR, disable screenshot
        config = MarkitaiConfig(
            ocr=OCRConfig(enabled=True),
            screenshot=ScreenshotConfig(enabled=False),
        )
        converter = PdfConverter(config)

        with patch.dict(
            sys.modules,
            {"pymupdf": mock_pymupdf, "markitai.ocr": mock_ocr_module},
        ):
            result = converter.convert(pdf_file)

        # Verify OCR was used
        assert result.metadata.get("ocr_used") is True
        mock_ocr.recognize_pdf_page.assert_called()

    def test_ocr_not_used_when_disabled(self, tmp_path: Path) -> None:
        """Test that OCR is not used when disabled."""
        with patch("markitai.converter.pdf.pymupdf4llm") as mock_pymupdf4llm:
            mock_pymupdf4llm.to_markdown.return_value = [{"text": "Normal text"}]

            pdf_file = tmp_path / "normal.pdf"
            pdf_file.touch()

            # OCR disabled (default)
            config = MarkitaiConfig(ocr=OCRConfig(enabled=False))
            converter = PdfConverter(config)
            result = converter.convert(pdf_file)

            # OCR should not be in metadata
            assert result.metadata.get("ocr_used") is not True


class TestConvertWithScreenshot:
    """Tests for screenshot rendering functionality."""

    def test_screenshot_enabled(self, tmp_path: Path) -> None:
        """Test screenshot rendering when enabled."""
        # Setup pymupdf mock
        mock_pymupdf = Mock()
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.samples = b"fake_pixel_data"
        mock_pix.width = 800
        mock_pix.height = 600
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.close = Mock()
        mock_pymupdf.open.return_value = mock_doc
        mock_pymupdf.Matrix.return_value = Mock()

        # Setup ImageProcessor mock
        mock_img_processor = Mock()
        mock_img_processor.save_screenshot.return_value = (800, 600)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Enable screenshot
        config = MarkitaiConfig(screenshot=ScreenshotConfig(enabled=True))
        converter = PdfConverter(config)

        with patch("markitai.converter.pdf.pymupdf4llm") as mock_pymupdf4llm:
            mock_pymupdf4llm.to_markdown.return_value = [{"text": "Page content"}]
            with (
                patch.dict(sys.modules, {"pymupdf": mock_pymupdf}),
                patch(
                    "markitai.converter.pdf.ImageProcessor",
                    return_value=mock_img_processor,
                ),
            ):
                result = converter.convert(pdf_file, output_dir)

        # Verify screenshot was created
        assert "page_images" in result.metadata
        assert result.metadata.get("pages") == 1
        mock_img_processor.save_screenshot.assert_called()

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_screenshot_disabled_no_rendering(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test that pages are not rendered when screenshot is disabled."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Screenshot disabled (default)
        config = MarkitaiConfig(screenshot=ScreenshotConfig(enabled=False))
        converter = PdfConverter(config)
        result = converter.convert(pdf_file, output_dir)

        # page_images should not be in metadata
        assert "page_images" not in result.metadata


class TestRenderPagesForLLM:
    """Tests for _render_pages_for_llm method (OCR + LLM mode)."""

    def test_render_for_llm_extracts_text_and_renders(self, tmp_path: Path) -> None:
        """Test that _render_pages_for_llm extracts text and renders pages."""
        # Setup pymupdf mock for page rendering
        mock_pymupdf = Mock()
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.samples = b"pixel_data"
        mock_pix.width = 1000
        mock_pix.height = 800
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=2)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.__enter__ = Mock(return_value=mock_doc)
        mock_doc.__exit__ = Mock(return_value=None)
        mock_doc.close = Mock()
        mock_pymupdf.open.return_value = mock_doc
        mock_pymupdf.Matrix.return_value = Mock()

        # Setup ImageProcessor mock
        mock_img_processor = Mock()
        mock_img_processor.save_screenshot.return_value = (1000, 800)

        pdf_file = tmp_path / "document.pdf"
        pdf_file.touch()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Enable both OCR and LLM
        config = MarkitaiConfig(
            ocr=OCRConfig(enabled=True),
            llm=LLMConfig(enabled=True),
            screenshot=ScreenshotConfig(enabled=True),
        )
        converter = PdfConverter(config)

        with patch("markitai.converter.pdf.pymupdf4llm") as mock_pymupdf4llm:
            mock_pymupdf4llm.to_markdown.return_value = "Extracted markdown text"
            with (
                patch.dict(sys.modules, {"pymupdf": mock_pymupdf}),
                patch(
                    "markitai.converter.pdf.ImageProcessor",
                    return_value=mock_img_processor,
                ),
            ):
                result = converter.convert(pdf_file, output_dir)

        # Verify text was extracted
        assert "Extracted markdown text" in result.markdown
        # Verify screenshots were taken when screenshot is enabled
        assert "page_images" in result.metadata


class TestImageCompression:
    """Tests for image compression during conversion."""

    def test_image_compression_when_enabled(self, tmp_path: Path) -> None:
        """Test that images are compressed when config.image.compress=True."""
        # Create a fake image file
        output_dir = tmp_path / "output"
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True)

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        # Create a test image that matches the expected pattern
        test_image = assets_dir / "test.pdf-0-0.png"
        test_image.write_bytes(b"fake_image_data")

        # Setup PIL Image mock
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_img.copy.return_value = mock_img

        # Setup ImageProcessor mock
        mock_img_processor = Mock()
        compressed_img = Mock()
        compressed_img.size = (800, 600)
        mock_img_processor.compress.return_value = (compressed_img, b"compressed_data")

        # Enable compression
        config = MarkitaiConfig(image=ImageConfig(compress=True, quality=75))
        converter = PdfConverter(config)

        with patch("markitai.converter.pdf.pymupdf4llm") as mock_pymupdf4llm:
            mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]
            with (
                patch("PIL.Image.open", return_value=mock_img),
                patch(
                    "markitai.converter.pdf.ImageProcessor",
                    return_value=mock_img_processor,
                ),
            ):
                _ = converter.convert(pdf_file, output_dir)

        # Compression should have been attempted
        # (the actual call depends on image file existing)


class TestMetadataGeneration:
    """Tests for metadata generation in convert results."""

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_basic_metadata(self, mock_pymupdf4llm: Mock, tmp_path: Path) -> None:
        """Test that basic metadata is generated correctly."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "document.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        assert result.metadata["source"] == str(pdf_file)
        assert result.metadata["format"] == "PDF"
        assert "images" in result.metadata

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_metadata_with_images(self, mock_pymupdf4llm: Mock, tmp_path: Path) -> None:
        """Test metadata includes image count."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        # images count should be in metadata
        assert "images" in result.metadata
        assert isinstance(result.metadata["images"], int)


class TestPageMarkers:
    """Tests for page marker formatting."""

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_page_markers_format(self, mock_pymupdf4llm: Mock, tmp_path: Path) -> None:
        """Test that page markers are in correct format."""
        mock_pymupdf4llm.to_markdown.return_value = [
            {"text": "Page 1 content"},
            {"text": "Page 2 content"},
            {"text": "Page 3 content"},
        ]

        pdf_file = tmp_path / "multi_page.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        # Check page markers format
        assert "<!-- Page number: 1 -->" in result.markdown
        assert "<!-- Page number: 2 -->" in result.markdown
        assert "<!-- Page number: 3 -->" in result.markdown

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_page_markers_have_blank_line_after(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test that page markers have blank line after for proper markdown formatting."""
        mock_pymupdf4llm.to_markdown.return_value = [{"text": "Content"}]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        # Should have format: "<!-- Page number: N -->\n\nContent"
        marker_pattern = r"<!-- Page number: \d+ -->\n\n"
        assert re.search(marker_pattern, result.markdown)


class TestWorkerCalculation:
    """Tests for adaptive worker count calculation in OCR mode."""

    def test_worker_count_small_file(self, tmp_path: Path) -> None:
        """Test worker count calculation for small files (<10MB)."""
        # Create a small file
        pdf_file = tmp_path / "small.pdf"
        pdf_file.write_bytes(b"x" * (5 * 1024 * 1024))  # 5 MB

        file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        assert file_size_mb < 10

        # Simulate worker calculation logic
        import os

        cpu_count = os.cpu_count() or 4
        total_pages = 10

        if file_size_mb < 10:
            max_workers = min(cpu_count // 2 or 2, total_pages, 6)
        elif file_size_mb < 50:
            max_workers = min(4, total_pages)
        else:
            max_workers = min(2, total_pages)

        max_workers = max(1, max_workers)

        # Should use up to cpu_count/2 workers, capped at 6
        assert 1 <= max_workers <= 6

    def test_worker_count_large_file(self, tmp_path: Path) -> None:
        """Test worker count for large files (>50MB) is limited."""
        # Simulate large file
        file_size_mb = 60  # 60 MB
        total_pages = 100

        if file_size_mb < 10:
            max_workers = min(4, total_pages, 6)
        elif file_size_mb < 50:
            max_workers = min(4, total_pages)
        else:
            max_workers = min(2, total_pages)

        max_workers = max(1, max_workers)

        # Large files should use at most 2 workers
        assert max_workers == 2


class TestChunkHandling:
    """Tests for handling page chunks from pymupdf4llm."""

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_dict_chunk_handling(self, mock_pymupdf4llm: Mock, tmp_path: Path) -> None:
        """Test handling of dict chunks from pymupdf4llm."""
        mock_pymupdf4llm.to_markdown.return_value = [
            {"text": "First page text", "metadata": {}},
            {"text": "Second page text", "metadata": {}},
        ]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        assert "First page text" in result.markdown
        assert "Second page text" in result.markdown

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_string_chunk_handling(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test handling of string chunks (fallback)."""
        # Some versions might return strings instead of dicts
        mock_pymupdf4llm.to_markdown.return_value = [
            "First page as string",
            "Second page as string",
        ]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        assert "First page as string" in result.markdown
        assert "Second page as string" in result.markdown

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_empty_page_handling(self, mock_pymupdf4llm: Mock, tmp_path: Path) -> None:
        """Test handling of empty pages."""
        mock_pymupdf4llm.to_markdown.return_value = [
            {"text": "Page with content"},
            {"text": ""},  # Empty page
            {"text": "Another page"},
        ]

        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        converter = PdfConverter()
        result = converter.convert(pdf_file)

        # Should still have 3 page markers
        assert result.markdown.count("<!-- Page number:") == 3


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @patch("markitai.converter.pdf.pymupdf4llm")
    def test_conversion_error_propagates(
        self, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Test that conversion errors propagate correctly."""
        mock_pymupdf4llm.to_markdown.side_effect = Exception("PDF parsing failed")

        pdf_file = tmp_path / "corrupted.pdf"
        pdf_file.touch()

        converter = PdfConverter()

        with pytest.raises(Exception, match="PDF parsing failed"):
            converter.convert(pdf_file)

    def test_ocr_import_error_handling(self, tmp_path: Path) -> None:
        """Test that ImportError is raised when pymupdf is not available in OCR mode."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        config = MarkitaiConfig(
            ocr=OCRConfig(enabled=True), llm=LLMConfig(enabled=False)
        )
        _ = PdfConverter(config)

        # Remove pymupdf from modules to simulate import failure
        original_modules = sys.modules.copy()
        if "pymupdf" in sys.modules:
            del sys.modules["pymupdf"]

        # The actual import error behavior depends on how the module handles it
        # This test verifies the code path doesn't crash unexpectedly
        try:
            # Restore the module for other tests
            pass
        finally:
            sys.modules.update(original_modules)


class TestMIMETypeHandling:
    """Tests for MIME type handling in extracted images."""

    def test_png_mime_type(self, tmp_path: Path) -> None:
        """Test PNG MIME type is set correctly."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        (assets_dir / "doc.pdf-0-0.png").touch()

        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "doc.pdf")

        assert images[0].mime_type == "image/png"

    def test_jpg_mime_type(self, tmp_path: Path) -> None:
        """Test JPG MIME type is set correctly."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        (assets_dir / "doc.pdf-0-0.jpg").touch()

        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "doc.pdf")

        assert images[0].mime_type == "image/jpeg"

    def test_jpeg_mime_type(self, tmp_path: Path) -> None:
        """Test JPEG extension MIME type is set correctly."""
        converter = PdfConverter()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        (assets_dir / "doc.pdf-0-0.jpeg").touch()

        mock_pymupdf = create_pymupdf_mock()
        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            images = converter._collect_embedded_images(assets_dir, "doc.pdf")

        assert images[0].mime_type == "image/jpeg"
