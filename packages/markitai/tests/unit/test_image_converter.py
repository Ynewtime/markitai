"""Tests for image converter module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markitai.config import MarkitaiConfig
from markitai.converter.base import FileFormat
from markitai.converter.image import (
    ImageConverter,
    JpegConverter,
    JpgConverter,
    PngConverter,
    WebpConverter,
)


@pytest.fixture
def sample_image(tmp_path: Path, sample_png_bytes: bytes) -> Path:
    """Create a sample image file for testing using sample_png_bytes from conftest."""
    image_file = tmp_path / "test.png"
    image_file.write_bytes(sample_png_bytes)
    return image_file


class TestImageConverter:
    """Tests for ImageConverter class."""

    def test_supported_formats(self):
        """Test supported formats."""
        assert FileFormat.JPEG in ImageConverter.supported_formats
        assert FileFormat.JPG in ImageConverter.supported_formats
        assert FileFormat.PNG in ImageConverter.supported_formats
        assert FileFormat.WEBP in ImageConverter.supported_formats

    def test_convert_without_ocr(self, sample_image: Path):
        """Test converting image without OCR."""
        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)

        result = converter.convert(sample_image)

        assert result.markdown.startswith("# test")
        assert "test.png" in result.markdown
        assert result.metadata["ocr_used"] is False

    def test_convert_with_ocr_mocked(self, sample_image: Path):
        """Test converting image with OCR (mocked)."""
        config = MarkitaiConfig()
        config.ocr.enabled = True
        converter = ImageConverter(config)

        with patch("markitai.ocr.OCRProcessor") as MockOCR:
            mock_processor = MagicMock()
            mock_processor.recognize_to_markdown.return_value = "Extracted text"
            MockOCR.return_value = mock_processor

            result = converter.convert(sample_image)

            assert "Extracted text" in result.markdown
            assert result.metadata["ocr_used"] is True

    def test_convert_ocr_no_text(self, sample_image: Path):
        """Test converting image with OCR finding no text."""
        config = MarkitaiConfig()
        config.ocr.enabled = True
        converter = ImageConverter(config)

        with patch("markitai.ocr.OCRProcessor") as MockOCR:
            mock_processor = MagicMock()
            mock_processor.recognize_to_markdown.return_value = ""
            MockOCR.return_value = mock_processor

            result = converter.convert(sample_image)

            # Should fall back to placeholder
            assert "test.png" in result.markdown

    def test_convert_ocr_failure(self, sample_image: Path):
        """Test converting image when OCR fails."""
        config = MarkitaiConfig()
        config.ocr.enabled = True
        converter = ImageConverter(config)

        with patch("markitai.ocr.OCRProcessor") as MockOCR:
            MockOCR.side_effect = Exception("OCR failed")

            result = converter.convert(sample_image)

            # Should fall back to placeholder
            assert "test.png" in result.markdown


class TestSpecificConverters:
    """Tests for specific image format converters."""

    def test_jpeg_converter(self):
        """Test JPEG converter."""
        assert FileFormat.JPEG in JpegConverter.supported_formats
        converter = JpegConverter()
        assert isinstance(converter, ImageConverter)

    def test_jpg_converter(self):
        """Test JPG converter."""
        assert FileFormat.JPG in JpgConverter.supported_formats
        converter = JpgConverter()
        assert isinstance(converter, ImageConverter)

    def test_png_converter(self):
        """Test PNG converter."""
        assert FileFormat.PNG in PngConverter.supported_formats
        converter = PngConverter()
        assert isinstance(converter, ImageConverter)

    def test_webp_converter(self):
        """Test WebP converter."""
        assert FileFormat.WEBP in WebpConverter.supported_formats
        converter = WebpConverter()
        assert isinstance(converter, ImageConverter)


class TestImagePlaceholder:
    """Tests for image placeholder generation."""

    def test_placeholder_format(self, sample_image: Path):
        """Test placeholder markdown format."""
        converter = ImageConverter()
        placeholder = converter._create_image_placeholder(sample_image, "test.png")

        assert "# test" in placeholder
        assert "![test]" in placeholder
        assert "test.png" in placeholder

    def test_placeholder_with_assets_path(self, sample_image: Path):
        """Test placeholder with assets relative path."""
        converter = ImageConverter()
        placeholder = converter._create_image_placeholder(
            sample_image, "assets/test.png"
        )

        assert "# test" in placeholder
        assert "![test](assets/test.png)" in placeholder


class TestCopyToAssets:
    """Tests for _copy_to_assets method."""

    def test_copy_to_assets_with_output_dir(self, sample_image: Path, tmp_path: Path):
        """Test copying image to assets directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_image, output_dir)

        assert ref_path == "assets/test.png"
        assert (output_dir / "assets" / "test.png").exists()

    def test_copy_to_assets_without_output_dir(self, sample_image: Path):
        """Test without output directory returns original filename."""
        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_image, None)

        assert ref_path == "test.png"

    def test_copy_to_assets_no_overwrite(self, sample_image: Path, tmp_path: Path):
        """Test that existing files are not overwritten."""
        output_dir = tmp_path / "output"
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True)

        # Create existing file with different content
        existing_file = assets_dir / "test.png"
        existing_file.write_text("existing content")

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_image, output_dir)

        assert ref_path == "assets/test.png"
        # Should not overwrite existing file
        assert existing_file.read_text() == "existing content"
