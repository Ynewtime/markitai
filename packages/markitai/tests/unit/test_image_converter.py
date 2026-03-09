"""Tests for image converter module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from markitai.config import MarkitaiConfig
from markitai.converter.base import FileFormat
from markitai.converter.image import ImageConverter


@pytest.fixture
def sample_image(tmp_path: Path, sample_png_bytes: bytes) -> Path:
    """Create a sample image file for testing using sample_png_bytes from conftest."""
    image_file = tmp_path / "test.png"
    image_file.write_bytes(sample_png_bytes)
    return image_file


@pytest.fixture
def sample_bmp_image(tmp_path: Path) -> Path:
    """Create a BMP image for compatibility conversion tests."""
    image_file = tmp_path / "test.bmp"
    Image.new("RGB", (8, 8), color="red").save(image_file, format="BMP")
    return image_file


@pytest.fixture
def sample_tiff_image(tmp_path: Path) -> Path:
    """Create a TIFF image for compatibility conversion tests."""
    image_file = tmp_path / "test.tiff"
    Image.new("RGB", (8, 8), color="blue").save(image_file, format="TIFF")
    return image_file


@pytest.fixture
def sample_tif_image(tmp_path: Path) -> Path:
    """Create a TIF image for compatibility conversion tests."""
    image_file = tmp_path / "test.tif"
    Image.new("RGB", (8, 8), color="green").save(image_file, format="TIFF")
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


class TestImageConverterFormats:
    """Tests for ImageConverter format support and registration."""

    def test_jpeg_format_supported(self):
        """Test JPEG format is supported by ImageConverter."""
        assert FileFormat.JPEG in ImageConverter.supported_formats

    def test_jpg_format_supported(self):
        """Test JPG format is supported by ImageConverter."""
        assert FileFormat.JPG in ImageConverter.supported_formats

    def test_png_format_supported(self):
        """Test PNG format is supported by ImageConverter."""
        assert FileFormat.PNG in ImageConverter.supported_formats

    def test_webp_format_supported(self):
        """Test WebP format is supported by ImageConverter."""
        assert FileFormat.WEBP in ImageConverter.supported_formats

    def test_all_formats_registered(self):
        """Test all image formats are registered in converter registry."""
        from markitai.converter.base import _converter_registry

        for fmt in (FileFormat.JPEG, FileFormat.JPG, FileFormat.PNG, FileFormat.WEBP):
            assert fmt in _converter_registry
            assert _converter_registry[fmt] is ImageConverter


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
            sample_image, ".markitai/assets/test.png"
        )

        assert "# test" in placeholder
        assert "![test](.markitai/assets/test.png)" in placeholder


class TestCopyToAssets:
    """Tests for _copy_to_assets method."""

    @pytest.mark.parametrize(
        ("file_name", "image_format"),
        [
            ("test.jpg", "JPEG"),
            ("test.jpeg", "JPEG"),
            ("test.gif", "GIF"),
            ("test.webp", "WEBP"),
        ],
    )
    def test_copy_to_assets_preserves_preview_compatible_formats(
        self, tmp_path: Path, file_name: str, image_format: str
    ):
        """Test preview-compatible formats keep their original asset extension."""
        input_path = tmp_path / file_name
        Image.new("RGB", (8, 8), color="purple").save(input_path, format=image_format)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(input_path, output_dir)

        assert ref_path == f".markitai/assets/{file_name}"
        assert (output_dir / ".markitai" / "assets" / file_name).exists()

    def test_copy_to_assets_with_output_dir(self, sample_image: Path, tmp_path: Path):
        """Test copying image to assets directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_image, output_dir)

        assert ref_path == ".markitai/assets/test.png"
        assert (output_dir / ".markitai" / "assets" / "test.png").exists()

    def test_copy_to_assets_without_output_dir(self, sample_image: Path):
        """Test without output directory returns original filename."""
        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_image, None)

        assert ref_path == "test.png"

    def test_copy_to_assets_no_overwrite(self, sample_image: Path, tmp_path: Path):
        """Test that existing files are not overwritten."""
        output_dir = tmp_path / "output"
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

        # Create existing file with different content
        existing_file = assets_dir / "test.png"
        existing_file.write_text("existing content")

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_image, output_dir)

        assert ref_path == ".markitai/assets/test.png"
        # Should not overwrite existing file
        assert existing_file.read_text() == "existing content"

    def test_copy_to_assets_transcodes_bmp_to_png(
        self, sample_bmp_image: Path, tmp_path: Path
    ):
        """Test BMP assets are converted to PNG for markdown preview compatibility."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_bmp_image, output_dir)

        converted_file = output_dir / ref_path
        assert ref_path.startswith(".markitai/assets/test-")
        assert ref_path.endswith(".png")
        assert converted_file.exists()
        assert not (output_dir / ".markitai" / "assets" / "test.bmp").exists()
        with Image.open(converted_file) as image:
            assert image.format == "PNG"

    def test_copy_to_assets_transcodes_tiff_to_png(
        self, sample_tiff_image: Path, tmp_path: Path
    ):
        """Test TIFF assets are converted to PNG for markdown preview compatibility."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_tiff_image, output_dir)

        converted_file = output_dir / ref_path
        assert ref_path.startswith(".markitai/assets/test-")
        assert ref_path.endswith(".png")
        assert converted_file.exists()
        assert not (output_dir / ".markitai" / "assets" / "test.tiff").exists()
        with Image.open(converted_file) as image:
            assert image.format == "PNG"

    def test_copy_to_assets_transcodes_tif_to_png(
        self, sample_tif_image: Path, tmp_path: Path
    ):
        """Test TIF assets are converted to PNG for markdown preview compatibility."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_tif_image, output_dir)

        converted_file = output_dir / ref_path
        assert ref_path.startswith(".markitai/assets/test-")
        assert ref_path.endswith(".png")
        assert converted_file.exists()
        assert not (output_dir / ".markitai" / "assets" / "test.tif").exists()
        with Image.open(converted_file) as image:
            assert image.format == "PNG"

    def test_copy_to_assets_transcoded_names_stay_unique_for_same_stem(
        self, tmp_path: Path
    ):
        """Different source files with same stem should not reuse the same PNG."""
        bmp_path = tmp_path / "scan.bmp"
        tiff_path = tmp_path / "scan.tiff"
        Image.new("RGB", (8, 8), color="red").save(bmp_path, format="BMP")
        Image.new("RGB", (8, 8), color="blue").save(tiff_path, format="TIFF")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        bmp_ref = converter._copy_to_assets(bmp_path, output_dir)
        tiff_ref = converter._copy_to_assets(tiff_path, output_dir)

        assert bmp_ref != tiff_ref
        bmp_asset = output_dir / bmp_ref
        tiff_asset = output_dir / tiff_ref
        assert bmp_asset.exists()
        assert tiff_asset.exists()
        with Image.open(bmp_asset) as bmp_image, Image.open(tiff_asset) as tiff_image:
            assert bmp_image.tobytes() != tiff_image.tobytes()

    def test_copy_to_assets_without_output_dir_keeps_original_bmp_name(
        self, sample_bmp_image: Path
    ):
        """Test BMP keeps original filename when no output dir is provided."""
        converter = ImageConverter()

        ref_path = converter._copy_to_assets(sample_bmp_image, None)

        assert ref_path == "test.bmp"

    def test_copy_to_assets_does_not_rewrite_existing_transcoded_png(
        self, sample_tiff_image: Path, tmp_path: Path
    ):
        """Test existing converted PNG targets are left untouched."""
        output_dir = tmp_path / "output"
        converter = ImageConverter()
        ref_path = converter._copy_to_assets(sample_tiff_image, output_dir)
        existing_file = output_dir / ref_path
        existing_file.write_bytes(b"existing png content")
        ref_path_second = converter._copy_to_assets(sample_tiff_image, output_dir)

        assert ref_path.startswith(".markitai/assets/test-")
        assert ref_path.endswith(".png")
        assert ref_path_second == ref_path
        assert existing_file.read_bytes() == b"existing png content"

    def test_convert_uses_transcoded_asset_path_for_bmp_markdown(
        self, sample_bmp_image: Path, tmp_path: Path
    ):
        """Test markdown points to the converted PNG path for BMP inputs."""
        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(sample_bmp_image, output_dir)

        assert result.metadata["asset_path"].startswith(".markitai/assets/test-")
        assert result.metadata["asset_path"].endswith(".png")
        assert f"![test]({result.metadata['asset_path']})" in result.markdown

    def test_copy_to_assets_preserves_svg(self, tmp_path: Path):
        """SVG is NOT transcoded — copied as-is to assets directory."""
        svg_path = tmp_path / "diagram.svg"
        svg_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>Hi</text></svg>'
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        converter = ImageConverter()
        ref_path = converter._copy_to_assets(svg_path, output_dir)

        assert ref_path == ".markitai/assets/diagram.svg"
        assert (output_dir / ".markitai" / "assets" / "diagram.svg").exists()


class TestImageConverterSVG:
    """Tests for SVG support in ImageConverter."""

    @pytest.fixture
    def sample_svg(self, tmp_path: Path) -> Path:
        """Create a minimal SVG file."""
        svg_file = tmp_path / "test.svg"
        svg_file.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            "<title>Test SVG</title>"
            '<circle cx="50" cy="50" r="40" fill="red"/>'
            "<text x='50' y='55'>Hello</text>"
            "</svg>"
        )
        return svg_file

    def test_svg_format_supported(self):
        """SVG must be in ImageConverter.supported_formats."""
        assert FileFormat.SVG in ImageConverter.supported_formats

    def test_svg_registered_in_converter_registry(self):
        """SVG must be registered to ImageConverter in the global registry."""
        from markitai.converter.base import _converter_registry

        assert _converter_registry.get(FileFormat.SVG) is ImageConverter

    def test_convert_svg_produces_image_reference(
        self, sample_svg: Path, tmp_path: Path
    ):
        """Converting SVG should produce markdown with image reference."""
        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(sample_svg, output_dir)

        assert "![test](.markitai/assets/test.svg)" in result.markdown
        assert result.metadata["asset_path"] == ".markitai/assets/test.svg"
        assert (output_dir / ".markitai" / "assets" / "test.svg").exists()

    def test_svg_not_in_kreuzberg_formats(self):
        """SVG should no longer be handled by KreuzbergConverter."""
        from markitai.converter.kreuzberg import KREUZBERG_FORMATS

        assert FileFormat.SVG not in KREUZBERG_FORMATS
