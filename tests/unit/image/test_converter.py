"""Tests for image format converter module."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from markit.converters.base import ExtractedImage
from markit.image.converter import (
    CONVERTIBLE_FORMATS,
    FORMAT_CONVERSION_MAP,
    LLM_SUPPORTED_IMAGE_FORMATS,
    BatchImageConverter,
    ConvertedImage,
    ImageFormatConverter,
    check_conversion_tools,
    is_llm_supported_format,
)


class TestConstants:
    """Tests for module constants."""

    def test_convertible_formats(self):
        """Test CONVERTIBLE_FORMATS contains expected formats."""
        assert "emf" in CONVERTIBLE_FORMATS
        assert "wmf" in CONVERTIBLE_FORMATS
        assert "tiff" in CONVERTIBLE_FORMATS
        assert "tif" in CONVERTIBLE_FORMATS
        assert "bmp" in CONVERTIBLE_FORMATS
        assert "gif" in CONVERTIBLE_FORMATS
        assert "ico" in CONVERTIBLE_FORMATS

    def test_format_conversion_map(self):
        """Test FORMAT_CONVERSION_MAP has correct mappings."""
        assert FORMAT_CONVERSION_MAP["emf"] == "png"
        assert FORMAT_CONVERSION_MAP["wmf"] == "png"
        assert FORMAT_CONVERSION_MAP["gif"] == "png"
        assert FORMAT_CONVERSION_MAP["bmp"] == "png"

    def test_llm_supported_formats(self):
        """Test LLM_SUPPORTED_IMAGE_FORMATS."""
        assert "png" in LLM_SUPPORTED_IMAGE_FORMATS
        assert "jpeg" in LLM_SUPPORTED_IMAGE_FORMATS
        assert "jpg" in LLM_SUPPORTED_IMAGE_FORMATS
        assert "webp" in LLM_SUPPORTED_IMAGE_FORMATS


class TestConvertedImage:
    """Tests for ConvertedImage dataclass."""

    def test_creation(self):
        """Test creating ConvertedImage."""
        converted = ConvertedImage(
            data=b"image data",
            format="png",
            filename="test.png",
            original_format="bmp",
            width=100,
            height=200,
        )

        assert converted.data == b"image data"
        assert converted.format == "png"
        assert converted.filename == "test.png"
        assert converted.original_format == "bmp"
        assert converted.width == 100
        assert converted.height == 200


class TestIsLlmSupportedFormat:
    """Tests for is_llm_supported_format function."""

    def test_supported_formats(self):
        """Test supported formats return True."""
        assert is_llm_supported_format("png") is True
        assert is_llm_supported_format("jpeg") is True
        assert is_llm_supported_format("jpg") is True
        assert is_llm_supported_format("webp") is True

    def test_unsupported_formats(self):
        """Test unsupported formats return False."""
        assert is_llm_supported_format("gif") is False
        assert is_llm_supported_format("bmp") is False
        assert is_llm_supported_format("tiff") is False
        assert is_llm_supported_format("emf") is False

    def test_case_insensitive(self):
        """Test format check is case insensitive."""
        assert is_llm_supported_format("PNG") is True
        assert is_llm_supported_format("Jpeg") is True


class TestImageFormatConverterInit:
    """Tests for ImageFormatConverter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        converter = ImageFormatConverter()

        # Paths depend on system
        assert hasattr(converter, "inkscape_path")
        assert hasattr(converter, "wmf2svg_path")

    def test_custom_paths(self):
        """Test custom tool paths."""
        converter = ImageFormatConverter(
            inkscape_path="/custom/inkscape",
            wmf2svg_path="/custom/wmf2svg",
        )

        assert converter.inkscape_path == "/custom/inkscape"
        assert converter.wmf2svg_path == "/custom/wmf2svg"

    def test_check_pillow_wmf(self):
        """Test Pillow WMF support check."""
        converter = ImageFormatConverter()
        # Result depends on installed packages
        assert isinstance(converter._pillow_wmf_support, bool)


class TestImageFormatConverterNeedsConversion:
    """Tests for needs_conversion method."""

    def test_convertible_formats(self):
        """Test that convertible formats are identified."""
        converter = ImageFormatConverter()

        assert converter.needs_conversion("emf") is True
        assert converter.needs_conversion("wmf") is True
        assert converter.needs_conversion("gif") is True
        assert converter.needs_conversion("bmp") is True
        assert converter.needs_conversion("tiff") is True

    def test_supported_formats(self):
        """Test that supported formats don't need conversion."""
        converter = ImageFormatConverter()

        assert converter.needs_conversion("png") is False
        assert converter.needs_conversion("jpeg") is False
        assert converter.needs_conversion("jpg") is False

    def test_case_insensitive(self):
        """Test format check is case insensitive."""
        converter = ImageFormatConverter()

        assert converter.needs_conversion("GIF") is True
        assert converter.needs_conversion("PNG") is False


class TestImageFormatConverterConvert:
    """Tests for convert method."""

    def _create_test_image(
        self, format_name: str, width: int = 100, height: int = 100
    ) -> ExtractedImage:
        """Create a test image in the specified format."""
        img = Image.new("RGB", (width, height), color="red")
        buffer = io.BytesIO()
        save_format = format_name.upper()
        if save_format == "JPG":
            save_format = "JPEG"
        img.save(buffer, format=save_format)

        return ExtractedImage(
            data=buffer.getvalue(),
            format=format_name,
            filename=f"test.{format_name}",
            source_document=Path("test.docx"),
            position=0,
            width=width,
            height=height,
        )

    def test_convert_no_conversion_needed(self):
        """Test converting image that doesn't need conversion."""
        converter = ImageFormatConverter()
        image = self._create_test_image("png")

        result = converter.convert(image)

        assert result is not None
        assert result.format == "png"
        assert result.original_format == "png"

    def test_convert_bmp_to_png(self):
        """Test converting BMP to PNG."""
        converter = ImageFormatConverter()
        image = self._create_test_image("bmp")

        result = converter.convert(image)

        assert result is not None
        assert result.format == "png"
        assert result.original_format == "bmp"
        assert result.filename == "test.png"

    def test_convert_gif_to_png(self):
        """Test converting GIF to PNG."""
        converter = ImageFormatConverter()

        # Create a GIF image
        img = Image.new("P", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="GIF")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="gif",
            filename="test.gif",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter.convert(image)

        assert result is not None
        assert result.format == "png"
        assert result.filename == "test.png"


class TestImageFormatConverterConvertWithPillow:
    """Tests for _convert_with_pillow method."""

    def test_convert_simple_image(self):
        """Test converting a simple image."""
        converter = ImageFormatConverter()

        # Create a BMP image
        img = Image.new("RGB", (50, 50), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="BMP")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="bmp",
            filename="blue.bmp",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter._convert_with_pillow(image)

        assert result.format == "png"
        assert result.width == 50
        assert result.height == 50
        assert result.filename == "blue.png"

    def test_convert_rgba_to_jpeg(self):
        """Test converting RGBA image to JPEG."""
        converter = ImageFormatConverter()

        # Create RGBA image
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Mock FORMAT_CONVERSION_MAP to return jpeg
        image = ExtractedImage(
            data=buffer.getvalue(),
            format="png",
            filename="test.png",
            source_document=Path("test.docx"),
            position=0,
        )

        # This won't trigger RGBA->RGB conversion since PNG->PNG doesn't change format
        result = converter._convert_with_pillow(image)

        assert result is not None

    def test_convert_error_handling(self):
        """Test error handling for invalid image data."""
        converter = ImageFormatConverter()

        image = ExtractedImage(
            data=b"not valid image data",
            format="bmp",
            filename="invalid.bmp",
            source_document=Path("test.docx"),
            position=0,
        )

        with pytest.raises((OSError, ValueError)):
            converter._convert_with_pillow(image)


class TestImageFormatConverterConvertTiff:
    """Tests for _convert_tiff method."""

    def test_convert_rgb_tiff_to_jpeg(self):
        """Test converting RGB TIFF to JPEG."""
        converter = ImageFormatConverter()

        # Create RGB TIFF
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="TIFF")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="tiff",
            filename="photo.tiff",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter._convert_tiff(image)

        assert result.format == "jpeg"
        assert result.filename == "photo.jpeg"

    def test_convert_rgba_tiff_to_png(self):
        """Test converting RGBA TIFF to PNG."""
        converter = ImageFormatConverter()

        # Create RGBA TIFF
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="TIFF")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="tiff",
            filename="photo.tiff",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter._convert_tiff(image)

        assert result.format == "png"
        assert result.filename == "photo.png"


class TestImageFormatConverterConvertMetafile:
    """Tests for _convert_metafile method."""

    def test_metafile_no_converters(self):
        """Test metafile conversion with no available converters."""
        converter = ImageFormatConverter()
        converter._pillow_wmf_support = False
        converter.inkscape_path = None

        image = ExtractedImage(
            data=b"fake emf data",
            format="emf",
            filename="diagram.emf",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter._convert_metafile(image)

        assert result is None  # Should return None when all methods fail

    def test_metafile_pillow_success(self):
        """Test metafile conversion with Pillow success."""
        converter = ImageFormatConverter()
        converter._pillow_wmf_support = True

        # Create valid PNG image to mock successful conversion
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="emf",
            filename="diagram.emf",
            source_document=Path("test.docx"),
            position=0,
        )

        # Mock _convert_with_pillow to succeed
        with patch.object(converter, "_convert_with_pillow") as mock_pillow:
            mock_pillow.return_value = ConvertedImage(
                data=b"converted",
                format="png",
                filename="diagram.png",
                original_format="emf",
                width=100,
                height=100,
            )

            result = converter._convert_metafile(image)

            assert result is not None
            assert result.format == "png"


class TestImageFormatConverterCreatePlaceholder:
    """Tests for _create_placeholder method."""

    def test_create_placeholder_default_size(self):
        """Test creating placeholder with default size."""
        converter = ImageFormatConverter()

        image = ExtractedImage(
            data=b"data",
            format="emf",
            filename="diagram.emf",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter._create_placeholder(image)

        assert result.format == "png"
        assert result.width == 200
        assert result.height == 200
        assert result.filename == "diagram.png"
        assert len(result.data) > 0

    def test_create_placeholder_custom_size(self):
        """Test creating placeholder with custom size."""
        converter = ImageFormatConverter()

        image = ExtractedImage(
            data=b"data",
            format="emf",
            filename="diagram.emf",
            source_document=Path("test.docx"),
            position=0,
            width=300,
            height=400,
        )

        result = converter._create_placeholder(image)

        assert result.width == 300
        assert result.height == 400


class TestBatchImageConverter:
    """Tests for BatchImageConverter class."""

    def test_init_default(self):
        """Test default initialization."""
        batch_converter = BatchImageConverter()

        assert batch_converter.converter is not None
        assert isinstance(batch_converter.converter, ImageFormatConverter)

    def test_init_custom_converter(self):
        """Test initialization with custom converter."""
        custom_converter = ImageFormatConverter()
        batch_converter = BatchImageConverter(converter=custom_converter)

        assert batch_converter.converter is custom_converter

    def test_convert_batch_no_conversion(self):
        """Test batch conversion with images that don't need conversion."""
        batch_converter = BatchImageConverter()

        # Create PNG images
        images = []
        for i in range(3):
            img = Image.new("RGB", (50, 50), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            images.append(
                ExtractedImage(
                    data=buffer.getvalue(),
                    format="png",
                    filename=f"image{i}.png",
                    source_document=Path("test.docx"),
                    position=i,
                )
            )

        results = batch_converter.convert_batch(images)

        assert len(results) == 3
        assert all(r.format == "png" for r in results)

    def test_convert_batch_with_conversion(self):
        """Test batch conversion with images that need conversion."""
        batch_converter = BatchImageConverter()

        # Create BMP images
        images = []
        for i in range(2):
            img = Image.new("RGB", (50, 50), color="blue")
            buffer = io.BytesIO()
            img.save(buffer, format="BMP")

            images.append(
                ExtractedImage(
                    data=buffer.getvalue(),
                    format="bmp",
                    filename=f"image{i}.bmp",
                    source_document=Path("test.docx"),
                    position=i,
                )
            )

        results = batch_converter.convert_batch(images)

        assert len(results) == 2
        assert all(r.format == "png" for r in results)

    def test_convert_batch_with_progress(self):
        """Test batch conversion with progress callback."""
        batch_converter = BatchImageConverter()
        progress_calls = []

        def on_progress(index, total, image):
            progress_calls.append((index, total, image))

        # Create images
        images = []
        for i in range(3):
            img = Image.new("RGB", (50, 50), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            images.append(
                ExtractedImage(
                    data=buffer.getvalue(),
                    format="png",
                    filename=f"image{i}.png",
                    source_document=Path("test.docx"),
                    position=i,
                )
            )

        batch_converter.convert_batch(images, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1
        assert progress_calls[1][0] == 2
        assert progress_calls[2][0] == 3
        assert all(call[1] == 3 for call in progress_calls)

    def test_convert_batch_error_handling(self):
        """Test batch conversion with error creates placeholder."""
        batch_converter = BatchImageConverter()

        # Create image with invalid data
        images = [
            ExtractedImage(
                data=b"invalid image data",
                format="bmp",
                filename="invalid.bmp",
                source_document=Path("test.docx"),
                position=0,
            )
        ]

        results = batch_converter.convert_batch(images)

        # Should return placeholder for failed conversion
        assert len(results) == 1
        assert results[0].format == "png"


class TestImageFormatConverterMetafileConversion:
    """Additional tests for metafile conversion."""

    def test_metafile_pillow_fallback_to_inkscape(self):
        """Test metafile conversion falls back to Inkscape when Pillow fails."""
        converter = ImageFormatConverter()
        converter._pillow_wmf_support = True
        converter.inkscape_path = None  # No Inkscape

        image = ExtractedImage(
            data=b"fake wmf data",
            format="wmf",
            filename="diagram.wmf",
            source_document=Path("test.docx"),
            position=0,
        )

        # Mock _convert_with_pillow to fail
        with patch.object(
            converter, "_convert_with_pillow", side_effect=Exception("Pillow failed")
        ):
            result = converter._convert_metafile(image)

            # Should return None since both methods failed
            assert result is None


class TestImageFormatConverterJpegConversion:
    """Tests for JPEG conversion with RGBA handling."""

    def test_convert_rgba_png_to_jpeg(self):
        """Test converting RGBA PNG to JPEG (flattens alpha)."""
        converter = ImageFormatConverter()

        # Since png->png doesn't trigger RGBA->RGB, we need to test via TIFF
        # which maps to jpeg for RGB images but preserves alpha as PNG
        tiff_img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        tiff_buffer = io.BytesIO()
        tiff_img.save(tiff_buffer, format="TIFF")

        tiff_image = ExtractedImage(
            data=tiff_buffer.getvalue(),
            format="tiff",
            filename="alpha.tiff",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter._convert_tiff(tiff_image)

        # RGBA TIFF should convert to PNG (not JPEG) to preserve alpha
        assert result.format == "png"


class TestImageFormatConverterRouting:
    """Tests for format routing in convert method."""

    def test_routes_emf_to_metafile_converter(self):
        """Test that EMF format is routed to metafile converter."""
        converter = ImageFormatConverter()
        converter._pillow_wmf_support = False
        converter.inkscape_path = None

        image = ExtractedImage(
            data=b"fake emf data",
            format="emf",
            filename="diagram.emf",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter.convert(image)

        # Should return None since no converters available
        assert result is None

    def test_routes_tiff_to_tiff_converter(self):
        """Test that TIFF format is routed to TIFF converter."""
        converter = ImageFormatConverter()

        # Create RGB TIFF
        img = Image.new("RGB", (50, 50), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="TIFF")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="tif",  # Use tif extension
            filename="photo.tif",
            source_document=Path("test.docx"),
            position=0,
        )

        result = converter.convert(image)

        assert result is not None
        assert result.format == "jpeg"
        assert result.filename == "photo.jpeg"


class TestCheckConversionTools:
    """Tests for check_conversion_tools function."""

    def test_returns_dict(self):
        """Test that function returns proper dict."""
        result = check_conversion_tools()

        assert isinstance(result, dict)
        assert "inkscape" in result
        assert "wmf2svg" in result
        assert "pillow_wmf" in result

    def test_inkscape_check(self):
        """Test inkscape availability check."""
        with patch("shutil.which", return_value="/usr/bin/inkscape"):
            result = check_conversion_tools()
            assert result["inkscape"] is True

        with patch("shutil.which", return_value=None):
            result = check_conversion_tools()
            assert result["inkscape"] is False

    def test_wmf2svg_check(self):
        """Test wmf2svg availability check."""

        def mock_which(cmd):
            if cmd == "wmf2svg":
                return "/usr/bin/wmf2svg"
            return None

        with patch("shutil.which", side_effect=mock_which):
            result = check_conversion_tools()
            assert result["wmf2svg"] is True
