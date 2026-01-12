"""Tests for image compression module."""

import io
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from markit.converters.base import ExtractedImage
from markit.image.compressor import (
    CompressedImage,
    CompressionConfig,
    ImageCompressor,
    compress_image_data,
)


class TestCompressionConfig:
    """Tests for CompressionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CompressionConfig()

        assert config.png_optimization_level == 2
        assert config.jpeg_quality == 85
        assert config.max_dimension == 2048
        assert config.skip_if_smaller_than == 10240

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CompressionConfig(
            png_optimization_level=4,
            jpeg_quality=75,
            max_dimension=1024,
            skip_if_smaller_than=5000,
        )

        assert config.png_optimization_level == 4
        assert config.jpeg_quality == 75
        assert config.max_dimension == 1024
        assert config.skip_if_smaller_than == 5000


class TestCompressedImage:
    """Tests for CompressedImage dataclass."""

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        compressed = CompressedImage(
            data=b"x" * 50,
            format="png",
            filename="test.png",
            original_size=100,
            compressed_size=50,
            width=100,
            height=100,
        )

        assert compressed.compression_ratio == 0.5

    def test_compression_ratio_zero_original(self):
        """Test compression ratio with zero original size."""
        compressed = CompressedImage(
            data=b"",
            format="png",
            filename="test.png",
            original_size=0,
            compressed_size=0,
            width=0,
            height=0,
        )

        assert compressed.compression_ratio == 1.0

    def test_savings_percent(self):
        """Test savings percentage calculation."""
        compressed = CompressedImage(
            data=b"x" * 25,
            format="png",
            filename="test.png",
            original_size=100,
            compressed_size=25,
            width=100,
            height=100,
        )

        assert compressed.savings_percent == 75.0


class TestCompressImageData:
    """Tests for compress_image_data function."""

    def test_compress_jpeg(self):
        """Test JPEG compression."""
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=100)

        data, fmt, width, height = compress_image_data(
            buffer.getvalue(),
            "jpeg",
            jpeg_quality=85,
        )

        assert fmt == "jpeg"
        assert width == 100
        assert height == 100

    def test_compress_png(self):
        """Test PNG compression."""
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        data, fmt, width, height = compress_image_data(
            buffer.getvalue(),
            "png",
            use_oxipng=False,
        )

        assert fmt == "png"
        assert width == 100
        assert height == 100

    def test_resize_large_image(self):
        """Test resizing of large images."""
        img = Image.new("RGB", (4000, 3000), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        data, fmt, width, height = compress_image_data(
            buffer.getvalue(),
            "jpeg",
            max_dimension=2048,
        )

        assert max(width, height) <= 2048

    def test_convert_rgba_jpeg(self):
        """Test RGBA to RGB conversion for JPEG."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        data, fmt, width, height = compress_image_data(
            buffer.getvalue(),
            "jpeg",
            jpeg_quality=85,
        )

        assert fmt == "jpeg"
        # Result should be valid JPEG
        result_img = Image.open(io.BytesIO(data))
        assert result_img.mode == "RGB"

    def test_compress_other_format(self):
        """Test compression of other formats (gif, webp)."""
        img = Image.new("P", (50, 50))
        buffer = io.BytesIO()
        img.save(buffer, format="GIF")

        data, fmt, width, height = compress_image_data(
            buffer.getvalue(),
            "gif",
        )

        assert fmt == "gif"
        assert width == 50
        assert height == 50


class TestImageCompressor:
    """Tests for ImageCompressor class."""

    def _create_test_image(self, size: int = 100, format_name: str = "png") -> ExtractedImage:
        """Create a test image."""
        img = Image.new("RGB", (size, size), color="red")
        buffer = io.BytesIO()
        if format_name.lower() in ("jpg", "jpeg"):
            img.save(buffer, format="JPEG")
        else:
            img.save(buffer, format=format_name.upper())

        # Make sure data is large enough to not be skipped
        data = buffer.getvalue()
        if len(data) < 10240:
            # Pad with comments to make larger
            img = Image.new("RGB", (200, 200), color="blue")
            buffer = io.BytesIO()
            if format_name.lower() in ("jpg", "jpeg"):
                img.save(buffer, format="JPEG", quality=100)
            else:
                img.save(buffer, format=format_name.upper())
            data = buffer.getvalue()

        return ExtractedImage(
            data=data,
            format=format_name,
            filename=f"test.{format_name}",
            source_document=Path("test.docx"),
            position=0,
        )

    def test_init_default_config(self):
        """Test default initialization."""
        compressor = ImageCompressor()
        assert compressor.config is not None
        assert isinstance(compressor.config, CompressionConfig)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = CompressionConfig(jpeg_quality=50)
        compressor = ImageCompressor(config=config)
        assert compressor.config.jpeg_quality == 50

    def test_compress_png(self):
        """Test compressing a PNG image."""
        compressor = ImageCompressor()
        image = self._create_test_image(200, "png")

        with patch("markit.image.compressor.shutil.which", return_value=None):
            result = compressor.compress(image)

        assert isinstance(result, CompressedImage)
        assert result.format == "png"
        assert result.width > 0
        assert result.height > 0

    def test_compress_jpeg(self):
        """Test compressing a JPEG image."""
        compressor = ImageCompressor()
        image = self._create_test_image(200, "jpeg")

        result = compressor.compress(image)

        assert isinstance(result, CompressedImage)
        assert result.format == "jpeg"

    def test_skip_small_image(self):
        """Test that small images are skipped."""
        config = CompressionConfig(skip_if_smaller_than=100000)
        compressor = ImageCompressor(config=config)

        # Create small image
        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = ExtractedImage(
            data=buffer.getvalue(),
            format="png",
            filename="small.png",
            source_document=Path("test.docx"),
            position=0,
        )

        result = compressor.compress(image)

        # Should return original data
        assert result.original_size == result.compressed_size
