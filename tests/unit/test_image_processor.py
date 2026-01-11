"""Tests for ImageProcessingService."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from markit.services.image_processor import (
    ImageProcessingConfig,
    ImageProcessingService,
    _compress_single_image_process,
    _generate_image_filename,
    _sanitize_filename,
)


class TestSanitizeFilename:
    """Tests for _sanitize_filename function."""

    def test_removes_special_chars(self):
        """Special characters are replaced with underscores."""
        assert _sanitize_filename("file<>:name.png") == "file_name.png"

    def test_removes_multiple_underscores(self):
        """Multiple underscores/spaces are collapsed to single underscore."""
        assert _sanitize_filename("file___name.png") == "file_name.png"
        assert _sanitize_filename("file   name.png") == "file_name.png"
        assert _sanitize_filename("file _ _ name.png") == "file_name.png"

    def test_handles_unicode(self):
        """Unicode characters are preserved."""
        assert _sanitize_filename("中文文件名.png") == "中文文件名.png"
        assert _sanitize_filename("日本語ファイル.jpg") == "日本語ファイル.jpg"

    def test_strips_leading_trailing_underscores(self):
        """Leading and trailing underscores are removed."""
        assert _sanitize_filename("_filename_.png") == "filename_.png"
        assert _sanitize_filename("__test__") == "test"

    def test_handles_empty_string(self):
        """Empty string returns empty string."""
        assert _sanitize_filename("") == ""

    def test_handles_only_special_chars(self):
        """String with only special chars becomes empty."""
        result = _sanitize_filename("<>:?*|")
        assert result == ""

    def test_preserves_dots(self):
        """Dots are preserved."""
        assert _sanitize_filename("file.name.png") == "file.name.png"


class TestGenerateImageFilename:
    """Tests for _generate_image_filename function."""

    def test_basic_filename_generation(self):
        """Basic filename generation follows pattern."""
        filename = _generate_image_filename(Path("document.docx"), 1, "png")
        assert filename == "document.docx.001.png"

    def test_different_index(self):
        """Different indices are properly formatted."""
        filename = _generate_image_filename(Path("document.docx"), 42, "jpeg")
        assert filename == "document.docx.042.jpeg"
        filename = _generate_image_filename(Path("document.docx"), 999, "png")
        assert filename == "document.docx.999.png"

    def test_legacy_doc_format(self):
        """Legacy .doc files are handled correctly."""
        filename = _generate_image_filename(Path("file-sample_100kB.doc"), 1, "jpeg")
        assert filename == "file-sample_100kB.doc.001.jpeg"

    def test_pdf_format(self):
        """PDF files are handled correctly."""
        filename = _generate_image_filename(Path("report.pdf"), 3, "png")
        assert filename == "report.pdf.003.png"

    def test_sanitizes_special_chars(self):
        """Special characters in filename are sanitized."""
        filename = _generate_image_filename(Path("file<with>special:chars.pdf"), 1, "png")
        assert "<" not in filename
        assert ">" not in filename
        assert ":" not in filename


class TestImageProcessingConfig:
    """Tests for ImageProcessingConfig dataclass."""

    def test_default_values(self):
        """Default configuration values are correct."""
        config = ImageProcessingConfig()
        assert config.compress_images is True
        assert config.use_process_pool is True
        assert config.process_pool_threshold > 0
        assert config.max_workers > 0

    def test_custom_values(self):
        """Custom values are set correctly."""
        config = ImageProcessingConfig(
            compress_images=False,
            png_optimization_level=3,
            jpeg_quality=90,
            max_dimension=1024,
            use_process_pool=False,
            process_pool_threshold=10,
        )
        assert config.compress_images is False
        assert config.png_optimization_level == 3
        assert config.jpeg_quality == 90
        assert config.max_dimension == 1024
        assert config.use_process_pool is False
        assert config.process_pool_threshold == 10


class TestImageProcessingService:
    """Tests for ImageProcessingService class."""

    @pytest.fixture
    def default_service(self):
        """Create service with default configuration."""
        return ImageProcessingService()

    @pytest.fixture
    def custom_service(self):
        """Create service with custom configuration."""
        config = ImageProcessingConfig(
            compress_images=True,
            use_process_pool=True,
            process_pool_threshold=5,
        )
        return ImageProcessingService(config)

    # --- Pool Selection Tests ---

    def test_should_use_thread_pool_when_below_threshold(self, custom_service):
        """Thread pool is used when image count is below threshold."""
        assert custom_service._should_use_process_pool(3) is False
        assert custom_service._should_use_process_pool(4) is False

    def test_should_use_process_pool_when_above_threshold(self, custom_service):
        """Process pool is used when image count reaches threshold."""
        assert custom_service._should_use_process_pool(5) is True
        assert custom_service._should_use_process_pool(100) is True

    def test_should_use_thread_pool_when_disabled(self, default_service):
        """Thread pool is used when process pool is disabled."""
        default_service.config.use_process_pool = False
        assert default_service._should_use_process_pool(100) is False

    # --- Lazy Initialization Tests ---

    def test_image_compressor_lazy_init(self, default_service):
        """Image compressor is lazily initialized."""
        assert default_service._image_compressor is None
        compressor = default_service._get_image_compressor()
        assert compressor is not None
        assert default_service._image_compressor is compressor

    def test_process_pool_lazy_init(self, default_service):
        """Process pool is lazily initialized."""
        assert default_service._process_pool is None
        pool = default_service._get_process_pool()
        assert pool is not None
        assert default_service._process_pool is pool
        # Clean up
        default_service.shutdown()

    def test_shutdown_cleans_up_pool(self, default_service):
        """Shutdown properly cleans up process pool."""
        # Create the pool
        _ = default_service._get_process_pool()
        assert default_service._process_pool is not None
        # Shutdown
        default_service.shutdown()
        assert default_service._process_pool is None

    # --- Optimize Images Tests ---

    async def test_optimize_images_empty_list(self, default_service):
        """Empty image list returns empty results."""
        result, filename_map = await default_service.optimize_images_parallel([], Path("test.pdf"))
        assert result == []
        assert filename_map == {}

    async def test_optimize_images_deduplication(self, custom_service):
        """Duplicate images (same content) are deduplicated."""
        # Create mock images with same data (should be deduplicated)
        from markit.converters.base import ExtractedImage

        images = [
            ExtractedImage(
                data=b"same_content_123",
                format="png",
                filename="img1.png",
                source_document=Path("test.pdf"),
                position=1,
            ),
            ExtractedImage(
                data=b"same_content_123",  # Same content
                format="png",
                filename="img2.png",
                source_document=Path("test.pdf"),
                position=2,
            ),
            ExtractedImage(
                data=b"different_content",
                format="png",
                filename="img3.png",
                source_document=Path("test.pdf"),
                position=3,
            ),
        ]

        # Mock the compressor to avoid actual compression
        mock_compressor = Mock()
        mock_compressor.compress = Mock(
            side_effect=lambda img: Mock(
                data=img.data,
                format=img.format,
                filename=img.filename,
                width=100,
                height=100,
            )
        )
        custom_service._image_compressor = mock_compressor

        result, filename_map = await custom_service.optimize_images_parallel(
            images, Path("test.pdf")
        )

        # Should only have 2 unique images
        assert len(result) == 2
        # All 3 original filenames should have mappings
        assert len(filename_map) == 3
        # img1 and img2 should map to the same filename
        assert filename_map["img1.png"] == filename_map["img2.png"]

    # --- Markdown Reference Update Tests ---

    def test_update_markdown_references(self, default_service):
        """Markdown references are correctly updated."""
        markdown = """# Test Document

![Image 1](assets/old_image.png)
![Image 2](assets/another_old.jpg)

Some text with inline ![](old_inline.png) reference.
"""
        filename_map = {
            "old_image.png": "test.pdf.001.png",
            "another_old.jpg": "test.pdf.002.jpeg",
            "old_inline.png": "new_inline.png",
        }

        result = default_service.update_markdown_references(markdown, filename_map)

        assert "assets/test.pdf.001.png" in result
        assert "assets/test.pdf.002.jpeg" in result
        assert "assets/old_image.png" not in result
        assert "assets/another_old.jpg" not in result

    def test_update_markdown_references_removes_failed_images(self, default_service):
        """Failed image references are removed from markdown."""
        markdown = """![Working](assets/working.png)
![Failed](assets/failed.png)
"""
        filename_map = {
            "working.png": "test.pdf.001.png",
            "failed.png": None,  # Processing failed
        }

        result = default_service.update_markdown_references(markdown, filename_map)

        assert "assets/test.pdf.001.png" in result
        assert "![Failed](assets/failed.png)" not in result

    # --- Prepare for Analysis Tests ---

    def test_prepare_for_analysis(self, default_service):
        """Processed images are correctly prepared for analysis."""
        from markit.converters.base import ExtractedImage

        images = [
            ExtractedImage(
                data=b"image_data_1",
                format="png",
                filename="test.pdf.001.png",
                source_document=Path("test.pdf"),
                position=1,
                width=100,
                height=200,
            ),
            ExtractedImage(
                data=b"image_data_2",
                format="jpeg",
                filename="test.pdf.002.jpeg",
                source_document=Path("test.pdf"),
                position=2,
                width=300,
                height=400,
            ),
        ]

        result = default_service.prepare_for_analysis(images)

        assert len(result) == 2
        assert result[0].filename == "test.pdf.001.png"
        assert result[0].format == "png"
        assert result[0].width == 100
        assert result[0].height == 200
        assert result[1].filename == "test.pdf.002.jpeg"
        assert result[1].format == "jpeg"


class TestCompressSingleImageProcess:
    """Tests for _compress_single_image_process module-level function."""

    @pytest.fixture
    def sample_png_data(self):
        """Create a simple valid PNG image."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    @pytest.fixture
    def sample_jpeg_data(self):
        """Create a simple valid JPEG image."""
        from io import BytesIO

        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_compress_png(self, sample_png_data):
        """PNG compression returns valid PNG data."""
        compressed, fmt, width, height = _compress_single_image_process(
            sample_png_data,
            "png",
            png_optimization_level=2,
            jpeg_quality=85,
            max_dimension=2048,
        )

        assert fmt == "png"
        assert width == 100
        assert height == 100
        assert len(compressed) > 0

        # Verify it's valid PNG
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(compressed))
        assert img.format == "PNG"

    def test_compress_jpeg(self, sample_jpeg_data):
        """JPEG compression returns valid JPEG data."""
        compressed, fmt, width, height = _compress_single_image_process(
            sample_jpeg_data,
            "jpeg",
            png_optimization_level=2,
            jpeg_quality=85,
            max_dimension=2048,
        )

        assert fmt == "jpeg"
        assert width == 100
        assert height == 100
        assert len(compressed) > 0

        # Verify it's valid JPEG
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(compressed))
        assert img.format == "JPEG"

    def test_resize_large_image(self):
        """Large images are resized to max dimension."""
        from io import BytesIO

        from PIL import Image

        # Create a large image
        img = Image.new("RGB", (4000, 2000), color=(0, 255, 0))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        large_data = buffer.getvalue()

        compressed, fmt, width, height = _compress_single_image_process(
            large_data,
            "png",
            png_optimization_level=2,
            jpeg_quality=85,
            max_dimension=1024,  # Max 1024
        )

        # Should be resized
        assert width <= 1024
        assert height <= 1024
        # Aspect ratio should be maintained
        assert abs(width / height - 2.0) < 0.01  # Original was 4000x2000

    def test_convert_rgba_to_rgb_for_jpeg(self):
        """RGBA images are converted to RGB for JPEG output."""
        from io import BytesIO

        from PIL import Image

        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        rgba_data = buffer.getvalue()

        compressed, fmt, width, height = _compress_single_image_process(
            rgba_data,
            "jpeg",  # Request JPEG output
            png_optimization_level=2,
            jpeg_quality=85,
            max_dimension=2048,
        )

        assert fmt == "jpeg"
        # Should be valid JPEG
        result_img = Image.open(BytesIO(compressed))
        assert result_img.format == "JPEG"
        assert result_img.mode == "RGB"
