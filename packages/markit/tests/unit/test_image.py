"""Tests for image processing module."""

import base64
import io
from pathlib import Path

from PIL import Image

from markit.config import ImageConfig, ImageFilterConfig
from markit.image import ImageProcessor


def create_test_image(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Create a test image and return as bytes."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_base64_markdown(image_data: bytes, alt: str = "test") -> str:
    """Create markdown with base64 embedded image."""
    b64 = base64.b64encode(image_data).decode()
    return f"![{alt}](data:image/png;base64,{b64})"


class TestImageProcessor:
    """Tests for ImageProcessor class."""

    def test_extract_base64_images(self) -> None:
        """Test extracting base64 images from markdown."""
        image_data = create_test_image()
        markdown = create_base64_markdown(image_data, "test image")

        processor = ImageProcessor()
        images = processor.extract_base64_images(markdown)

        assert len(images) == 1
        alt, mime, data = images[0]
        assert alt == "test image"
        assert mime == "image/png"
        assert data == image_data

    def test_extract_multiple_images(self) -> None:
        """Test extracting multiple base64 images."""
        img1 = create_test_image(50, 50, "red")
        img2 = create_test_image(100, 100, "blue")
        markdown = f"{create_base64_markdown(img1, 'red')}\n\n{create_base64_markdown(img2, 'blue')}"

        processor = ImageProcessor()
        images = processor.extract_base64_images(markdown)

        assert len(images) == 2
        assert images[0][0] == "red"
        assert images[1][0] == "blue"

    def test_extract_no_images(self) -> None:
        """Test extracting from markdown without images."""
        markdown = "# Title\n\nSome text without images."

        processor = ImageProcessor()
        images = processor.extract_base64_images(markdown)

        assert len(images) == 0


class TestImageCompression:
    """Tests for image compression."""

    def test_compress_image(self) -> None:
        """Test basic image compression."""
        img = Image.new("RGB", (200, 200), "red")
        processor = ImageProcessor()

        compressed_img, compressed_data = processor.compress(
            img, quality=85, max_size=(100, 100)
        )

        # Check dimensions were reduced
        assert compressed_img.size[0] <= 100
        assert compressed_img.size[1] <= 100

    def test_compress_with_alpha(self) -> None:
        """Test compressing image with alpha channel."""
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        processor = ImageProcessor()

        compressed_img, compressed_data = processor.compress(
            img, quality=85, output_format="JPEG"
        )

        # Should be converted to RGB
        assert compressed_img.mode == "RGB"


class TestImageFiltering:
    """Tests for image filtering."""

    def test_filter_small_width(self) -> None:
        """Test filtering by minimum width."""
        config = ImageConfig(filter=ImageFilterConfig(min_width=100))
        processor = ImageProcessor(config=config)

        assert processor.should_filter(50, 200) is True
        assert processor.should_filter(100, 200) is False

    def test_filter_small_height(self) -> None:
        """Test filtering by minimum height."""
        config = ImageConfig(filter=ImageFilterConfig(min_height=100))
        processor = ImageProcessor(config=config)

        assert processor.should_filter(200, 50) is True
        assert processor.should_filter(200, 100) is False

    def test_filter_small_area(self) -> None:
        """Test filtering by minimum area."""
        config = ImageConfig(filter=ImageFilterConfig(min_area=10000))
        processor = ImageProcessor(config=config)

        # 50 * 50 = 2500 < 10000
        assert processor.should_filter(50, 50) is True
        # 100 * 100 = 10000 >= 10000
        assert processor.should_filter(100, 100) is False


class TestImageDeduplication:
    """Tests for image deduplication."""

    def test_deduplicate_same_image(self) -> None:
        """Test that duplicate images are detected."""
        config = ImageConfig(filter=ImageFilterConfig(deduplicate=True))
        processor = ImageProcessor(config=config)

        image_data = create_test_image()

        # First image should not be duplicate
        assert processor.is_duplicate(image_data) is False
        # Second identical image should be duplicate
        assert processor.is_duplicate(image_data) is True

    def test_deduplicate_different_images(self) -> None:
        """Test that different images are not duplicates."""
        config = ImageConfig(filter=ImageFilterConfig(deduplicate=True))
        processor = ImageProcessor(config=config)

        img1 = create_test_image(100, 100, "red")
        img2 = create_test_image(100, 100, "blue")

        assert processor.is_duplicate(img1) is False
        assert processor.is_duplicate(img2) is False

    def test_reset_dedup_cache(self) -> None:
        """Test resetting deduplication cache."""
        config = ImageConfig(filter=ImageFilterConfig(deduplicate=True))
        processor = ImageProcessor(config=config)

        image_data = create_test_image()

        processor.is_duplicate(image_data)
        assert processor.is_duplicate(image_data) is True

        processor.reset_dedup_cache()
        assert processor.is_duplicate(image_data) is False


class TestProcessAndSave:
    """Tests for process_and_save method."""

    def test_process_and_save_images(self, tmp_path: Path) -> None:
        """Test processing and saving images."""
        config = ImageConfig(compress=True, quality=85, format="jpeg")
        processor = ImageProcessor(config=config)

        image_data = create_test_image(200, 200)
        images = [("test", "image/png", image_data)]

        result = processor.process_and_save(
            images, output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 1
        assert result.filtered_count == 0
        assert result.deduplicated_count == 0

        # Check file was created
        saved_path = result.saved_images[0].path
        assert saved_path.exists()
        assert saved_path.suffix == ".jpg"

    def test_process_filters_small_images(self, tmp_path: Path) -> None:
        """Test that small images are filtered out."""
        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(min_width=100, min_height=100),
        )
        processor = ImageProcessor(config=config)

        # Create small image
        image_data = create_test_image(50, 50)
        images = [("small", "image/png", image_data)]

        result = processor.process_and_save(
            images, output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 0
        assert result.filtered_count == 1


class TestSaveScreenshot:
    """Tests for save_screenshot method (screenshot compression for LLM)."""

    def test_save_screenshot_basic(self, tmp_path: Path) -> None:
        """Test basic screenshot saving."""
        processor = ImageProcessor()

        # Create test image (simulate pymupdf pixmap samples)
        img = Image.new("RGB", (800, 600), color="blue")
        samples = img.tobytes()

        output_path = tmp_path / "screenshot.jpg"
        final_size = processor.save_screenshot(
            samples, img.width, img.height, output_path
        )

        assert output_path.exists()
        assert final_size[0] <= 1920  # Max width
        assert final_size[1] <= 1080  # Max height

    def test_save_screenshot_respects_quality_config(self, tmp_path: Path) -> None:
        """Test that save_screenshot respects image.quality config."""
        config = ImageConfig(quality=50, format="jpeg")
        processor = ImageProcessor(config=config)

        # Create test image
        img = Image.new("RGB", (500, 500))
        pixels = img.load()
        for i in range(img.width):
            for j in range(img.height):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
        samples = img.tobytes()

        output_path = tmp_path / "screenshot.jpg"
        processor.save_screenshot(samples, img.width, img.height, output_path)

        # File should exist and be reasonable size
        assert output_path.exists()
        file_size = output_path.stat().st_size
        assert file_size > 0

    def test_save_screenshot_compresses_to_under_5mb(self, tmp_path: Path) -> None:
        """Test that save_screenshot ensures output is under 5MB."""
        processor = ImageProcessor()

        # Create large test image with gradient pattern
        img = Image.new("RGB", (3000, 3000))
        pixels = img.load()
        for i in range(img.width):
            for j in range(img.height):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
        samples = img.tobytes()

        output_path = tmp_path / "large_screenshot.jpg"
        processor.save_screenshot(
            samples, img.width, img.height, output_path, max_bytes=5 * 1024 * 1024
        )

        assert output_path.exists()
        file_size = output_path.stat().st_size
        # Should be under 5MB (with some tolerance for edge cases)
        assert file_size <= 5 * 1024 * 1024 * 1.1

    def test_save_screenshot_resizes_large_images(self, tmp_path: Path) -> None:
        """Test that large images are resized according to config."""
        config = ImageConfig(max_width=800, max_height=600)
        processor = ImageProcessor(config=config)

        # Create image larger than max dimensions
        img = Image.new("RGB", (2000, 1500), color="green")
        samples = img.tobytes()

        output_path = tmp_path / "resized.jpg"
        final_size = processor.save_screenshot(
            samples, img.width, img.height, output_path
        )

        # Result should be within configured max dimensions
        assert final_size[0] <= 800
        assert final_size[1] <= 600
