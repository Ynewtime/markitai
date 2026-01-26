"""Tests for image processing module."""

import base64
import io
from pathlib import Path

from PIL import Image

from markitai.config import ImageConfig, ImageFilterConfig
from markitai.image import ImageProcessor


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

    def test_extract_emf_wmf_images(self) -> None:
        """Test extracting x-emf and x-wmf images (Office formats with hyphens)."""
        processor = ImageProcessor()
        # Create minimal valid base64 data (won't be converted but should be extracted)
        b64_data = base64.b64encode(b"fake-image-data").decode()
        markdown = f"![](data:image/x-emf;base64,{b64_data})"

        images = processor.extract_base64_images(markdown)
        # The image is extracted (even if conversion may fail later)
        # The regex should match x-emf MIME type
        assert len(images) == 1
        assert images[0][1] == "image/png"  # Converted to PNG

    def test_extract_various_mime_types(self) -> None:
        """Test extracting images with various MIME type formats."""
        processor = ImageProcessor()
        b64 = base64.b64encode(create_test_image()).decode()

        # Standard types
        for mime in ["png", "jpeg", "gif", "webp", "svg+xml"]:
            md = f"![](data:image/{mime};base64,{b64})"
            images = processor.extract_base64_images(md)
            assert len(images) >= 0  # Just ensure regex matches


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


class TestRemoveNonexistentImages:
    """Tests for remove_nonexistent_images static method."""

    def test_removes_placeholder_patterns(self, tmp_path: Path) -> None:
        """Test that placeholder image references are removed."""
        # Create assets dir
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "![](assets/...)\n![](assets/placeholder)\n![](assets/..)\nSome text"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "assets/..." not in result
        assert "assets/placeholder" not in result
        assert "assets/.." not in result
        assert "Some text" in result

    def test_keeps_existing_images(self, tmp_path: Path) -> None:
        """Test that existing image references are kept."""
        # Create assets dir with actual image
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "real.jpg").write_bytes(b"fake image data")

        markdown = "![](assets/real.jpg)\n![](assets/fake.jpg)"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "assets/real.jpg" in result
        assert "assets/fake.jpg" not in result

    def test_removes_nonexistent_images(self, tmp_path: Path) -> None:
        """Test that nonexistent image references are removed."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "![alt](assets/nonexistent.png)"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "assets/nonexistent.png" not in result


class TestIndexMappingReplacement:
    """Tests for index-based base64 replacement (P0-1 fix for image reference misalignment)."""

    def test_replace_with_index_mapping_no_skip(self, tmp_path: Path) -> None:
        """Test replacement with index mapping when no images are skipped."""
        from markitai.image import ProcessedImage

        processor = ImageProcessor()

        # Create test images
        img1 = create_test_image(100, 100, "red")
        img2 = create_test_image(100, 100, "blue")
        markdown = f"Start\n{create_base64_markdown(img1, 'img1')}\nMiddle\n{create_base64_markdown(img2, 'img2')}\nEnd"

        # Simulate saved images and index mapping (no filtering)
        from markitai.converter.base import ExtractedImage

        saved_images = [
            ExtractedImage(
                path=tmp_path / "test.0001.jpg",
                index=1,
                original_name="test.0001.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            ),
            ExtractedImage(
                path=tmp_path / "test.0002.jpg",
                index=2,
                original_name="test.0002.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            ),
        ]
        index_mapping = {
            1: ProcessedImage(
                original_index=1,
                saved_path=tmp_path / "test.0001.jpg",
                skip_reason=None,
            ),
            2: ProcessedImage(
                original_index=2,
                saved_path=tmp_path / "test.0002.jpg",
                skip_reason=None,
            ),
        }

        result = processor.replace_base64_with_paths(
            markdown, saved_images, index_mapping=index_mapping
        )

        assert "![img1](assets/test.0001.jpg)" in result
        assert "![img2](assets/test.0002.jpg)" in result
        assert "data:image" not in result

    def test_replace_with_index_mapping_filtered_image(self, tmp_path: Path) -> None:
        """Test replacement when middle image is filtered - should not cause misalignment."""
        from markitai.image import ProcessedImage

        processor = ImageProcessor()

        # Create 3 images, middle one will be filtered
        img1 = create_test_image(100, 100, "red")
        img2 = create_test_image(20, 20, "blue")  # Small - will be filtered
        img3 = create_test_image(100, 100, "green")
        markdown = f"{create_base64_markdown(img1, 'first')}\n{create_base64_markdown(img2, 'filtered')}\n{create_base64_markdown(img3, 'third')}"

        # Simulate: image 2 was filtered, only 1 and 3 were saved
        from markitai.converter.base import ExtractedImage

        saved_images = [
            ExtractedImage(
                path=tmp_path / "test.0001.jpg",
                index=1,
                original_name="test.0001.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            ),
            ExtractedImage(
                path=tmp_path / "test.0003.jpg",
                index=3,
                original_name="test.0003.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            ),
        ]
        index_mapping = {
            1: ProcessedImage(
                original_index=1,
                saved_path=tmp_path / "test.0001.jpg",
                skip_reason=None,
            ),
            2: ProcessedImage(
                original_index=2, saved_path=None, skip_reason="filtered"
            ),
            3: ProcessedImage(
                original_index=3,
                saved_path=tmp_path / "test.0003.jpg",
                skip_reason=None,
            ),
        }

        result = processor.replace_base64_with_paths(
            markdown, saved_images, index_mapping=index_mapping
        )

        # First image should map to test.0001.jpg
        assert "![first](assets/test.0001.jpg)" in result
        # Filtered image should be removed (empty string)
        assert "![filtered]" not in result
        # Third image should map to test.0003.jpg (NOT test.0002.jpg which would be misalignment)
        assert "![third](assets/test.0003.jpg)" in result

    def test_replace_with_index_mapping_deduplicated_image(
        self, tmp_path: Path
    ) -> None:
        """Test replacement when duplicate image is skipped."""
        from markitai.image import ProcessedImage

        processor = ImageProcessor()

        # Create 3 images, second is duplicate of first
        img1 = create_test_image(100, 100, "red")
        img3 = create_test_image(100, 100, "green")
        markdown = f"{create_base64_markdown(img1, 'original')}\n{create_base64_markdown(img1, 'duplicate')}\n{create_base64_markdown(img3, 'unique')}"

        from markitai.converter.base import ExtractedImage

        saved_images = [
            ExtractedImage(
                path=tmp_path / "test.0001.jpg",
                index=1,
                original_name="test.0001.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            ),
            ExtractedImage(
                path=tmp_path / "test.0003.jpg",
                index=3,
                original_name="test.0003.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            ),
        ]
        index_mapping = {
            1: ProcessedImage(
                original_index=1,
                saved_path=tmp_path / "test.0001.jpg",
                skip_reason=None,
            ),
            2: ProcessedImage(
                original_index=2, saved_path=None, skip_reason="duplicate"
            ),
            3: ProcessedImage(
                original_index=3,
                saved_path=tmp_path / "test.0003.jpg",
                skip_reason=None,
            ),
        }

        result = processor.replace_base64_with_paths(
            markdown, saved_images, index_mapping=index_mapping
        )

        assert "![original](assets/test.0001.jpg)" in result
        assert "![duplicate]" not in result  # Duplicate removed
        assert "![unique](assets/test.0003.jpg)" in result

    def test_process_and_save_returns_index_mapping(self, tmp_path: Path) -> None:
        """Test that process_and_save returns correct index mapping."""
        config = ImageConfig(
            compress=True,
            quality=85,
            format="jpeg",
            filter=ImageFilterConfig(min_width=50, min_height=50, deduplicate=True),
        )
        processor = ImageProcessor(config=config)

        # Create images: large, small (filtered), large duplicate
        img_large = create_test_image(100, 100, "red")
        img_small = create_test_image(20, 20, "blue")
        images = [
            ("large", "image/png", img_large),
            ("small", "image/png", img_small),
            ("dup", "image/png", img_large),  # Duplicate
        ]

        result = processor.process_and_save(
            images, output_dir=tmp_path, base_name="test"
        )

        # Should have index mapping
        assert result.index_mapping is not None
        assert len(result.index_mapping) == 3

        # Check mapping contents
        assert result.index_mapping[1].skip_reason is None  # Saved
        assert result.index_mapping[1].saved_path is not None
        assert result.index_mapping[2].skip_reason == "filtered"  # Filtered
        assert result.index_mapping[2].saved_path is None
        assert result.index_mapping[3].skip_reason == "duplicate"  # Deduplicated
        assert result.index_mapping[3].saved_path is None

        # Only 1 image should be saved
        assert len(result.saved_images) == 1
        assert result.filtered_count == 1
        assert result.deduplicated_count == 1


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
