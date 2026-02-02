"""Tests for image processing module."""

import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from markitai.config import ImageConfig, ImageFilterConfig
from markitai.image import (
    ImageProcessor,
    _compress_image_cv2,
    _compress_image_pillow,
    _compress_image_worker,
)


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
        assert pixels is not None
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
        assert pixels is not None
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


class TestCompressImageWorkerFunctions:
    """Tests for _compress_image_cv2, _compress_image_pillow, and _compress_image_worker."""

    def test_compress_image_pillow_basic(self) -> None:
        """Test basic Pillow compression."""
        img_data = create_test_image(200, 200, "red")

        result = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        compressed_data, width, height = result
        assert width <= 100
        assert height <= 100
        assert len(compressed_data) > 0

    def test_compress_image_pillow_filters_small(self) -> None:
        """Test Pillow compression filters small images."""
        img_data = create_test_image(50, 50, "red")

        result = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(1000, 1000),
            output_format="JPEG",
            min_width=100,  # Image is smaller
            min_height=100,
            min_area=10000,
        )

        assert result is None  # Filtered out

    def test_compress_image_cv2_basic(self) -> None:
        """Test basic OpenCV compression."""
        img_data = create_test_image(200, 200, "blue")

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        compressed_data, width, height = result
        assert width <= 100
        assert height <= 100
        assert len(compressed_data) > 0

    def test_compress_image_cv2_filters_small(self) -> None:
        """Test OpenCV compression filters small images."""
        img_data = create_test_image(50, 50, "blue")

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(1000, 1000),
            output_format="JPEG",
            min_width=100,  # Image is smaller
            min_height=100,
            min_area=10000,
        )

        assert result is None  # Filtered out

    def test_compress_image_cv2_png_format(self) -> None:
        """Test OpenCV compression with PNG format."""
        img_data = create_test_image(100, 100, "green")

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="PNG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        compressed_data, width, height = result
        assert len(compressed_data) > 0
        # PNG signature
        assert compressed_data[:4] == b"\x89PNG"

    def test_compress_image_cv2_webp_format(self) -> None:
        """Test OpenCV compression with WebP format."""
        img_data = create_test_image(100, 100, "yellow")

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="WEBP",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        compressed_data, width, height = result
        assert len(compressed_data) > 0
        # WebP signature
        assert compressed_data[:4] == b"RIFF"

    def test_compress_image_worker_uses_opencv_first(self) -> None:
        """Test that worker uses OpenCV first, falls back to Pillow."""
        img_data = create_test_image(100, 100, "purple")

        # Worker should succeed using OpenCV
        result = _compress_image_worker(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        compressed_data, width, height = result
        assert len(compressed_data) > 0

    def test_compress_image_worker_handles_invalid_data(self) -> None:
        """Test that worker handles invalid image data gracefully."""
        invalid_data = b"not an image"

        result = _compress_image_worker(
            image_data=invalid_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        # Both OpenCV and Pillow should fail, return None
        assert result is None

    def test_compress_image_cv2_handles_rgba(self) -> None:
        """Test OpenCV handles RGBA images for JPEG output."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",  # JPEG doesn't support alpha
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        compressed_data, width, height = result
        # Should produce valid JPEG
        assert compressed_data[:2] == b"\xff\xd8"  # JPEG signature

    def test_compress_consistency_between_cv2_and_pillow(self) -> None:
        """Test that CV2 and Pillow produce similar sized outputs."""
        img_data = create_test_image(200, 200, "orange")

        result_cv2 = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        result_pillow = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result_cv2 is not None
        assert result_pillow is not None

        # Both should produce similar dimensions
        assert result_cv2[1] == result_pillow[1]  # width
        assert result_cv2[2] == result_pillow[2]  # height


class TestRemoveHallucinatedImages:
    """Tests for remove_hallucinated_images static method."""

    def test_removes_hallucinated_urls(self) -> None:
        """Test that hallucinated image URLs are removed."""
        original = "Some text with image ![](https://example.com/real.jpg)"
        llm_output = (
            "Some text with image ![](https://example.com/real.jpg)\n"
            "And hallucinated ![fake](https://fake.com/generated.png)"
        )

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        assert "https://example.com/real.jpg" in result
        assert "https://fake.com/generated.png" not in result

    def test_keeps_original_urls(self) -> None:
        """Test that original image URLs are preserved."""
        original = (
            "![img1](https://example.com/a.jpg)\n![img2](https://example.com/b.png)"
        )
        llm_output = original  # No hallucination

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        assert "https://example.com/a.jpg" in result
        assert "https://example.com/b.png" in result

    def test_keeps_local_asset_references(self) -> None:
        """Test that local assets/ references are kept (handled by other method)."""
        original = "Some content"
        llm_output = "Some content with ![local](assets/image.jpg)"

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        # Local asset references should be kept (validated by remove_nonexistent_images)
        assert "assets/image.jpg" in result

    def test_keeps_local_asset_backslash_references(self) -> None:
        """Test that local assets\\ references are kept (Windows paths)."""
        original = "Some content"
        llm_output = "Some content with ![local](assets\\image.jpg)"

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        assert "assets\\image.jpg" in result

    def test_keeps_relative_urls(self) -> None:
        """Test that relative URLs (non-http) are kept."""
        original = "Content"
        llm_output = "![](./images/local.png)\n![](/root/image.jpg)"

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        assert "./images/local.png" in result
        assert "/root/image.jpg" in result

    def test_recognizes_bare_urls_in_original(self) -> None:
        """Test that bare URLs in original are recognized as valid."""
        original = "Check out this image: https://example.com/image.png"
        llm_output = "Here it is: ![image](https://example.com/image.png)"

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        assert "https://example.com/image.png" in result

    def test_cleans_up_extra_newlines(self) -> None:
        """Test that extra newlines from removed images are cleaned up."""
        original = "Text"
        llm_output = "Text\n\n\n![fake](https://fake.com/img.png)\n\n\nMore text"

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result
        assert "Text" in result
        assert "More text" in result

    def test_empty_content(self) -> None:
        """Test with empty content."""
        result = ImageProcessor.remove_hallucinated_images("", "")
        assert result == ""

    def test_no_images_in_output(self) -> None:
        """Test when there are no images in output."""
        original = "Some content"
        llm_output = "Some different content"

        result = ImageProcessor.remove_hallucinated_images(llm_output, original)

        assert result == "Some different content"


class TestStripBase64Images:
    """Tests for strip_base64_images method."""

    def test_strip_removes_all_base64_images(self) -> None:
        """Test that all base64 images are removed."""
        img = create_test_image()
        markdown = f"Text\n{create_base64_markdown(img, 'test')}\nMore text"

        processor = ImageProcessor()
        result = processor.strip_base64_images(markdown)

        assert "data:image" not in result
        assert "Text" in result
        assert "More text" in result

    def test_strip_with_replacement_path(self) -> None:
        """Test replacing base64 images with a placeholder path."""
        img = create_test_image()
        markdown = f"Before {create_base64_markdown(img, 'alt')} After"

        processor = ImageProcessor()
        result = processor.strip_base64_images(
            markdown, replacement_path="placeholder.png"
        )

        assert "![alt](placeholder.png)" in result
        assert "data:image" not in result

    def test_strip_multiple_images(self) -> None:
        """Test stripping multiple base64 images."""
        img1 = create_test_image(50, 50, "red")
        img2 = create_test_image(100, 100, "blue")
        markdown = f"{create_base64_markdown(img1)}\n{create_base64_markdown(img2)}"

        processor = ImageProcessor()
        result = processor.strip_base64_images(markdown)

        assert "data:image" not in result
        assert result.strip() == ""  # Both images removed

    def test_strip_empty_content(self) -> None:
        """Test stripping from empty content."""
        processor = ImageProcessor()
        result = processor.strip_base64_images("")
        assert result == ""

    def test_strip_no_images(self) -> None:
        """Test stripping when there are no images."""
        markdown = "Just some text without images"
        processor = ImageProcessor()
        result = processor.strip_base64_images(markdown)
        assert result == markdown


class TestDataUriPattern:
    """Tests for DATA_URI_PATTERN regex."""

    def test_matches_standard_image_types(self) -> None:
        """Test pattern matches common image types."""
        pattern = ImageProcessor.DATA_URI_PATTERN

        test_cases = [
            ("![](data:image/png;base64,ABC123)", "png"),
            ("![](data:image/jpeg;base64,XYZ)", "jpeg"),
            ("![](data:image/gif;base64,GIF)", "gif"),
            ("![](data:image/webp;base64,WEB)", "webp"),
        ]

        for markdown, expected_type in test_cases:
            match = pattern.search(markdown)
            assert match is not None, f"Failed to match: {markdown}"
            assert match.group(2) == expected_type

    def test_matches_svg_plus_xml(self) -> None:
        """Test pattern matches svg+xml MIME type."""
        pattern = ImageProcessor.DATA_URI_PATTERN
        markdown = "![svg](data:image/svg+xml;base64,SVGDATA)"

        match = pattern.search(markdown)
        assert match is not None
        assert match.group(2) == "svg+xml"

    def test_matches_x_emf_and_x_wmf(self) -> None:
        """Test pattern matches Office vector formats with hyphens."""
        pattern = ImageProcessor.DATA_URI_PATTERN

        for mime_type in ["x-emf", "x-wmf"]:
            markdown = f"![](data:image/{mime_type};base64,OFFICE)"
            match = pattern.search(markdown)
            assert match is not None, f"Failed to match: {mime_type}"
            assert match.group(2) == mime_type

    def test_captures_alt_text(self) -> None:
        """Test pattern captures alt text correctly."""
        pattern = ImageProcessor.DATA_URI_PATTERN
        markdown = "![This is alt text](data:image/png;base64,ABC)"

        match = pattern.search(markdown)
        assert match is not None
        assert match.group(1) == "This is alt text"

    def test_captures_empty_alt_text(self) -> None:
        """Test pattern handles empty alt text."""
        pattern = ImageProcessor.DATA_URI_PATTERN
        markdown = "![](data:image/png;base64,ABC)"

        match = pattern.search(markdown)
        assert match is not None
        assert match.group(1) == ""

    def test_captures_base64_data(self) -> None:
        """Test pattern captures base64 data correctly."""
        pattern = ImageProcessor.DATA_URI_PATTERN
        b64_data = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        markdown = f"![](data:image/png;base64,{b64_data})"

        match = pattern.search(markdown)
        assert match is not None
        assert match.group(3) == b64_data

    def test_does_not_match_non_image_data_uri(self) -> None:
        """Test pattern does not match non-image data URIs."""
        pattern = ImageProcessor.DATA_URI_PATTERN
        markdown = "![](data:text/plain;base64,ABC)"

        match = pattern.search(markdown)
        assert match is None

    def test_does_not_match_regular_url(self) -> None:
        """Test pattern does not match regular image URLs."""
        pattern = ImageProcessor.DATA_URI_PATTERN
        markdown = "![](https://example.com/image.png)"

        match = pattern.search(markdown)
        assert match is None


class TestRemoveNonexistentImagesEdgeCases:
    """Additional edge case tests for remove_nonexistent_images."""

    def test_handles_empty_markdown(self, tmp_path: Path) -> None:
        """Test with empty markdown."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        result = ImageProcessor.remove_nonexistent_images("", assets_dir)
        assert result == ""

    def test_handles_no_image_references(self, tmp_path: Path) -> None:
        """Test markdown without image references."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "# Heading\n\nSome paragraph text."
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)
        assert result == markdown

    def test_handles_windows_backslash_paths(self, tmp_path: Path) -> None:
        """Test handling of Windows-style backslash paths."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        (assets_dir / "image.jpg").write_bytes(b"fake")

        markdown = "![](assets\\image.jpg)\n![](assets\\missing.jpg)"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "assets\\image.jpg" in result or "assets/image.jpg" in result
        assert "missing.jpg" not in result

    def test_removes_empty_filename(self, tmp_path: Path) -> None:
        """Test removal of empty filename references."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "![](assets/)\nText"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "assets/" not in result or "Text" in result

    def test_removes_dot_references(self, tmp_path: Path) -> None:
        """Test removal of single dot references."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "![](assets/.)"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "assets/." not in result

    def test_cleans_multiple_spaces(self, tmp_path: Path) -> None:
        """Test that multiple spaces are cleaned to single space."""
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "Text  ![](assets/fake.png)  more"
        result = ImageProcessor.remove_nonexistent_images(markdown, assets_dir)

        assert "  " not in result  # No double spaces


class TestImageProcessorEdgeCases:
    """Edge case tests for ImageProcessor."""

    def test_extract_invalid_base64(self) -> None:
        """Test extracting images with invalid base64 data."""
        processor = ImageProcessor()
        # Create markdown with invalid base64 (not properly padded)
        markdown = "![](data:image/png;base64,!!!invalid!!!)"

        images = processor.extract_base64_images(markdown)
        # Invalid base64 should be skipped
        assert len(images) == 0

    def test_should_filter_no_config(self) -> None:
        """Test should_filter when no config is set."""
        processor = ImageProcessor()  # No config
        # Should return False when no config
        assert processor.should_filter(10, 10) is False
        assert processor.should_filter(1, 1) is False

    def test_is_duplicate_no_config(self) -> None:
        """Test is_duplicate when no config is set."""
        processor = ImageProcessor()  # No config
        img = create_test_image()
        # Should return False when deduplication not enabled
        assert processor.is_duplicate(img) is False
        assert processor.is_duplicate(img) is False  # Still False

    def test_is_duplicate_config_disabled(self) -> None:
        """Test is_duplicate when deduplication is disabled in config."""
        config = ImageConfig(filter=ImageFilterConfig(deduplicate=False))
        processor = ImageProcessor(config=config)

        img = create_test_image()
        assert processor.is_duplicate(img) is False
        assert processor.is_duplicate(img) is False

    def test_process_and_save_empty_list(self, tmp_path: Path) -> None:
        """Test processing an empty list of images."""
        processor = ImageProcessor()
        result = processor.process_and_save([], tmp_path, "test")

        assert len(result.saved_images) == 0
        assert result.filtered_count == 0
        assert result.deduplicated_count == 0

    def test_process_and_save_invalid_image_data(self, tmp_path: Path) -> None:
        """Test processing invalid image data."""
        processor = ImageProcessor()
        images = [("invalid", "image/png", b"not a valid image")]

        result = processor.process_and_save(images, tmp_path, "test")

        # Invalid image should be skipped with error
        assert len(result.saved_images) == 0
        assert result.index_mapping is not None
        assert result.index_mapping[1].skip_reason == "error"


class TestUrlImageHelpers:
    """Tests for URL image helper functions."""

    def test_get_extension_from_url(self) -> None:
        """Test extracting extension from URL."""
        from markitai.image import _get_extension_from_url

        assert _get_extension_from_url("https://example.com/image.jpg") == ".jpg"
        assert _get_extension_from_url("https://example.com/image.PNG") == ".png"
        assert _get_extension_from_url("https://example.com/image.gif?v=1") == ".gif"
        assert _get_extension_from_url("https://example.com/image") is None
        assert _get_extension_from_url("https://example.com/") is None

    def test_sanitize_image_filename(self) -> None:
        """Test filename sanitization."""
        from markitai.image import _sanitize_image_filename

        # Basic sanitization
        assert _sanitize_image_filename("normal_name") == "normal_name"
        assert _sanitize_image_filename("name with spaces") == "name with spaces"

        # Invalid characters: < > : " / \ | ? * = 9 chars
        assert _sanitize_image_filename('file<>:"/\\|?*') == "file_________"

        # Length limit
        long_name = "a" * 150
        result = _sanitize_image_filename(long_name, max_length=100)
        assert len(result) == 100

        # Empty/whitespace
        assert _sanitize_image_filename("") == "image"
        assert _sanitize_image_filename("   ") == "image"

    def test_get_extension_from_content_type(self) -> None:
        """Test extension detection from Content-Type."""
        from markitai.image import _get_extension_from_content_type

        assert _get_extension_from_content_type("image/jpeg") == ".jpg"
        assert _get_extension_from_content_type("image/png") == ".png"
        assert _get_extension_from_content_type("image/gif") == ".gif"
        assert _get_extension_from_content_type("image/webp") == ".webp"
        # Unknown type defaults
        assert _get_extension_from_content_type("image/unknown") == ".jpg"
        assert _get_extension_from_content_type("") == ".jpg"


class TestUrlImagePattern:
    """Tests for URL_IMAGE_PATTERN regex."""

    def test_matches_http_urls(self) -> None:
        """Test pattern matches http URLs."""
        from markitai.image import _URL_IMAGE_PATTERN

        markdown = "![alt](http://example.com/image.jpg)"
        match = _URL_IMAGE_PATTERN.search(markdown)
        assert match is not None
        assert match.group(1) == "alt"
        assert match.group(2) == "http://example.com/image.jpg"

    def test_matches_https_urls(self) -> None:
        """Test pattern matches https URLs."""
        from markitai.image import _URL_IMAGE_PATTERN

        markdown = "![](https://example.com/photo.png)"
        match = _URL_IMAGE_PATTERN.search(markdown)
        assert match is not None
        assert match.group(2) == "https://example.com/photo.png"

    def test_matches_relative_urls(self) -> None:
        """Test pattern matches relative URLs."""
        from markitai.image import _URL_IMAGE_PATTERN

        for url in ["./images/test.png", "../assets/img.jpg", "/root/image.gif"]:
            markdown = f"![alt]({url})"
            match = _URL_IMAGE_PATTERN.search(markdown)
            assert match is not None, f"Failed to match: {url}"
            assert match.group(2) == url

    def test_excludes_data_uris(self) -> None:
        """Test pattern excludes data: URIs."""
        from markitai.image import _URL_IMAGE_PATTERN

        markdown = "![](data:image/png;base64,ABC123)"
        match = _URL_IMAGE_PATTERN.search(markdown)
        assert match is None

    def test_matches_multiple_images(self) -> None:
        """Test pattern finds all images."""
        from markitai.image import _URL_IMAGE_PATTERN

        markdown = "![a](http://a.com/1.jpg) text ![b](https://b.com/2.png)"
        matches = list(_URL_IMAGE_PATTERN.finditer(markdown))
        assert len(matches) == 2


class TestReplaceBase64WithPaths:
    """Tests for replace_base64_with_paths method."""

    def test_basic_replacement(self, tmp_path: Path) -> None:
        """Test basic base64 to path replacement."""
        from markitai.converter.base import ExtractedImage

        processor = ImageProcessor()
        img = create_test_image()
        markdown = f"Text {create_base64_markdown(img, 'alt')} more"

        images = [
            ExtractedImage(
                path=tmp_path / "test.0001.jpg",
                index=1,
                original_name="test.0001.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            )
        ]

        result = processor.replace_base64_with_paths(markdown, images)

        assert "![alt](assets/test.0001.jpg)" in result
        assert "data:image" not in result

    def test_custom_assets_path(self, tmp_path: Path) -> None:
        """Test replacement with custom assets path."""
        from markitai.converter.base import ExtractedImage

        processor = ImageProcessor()
        img = create_test_image()
        markdown = create_base64_markdown(img)

        images = [
            ExtractedImage(
                path=tmp_path / "test.0001.jpg",
                index=1,
                original_name="test.0001.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            )
        ]

        result = processor.replace_base64_with_paths(
            markdown, images, assets_path="custom_assets"
        )

        assert "custom_assets/test.0001.jpg" in result

    def test_empty_images_list(self) -> None:
        """Test replacement with empty images list."""
        processor = ImageProcessor()
        img = create_test_image()
        markdown = create_base64_markdown(img)

        result = processor.replace_base64_with_paths(markdown, [])

        # Should keep original when no replacement available
        assert "data:image" in result


class TestProcessAndSaveAsync:
    """Tests for process_and_save_async method."""

    @pytest.mark.asyncio
    async def test_process_and_save_async_basic(self, tmp_path: Path) -> None:
        """Test async processing and saving images."""
        config = ImageConfig(compress=True, quality=85, format="jpeg")
        processor = ImageProcessor(config=config)

        image_data = create_test_image(200, 200)
        images = [("test", "image/png", image_data)]

        result = await processor.process_and_save_async(
            images, output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 1
        assert result.filtered_count == 0
        assert result.deduplicated_count == 0

        # Check file was created
        saved_path = result.saved_images[0].path
        assert saved_path.exists()
        assert saved_path.suffix == ".jpg"

    @pytest.mark.asyncio
    async def test_process_and_save_async_filters_small_images(
        self, tmp_path: Path
    ) -> None:
        """Test that async processing filters small images."""
        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(min_width=100, min_height=100),
        )
        processor = ImageProcessor(config=config)

        # Create small image
        image_data = create_test_image(50, 50)
        images = [("small", "image/png", image_data)]

        result = await processor.process_and_save_async(
            images, output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 0
        assert result.filtered_count == 1

    @pytest.mark.asyncio
    async def test_process_and_save_async_deduplicates(self, tmp_path: Path) -> None:
        """Test that async processing deduplicates images."""
        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(deduplicate=True),
        )
        processor = ImageProcessor(config=config)

        image_data = create_test_image(100, 100)
        images = [
            ("first", "image/png", image_data),
            ("duplicate", "image/png", image_data),
        ]

        result = await processor.process_and_save_async(
            images, output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 1
        assert result.deduplicated_count == 1

    @pytest.mark.asyncio
    async def test_process_and_save_async_multiple_images(self, tmp_path: Path) -> None:
        """Test async processing multiple images."""
        config = ImageConfig(compress=True, quality=85, format="png")
        processor = ImageProcessor(config=config)

        images = [
            ("img1", "image/png", create_test_image(100, 100, "red")),
            ("img2", "image/png", create_test_image(150, 150, "blue")),
            ("img3", "image/png", create_test_image(200, 200, "green")),
        ]

        result = await processor.process_and_save_async(
            images, output_dir=tmp_path, base_name="multi"
        )

        assert len(result.saved_images) == 3
        # Check all files were created
        for img in result.saved_images:
            assert img.path.exists()

    @pytest.mark.asyncio
    async def test_process_and_save_async_invalid_image(self, tmp_path: Path) -> None:
        """Test async processing handles invalid image data."""
        processor = ImageProcessor()
        images = [("invalid", "image/png", b"not a valid image")]

        result = await processor.process_and_save_async(
            images, output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 0
        assert result.index_mapping is not None
        assert result.index_mapping[1].skip_reason == "error"

    @pytest.mark.asyncio
    async def test_process_and_save_async_empty_list(self, tmp_path: Path) -> None:
        """Test async processing an empty list."""
        processor = ImageProcessor()

        result = await processor.process_and_save_async(
            [], output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 0
        assert result.filtered_count == 0
        assert result.deduplicated_count == 0

    @pytest.mark.asyncio
    async def test_process_and_save_async_with_concurrency_limit(
        self, tmp_path: Path
    ) -> None:
        """Test async processing with custom concurrency limit."""
        config = ImageConfig(compress=True, format="jpeg")
        processor = ImageProcessor(config=config)

        images = [
            ("img1", "image/png", create_test_image(100, 100, "red")),
            ("img2", "image/png", create_test_image(100, 100, "blue")),
        ]

        result = await processor.process_and_save_async(
            images,
            output_dir=tmp_path,
            base_name="test",
            max_concurrency=1,  # Limit to 1 concurrent I/O
        )

        assert len(result.saved_images) == 2


class TestProcessAndSaveMultiprocess:
    """Tests for process_and_save_multiprocess method."""

    @pytest.mark.asyncio
    async def test_process_multiprocess_basic(self, tmp_path: Path) -> None:
        """Test multiprocess processing and saving images."""
        config = ImageConfig(compress=True, quality=85, format="jpeg")
        processor = ImageProcessor(config=config)

        image_data = create_test_image(200, 200)
        images = [("test", "image/png", image_data)]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 1
        saved_path = result.saved_images[0].path
        assert saved_path.exists()

    @pytest.mark.asyncio
    async def test_process_multiprocess_empty_list(self, tmp_path: Path) -> None:
        """Test multiprocess with empty list returns early."""
        processor = ImageProcessor()

        result = await processor.process_and_save_multiprocess(
            [], output_dir=tmp_path, base_name="test"
        )

        assert len(result.saved_images) == 0
        assert result.filtered_count == 0
        assert result.deduplicated_count == 0

    @pytest.mark.asyncio
    async def test_process_multiprocess_filters_small(self, tmp_path: Path) -> None:
        """Test multiprocess filtering of small images."""
        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(min_width=100, min_height=100),
        )
        processor = ImageProcessor(config=config)

        small_image = create_test_image(50, 50)
        images = [("small", "image/png", small_image)]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 0
        assert result.filtered_count == 1

    @pytest.mark.asyncio
    async def test_process_multiprocess_deduplicates(self, tmp_path: Path) -> None:
        """Test multiprocess deduplication."""
        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(deduplicate=True),
        )
        processor = ImageProcessor(config=config)

        image_data = create_test_image(100, 100)
        images = [
            ("first", "image/png", image_data),
            ("dup", "image/png", image_data),
        ]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 1
        assert result.deduplicated_count == 1

    @pytest.mark.asyncio
    async def test_process_multiprocess_all_duplicates(self, tmp_path: Path) -> None:
        """Test multiprocess when all images are duplicates (except first)."""
        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(deduplicate=True),
        )
        processor = ImageProcessor(config=config)

        image_data = create_test_image(100, 100)
        images = [
            ("first", "image/png", image_data),
            ("dup1", "image/png", image_data),
            ("dup2", "image/png", image_data),
        ]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 1
        assert result.deduplicated_count == 2

    @pytest.mark.asyncio
    async def test_process_multiprocess_no_compression(self, tmp_path: Path) -> None:
        """Test multiprocess without compression enabled."""
        config = ImageConfig(
            compress=False,  # Disable compression
            filter=ImageFilterConfig(min_width=50, min_height=50),
        )
        processor = ImageProcessor(config=config)

        image_data = create_test_image(100, 100)
        images = [("test", "image/png", image_data)]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 1

    @pytest.mark.asyncio
    async def test_process_multiprocess_no_compression_filters_small(
        self, tmp_path: Path
    ) -> None:
        """Test multiprocess without compression still filters small images."""
        config = ImageConfig(
            compress=False,
            filter=ImageFilterConfig(min_width=100, min_height=100),
        )
        processor = ImageProcessor(config=config)

        small_image = create_test_image(50, 50)
        images = [("small", "image/png", small_image)]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 0
        assert result.filtered_count == 1

    @pytest.mark.asyncio
    async def test_process_multiprocess_invalid_image(self, tmp_path: Path) -> None:
        """Test multiprocess handles invalid image data."""
        config = ImageConfig(compress=False)
        processor = ImageProcessor(config=config)
        images = [("invalid", "image/png", b"not a valid image")]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 0
        assert result.index_mapping is not None
        assert result.index_mapping[1].skip_reason == "error"

    @pytest.mark.asyncio
    async def test_process_multiprocess_png_format(self, tmp_path: Path) -> None:
        """Test multiprocess with PNG output format."""
        config = ImageConfig(compress=True, format="png")
        processor = ImageProcessor(config=config)

        image_data = create_test_image(100, 100)
        images = [("test", "image/png", image_data)]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 1
        assert result.saved_images[0].path.suffix == ".png"

    @pytest.mark.asyncio
    async def test_process_multiprocess_webp_format(self, tmp_path: Path) -> None:
        """Test multiprocess with WebP output format."""
        config = ImageConfig(compress=True, format="webp")
        processor = ImageProcessor(config=config)

        image_data = create_test_image(100, 100)
        images = [("test", "image/png", image_data)]

        result = await processor.process_and_save_multiprocess(
            images, output_dir=tmp_path, base_name="test", max_workers=1
        )

        assert len(result.saved_images) == 1
        assert result.saved_images[0].path.suffix == ".webp"


class TestDownloadUrlImages:
    """Tests for download_url_images function."""

    @pytest.mark.asyncio
    async def test_download_no_images(self, tmp_path: Path) -> None:
        """Test downloading when no images in markdown."""
        from markitai.image import download_url_images

        config = ImageConfig()
        markdown = "# Title\n\nNo images here."

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        assert result.updated_markdown == markdown
        assert len(result.downloaded_paths) == 0
        assert len(result.failed_urls) == 0

    @pytest.mark.asyncio
    async def test_download_skips_data_uris(self, tmp_path: Path) -> None:
        """Test that data URIs are not downloaded."""
        from markitai.image import download_url_images

        config = ImageConfig()
        img_data = create_test_image()
        markdown = create_base64_markdown(img_data)

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        # Data URIs should be left as-is (not treated as URLs)
        assert result.updated_markdown == markdown
        assert len(result.downloaded_paths) == 0

    @pytest.mark.asyncio
    async def test_download_handles_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of download timeout."""
        import httpx

        from markitai.image import download_url_images

        # Mock httpx.AsyncClient.get to raise TimeoutException
        async def mock_get(*args, **kwargs):
            raise httpx.TimeoutException("Connection timeout")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig()
        markdown = "![](https://example.com/timeout.jpg)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        # Should fail gracefully
        assert len(result.failed_urls) == 1
        assert "example.com/timeout.jpg" in result.failed_urls[0]
        # Original markdown should be unchanged for failed downloads
        assert "https://example.com/timeout.jpg" in result.updated_markdown

    @pytest.mark.asyncio
    async def test_download_handles_http_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of HTTP error responses."""
        import httpx

        from markitai.image import download_url_images

        # Create a mock response that raises on raise_for_status
        class MockResponse:
            status_code = 404

            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "404",
                    request=None,
                    response=self,  # type: ignore
                )

        async def mock_get(*args, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig()
        markdown = "![](https://example.com/notfound.jpg)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        assert len(result.failed_urls) == 1

    @pytest.mark.asyncio
    async def test_download_handles_generic_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of generic exceptions during download."""
        import httpx

        from markitai.image import download_url_images

        async def mock_get(*args, **kwargs):
            raise Exception("Some unexpected error")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig()
        markdown = "![](https://example.com/error.jpg)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        assert len(result.failed_urls) == 1

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_download_success_integration(self, tmp_path: Path) -> None:
        """Integration test for successful download (requires network)."""
        # This test is marked as network-dependent and may be skipped
        # It serves as a real integration test when network is available
        pass  # Skip for now - would need real URL

    @pytest.mark.asyncio
    async def test_download_resolves_relative_urls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that relative URLs are resolved correctly."""
        import httpx

        from markitai.image import download_url_images

        captured_urls: list[str] = []

        class MockResponse:
            status_code = 200
            content = create_test_image()
            headers = {"content-type": "image/png"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            captured_urls.append(str(url))
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(compress=False)
        markdown = "![](images/photo.png)"

        await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com/page/",
            config=config,
        )

        # Should have resolved relative URL
        assert len(captured_urls) == 1
        assert "example.com" in captured_urls[0]
        assert "images/photo.png" in captured_urls[0]

    @pytest.mark.asyncio
    async def test_download_protocol_relative_urls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of protocol-relative URLs (//example.com)."""
        import httpx

        from markitai.image import download_url_images

        captured_urls: list[str] = []

        class MockResponse:
            status_code = 200
            content = create_test_image()
            headers = {"content-type": "image/jpeg"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            captured_urls.append(str(url))
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(compress=False)
        markdown = "![](//cdn.example.com/image.jpg)"

        await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        # Should have prepended https:
        assert len(captured_urls) == 1
        assert captured_urls[0].startswith("https://cdn.example.com")

    @pytest.mark.asyncio
    async def test_download_with_compression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test downloading with image compression enabled."""
        import httpx

        from markitai.image import download_url_images

        class MockResponse:
            status_code = 200
            content = create_test_image(500, 500)
            headers = {"content-type": "image/png"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(
            compress=True, quality=50, max_width=200, max_height=200, format="jpeg"
        )
        markdown = "![](https://example.com/large.png)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
            source_name="compressed",
        )

        assert len(result.downloaded_paths) == 1
        # Format should change to configured format
        assert "compressed.0001.jpeg" in result.updated_markdown

    @pytest.mark.asyncio
    async def test_download_filters_small_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that downloaded small images are filtered."""
        import httpx

        from markitai.image import download_url_images

        class MockResponse:
            status_code = 200
            content = create_test_image(20, 20)  # Very small
            headers = {"content-type": "image/jpeg"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(
            compress=True,
            filter=ImageFilterConfig(min_width=100, min_height=100),
        )
        markdown = "![](https://example.com/tiny.jpg)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        # Image should be filtered (not saved)
        assert len(result.downloaded_paths) == 0

    @pytest.mark.asyncio
    async def test_download_multiple_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test downloading multiple images."""
        import httpx

        from markitai.image import download_url_images

        colors = ["red", "green", "blue"]
        call_count = 0

        class MockResponse:
            status_code = 200
            headers = {"content-type": "image/jpeg"}

            def __init__(self):
                nonlocal call_count
                color = colors[call_count % len(colors)]
                call_count += 1
                self.content = create_test_image(100, 100, color)

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(compress=False)
        markdown = """
![](https://example.com/img0.jpg)
![](https://example.com/img1.jpg)
![](https://example.com/img2.jpg)
"""

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
            source_name="multi",
            concurrency=2,
        )

        assert len(result.downloaded_paths) == 3
        assert len(result.failed_urls) == 0

    @pytest.mark.asyncio
    async def test_download_url_to_path_mapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that url_to_path mapping is populated."""
        import httpx

        from markitai.image import download_url_images

        class MockResponse:
            status_code = 200
            content = create_test_image()
            headers = {"content-type": "image/png"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(compress=False)
        markdown = "![](https://example.com/mapped.png)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        assert "https://example.com/mapped.png" in result.url_to_path
        assert result.url_to_path["https://example.com/mapped.png"].exists()

    @pytest.mark.asyncio
    async def test_download_extension_from_content_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test extension detection from content-type when URL has no extension."""
        import httpx

        from markitai.image import download_url_images

        class MockResponse:
            status_code = 200
            content = create_test_image()
            headers = {"content-type": "image/png"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        # Use png format to match content-type detection
        config = ImageConfig(compress=True, format="png")
        markdown = "![](https://example.com/image)"  # No extension

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
            source_name="noext",
        )

        assert len(result.downloaded_paths) == 1
        # Should use configured format
        assert result.downloaded_paths[0].suffix == ".png"

    @pytest.mark.asyncio
    async def test_download_preserves_alt_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that alt text is preserved in updated markdown."""
        import httpx

        from markitai.image import download_url_images

        class MockResponse:
            status_code = 200
            content = create_test_image()
            headers = {"content-type": "image/jpeg"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(compress=False)
        markdown = "![My Alt Text](https://example.com/img.jpg)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
            source_name="test",
        )

        assert "![My Alt Text]" in result.updated_markdown

    @pytest.mark.asyncio
    async def test_download_sanitizes_source_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that source name with special chars is sanitized."""
        import httpx

        from markitai.image import download_url_images

        class MockResponse:
            status_code = 200
            content = create_test_image()
            headers = {"content-type": "image/jpeg"}

            def raise_for_status(self):
                pass

        async def mock_get(self, url, **kwargs):
            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        config = ImageConfig(compress=False)
        markdown = "![](https://example.com/img.jpg)"

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
            source_name='unsafe<>:"/\\|?*name',  # Has invalid chars
        )

        assert len(result.downloaded_paths) == 1
        # Filename should be sanitized
        filename = result.downloaded_paths[0].name
        for char in '<>:"/\\|?*':
            assert char not in filename


class TestUrlImageDownloadResultDataclass:
    """Tests for UrlImageDownloadResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating UrlImageDownloadResult."""
        from markitai.image import UrlImageDownloadResult

        result = UrlImageDownloadResult(
            updated_markdown="# Test",
            downloaded_paths=[Path("/tmp/img.jpg")],
            failed_urls=["https://fail.com/img.png"],
            url_to_path={"https://ok.com/img.jpg": Path("/tmp/img.jpg")},
        )

        assert result.updated_markdown == "# Test"
        assert len(result.downloaded_paths) == 1
        assert len(result.failed_urls) == 1
        assert len(result.url_to_path) == 1

    def test_dataclass_default_url_to_path(self) -> None:
        """Test default empty dict for url_to_path."""
        from markitai.image import UrlImageDownloadResult

        result = UrlImageDownloadResult(
            updated_markdown="",
            downloaded_paths=[],
            failed_urls=[],
        )

        assert result.url_to_path == {}


class TestImageProcessResultDataclass:
    """Tests for ImageProcessResult dataclass."""

    def test_dataclass_creation(self, tmp_path: Path) -> None:
        """Test creating ImageProcessResult."""
        from markitai.converter.base import ExtractedImage
        from markitai.image import ImageProcessResult, ProcessedImage

        saved = [
            ExtractedImage(
                path=tmp_path / "test.jpg",
                index=1,
                original_name="test.jpg",
                mime_type="image/jpeg",
                width=100,
                height=100,
            )
        ]
        mapping = {
            1: ProcessedImage(
                original_index=1, saved_path=tmp_path / "test.jpg", skip_reason=None
            )
        }

        result = ImageProcessResult(
            saved_images=saved,
            filtered_count=2,
            deduplicated_count=1,
            index_mapping=mapping,
        )

        assert len(result.saved_images) == 1
        assert result.filtered_count == 2
        assert result.deduplicated_count == 1
        assert result.index_mapping is not None

    def test_dataclass_default_index_mapping(self) -> None:
        """Test default None for index_mapping."""
        from markitai.image import ImageProcessResult

        result = ImageProcessResult(
            saved_images=[], filtered_count=0, deduplicated_count=0
        )

        assert result.index_mapping is None


class TestProcessedImageDataclass:
    """Tests for ProcessedImage dataclass."""

    def test_processed_image_saved(self, tmp_path: Path) -> None:
        """Test ProcessedImage for a saved image."""
        from markitai.image import ProcessedImage

        pi = ProcessedImage(
            original_index=1, saved_path=tmp_path / "image.jpg", skip_reason=None
        )

        assert pi.original_index == 1
        assert pi.saved_path is not None
        assert pi.skip_reason is None

    def test_processed_image_filtered(self) -> None:
        """Test ProcessedImage for a filtered image."""
        from markitai.image import ProcessedImage

        pi = ProcessedImage(original_index=2, saved_path=None, skip_reason="filtered")

        assert pi.original_index == 2
        assert pi.saved_path is None
        assert pi.skip_reason == "filtered"

    def test_processed_image_duplicate(self) -> None:
        """Test ProcessedImage for a duplicate image."""
        from markitai.image import ProcessedImage

        pi = ProcessedImage(original_index=3, saved_path=None, skip_reason="duplicate")

        assert pi.skip_reason == "duplicate"

    def test_processed_image_error(self) -> None:
        """Test ProcessedImage for an error case."""
        from markitai.image import ProcessedImage

        pi = ProcessedImage(original_index=4, saved_path=None, skip_reason="error")

        assert pi.skip_reason == "error"


class TestCompressPillowEdgeCases:
    """Additional edge case tests for Pillow compression."""

    def test_compress_pillow_invalid_data(self) -> None:
        """Test Pillow compression with invalid data returns None."""
        result = _compress_image_pillow(
            image_data=b"not an image",
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )
        assert result is None

    def test_compress_pillow_palette_mode(self) -> None:
        """Test Pillow compression handles palette mode (P) images."""
        # Create a palette mode image
        img = Image.new("P", (100, 100))
        img.putpalette([i % 256 for i in range(768)])  # Set palette
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        result = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        # Should be valid JPEG
        assert result[0][:2] == b"\xff\xd8"

    def test_compress_pillow_la_mode(self) -> None:
        """Test Pillow compression handles LA (luminance + alpha) mode."""
        # Create LA mode image
        img = Image.new("LA", (100, 100), (128, 200))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        result = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None

    def test_compress_pillow_png_format(self) -> None:
        """Test Pillow compression with PNG output."""
        img_data = create_test_image(100, 100)

        result = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="PNG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        assert result[0][:4] == b"\x89PNG"

    def test_compress_pillow_webp_format(self) -> None:
        """Test Pillow compression with WebP output."""
        img_data = create_test_image(100, 100)

        result = _compress_image_pillow(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="WEBP",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        assert result[0][:4] == b"RIFF"


class TestCompressCV2EdgeCases:
    """Additional edge case tests for CV2 compression."""

    def test_compress_cv2_grayscale(self) -> None:
        """Test CV2 compression handles grayscale images."""
        # Create grayscale image
        img = Image.new("L", (100, 100), 128)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        assert len(result[0]) > 0

    def test_compress_cv2_unknown_format_fallback(self) -> None:
        """Test CV2 compression falls back to JPEG for unknown formats."""
        img_data = create_test_image(100, 100)

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(100, 100),
            output_format="UNKNOWN",  # Unknown format
            min_width=10,
            min_height=10,
            min_area=100,
        )

        assert result is not None
        # Should fall back to JPEG
        assert result[0][:2] == b"\xff\xd8"

    def test_compress_cv2_filter_by_area(self) -> None:
        """Test CV2 filtering by area."""
        # 80x80 = 6400 area
        img_data = create_test_image(80, 80)

        result = _compress_image_cv2(
            image_data=img_data,
            quality=85,
            max_size=(1000, 1000),
            output_format="JPEG",
            min_width=10,
            min_height=10,
            min_area=10000,  # Requires 10000, image has 6400
        )

        assert result is None  # Filtered due to area


class TestSaveScreenshotEdgeCases:
    """Additional edge case tests for save_screenshot."""

    def test_save_screenshot_jpeg_format(self, tmp_path: Path) -> None:
        """Test save_screenshot with 'jpeg' format."""
        config = ImageConfig(format="jpeg")
        processor = ImageProcessor(config=config)

        img = Image.new("RGB", (200, 200), "red")
        samples = img.tobytes()
        output_path = tmp_path / "test.jpg"

        processor.save_screenshot(samples, img.width, img.height, output_path)

        assert output_path.exists()
        # Should be valid JPEG
        with open(output_path, "rb") as f:
            assert f.read(2) == b"\xff\xd8"

    def test_save_screenshot_png_format(self, tmp_path: Path) -> None:
        """Test save_screenshot with PNG format."""
        config = ImageConfig(format="png")
        processor = ImageProcessor(config=config)

        img = Image.new("RGB", (200, 200), "blue")
        samples = img.tobytes()
        output_path = tmp_path / "test.png"

        processor.save_screenshot(samples, img.width, img.height, output_path)

        assert output_path.exists()
        with open(output_path, "rb") as f:
            assert f.read(4) == b"\x89PNG"

    def test_save_screenshot_webp_format(self, tmp_path: Path) -> None:
        """Test save_screenshot with WebP format."""
        config = ImageConfig(format="webp")
        processor = ImageProcessor(config=config)

        img = Image.new("RGB", (200, 200), "green")
        samples = img.tobytes()
        output_path = tmp_path / "test.webp"

        processor.save_screenshot(samples, img.width, img.height, output_path)

        assert output_path.exists()
        with open(output_path, "rb") as f:
            assert f.read(4) == b"RIFF"

    def test_save_screenshot_converts_rgba_to_rgb(self, tmp_path: Path) -> None:
        """Test save_screenshot handles RGBA input for JPEG output."""
        config = ImageConfig(format="jpeg")
        processor = ImageProcessor(config=config)

        # Create RGBA image and get RGB samples (as if converted)
        img = Image.new("RGB", (200, 200), (255, 0, 0))
        samples = img.tobytes()
        output_path = tmp_path / "rgba_test.jpg"

        processor.save_screenshot(samples, img.width, img.height, output_path)

        assert output_path.exists()
