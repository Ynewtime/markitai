"""Image processing module for extraction, compression, and filtering."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from markit.config import ImageConfig
    from markit.converter.base import ExtractedImage


@dataclass
class ImageProcessResult:
    """Result of image processing."""

    saved_images: list[ExtractedImage]
    filtered_count: int
    deduplicated_count: int


class ImageProcessor:
    """Processor for image extraction, compression, and filtering."""

    # Regex pattern to match base64 data URIs in markdown
    DATA_URI_PATTERN = re.compile(
        r"!\[([^\]]*)\]\(data:image/([\w+]+);base64,([A-Za-z0-9+/=]+)\)"
    )

    def __init__(self, config: ImageConfig | None = None) -> None:
        """Initialize with optional image configuration."""
        self.config = config
        self._seen_hashes: set[str] = set()

    def extract_base64_images(self, markdown: str) -> list[tuple[str, str, bytes]]:
        """
        Extract base64-encoded images from markdown content.

        Args:
            markdown: Markdown content containing data URIs

        Returns:
            List of (alt_text, mime_type, image_data) tuples
        """
        images = []
        for match in self.DATA_URI_PATTERN.finditer(markdown):
            alt_text = match.group(1)
            image_type = match.group(2)
            base64_data = match.group(3)

            try:
                image_data = base64.b64decode(base64_data)
                mime_type = f"image/{image_type}"
                images.append((alt_text, mime_type, image_data))
            except Exception:
                # Skip invalid base64 data
                continue

        return images

    def replace_base64_with_paths(
        self,
        markdown: str,
        images: list[ExtractedImage],
        assets_path: str = "assets",
    ) -> str:
        """
        Replace base64 data URIs with file paths in markdown.

        Args:
            markdown: Original markdown with data URIs
            images: List of saved images with paths
            assets_path: Relative path to assets directory

        Returns:
            Markdown with data URIs replaced by file paths
        """
        result = markdown
        image_iter = iter(images)

        def replace_match(match: re.Match) -> str:
            try:
                img = next(image_iter)
                return f"![{match.group(1)}]({assets_path}/{img.path.name})"
            except StopIteration:
                return match.group(0)

        result = self.DATA_URI_PATTERN.sub(replace_match, result)
        return result

    def compress(
        self,
        image: Image.Image,
        quality: int = 85,
        max_size: tuple[int, int] = (1920, 1080),
        output_format: str = "JPEG",
    ) -> tuple[Image.Image, bytes]:
        """
        Compress an image.

        Args:
            image: PIL Image to compress
            quality: JPEG quality (1-100)
            max_size: Maximum dimensions (width, height)
            output_format: Output format (JPEG, PNG, WEBP)

        Returns:
            Tuple of (compressed image, compressed data)
        """
        # Resize if needed
        image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to RGB for JPEG (no alpha channel)
        if output_format.upper() == "JPEG" and image.mode in ("RGBA", "P", "LA"):
            # Create white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(
                image, mask=image.split()[-1] if image.mode == "RGBA" else None
            )
            image = background

        # Compress to bytes
        buffer = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": output_format}
        if output_format.upper() in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality
        if output_format.upper() == "PNG":
            save_kwargs["optimize"] = True

        image.save(buffer, **save_kwargs)
        compressed_data = buffer.getvalue()

        return image, compressed_data

    def should_filter(self, width: int, height: int) -> bool:
        """
        Check if an image should be filtered out based on size.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            True if image should be filtered out
        """
        if not self.config:
            return False

        filter_config = self.config.filter

        if width < filter_config.min_width:
            return True
        if height < filter_config.min_height:
            return True
        if width * height < filter_config.min_area:
            return True

        return False

    def is_duplicate(self, image_data: bytes) -> bool:
        """
        Check if image is a duplicate based on hash.

        Args:
            image_data: Raw image data

        Returns:
            True if image is a duplicate
        """
        if not self.config or not self.config.filter.deduplicate:
            return False

        image_hash = hashlib.md5(image_data).hexdigest()
        if image_hash in self._seen_hashes:
            return True

        self._seen_hashes.add(image_hash)
        return False

    def process_and_save(
        self,
        images: list[tuple[str, str, bytes]],
        output_dir: Path,
        base_name: str,
    ) -> ImageProcessResult:
        """
        Process and save a list of images.

        Args:
            images: List of (alt_text, mime_type, image_data) tuples
            output_dir: Directory to save images
            base_name: Base name for image files

        Returns:
            ImageProcessResult with saved images and statistics
        """
        # Delayed import to avoid circular import
        from markit.converter.base import ExtractedImage

        # Create assets directory
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        saved_images: list[ExtractedImage] = []
        filtered_count = 0
        deduplicated_count = 0

        # Determine output format
        output_format = "JPEG"
        extension = "jpg"
        if self.config:
            format_map = {
                "jpeg": ("JPEG", "jpg"),
                "png": ("PNG", "png"),
                "webp": ("WEBP", "webp"),
            }
            output_format, extension = format_map.get(
                self.config.format, ("JPEG", "jpg")
            )

        for idx, (_alt_text, _mime_type, image_data) in enumerate(images, start=1):
            # Check for duplicates
            if self.is_duplicate(image_data):
                deduplicated_count += 1
                continue

            # Load image
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    width, height = img.size

                    # Check filter
                    if self.should_filter(width, height):
                        filtered_count += 1
                        continue

                    # Compress
                    quality = self.config.quality if self.config else 85
                    max_size = (
                        (self.config.max_width, self.config.max_height)
                        if self.config
                        else (1920, 1080)
                    )

                    if self.config and self.config.compress:
                        compressed_img, compressed_data = self.compress(
                            img.copy(),
                            quality=quality,
                            max_size=max_size,
                            output_format=output_format,
                        )
                        final_width, final_height = compressed_img.size
                    else:
                        compressed_data = image_data
                        final_width, final_height = width, height

                    # Generate filename
                    filename = f"{base_name}.{idx:04d}.{extension}"
                    output_path = assets_dir / filename

                    # Save
                    output_path.write_bytes(compressed_data)

                    saved_images.append(
                        ExtractedImage(
                            path=output_path,
                            index=idx,
                            original_name=filename,
                            mime_type=f"image/{extension}",
                            width=final_width,
                            height=final_height,
                        )
                    )

            except Exception:
                # Skip invalid images
                continue

        return ImageProcessResult(
            saved_images=saved_images,
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
        )

    def reset_dedup_cache(self) -> None:
        """Reset the deduplication hash cache."""
        self._seen_hashes.clear()

    async def process_and_save_async(
        self,
        images: list[tuple[str, str, bytes]],
        output_dir: Path,
        base_name: str,
        max_concurrency: int = 4,
    ) -> ImageProcessResult:
        """Process and save a list of images with async I/O.

        This is an optimized version that uses asyncio for concurrent I/O
        operations while keeping CPU-bound image processing sequential.

        Args:
            images: List of (alt_text, mime_type, image_data) tuples
            output_dir: Directory to save images
            base_name: Base name for image files
            max_concurrency: Maximum concurrent I/O operations

        Returns:
            ImageProcessResult with saved images and statistics
        """
        import asyncio

        # Delayed imports to avoid circular import
        from markit.converter.base import ExtractedImage
        from markit.security import write_bytes_async

        # Create assets directory
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        saved_images: list[ExtractedImage] = []
        filtered_count = 0
        deduplicated_count = 0

        # Determine output format
        output_format = "JPEG"
        extension = "jpg"
        if self.config:
            format_map = {
                "jpeg": ("JPEG", "jpg"),
                "png": ("PNG", "png"),
                "webp": ("WEBP", "webp"),
            }
            output_format, extension = format_map.get(
                self.config.format, ("JPEG", "jpg")
            )

        # First pass: process images (CPU-bound, sequential)
        processed_images: list[tuple[int, bytes, int, int]] = []
        for idx, (_alt_text, _mime_type, image_data) in enumerate(images, start=1):
            # Check for duplicates
            if self.is_duplicate(image_data):
                deduplicated_count += 1
                continue

            # Load and process image
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    width, height = img.size

                    # Check filter
                    if self.should_filter(width, height):
                        filtered_count += 1
                        continue

                    # Compress
                    quality = self.config.quality if self.config else 85
                    max_size = (
                        (self.config.max_width, self.config.max_height)
                        if self.config
                        else (1920, 1080)
                    )

                    if self.config and self.config.compress:
                        compressed_img, compressed_data = self.compress(
                            img.copy(),
                            quality=quality,
                            max_size=max_size,
                            output_format=output_format,
                        )
                        final_width, final_height = compressed_img.size
                    else:
                        compressed_data = image_data
                        final_width, final_height = width, height

                    processed_images.append(
                        (idx, compressed_data, final_width, final_height)
                    )

            except Exception:
                # Skip invalid images
                continue

        # Second pass: save images concurrently (I/O-bound)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def save_image(
            idx: int, data: bytes, width: int, height: int
        ) -> ExtractedImage | None:
            filename = f"{base_name}.{idx:04d}.{extension}"
            output_path = assets_dir / filename

            async with semaphore:
                try:
                    await write_bytes_async(output_path, data)
                    return ExtractedImage(
                        path=output_path,
                        index=idx,
                        original_name=filename,
                        mime_type=f"image/{extension}",
                        width=width,
                        height=height,
                    )
                except Exception:
                    return None

        # Run all saves concurrently
        tasks = [
            save_image(idx, data, width, height)
            for idx, data, width, height in processed_images
        ]
        results = await asyncio.gather(*tasks)

        # Collect successful saves
        for result in results:
            if result is not None:
                saved_images.append(result)

        # Sort by index to maintain order
        saved_images.sort(key=lambda x: x.index)

        return ImageProcessResult(
            saved_images=saved_images,
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
        )
