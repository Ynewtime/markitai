"""Image processing module for extraction, compression, and filtering."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from markit.constants import (
    DEFAULT_IMAGE_IO_CONCURRENCY,
    DEFAULT_IMAGE_MAX_HEIGHT,
    DEFAULT_IMAGE_MAX_WIDTH,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_SCREENSHOT_MAX_BYTES,
)

if TYPE_CHECKING:
    from markit.config import ImageConfig
    from markit.converter.base import ExtractedImage


# Module-level function for multiprocessing (must be picklable)
def _compress_image_worker(
    image_data: bytes,
    quality: int,
    max_size: tuple[int, int],
    output_format: str,
    min_width: int,
    min_height: int,
    min_area: int,
) -> tuple[bytes, int, int] | None:
    """Compress a single image in a worker process.

    Args:
        image_data: Raw image bytes
        quality: JPEG quality (1-100)
        max_size: Maximum dimensions (width, height)
        output_format: Output format (JPEG, PNG, WEBP)
        min_width: Minimum width filter
        min_height: Minimum height filter
        min_area: Minimum area filter

    Returns:
        Tuple of (compressed_data, final_width, final_height) or None if filtered
    """
    try:
        with io.BytesIO(image_data) as buffer:
            img = Image.open(buffer)
            img.load()
            width, height = img.size

            # Apply filter
            if width < min_width or height < min_height or width * height < min_area:
                return None

            # Resize if needed
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to RGB for JPEG
            if output_format.upper() == "JPEG" and img.mode in ("RGBA", "P", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if img.mode in ("RGBA", "LA"):
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background

            # Compress to bytes
            out_buffer = io.BytesIO()
            save_kwargs: dict[str, Any] = {"format": output_format}
            if output_format.upper() in ("JPEG", "WEBP"):
                save_kwargs["quality"] = quality
            if output_format.upper() == "PNG":
                save_kwargs["optimize"] = True

            img.save(out_buffer, **save_kwargs)
            return out_buffer.getvalue(), img.size[0], img.size[1]
    except Exception:
        return None


@dataclass
class ImageProcessResult:
    """Result of image processing."""

    saved_images: list[ExtractedImage]
    filtered_count: int
    deduplicated_count: int


class ImageProcessor:
    """Processor for image extraction, compression, and filtering."""

    # Regex pattern to match base64 data URIs in markdown
    # Support MIME types like png, jpeg, x-emf, x-wmf (with hyphens)
    DATA_URI_PATTERN = re.compile(
        r"!\[([^\]]*)\]\(data:image/([\w+.-]+);base64,([A-Za-z0-9+/=]+)\)"
    )

    def __init__(self, config: ImageConfig | None = None) -> None:
        """Initialize with optional image configuration."""
        self.config = config
        self._seen_hashes: set[str] = set()

    def _convert_to_png(self, image_data: bytes, original_fmt: str) -> bytes:
        """Convert unsupported image formats (EMF/WMF) to PNG.

        On Windows, uses Pillow which has native EMF/WMF support.
        On other platforms, falls back to LibreOffice if available.
        """
        import platform

        # Normalize format name
        fmt_lower = original_fmt.lower().replace("x-", "")  # x-emf -> emf

        # On Windows, Pillow can natively read EMF/WMF files
        if platform.system() == "Windows" and fmt_lower in ("emf", "wmf"):
            try:
                with io.BytesIO(image_data) as buffer:
                    img = Image.open(buffer)
                    # Load at higher DPI for better quality
                    # WmfImagePlugin.load() accepts dpi parameter
                    img.load(dpi=150)  # type: ignore[call-arg]

                    # Convert to RGB if necessary (EMF/WMF loads as RGB)
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGB")

                    # Save as PNG
                    out_buffer = io.BytesIO()
                    img.save(out_buffer, format="PNG")
                    return out_buffer.getvalue()
            except Exception:
                # Fall through to LibreOffice fallback
                pass

        # Fallback to LibreOffice (for non-Windows or if Pillow fails)
        import subprocess
        import tempfile
        import uuid

        from markit.utils.office import find_libreoffice

        soffice = find_libreoffice()
        if not soffice:
            return image_data

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                # Ensure extension doesn't have special chars
                ext = re.sub(r"[^a-zA-Z0-9]", "", original_fmt)
                temp_in = temp_path / f"temp_{uuid.uuid4().hex[:8]}.{ext}"
                temp_in.write_bytes(image_data)

                cmd = [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "png",
                    "--outdir",
                    str(temp_path),
                    str(temp_in),
                ]

                subprocess.run(cmd, capture_output=True, timeout=30)

                # LibreOffice output filename depends on input filename
                temp_out = temp_path / f"{temp_in.stem}.png"
                if temp_out.exists():
                    return temp_out.read_bytes()
        except Exception:
            pass

        return image_data

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

                # Handle EMF/WMF conversion
                if image_type.lower() in ("x-emf", "emf", "x-wmf", "wmf"):
                    image_data = self._convert_to_png(image_data, image_type)
                    image_type = "png"

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

    def strip_base64_images(
        self,
        markdown: str,
        replacement_path: str | None = None,
    ) -> str:
        """
        Remove all base64 data URIs from markdown.

        Args:
            markdown: Markdown content with data URIs
            replacement_path: If provided, replace with this path; otherwise remove

        Returns:
            Markdown with base64 images removed or replaced
        """

        def replace_match(match: re.Match) -> str:
            alt_text = match.group(1)
            if replacement_path:
                return f"![{alt_text}]({replacement_path})"
            return ""  # Remove the image entirely

        return self.DATA_URI_PATTERN.sub(replace_match, markdown)

    @staticmethod
    def remove_nonexistent_images(
        markdown: str,
        assets_dir: Path,
    ) -> str:
        """
        Remove image references that don't exist in assets directory.

        LLM may hallucinate non-existent image references. This method
        validates each assets/ image reference and removes those that
        don't exist on disk.

        Args:
            markdown: Markdown content with image references
            assets_dir: Path to the assets directory

        Returns:
            Markdown with non-existent image references removed
        """
        # Pattern to match image references: ![alt](assets/filename) or ![alt](assets\filename)
        # Support both forward slash and backslash for Windows compatibility
        img_pattern = re.compile(r"!\[[^\]]*\]\(assets[/\\]([^)]+)\)")

        # Invalid filename patterns that indicate placeholders or hallucinations
        invalid_patterns = {"...", "..", ".", "placeholder", "image", "filename"}

        def validate_image(match: re.Match) -> str:
            filename = match.group(1)
            # Check for placeholder patterns
            if filename.strip() in invalid_patterns or filename.strip() == "":
                return ""
            image_path = assets_dir / filename
            if image_path.exists():
                return match.group(0)  # Keep existing image
            # Remove non-existent image reference
            return ""

        result = img_pattern.sub(validate_image, markdown)

        # Clean up any resulting double spaces or empty lines
        result = re.sub(r"  +", " ", result)  # Multiple spaces to single
        result = re.sub(r"\n{3,}", "\n\n", result)  # 3+ newlines to 2

        return result

    def compress(
        self,
        image: Image.Image,
        quality: int = DEFAULT_IMAGE_QUALITY,
        max_size: tuple[int, int] = (DEFAULT_IMAGE_MAX_WIDTH, DEFAULT_IMAGE_MAX_HEIGHT),
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

    def save_screenshot(
        self,
        pix_samples: bytes,
        width: int,
        height: int,
        output_path: Path,
        max_bytes: int = DEFAULT_SCREENSHOT_MAX_BYTES,
    ) -> tuple[int, int]:
        """
        Save a screenshot with compression to ensure it's under the size limit.

        Converts raw pixel data to PIL Image, compresses using config quality,
        and progressively reduces quality if needed to stay under max_bytes.

        Args:
            pix_samples: Raw RGB pixel data from pymupdf pixmap.samples
            width: Image width
            height: Image height
            output_path: Path to save the image
            max_bytes: Maximum file size in bytes (default 5MB for LLM providers)

        Returns:
            Tuple of (final_width, final_height) after any resizing
        """
        from loguru import logger

        # Convert raw samples to PIL Image
        image = Image.frombytes("RGB", (width, height), pix_samples)

        # Get quality from config or use default
        quality = self.config.quality if self.config else DEFAULT_IMAGE_QUALITY
        max_width = self.config.max_width if self.config else DEFAULT_IMAGE_MAX_WIDTH
        max_height = self.config.max_height if self.config else DEFAULT_IMAGE_MAX_HEIGHT
        output_format = (self.config.format if self.config else "jpeg").upper()
        if output_format == "JPG":
            output_format = "JPEG"

        # Resize to configured max dimensions
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Convert to RGB for JPEG
        if output_format == "JPEG" and image.mode in ("RGBA", "P", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            if image.mode in ("RGBA", "LA"):
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background

        # Try compressing with configured quality first
        for q in [quality, 70, 55, 40, 25]:
            buffer = io.BytesIO()
            save_kwargs: dict[str, Any] = {"format": output_format}
            if output_format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = q
                save_kwargs["optimize"] = True
            elif output_format == "PNG":
                save_kwargs["optimize"] = True

            image.save(buffer, **save_kwargs)
            data = buffer.getvalue()

            if len(data) <= max_bytes:
                output_path.write_bytes(data)
                if q < quality:
                    logger.debug(
                        f"Screenshot compressed: quality {quality}->{q}, "
                        f"size {len(data) / 1024:.1f}KB"
                    )
                return image.size

        # Last resort: aggressive resize
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=20, optimize=True)
        data = buffer.getvalue()
        output_path.write_bytes(data)
        logger.warning(f"Screenshot aggressively compressed: {len(data) / 1024:.1f}KB")
        return image.size

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
                # Use BytesIO as context manager to ensure buffer is released
                img_buffer = io.BytesIO(image_data)
                try:
                    img = Image.open(img_buffer)
                    # Load image data immediately so we can release the buffer
                    img.load()

                    width, height = img.size

                    # Check filter
                    if self.should_filter(width, height):
                        filtered_count += 1
                        img.close()
                        continue

                    # Compress
                    quality = (
                        self.config.quality if self.config else DEFAULT_IMAGE_QUALITY
                    )
                    max_size = (
                        (self.config.max_width, self.config.max_height)
                        if self.config
                        else (DEFAULT_IMAGE_MAX_WIDTH, DEFAULT_IMAGE_MAX_HEIGHT)
                    )

                    if self.config and self.config.compress:
                        # No need for img.copy() - compress can modify the image
                        # since we don't need the original after this
                        compressed_img, compressed_data = self.compress(
                            img,
                            quality=quality,
                            max_size=max_size,
                            output_format=output_format,
                        )
                        final_width, final_height = compressed_img.size
                        # Release the compressed image
                        compressed_img.close()
                    else:
                        compressed_data = image_data
                        final_width, final_height = width, height

                    # Close original image to release memory
                    img.close()

                    # Generate filename
                    filename = f"{base_name}.{idx:04d}.{extension}"
                    output_path = assets_dir / filename

                    # Save
                    output_path.write_bytes(compressed_data)

                    # Release compressed data reference
                    del compressed_data

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
                finally:
                    img_buffer.close()

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
        max_concurrency: int = DEFAULT_IMAGE_IO_CONCURRENCY,
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
                    quality = (
                        self.config.quality if self.config else DEFAULT_IMAGE_QUALITY
                    )
                    max_size = (
                        (self.config.max_width, self.config.max_height)
                        if self.config
                        else (DEFAULT_IMAGE_MAX_WIDTH, DEFAULT_IMAGE_MAX_HEIGHT)
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

    async def process_and_save_multiprocess(
        self,
        images: list[tuple[str, str, bytes]],
        output_dir: Path,
        base_name: str,
        max_workers: int | None = None,
        max_io_concurrency: int = DEFAULT_IMAGE_IO_CONCURRENCY,
    ) -> ImageProcessResult:
        """Process and save images using multiprocessing for CPU-bound compression.

        This version uses ProcessPoolExecutor to parallelize image compression
        across multiple CPU cores, bypassing the GIL limitation.

        Args:
            images: List of (alt_text, mime_type, image_data) tuples
            output_dir: Directory to save images
            base_name: Base name for image files
            max_workers: Max worker processes (default: cpu_count // 2)
            max_io_concurrency: Maximum concurrent I/O operations

        Returns:
            ImageProcessResult with saved images and statistics
        """
        import asyncio

        from markit.converter.base import ExtractedImage
        from markit.security import write_bytes_async

        if not images:
            return ImageProcessResult(
                saved_images=[], filtered_count=0, deduplicated_count=0
            )

        # Create assets directory
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

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

        # Get compression parameters
        quality = self.config.quality if self.config else DEFAULT_IMAGE_QUALITY
        max_size = (
            (self.config.max_width, self.config.max_height)
            if self.config
            else (DEFAULT_IMAGE_MAX_WIDTH, DEFAULT_IMAGE_MAX_HEIGHT)
        )
        compress_enabled = self.config.compress if self.config else True

        # Get filter parameters
        min_width = self.config.filter.min_width if self.config else 50
        min_height = self.config.filter.min_height if self.config else 50
        min_area = self.config.filter.min_area if self.config else 5000

        # Prepare work items (filter duplicates first)
        work_items: list[tuple[int, bytes]] = []
        deduplicated_count = 0
        for idx, (_alt_text, _mime_type, image_data) in enumerate(images, start=1):
            if self.is_duplicate(image_data):
                deduplicated_count += 1
                continue
            work_items.append((idx, image_data))

        if not work_items:
            return ImageProcessResult(
                saved_images=[], filtered_count=0, deduplicated_count=deduplicated_count
            )

        # Determine worker count (use half of CPUs to avoid system overload)
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 4) // 2)

        # Process images in parallel using ProcessPoolExecutor
        loop = asyncio.get_event_loop()
        processed_results: list[tuple[int, bytes, int, int]] = []
        filtered_count = 0

        # Use ProcessPoolExecutor for CPU-bound compression
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, image_data in work_items:
                if compress_enabled:
                    future = loop.run_in_executor(
                        executor,
                        _compress_image_worker,
                        image_data,
                        quality,
                        max_size,
                        output_format,
                        min_width,
                        min_height,
                        min_area,
                    )
                    futures.append((idx, future))
                else:
                    # No compression, just validate size
                    try:
                        with io.BytesIO(image_data) as buffer:
                            img = Image.open(buffer)
                            w, h = img.size
                            if w >= min_width and h >= min_height and w * h >= min_area:
                                processed_results.append((idx, image_data, w, h))
                            else:
                                filtered_count += 1
                    except Exception:
                        pass

            # Gather results from workers
            for idx, future in futures:
                try:
                    result = await future
                    if result is None:
                        filtered_count += 1
                    else:
                        compressed_data, final_w, final_h = result
                        processed_results.append(
                            (idx, compressed_data, final_w, final_h)
                        )
                except Exception:
                    filtered_count += 1

        # Second pass: save images concurrently (I/O-bound)
        semaphore = asyncio.Semaphore(max_io_concurrency)
        saved_images: list[ExtractedImage] = []

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
            for idx, data, width, height in processed_results
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
