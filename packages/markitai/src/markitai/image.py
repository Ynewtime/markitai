"""Image processing module for extraction, compression, and filtering."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger
from PIL import Image

from markitai.constants import (
    DEFAULT_IMAGE_IO_CONCURRENCY,
    DEFAULT_IMAGE_MAX_HEIGHT,
    DEFAULT_IMAGE_MAX_WIDTH,
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_SCREENSHOT_MAX_BYTES,
)
from markitai.utils.mime import get_extension_from_mime
from markitai.utils.paths import ensure_assets_dir

if TYPE_CHECKING:
    from markitai.config import ImageConfig
    from markitai.converter.base import ExtractedImage


# Module-level function for multiprocessing (must be picklable)
def _compress_image_cv2(
    image_data: bytes,
    quality: int,
    max_size: tuple[int, int],
    output_format: str,
    min_width: int,
    min_height: int,
    min_area: int,
) -> tuple[bytes, int, int] | None:
    """Compress a single image using OpenCV (releases GIL in C++ layer).

    OpenCV performs image operations in C++ which releases Python's GIL,
    making it more efficient for multi-threaded processing compared to Pillow.

    Args:
        image_data: Raw image bytes
        quality: JPEG/WEBP quality (1-100)
        max_size: Maximum dimensions (width, height)
        output_format: Output format (JPEG, PNG, WEBP)
        min_width: Minimum width filter
        min_height: Minimum height filter
        min_area: Minimum area filter

    Returns:
        Tuple of (compressed_data, final_width, final_height) or None if filtered
    """
    import cv2
    import numpy as np

    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        # Get dimensions (OpenCV uses height, width order)
        if len(img.shape) == 2:
            height, width = img.shape
            channels = 1
        else:
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1

        # Apply size filter
        if width < min_width or height < min_height or width * height < min_area:
            return None

        # Resize if needed (maintain aspect ratio like Pillow's thumbnail)
        max_w, max_h = max_size
        if width > max_w or height > max_h:
            scale = min(max_w / width, max_h / height)
            new_w, new_h = int(width * scale), int(height * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            width, height = new_w, new_h

        # Handle alpha channel for JPEG (convert BGRA/RGBA to BGR)
        fmt_upper = output_format.upper()
        if fmt_upper == "JPEG" and channels == 4:
            # Create white background and blend
            if img.shape[2] == 4:  # BGRA
                alpha = img[:, :, 3:4] / 255.0
                bgr = img[:, :, :3]
                white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
                img = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        # Encode to bytes
        if fmt_upper == "JPEG":
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, buffer = cv2.imencode(".jpg", img, encode_param)
        elif fmt_upper == "PNG":
            # PNG compression level 0-9, map quality 100->0, 0->9
            compression = max(0, min(9, 9 - quality // 11))
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, compression]
            success, buffer = cv2.imencode(".png", img, encode_param)
        elif fmt_upper == "WEBP":
            encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
            success, buffer = cv2.imencode(".webp", img, encode_param)
        else:
            # Fallback to JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, buffer = cv2.imencode(".jpg", img, encode_param)

        if not success:
            return None

        return buffer.tobytes(), width, height

    except Exception:
        return None


def _compress_image_pillow(
    image_data: bytes,
    quality: int,
    max_size: tuple[int, int],
    output_format: str,
    min_width: int,
    min_height: int,
    min_area: int,
) -> tuple[bytes, int, int] | None:
    """Compress a single image using Pillow (fallback implementation).

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

    Prefers OpenCV for better multi-threaded performance (releases GIL),
    falls back to Pillow if OpenCV fails.

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
    # Try OpenCV first (releases GIL, better for multi-threading)
    try:
        result = _compress_image_cv2(
            image_data,
            quality,
            max_size,
            output_format,
            min_width,
            min_height,
            min_area,
        )
        if result is not None:
            return result
    except ImportError:
        pass  # OpenCV not installed, fall through to Pillow
    except Exception:
        pass  # OpenCV failed, fall through to Pillow

    # Fallback to Pillow
    return _compress_image_pillow(
        image_data, quality, max_size, output_format, min_width, min_height, min_area
    )


@dataclass
class ProcessedImage:
    """Result of processing a single image.

    Tracks the original position and processing outcome for each image,
    enabling correct mapping during base64 replacement.
    """

    original_index: int  # 1-indexed position in original markdown
    saved_path: Path | None  # None if filtered/deduplicated
    skip_reason: str | None  # "duplicate" | "filtered" | None


@dataclass
class ImageProcessResult:
    """Result of image processing."""

    saved_images: list[ExtractedImage]
    filtered_count: int
    deduplicated_count: int
    # Mapping from original 1-indexed position to processing result
    # This enables correct base64 replacement even when images are filtered
    index_mapping: dict[int, ProcessedImage] | None = None


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

        from markitai.utils.office import find_libreoffice

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

                # Create isolated user profile for concurrent LibreOffice execution
                profile_path = temp_path / "lo_profile"
                profile_path.mkdir()
                profile_url = profile_path.as_uri()

                cmd = [
                    soffice,
                    "--headless",
                    f"-env:UserInstallation={profile_url}",
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
        index_mapping: dict[int, ProcessedImage] | None = None,
    ) -> str:
        """
        Replace base64 data URIs with file paths in markdown.

        When index_mapping is provided, uses position-based replacement to ensure
        each base64 image is replaced with the correct saved image, even when
        some images were filtered or deduplicated.

        Args:
            markdown: Original markdown with data URIs
            images: List of saved images with paths
            assets_path: Relative path to assets directory
            index_mapping: Optional mapping from original index to ProcessedImage

        Returns:
            Markdown with data URIs replaced by file paths (filtered images removed)
        """
        if index_mapping:
            # Use position-based replacement for correct mapping
            current_index = 0

            def replace_match_indexed(match: re.Match) -> str:
                nonlocal current_index
                current_index += 1  # 1-indexed
                processed = index_mapping.get(current_index)
                if processed is None:
                    # No mapping for this index, keep original
                    return match.group(0)
                if processed.saved_path is None:
                    # Image was filtered/deduplicated, remove from output
                    return ""
                return f"![{match.group(1)}]({assets_path}/{processed.saved_path.name})"

            return self.DATA_URI_PATTERN.sub(replace_match_indexed, markdown)

        # Legacy: sequential iteration (for backward compatibility)
        image_iter = iter(images)

        def replace_match(match: re.Match) -> str:
            try:
                img = next(image_iter)
                return f"![{match.group(1)}]({assets_path}/{img.path.name})"
            except StopIteration:
                return match.group(0)

        return self.DATA_URI_PATTERN.sub(replace_match, markdown)

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

    @staticmethod
    def remove_hallucinated_images(
        llm_output: str,
        original_content: str,
    ) -> str:
        """Remove hallucinated image URLs from LLM output.

        LLM may hallucinate image URLs that don't exist in the original content.
        This method compares image URLs in the LLM output against the original
        and removes any that weren't present originally.

        Args:
            llm_output: LLM processed markdown content
            original_content: Original markdown before LLM processing

        Returns:
            LLM output with hallucinated image references removed
        """
        # Extract all image URLs from original content
        img_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
        original_urls = set(img_pattern.findall(original_content))

        # Also extract URLs without markdown syntax (bare URLs in original)
        url_pattern = re.compile(r"https?://[^\s\)\"'>]+")
        original_urls.update(url_pattern.findall(original_content))

        def validate_image(match: re.Match) -> str:
            full_match = match.group(0)
            url = match.group(1)

            # Keep local asset references (handled by remove_nonexistent_images)
            if url.startswith("assets/") or url.startswith("assets\\"):
                return full_match

            # Keep relative URLs (likely internal links)
            if not url.startswith("http://") and not url.startswith("https://"):
                return full_match

            # Check if this URL existed in original
            if url in original_urls:
                return full_match

            # URL is hallucinated - remove it
            logger.debug(f"Removing hallucinated image URL: {url}")
            return ""

        result = img_pattern.sub(validate_image, llm_output)

        # Clean up any resulting empty lines
        result = re.sub(r"\n{3,}", "\n\n", result)

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

        image_hash = hashlib.md5(image_data, usedforsecurity=False).hexdigest()
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
            ImageProcessResult with saved images, statistics, and index mapping
        """
        # Delayed import to avoid circular import
        from markitai.converter.base import ExtractedImage

        # Create assets directory
        assets_dir = ensure_assets_dir(output_dir)

        saved_images: list[ExtractedImage] = []
        filtered_count = 0
        deduplicated_count = 0
        index_mapping: dict[int, ProcessedImage] = {}

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
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="duplicate"
                )
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
                        index_mapping[idx] = ProcessedImage(
                            original_index=idx, saved_path=None, skip_reason="filtered"
                        )
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

                    extracted = ExtractedImage(
                        path=output_path,
                        index=idx,
                        original_name=filename,
                        mime_type=f"image/{extension}",
                        width=final_width,
                        height=final_height,
                    )
                    saved_images.append(extracted)
                    index_mapping[idx] = ProcessedImage(
                        original_index=idx, saved_path=output_path, skip_reason=None
                    )
                finally:
                    img_buffer.close()

            except Exception:
                # Skip invalid images - record as filtered
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="error"
                )
                continue

        return ImageProcessResult(
            saved_images=saved_images,
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
            index_mapping=index_mapping,
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
            ImageProcessResult with saved images, statistics, and index mapping
        """
        import asyncio

        # Delayed imports to avoid circular import
        from markitai.converter.base import ExtractedImage
        from markitai.security import write_bytes_async

        # Create assets directory
        assets_dir = ensure_assets_dir(output_dir)

        saved_images: list[ExtractedImage] = []
        filtered_count = 0
        deduplicated_count = 0
        index_mapping: dict[int, ProcessedImage] = {}

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
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="duplicate"
                )
                continue

            # Load and process image
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    width, height = img.size

                    # Check filter
                    if self.should_filter(width, height):
                        filtered_count += 1
                        index_mapping[idx] = ProcessedImage(
                            original_index=idx, saved_path=None, skip_reason="filtered"
                        )
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
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="error"
                )
                continue

        # Second pass: save images concurrently (I/O-bound)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def save_image(
            idx: int, data: bytes, width: int, height: int
        ) -> tuple[int, ExtractedImage | None, Path | None]:
            filename = f"{base_name}.{idx:04d}.{extension}"
            output_path = assets_dir / filename

            async with semaphore:
                try:
                    await write_bytes_async(output_path, data)
                    return (
                        idx,
                        ExtractedImage(
                            path=output_path,
                            index=idx,
                            original_name=filename,
                            mime_type=f"image/{extension}",
                            width=width,
                            height=height,
                        ),
                        output_path,
                    )
                except Exception:
                    return idx, None, None

        # Run all saves concurrently
        tasks = [
            save_image(idx, data, width, height)
            for idx, data, width, height in processed_images
        ]
        results = await asyncio.gather(*tasks)

        # Collect successful saves and build index mapping
        for idx, extracted, output_path in results:
            if extracted is not None:
                saved_images.append(extracted)
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=output_path, skip_reason=None
                )
            else:
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="error"
                )

        # Sort by index to maintain order
        saved_images.sort(key=lambda x: x.index)

        return ImageProcessResult(
            saved_images=saved_images,
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
            index_mapping=index_mapping,
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
            ImageProcessResult with saved images, statistics, and index mapping
        """
        import asyncio

        from markitai.converter.base import ExtractedImage
        from markitai.security import write_bytes_async

        if not images:
            return ImageProcessResult(
                saved_images=[],
                filtered_count=0,
                deduplicated_count=0,
                index_mapping={},
            )

        # Create assets directory
        assets_dir = ensure_assets_dir(output_dir)

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
        index_mapping: dict[int, ProcessedImage] = {}
        for idx, (_alt_text, _mime_type, image_data) in enumerate(images, start=1):
            if self.is_duplicate(image_data):
                deduplicated_count += 1
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="duplicate"
                )
                continue
            work_items.append((idx, image_data))

        if not work_items:
            return ImageProcessResult(
                saved_images=[],
                filtered_count=0,
                deduplicated_count=deduplicated_count,
                index_mapping=index_mapping,
            )

        # Determine worker count (use half of CPUs to avoid system overload)
        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 4) // 2)

        # Process images in parallel using ProcessPoolExecutor
        loop = asyncio.get_running_loop()
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
                                index_mapping[idx] = ProcessedImage(
                                    original_index=idx,
                                    saved_path=None,
                                    skip_reason="filtered",
                                )
                    except Exception:
                        index_mapping[idx] = ProcessedImage(
                            original_index=idx, saved_path=None, skip_reason="error"
                        )

            # Gather results from workers
            for idx, future in futures:
                try:
                    result = await future
                    if result is None:
                        filtered_count += 1
                        index_mapping[idx] = ProcessedImage(
                            original_index=idx, saved_path=None, skip_reason="filtered"
                        )
                    else:
                        compressed_data, final_w, final_h = result
                        processed_results.append(
                            (idx, compressed_data, final_w, final_h)
                        )
                except Exception:
                    filtered_count += 1
                    index_mapping[idx] = ProcessedImage(
                        original_index=idx, saved_path=None, skip_reason="error"
                    )

        # Second pass: save images concurrently (I/O-bound)
        semaphore = asyncio.Semaphore(max_io_concurrency)
        saved_images: list[ExtractedImage] = []

        async def save_image(
            idx: int, data: bytes, width: int, height: int
        ) -> tuple[int, ExtractedImage | None, Path | None]:
            filename = f"{base_name}.{idx:04d}.{extension}"
            output_path = assets_dir / filename

            async with semaphore:
                try:
                    await write_bytes_async(output_path, data)
                    return (
                        idx,
                        ExtractedImage(
                            path=output_path,
                            index=idx,
                            original_name=filename,
                            mime_type=f"image/{extension}",
                            width=width,
                            height=height,
                        ),
                        output_path,
                    )
                except Exception:
                    return idx, None, None

        # Run all saves concurrently
        tasks = [
            save_image(idx, data, width, height)
            for idx, data, width, height in processed_results
        ]
        results = await asyncio.gather(*tasks)

        # Collect successful saves and build index mapping
        for idx, extracted, output_path in results:
            if extracted is not None:
                saved_images.append(extracted)
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=output_path, skip_reason=None
                )
            else:
                index_mapping[idx] = ProcessedImage(
                    original_index=idx, saved_path=None, skip_reason="error"
                )

        # Sort by index to maintain order
        saved_images.sort(key=lambda x: x.index)

        return ImageProcessResult(
            saved_images=saved_images,
            filtered_count=filtered_count,
            deduplicated_count=deduplicated_count,
            index_mapping=index_mapping,
        )


# =============================================================================
# URL Image Download
# =============================================================================

# Pattern to match markdown images: ![alt](url)
# Excludes data: URIs (base64 encoded images)
_URL_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\((?!data:)([^)]+)\)",
    re.IGNORECASE,
)

# Common image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico"}


@dataclass
class UrlImageDownloadResult:
    """Result of downloading images from URLs."""

    updated_markdown: str
    downloaded_paths: list[Path]
    failed_urls: list[str]
    url_to_path: dict[str, Path] = field(
        default_factory=dict
    )  # URL -> local path mapping


def _get_extension_from_content_type(content_type: str) -> str:
    """Get file extension from content-type header."""
    return get_extension_from_mime(content_type)


def _get_extension_from_url(url: str) -> str | None:
    """Extract image extension from URL path."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    # Remove query params from path
    path = path.split("?")[0]
    for ext in _IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return None


def _sanitize_image_filename(name: str, max_length: int = 100) -> str:
    """Sanitize filename for cross-platform compatibility."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    # Remove control characters
    name = "".join(c for c in name if ord(c) >= 32)
    # Limit length
    if len(name) > max_length:
        name = name[:max_length]
    return name.strip() or "image"


async def download_url_images(
    markdown: str,
    output_dir: Path,
    base_url: str,
    config: ImageConfig,
    source_name: str = "url",
    concurrency: int = 5,
    timeout: int = 30,
) -> UrlImageDownloadResult:
    """Download images from URLs in markdown and save to assets directory.

    This function:
    1. Finds all image URLs in markdown (excluding data: URIs)
    2. Downloads images concurrently with rate limiting
    3. Saves to assets directory with proper naming
    4. Replaces URLs with local paths in markdown
    5. Skips failed downloads (keeps original URL, logs warning)

    Args:
        markdown: Markdown content with image URLs
        output_dir: Output directory (assets will be created inside)
        base_url: Base URL for resolving relative image paths
        config: Image configuration (format, quality, etc.)
        source_name: Source identifier for naming images
        concurrency: Max concurrent downloads (default 5)
        timeout: HTTP request timeout in seconds (default 30)

    Returns:
        UrlImageDownloadResult with:
        - updated_markdown: Markdown with local paths for downloaded images
        - downloaded_paths: List of successfully downloaded image paths
        - failed_urls: List of URLs that failed to download
    """
    # Find all image URLs
    matches = list(_URL_IMAGE_PATTERN.finditer(markdown))
    if not matches:
        return UrlImageDownloadResult(
            updated_markdown=markdown,
            downloaded_paths=[],
            failed_urls=[],
        )

    # Create assets directory
    assets_dir = ensure_assets_dir(output_dir)

    # Prepare download tasks
    semaphore = asyncio.Semaphore(concurrency)
    downloaded_paths: list[Path] = []
    failed_urls: list[str] = []
    replacements: dict[str, str] = {}  # original_match -> replacement
    url_to_path: dict[str, Path] = {}  # image_url -> local_path mapping

    # Sanitize source name for filenames
    safe_source = _sanitize_image_filename(source_name, max_length=50)

    async def download_single(
        client: httpx.AsyncClient,
        match: re.Match,
        index: int,
    ) -> None:
        """Download a single image."""
        alt_text = match.group(1)
        image_url = match.group(2).strip()
        original_match = match.group(0)

        # Resolve relative URLs
        if not image_url.startswith(("http://", "https://", "//")):
            image_url = urljoin(base_url, image_url)
        elif image_url.startswith("//"):
            # Protocol-relative URL
            parsed_base = urlparse(base_url)
            image_url = f"{parsed_base.scheme}:{image_url}"

        async with semaphore:
            try:
                response = await client.get(
                    image_url,
                    follow_redirects=True,
                    timeout=timeout,
                )
                response.raise_for_status()

                # Determine file extension
                content_type = response.headers.get("content-type", "")
                ext = _get_extension_from_url(image_url)
                if not ext:
                    ext = _get_extension_from_content_type(content_type)

                # Generate filename: source_name.NNNN.ext (1-indexed, 4 digits)
                filename = f"{safe_source}.{index + 1:04d}{ext}"
                output_path = assets_dir / filename

                # Process image (apply quality settings if configured)
                image_data = response.content
                if ext.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    try:
                        processed = _compress_image_worker(
                            image_data,
                            quality=config.quality,
                            max_size=(config.max_width, config.max_height),
                            output_format=config.format.upper(),
                            min_width=config.filter.min_width,
                            min_height=config.filter.min_height,
                            min_area=config.filter.min_area,
                        )
                        if processed:
                            image_data, _, _ = processed
                            # Update extension if format changed
                            if config.format.lower() != ext[1:].lower():
                                ext = f".{config.format.lower()}"
                                filename = f"{safe_source}.{index + 1:04d}{ext}"
                                output_path = assets_dir / filename
                        else:
                            # Image was filtered out (too small)
                            logger.debug(
                                f"Image filtered (too small): {image_url[:60]}..."
                            )
                            return
                    except Exception as e:
                        logger.debug(f"Image processing failed, saving original: {e}")

                # Save to file
                output_path.write_bytes(image_data)
                downloaded_paths.append(output_path)

                # Prepare replacement with local path
                local_path = f"assets/{filename}"
                replacements[original_match] = f"![{alt_text}]({local_path})"

                # Track URL to path mapping for post-processing
                url_to_path[image_url] = output_path

                logger.debug(f"Downloaded: {image_url[:60]}... -> {output_path}")

            except httpx.TimeoutException:
                logger.warning(f"Timeout downloading image: {image_url[:80]}...")
                failed_urls.append(image_url)
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"HTTP {e.response.status_code} downloading: {image_url[:80]}..."
                )
                failed_urls.append(image_url)
            except Exception as e:
                logger.warning(f"Failed to download image: {image_url[:80]}... - {e}")
                failed_urls.append(image_url)

    # Download all images concurrently
    async with httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; markitai/0.4.0; +https://github.com/Ynewtime/markitai)"
        },
        follow_redirects=True,
    ) as client:
        tasks = [
            download_single(client, match, idx) for idx, match in enumerate(matches)
        ]
        await asyncio.gather(*tasks)

    # Apply replacements to markdown
    updated_markdown = markdown
    for original, replacement in replacements.items():
        updated_markdown = updated_markdown.replace(original, replacement)

    return UrlImageDownloadResult(
        updated_markdown=updated_markdown,
        downloaded_paths=downloaded_paths,
        failed_urls=failed_urls,
        url_to_path=url_to_path,
    )
