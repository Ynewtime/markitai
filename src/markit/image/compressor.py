"""Image compression utilities."""

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from markit.converters.base import ExtractedImage

from markit.utils.logging import get_logger

log = get_logger(__name__)


# =============================================================================
# Core compression functions (module-level for process pool compatibility)
# =============================================================================


def compress_image_data(
    data: bytes,
    image_format: str,
    jpeg_quality: int = 85,
    png_optimization_level: int = 2,
    max_dimension: int = 2048,
    use_oxipng: bool = True,
) -> tuple[bytes, str, int, int]:
    """Core image compression logic (picklable, can be called by process pool).

    This is the unified compression function used by both ImageCompressor
    and process pool workers.

    Args:
        data: Raw image bytes
        image_format: Image format (png, jpeg, etc.)
        jpeg_quality: JPEG quality (0-100)
        png_optimization_level: PNG optimization level (0-6)
        max_dimension: Maximum dimension for resizing
        use_oxipng: Whether to use oxipng for PNG optimization (disable for process pool)

    Returns:
        Tuple of (compressed_data, format, width, height)
    """
    # Open image
    img = Image.open(io.BytesIO(data))
    original_format = image_format.lower()

    # Resize if needed
    width, height = img.size
    if max(width, height) > max_dimension:
        ratio = max_dimension / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        width, height = img.size

    # Compress based on format
    output = io.BytesIO()
    if original_format in ("jpg", "jpeg"):
        # Convert to RGB if necessary (JPEG doesn't support alpha)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGB")
        img.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        result_format = "jpeg"
        compressed_data = output.getvalue()
    elif original_format == "png":
        img.save(output, format="PNG", optimize=True)
        compressed_data = output.getvalue()
        result_format = "png"

        # Apply oxipng if available and requested
        if use_oxipng:
            compressed_data = _apply_oxipng(compressed_data, png_optimization_level)
    else:
        # For other formats, just save as-is
        format_map = {
            "gif": "GIF",
            "webp": "WEBP",
            "bmp": "BMP",
        }
        pil_format = format_map.get(original_format, original_format.upper())
        img.save(output, format=pil_format)
        result_format = original_format
        compressed_data = output.getvalue()

    return compressed_data, result_format, width, height


def _apply_oxipng(data: bytes, optimization_level: int = 2) -> bytes:
    """Apply oxipng optimization to PNG data if available.

    Args:
        data: PNG image data
        optimization_level: Optimization level (0-6)

    Returns:
        Optimized PNG data, or original data if oxipng is unavailable
    """
    if not shutil.which("oxipng"):
        return data

    import sys
    import time

    # Create a temporary directory instead of a file
    # This avoids issues with file handle retention on Windows
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir) / "image.png"

        # Write data to temporary file
        tmp_path.write_bytes(data)

        # On Windows, add a small delay to ensure file handle is released
        if sys.platform == "win32":
            time.sleep(0.01)  # 10ms delay

        # Run oxipng
        try:
            subprocess.run(
                [
                    "oxipng",
                    "-o",
                    str(optimization_level),
                    "--strip",
                    "safe",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
            )
            return tmp_path.read_bytes()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.debug("oxipng optimization failed, using original", error=str(e))
            return data


@dataclass
class CompressionConfig:
    """Configuration for image compression."""

    png_optimization_level: int = 2  # oxipng: 0-6
    jpeg_quality: int = 85  # mozjpeg: 0-100
    max_dimension: int = 2048  # Maximum width or height
    skip_if_smaller_than: int = 10240  # Skip compression for files < 10KB


@dataclass
class CompressedImage:
    """Result of image compression."""

    data: bytes
    format: str
    filename: str
    original_size: int
    compressed_size: int
    width: int
    height: int

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (0-1, lower is better)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size

    @property
    def savings_percent(self) -> float:
        """Calculate percentage of space saved."""
        return (1 - self.compression_ratio) * 100


class ImageCompressor:
    """Compress images using various tools.

    This class wraps the core `compress_image_data` function and provides
    a convenient interface for compressing ExtractedImage objects.
    """

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """Initialize the compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()

    def compress(self, image: "ExtractedImage") -> CompressedImage:
        """Compress an image.

        Args:
            image: The image to compress

        Returns:
            Compressed image result
        """
        original_size = len(image.data)

        # Skip small images
        if original_size < self.config.skip_if_smaller_than:
            log.debug(
                "Skipping compression for small image",
                filename=image.filename,
                size=original_size,
            )
            return self._create_result(image, image.data)

        # Use the unified compression function
        compressed_data, result_format, width, height = compress_image_data(
            data=image.data,
            image_format=image.format,
            jpeg_quality=self.config.jpeg_quality,
            png_optimization_level=self.config.png_optimization_level,
            max_dimension=self.config.max_dimension,
            use_oxipng=True,  # Use oxipng when available
        )

        compressed_size = len(compressed_data)
        log.debug(
            "Image compressed",
            filename=image.filename,
            original_size=original_size,
            compressed_size=compressed_size,
            savings=f"{(1 - compressed_size / original_size) * 100:.1f}%",
        )

        return CompressedImage(
            data=compressed_data,
            format=result_format,
            filename=image.filename,
            original_size=original_size,
            compressed_size=compressed_size,
            width=width,
            height=height,
        )

    def _create_result(self, image: "ExtractedImage", data: bytes) -> CompressedImage:
        """Create a CompressedImage result from raw data (for skipped images)."""
        img = Image.open(io.BytesIO(data))
        return CompressedImage(
            data=data,
            format=image.format,
            filename=image.filename,
            original_size=len(image.data),
            compressed_size=len(data),
            width=img.size[0],
            height=img.size[1],
        )
