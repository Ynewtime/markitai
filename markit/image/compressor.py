"""Image compression utilities."""

import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from markit.converters.base import ExtractedImage
from markit.utils.logging import get_logger

log = get_logger(__name__)


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
    """Compress images using various tools."""

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """Initialize the compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or CompressionConfig()
        self._oxipng_available = self._check_oxipng()
        self._mozjpeg_available = self._check_mozjpeg()

    def _check_oxipng(self) -> bool:
        """Check if oxipng is available."""
        return shutil.which("oxipng") is not None

    def _check_mozjpeg(self) -> bool:
        """Check if mozjpeg (cjpeg) is available."""
        return shutil.which("cjpeg") is not None or shutil.which("jpegtran") is not None

    def compress(self, image: ExtractedImage) -> CompressedImage:
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

        # Load image with Pillow
        img = Image.open(io.BytesIO(image.data))
        original_width, original_height = img.size

        # Resize if needed
        img = self._resize_if_needed(img)

        # Compress based on format
        fmt = image.format.lower()
        if fmt in ("png",):
            compressed_data = self._compress_png(img)
        elif fmt in ("jpeg", "jpg"):
            compressed_data = self._compress_jpeg(img)
        else:
            # For other formats, just resize
            compressed_data = self._save_image(img, fmt)

        compressed_size = len(compressed_data)
        # Changed to debug to reduce log noise when processing many images
        log.debug(
            "Image compressed",
            filename=image.filename,
            original_size=original_size,
            compressed_size=compressed_size,
            savings=f"{(1 - compressed_size / original_size) * 100:.1f}%",
        )

        return CompressedImage(
            data=compressed_data,
            format=image.format,
            filename=image.filename,
            original_size=original_size,
            compressed_size=compressed_size,
            width=img.size[0],
            height=img.size[1],
        )

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        """Resize image if it exceeds maximum dimensions."""
        width, height = img.size
        max_dim = self.config.max_dimension

        if width <= max_dim and height <= max_dim:
            return img

        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))

        log.debug(
            "Resizing image",
            original=f"{width}x{height}",
            new=f"{new_width}x{new_height}",
        )

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _compress_png(self, img: Image.Image) -> bytes:
        """Compress PNG image using oxipng if available, else Pillow."""
        if self._oxipng_available:
            return self._compress_png_oxipng(img)
        else:
            return self._compress_png_pillow(img)

    def _compress_png_oxipng(self, img: Image.Image) -> bytes:
        """Compress PNG using oxipng.

        On Windows, special care is taken to ensure temporary files are properly
        closed before oxipng processes them, avoiding WinError 32 (file in use).
        """
        import sys
        import time

        # Create a temporary directory instead of a file
        # This avoids issues with file handle retention on Windows
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir) / "image.png"

            # Save image to temporary file
            # Use explicit close to ensure file handle is released
            img.save(str(tmp_path), format="PNG")

            # On Windows, add a small delay to ensure file handle is released
            if sys.platform == "win32":
                time.sleep(0.01)  # 10ms delay

            # Run oxipng
            try:
                subprocess.run(
                    [
                        "oxipng",
                        "-o",
                        str(self.config.png_optimization_level),
                        "--strip",
                        "safe",
                        str(tmp_path),
                    ],
                    check=True,
                    capture_output=True,
                )
                compressed_data = tmp_path.read_bytes()
            except subprocess.CalledProcessError as e:
                log.warning("oxipng failed, using Pillow", error=str(e))
                compressed_data = self._compress_png_pillow(img)
            except FileNotFoundError:
                # oxipng not found
                log.debug("oxipng not available, using Pillow")
                compressed_data = self._compress_png_pillow(img)

            return compressed_data

    def _compress_png_pillow(self, img: Image.Image) -> bytes:
        """Compress PNG using Pillow."""
        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _compress_jpeg(self, img: Image.Image) -> bytes:
        """Compress JPEG image."""
        # Convert to RGB if necessary (JPEG doesn't support alpha)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGB")

        output = io.BytesIO()
        img.save(output, format="JPEG", quality=self.config.jpeg_quality, optimize=True)
        return output.getvalue()

    def _save_image(self, img: Image.Image, fmt: str) -> bytes:
        """Save image in specified format."""
        output = io.BytesIO()

        # Map format names
        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG",
            "gif": "GIF",
            "webp": "WEBP",
            "bmp": "BMP",
        }
        pil_format = format_map.get(fmt.lower(), fmt.upper())

        img.save(output, format=pil_format)
        return output.getvalue()

    def _create_result(self, image: ExtractedImage, data: bytes) -> CompressedImage:
        """Create a CompressedImage result from raw data."""
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
