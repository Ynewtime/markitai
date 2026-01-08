"""Image format converter for incompatible formats.

Converts legacy/incompatible image formats to web-friendly formats:
- EMF (Enhanced Metafile) -> PNG
- WMF (Windows Metafile) -> PNG
- TIFF -> PNG/JPEG
- BMP -> PNG
- ICO -> PNG
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from markit.converters.base import ExtractedImage
from markit.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# Formats that need conversion (including GIF for LLM compatibility)
CONVERTIBLE_FORMATS = {"emf", "wmf", "tiff", "tif", "bmp", "ico", "dib", "gif"}

# Target format mapping
FORMAT_CONVERSION_MAP = {
    "emf": "png",
    "wmf": "png",
    "tiff": "png",  # or jpeg if no alpha
    "tif": "png",
    "bmp": "png",
    "ico": "png",
    "dib": "png",
    "gif": "png",  # GIF -> PNG for LLM compatibility (Gemini doesn't support GIF)
}

# Formats supported by LLM vision APIs (Gemini, OpenAI, etc.)
LLM_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "jpg", "webp", "heic", "heif"}


@dataclass
class ConvertedImage:
    """Result of image format conversion."""

    data: bytes
    format: str
    filename: str
    original_format: str
    width: int
    height: int


def is_llm_supported_format(format_name: str) -> bool:
    """Check if image format is supported by LLM vision APIs.

    Args:
        format_name: Image format name (e.g., 'png', 'gif', 'jpeg')

    Returns:
        True if format is supported by LLM APIs like Gemini
    """
    return format_name.lower() in LLM_SUPPORTED_IMAGE_FORMATS


class ImageFormatConverter:
    """Convert incompatible image formats to web-friendly formats."""

    def __init__(
        self,
        inkscape_path: str | None = None,
        wmf2svg_path: str | None = None,
    ) -> None:
        """Initialize the image format converter.

        Args:
            inkscape_path: Path to Inkscape (for EMF/WMF)
            wmf2svg_path: Path to wmf2svg tool
        """
        self.inkscape_path = inkscape_path or shutil.which("inkscape")
        self.wmf2svg_path = wmf2svg_path or shutil.which("wmf2svg")
        self._pillow_wmf_support = self._check_pillow_wmf()

    def _check_pillow_wmf(self) -> bool:
        """Check if Pillow can handle WMF/EMF."""
        try:
            # Try to import Pillow's WMF handler
            from PIL import WmfImagePlugin  # noqa: F401

            return True
        except ImportError:
            return False

    def needs_conversion(self, format_name: str) -> bool:
        """Check if format needs conversion.

        Args:
            format_name: Image format name

        Returns:
            True if conversion is needed
        """
        return format_name.lower() in CONVERTIBLE_FORMATS

    def convert(self, image: ExtractedImage) -> ConvertedImage | None:
        """Convert image to web-friendly format.

        Args:
            image: Extracted image to convert

        Returns:
            Converted image
        """
        format_lower = image.format.lower()

        if not self.needs_conversion(format_lower):
            # No conversion needed, return as-is
            return ConvertedImage(
                data=image.data,
                format=image.format,
                filename=image.filename,
                original_format=image.format,
                width=image.width or 0,
                height=image.height or 0,
            )

        log.info(
            "Converting image format",
            filename=image.filename,
            from_format=format_lower,
        )

        # Route to appropriate converter
        if format_lower in ("emf", "wmf"):
            return self._convert_metafile(image)
        elif format_lower in ("tiff", "tif"):
            return self._convert_tiff(image)
        else:
            return self._convert_with_pillow(image)

    def _convert_with_pillow(self, image: ExtractedImage) -> ConvertedImage:
        """Convert image using Pillow."""
        try:
            img = Image.open(io.BytesIO(image.data))

            # Determine output format
            target_format = FORMAT_CONVERSION_MAP.get(image.format.lower(), "png")

            # Convert RGBA to RGB for JPEG
            if target_format == "jpeg" and img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background

            # Save to buffer
            buffer = io.BytesIO()
            save_format = target_format.upper()
            if save_format == "JPG":
                save_format = "JPEG"

            img.save(buffer, format=save_format)
            converted_data = buffer.getvalue()

            # Generate new filename
            new_filename = Path(image.filename).stem + f".{target_format}"

            return ConvertedImage(
                data=converted_data,
                format=target_format,
                filename=new_filename,
                original_format=image.format,
                width=img.size[0],
                height=img.size[1],
            )

        except Exception as e:
            log.error(
                "Pillow conversion failed",
                filename=image.filename,
                error=str(e),
            )
            raise

    def _convert_metafile(self, image: ExtractedImage) -> ConvertedImage | None:
        """Convert EMF/WMF metafile to PNG."""
        # Try Pillow first (may work on Windows)
        if self._pillow_wmf_support:
            try:
                return self._convert_with_pillow(image)
            except Exception as e:
                log.warning(
                    "Pillow WMF conversion failed, trying Inkscape",
                    error=str(e),
                )

        # Try Inkscape
        if self.inkscape_path:
            try:
                return self._convert_with_inkscape(image)
            except Exception as e:
                log.warning(
                    "Inkscape conversion failed",
                    error=str(e),
                )

        # Cannot convert - return None to signal skip
        log.warning(
            "Cannot convert metafile, skipping image",
            filename=image.filename,
        )
        return None

    def _convert_with_inkscape(self, image: ExtractedImage) -> ConvertedImage:
        """Convert EMF/WMF using Inkscape."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / f"input.{image.format}"
            output_file = temp_path / "output.png"

            # Write input file
            input_file.write_bytes(image.data)

            # Run Inkscape
            cmd = [
                self.inkscape_path,
                str(input_file),
                "--export-type=png",
                f"--export-filename={output_file}",
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            if not output_file.exists():
                raise RuntimeError("Inkscape did not produce output")

            # Read result
            converted_data = output_file.read_bytes()

            # Get dimensions
            img = Image.open(output_file)
            width, height = img.size

            new_filename = Path(image.filename).stem + ".png"

            return ConvertedImage(
                data=converted_data,
                format="png",
                filename=new_filename,
                original_format=image.format,
                width=width,
                height=height,
            )

    def _convert_tiff(self, image: ExtractedImage) -> ConvertedImage:
        """Convert TIFF image."""
        img = Image.open(io.BytesIO(image.data))

        # Determine target format based on alpha channel
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

        if has_alpha:
            target_format = "png"
        else:
            target_format = "jpeg"
            # Convert to RGB for JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")

        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format=target_format.upper())
        converted_data = buffer.getvalue()

        new_filename = Path(image.filename).stem + f".{target_format}"

        return ConvertedImage(
            data=converted_data,
            format=target_format,
            filename=new_filename,
            original_format=image.format,
            width=img.size[0],
            height=img.size[1],
        )

    def _create_placeholder(self, image: ExtractedImage) -> ConvertedImage:
        """Create a placeholder image when conversion fails."""
        # Create a simple gray placeholder
        width = image.width or 200
        height = image.height or 200

        placeholder = Image.new("RGB", (width, height), (200, 200, 200))

        # Add text if possible
        try:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(placeholder)
            text = f"[{image.format.upper()}]"
            draw.text((10, 10), text, fill=(100, 100, 100))
        except Exception:
            pass

        buffer = io.BytesIO()
        placeholder.save(buffer, format="PNG")

        new_filename = Path(image.filename).stem + ".png"

        return ConvertedImage(
            data=buffer.getvalue(),
            format="png",
            filename=new_filename,
            original_format=image.format,
            width=width,
            height=height,
        )


class BatchImageConverter:
    """Convert multiple images with progress tracking."""

    def __init__(self, converter: ImageFormatConverter | None = None):
        """Initialize batch converter.

        Args:
            converter: Image format converter to use
        """
        self.converter = converter or ImageFormatConverter()

    def convert_batch(
        self,
        images: list[ExtractedImage],
        on_progress: Callable | None = None,
    ) -> list[ConvertedImage]:
        """Convert a batch of images.

        Args:
            images: Images to convert
            on_progress: Progress callback (index, total, image)

        Returns:
            List of converted images
        """
        results = []
        total = len(images)

        for idx, image in enumerate(images):
            try:
                if self.converter.needs_conversion(image.format):
                    converted = self.converter.convert(image)
                else:
                    # No conversion needed
                    converted = ConvertedImage(
                        data=image.data,
                        format=image.format,
                        filename=image.filename,
                        original_format=image.format,
                        width=image.width or 0,
                        height=image.height or 0,
                    )

                results.append(converted)

                if on_progress:
                    on_progress(idx + 1, total, image)

            except Exception as e:
                log.error(
                    "Image conversion failed",
                    filename=image.filename,
                    error=str(e),
                )
                # Create placeholder for failed conversions
                placeholder = self.converter._create_placeholder(image)
                results.append(placeholder)

        return results


def check_conversion_tools() -> dict[str, bool]:
    """Check availability of image conversion tools.

    Returns:
        Dictionary with tool availability status
    """
    return {
        "inkscape": shutil.which("inkscape") is not None,
        "wmf2svg": shutil.which("wmf2svg") is not None,
        "pillow_wmf": _check_pillow_wmf(),
    }


def _check_pillow_wmf() -> bool:
    """Check if Pillow has WMF support."""
    try:
        from PIL import WmfImagePlugin  # noqa: F401

        return True
    except ImportError:
        return False
