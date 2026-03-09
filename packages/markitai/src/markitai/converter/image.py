"""Image file converters using OCR or LLM Vision."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from loguru import logger

from markitai.constants import ASSETS_REL_PATH
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)
from markitai.utils.paths import ensure_assets_dir


class ImageConverter(BaseConverter):
    """Converter for image files using OCR or LLM Vision.

    Extracts text from images using RapidOCR by default,
    or LLM Vision when --llm --alt|--desc flags are used.
    """

    supported_formats = [
        FileFormat.JPEG,
        FileFormat.JPG,
        FileFormat.PNG,
        FileFormat.WEBP,
        FileFormat.GIF,
        FileFormat.BMP,
        FileFormat.TIFF,
        FileFormat.SVG,
    ]
    _preview_transcode_suffixes = {".bmp", ".tif", ".tiff"}

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert image to Markdown.

        Modes based on config:
        - OCR + LLM: Return placeholder for later LLM vision analysis
        - OCR only: Extract text via RapidOCR
        - Neither: Return image reference placeholder

        Args:
            input_path: Path to the image file
            output_dir: Optional output directory for copying image

        Returns:
            ConvertResult containing markdown with OCR text or placeholder
        """
        input_path = Path(input_path)

        # Check if OCR and LLM are enabled in config
        use_ocr = self.config and self.config.ocr.enabled
        use_llm = self.config and self.config.llm.enabled

        # Copy image to assets directory and get relative path
        image_ref_path = self._copy_to_assets(input_path, output_dir)

        if use_ocr and use_llm:
            # --ocr --llm: Skip OCR, let LLM Vision analyze the image later
            # Just return a placeholder - LLM will process it in cli.py
            markdown = self._create_image_placeholder(input_path, image_ref_path)
            return ConvertResult(
                markdown=markdown,
                images=[],
                metadata={
                    "format": input_path.suffix.lstrip(".").upper(),
                    "source": str(input_path),
                    "asset_path": image_ref_path,
                },
            )
        elif use_ocr:
            # --ocr only: Use RapidOCR
            markdown = self._convert_with_ocr(input_path, image_ref_path)
        else:
            # Just return a placeholder with image reference
            markdown = self._create_image_placeholder(input_path, image_ref_path)

        return ConvertResult(
            markdown=markdown,
            images=[],  # No embedded images to extract
            metadata={
                "format": input_path.suffix.lstrip(".").upper(),
                "source": str(input_path),
                "ocr_used": use_ocr and not use_llm,
                "asset_path": image_ref_path,
            },
        )

    def _copy_to_assets(self, input_path: Path, output_dir: Path | None) -> str:
        """Copy image to assets directory and return relative path.

        Args:
            input_path: Path to the source image file
            output_dir: Output directory (assets will be created inside)

        Returns:
            Relative path to use in markdown (e.g., "assets/image.jpg")
        """
        if output_dir is None:
            # No output directory specified, use original filename
            return input_path.name

        assets_dir = ensure_assets_dir(output_dir)

        if input_path.suffix.lower() in self._preview_transcode_suffixes:
            dest_path = assets_dir / self._transcoded_asset_name(input_path)
            if not dest_path.exists():
                self._transcode_to_png(input_path, dest_path)
                logger.debug(f"Transcoded {input_path.name} to {dest_path.name}")
            return f"{ASSETS_REL_PATH}/{dest_path.name}"

        # Copy image to assets directory
        dest_path = assets_dir / input_path.name
        if not dest_path.exists():
            shutil.copy2(input_path, dest_path)
            logger.debug(f"Copied {input_path.name} to {dest_path}")

        return f"{ASSETS_REL_PATH}/{input_path.name}"

    def _transcode_to_png(self, input_path: Path, dest_path: Path) -> None:
        """Transcode less-compatible image formats to PNG for markdown previews."""
        from PIL import Image

        with Image.open(input_path) as image:
            image.save(dest_path, format="PNG")

    def _transcoded_asset_name(self, input_path: Path) -> str:
        """Build a stable, unique preview asset name for transcoded images."""
        source_id = hashlib.sha256(
            str(input_path.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        return f"{input_path.stem}-{source_id}.png"

    def _convert_with_ocr(self, input_path: Path, image_ref_path: str) -> str:
        """Convert image using OCR.

        Args:
            input_path: Path to the image file
            image_ref_path: Relative path for image reference in markdown

        Returns:
            Markdown with OCR extracted text
        """
        try:
            from markitai.ocr import OCRProcessor

            processor = OCRProcessor(self.config.ocr if self.config else None)
            result = processor.recognize_to_markdown(input_path)

            if result.strip():
                logger.debug(f"OCR extracted text from {input_path.name}")
                return f"# {input_path.stem}\n\n{result}"
            else:
                logger.warning(f"OCR found no text in {input_path.name}")
                return self._create_image_placeholder(input_path, image_ref_path)

        except ImportError:
            logger.warning("RapidOCR not available, returning placeholder")
            return self._create_image_placeholder(input_path, image_ref_path)
        except Exception as e:
            logger.warning(f"OCR failed for {input_path.name}: {e}")
            return self._create_image_placeholder(input_path, image_ref_path)

    def _create_image_placeholder(self, input_path: Path, image_ref_path: str) -> str:
        """Create a placeholder markdown for the image.

        Args:
            input_path: Path to the image file
            image_ref_path: Relative path for image reference in markdown

        Returns:
            Markdown with image placeholder
        """
        return f"# {input_path.stem}\n\n![{input_path.stem}]({image_ref_path})\n"


# Register ImageConverter for all supported image formats
for _fmt in (
    FileFormat.JPEG,
    FileFormat.JPG,
    FileFormat.PNG,
    FileFormat.WEBP,
    FileFormat.GIF,
    FileFormat.BMP,
    FileFormat.TIFF,
    FileFormat.SVG,
):
    register_converter(_fmt)(ImageConverter)
