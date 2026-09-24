"""Image file converters using OCR or LLM Vision."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import ASSETS_REL_PATH, page_marker
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)
from markitai.converter.heif import HEIF_SUFFIXES, decode_to_png, ensure_heif_ready
from markitai.notices import user_notice
from markitai.ocr import (
    OCR_INSTALL_HINT,
    OCRBackendMissing,
    OCRError,
    OCRLanguageError,
    is_ocr_available,
)
from markitai.utils.paths import ensure_assets_dir
from markitai.utils.text import markdown_image_reference
from markitai.vision_consent import ensure_vlm_ocr_disclosed, vlm_ocr_allowed

if TYPE_CHECKING:
    from markitai.ocr import OCRProcessor

# Formats whose frames are pages of one document (fax/scan TIFFs), as opposed
# to animation frames (GIF/WebP), where the first frame stands for the image.
_MULTI_PAGE_SUFFIXES = {".tif", ".tiff"}

# Rasterization scale for SVG before OCR (1 SVG px -> 2 bitmap px), so text
# drawn at ordinary sizes is tall enough for the recognizer.
_SVG_OCR_SCALE = 2.0


def _rasterize_svg(path: Path) -> Any:
    """Render an SVG to an RGB array (PyMuPDF opens SVG as a document)."""
    import numpy as np
    import pymupdf

    doc = pymupdf.open(path)
    try:
        if len(doc) == 0:
            raise OCRError(f"{path.name} has nothing to render")
        pix = doc[0].get_pixmap(
            matrix=pymupdf.Matrix(_SVG_OCR_SCALE, _SVG_OCR_SCALE), alpha=False
        )
        return (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape((pix.height, pix.width, pix.n))[:, :, :3]
            .copy()
        )
    finally:
        doc.close()


def _frame_rgb_array(image: Any, index: int) -> Any:
    """One frame of a multi-frame PIL image as an RGB array."""
    import numpy as np
    from PIL import ImageOps

    image.seek(index)
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


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
        FileFormat.HEIC,
        FileFormat.HEIF,
        FileFormat.AVIF,
    ]
    _preview_transcode_suffixes = {".bmp", ".tif", ".tiff"} | HEIF_SUFFIXES

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

        # HEIC/HEIF/AVIF need pillow-heif to decode — fail early with an
        # actionable error naming the extra (skipped for mislabeled files
        # whose content is not actually a HEIF container).
        if input_path.suffix.lower() in HEIF_SUFFIXES:
            ensure_heif_ready(input_path)

        # Check if OCR and LLM are enabled in config
        use_ocr = self.config and self.config.ocr.enabled
        use_llm = self.config and self.config.llm.enabled

        # Copy image to assets directory and get relative path
        image_ref_path = self._copy_to_assets(input_path, output_dir)

        if use_ocr and use_llm:
            # --ocr --llm: Skip OCR, let LLM Vision analyze the image later.
            # The image goes to the vision model for OCR reading — disclose
            # once per process, and honor the MARKITAI_NO_VLM_OCR opt-out.
            if not vlm_ocr_allowed():
                return self._degrade_vlm_ocr(input_path, image_ref_path, output_dir)
            ensure_vlm_ocr_disclosed(self.config, page_count=1)
            markdown = self._create_image_placeholder(input_path, image_ref_path)
            return ConvertResult(
                markdown=markdown,
                images=[],
                metadata={
                    "format": input_path.suffix.lstrip(".").upper(),
                    "source": str(input_path),
                    "ocr_path": "vlm",
                    "asset_path": image_ref_path,
                },
            )
        elif use_ocr:
            # --ocr only: Use RapidOCR
            markdown = self._convert_with_ocr(
                input_path,
                image_ref_path,
                ocr_source=self._ocr_source(input_path, output_dir),
            )
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
                "ocr_path": "rapidocr" if use_ocr else "none",
                "asset_path": image_ref_path,
            },
        )

    def _degrade_vlm_ocr(
        self,
        input_path: Path,
        image_ref_path: str,
        output_dir: Path | None,
    ) -> ConvertResult:
        """Fall back to local OCR when MARKITAI_NO_VLM_OCR blocks the VLM path.

        Privacy-preserving degrade: never send the image to a remote model.
        Uses RapidOCR when installed, otherwise fails with an actionable
        error that names both ways out (unset the env var, or install
        RapidOCR).
        """
        if is_ocr_available():
            logger.warning(
                "[VLM OCR] Disabled by MARKITAI_NO_VLM_OCR; "
                "falling back to local RapidOCR for {}",
                input_path.name,
            )
            markdown = self._convert_with_ocr(
                input_path,
                image_ref_path,
                ocr_source=self._ocr_source(input_path, output_dir),
            )
            return ConvertResult(
                markdown=markdown,
                images=[],
                metadata={
                    "format": input_path.suffix.lstrip(".").upper(),
                    "source": str(input_path),
                    "ocr_used": True,
                    "ocr_path": "rapidocr",
                    "asset_path": image_ref_path,
                },
            )
        raise OCRBackendMissing(
            "VLM OCR is disabled by MARKITAI_NO_VLM_OCR=1 and the local "
            "RapidOCR backend is not installed. Either unset "
            "MARKITAI_NO_VLM_OCR to use the vision LLM, or install "
            f"RapidOCR ({OCR_INSTALL_HINT})."
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
        dest_path = assets_dir / self._asset_name(input_path)
        if not dest_path.exists():
            shutil.copy2(input_path, dest_path)
            logger.debug(f"Copied {input_path.name} to {dest_path}")

        return f"{ASSETS_REL_PATH}/{dest_path.name}"

    def _asset_name(self, input_path: Path) -> str:
        """Name of the image's copy in the shared assets directory.

        Derived from the resolved output name (``BaseConverter.asset_prefix``)
        so a renamed re-run (``photo.jpg.v2.md``) gets its own copy instead of
        sharing the one ``photo.jpg.md`` references. The image's extension is
        kept so the copy stays viewable; the default output keeps the plain
        input name.
        """
        prefix = self.asset_prefix or input_path.name
        suffix = input_path.suffix
        if suffix and prefix.lower().endswith(suffix.lower()):
            return prefix
        return f"{prefix}{suffix}"

    def _transcode_to_png(self, input_path: Path, dest_path: Path) -> None:
        """Transcode less-compatible image formats to PNG for markdown previews."""
        if input_path.suffix.lower() in HEIF_SUFFIXES:
            # Decode once at the boundary (registers pillow-heif lazily,
            # applies EXIF orientation) — downstream sees a plain PNG.
            decode_to_png(input_path, dest_path)
            return

        from PIL import Image

        with Image.open(input_path) as image:
            image.save(dest_path, format="PNG")

    def _ocr_source(self, input_path: Path, output_dir: Path | None) -> Path:
        """Return the path OCR should read (HEIF is decoded to PNG first).

        RapidOCR cannot read HEIF-family containers, so those are decoded to
        PNG once and OCR runs on the PNG. Reuses the transcoded asset when
        available; otherwise decodes into a temporary file.
        """
        if input_path.suffix.lower() not in HEIF_SUFFIXES:
            return input_path

        if output_dir is not None:
            transcoded = ensure_assets_dir(output_dir) / self._transcoded_asset_name(
                input_path
            )
            if transcoded.exists():
                return transcoded

        import tempfile

        tmp_path = Path(tempfile.mkdtemp(prefix="markitai-heif-")) / (
            input_path.stem + ".png"
        )
        decode_to_png(input_path, tmp_path)
        return tmp_path

    def _transcoded_asset_name(self, input_path: Path) -> str:
        """Build a stable, unique preview asset name for transcoded images."""
        source_id = hashlib.sha256(
            str(input_path.resolve()).encode("utf-8")
        ).hexdigest()[:12]
        # A renamed output (asset_prefix other than the input name) gets its
        # own preview rather than sharing the older output's.
        stem = input_path.stem
        if self.asset_prefix and self.asset_prefix != input_path.name:
            stem = self.asset_prefix
        return f"{stem}-{source_id}.png"

    def _convert_with_ocr(
        self,
        input_path: Path,
        image_ref_path: str,
        ocr_source: Path | None = None,
    ) -> str:
        """Convert image using OCR.

        Args:
            input_path: Path to the image file
            image_ref_path: Relative path for image reference in markdown
            ocr_source: Optional decoded stand-in to run OCR on (e.g. the
                PNG transcoded from a HEIF file); defaults to input_path

        Returns:
            Markdown with OCR extracted text; the plain image reference
            (with a user notice) when the image holds no readable text

        Raises:
            OCRError: The image could not be decoded or the OCR engine
                failed. The item fails: a text-free placeholder reported as
                a successful conversion hid that nothing was read.
        """
        from markitai.ocr import OCRProcessor

        try:
            processor = OCRProcessor(self.config.ocr if self.config else None)
            result = self._recognize(processor, input_path, ocr_source or input_path)
        except (OCRBackendMissing, OCRLanguageError, OCRError):
            # Missing backend: the user asked for OCR and it is one command
            # away. Bad ocr.lang: a setting to fix. Both already say what to
            # do, and neither is this image's fault.
            raise
        except Exception as e:
            raise OCRError(f"OCR failed for {input_path.name}: {e}") from e

        if result.strip():
            logger.debug(f"OCR extracted text from {input_path.name}")
            return (
                f"# {input_path.stem}\n\n"
                f"{markdown_image_reference(input_path.stem, image_ref_path)}\n\n"
                f"{result}"
            )
        # A blank image is a legitimate input, not a failure -- but the user
        # asked for its text, so say that there was none.
        user_notice(
            "[OCR] No text found in {}; the output only references the image",
            input_path.name,
        )
        return self._create_image_placeholder(input_path, image_ref_path)

    def _recognize(
        self, processor: OCRProcessor, input_path: Path, source: Path
    ) -> str:
        """Run OCR on every page of the image and return its Markdown.

        RapidOCR opens ordinary raster files itself. SVG is vector and is
        rasterized first (PyMuPDF reads SVG). A multi-page TIFF is read
        frame by frame, each frame under its own page marker: RapidOCR alone
        only ever saw the first frame and dropped the rest silently. One
        frame is decoded at a time, and recognized before the next: decoded
        all at once, a 300-page A4 scan at 300 dpi took about 8 GB.
        """
        if source.suffix.lower() == ".svg":
            return processor.recognize_array_to_markdown(_rasterize_svg(source))

        from PIL import Image

        # Opening parses the header only: an empty, unidentifiable or
        # decompression-bomb file fails here with PIL's own explanation;
        # undecodable pixel data fails in RapidOCR's decoder below.
        with Image.open(source) as image:
            frame_count = getattr(image, "n_frames", 1)
            if source.suffix.lower() not in _MULTI_PAGE_SUFFIXES or frame_count < 2:
                pages = None
            else:
                logger.debug(
                    f"OCR reading {frame_count} TIFF pages of {input_path.name}"
                )
                pages = [
                    processor.recognize_array_to_markdown(_frame_rgb_array(image, i))
                    for i in range(frame_count)
                ]

        if pages is None:
            return processor.recognize_to_markdown(source)
        if not any(page.strip() for page in pages):
            return ""
        return "\n\n".join(
            f"{page_marker(number)}\n\n{page}".rstrip()
            for number, page in enumerate(pages, 1)
        )

    def _create_image_placeholder(self, input_path: Path, image_ref_path: str) -> str:
        """Create a placeholder markdown for the image.

        Args:
            input_path: Path to the image file
            image_ref_path: Relative path for image reference in markdown

        Returns:
            Markdown with image placeholder
        """
        return (
            f"# {input_path.stem}\n\n"
            f"{markdown_image_reference(input_path.stem, image_ref_path)}\n"
        )


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
    FileFormat.HEIC,
    FileFormat.HEIF,
    FileFormat.AVIF,
):
    register_converter(_fmt)(ImageConverter)
