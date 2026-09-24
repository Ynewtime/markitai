"""Office document converters (DOCX, PPTX, XLSX, XLS)."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import DEFAULT_RENDER_DPI, SCREENSHOTS_REL_PATH
from markitai.converter._patches import (
    apply_all_patches,
    apply_openpyxl_patches,
    apply_pptx_patches,
)
from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    append_screenshot_comments,
    register_converter,
)
from markitai.notices import user_notice
from markitai.ocr import (
    OCR_INSTALL_HINT,
    OCRBackendMissing,
    OCRLanguageError,
    is_ocr_available,
)
from markitai.utils import office_mac
from markitai.utils.mime import get_mime_type, normalize_image_extension
from markitai.utils.office import find_libreoffice, has_ms_office
from markitai.utils.paths import create_tracked_temp_dir, ensure_screenshots_dir
from markitai.vision_consent import ensure_vlm_ocr_disclosed, vlm_ocr_allowed

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

# A picture embedded in a slide as a data URI (python-pptx reader output).
_DATA_URI_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(data:(?P<mime>image/[\w.+-]+);base64,(?P<data>[A-Za-z0-9+/=]+)\)"
)

# Pictures smaller than this many pixels (icons, logos) are not OCR'd.
_PICTURE_OCR_MIN_PIXELS = 40_000

# The slide boundary the PPTX readers emit (``<!-- Slide number: N -->``).
_SLIDE_MARKER_RE = re.compile(r"<!--\s*Slide number:\s*(\d+)\s*-->")


class OfficeConverter(BaseConverter):
    """Base converter for Office documents.

    Uses MarkItDown for text extraction (cross-platform).
    COM is only used for slide/page rendering when needed.
    """

    def __init__(self, config: MarkitaiConfig | None = None) -> None:
        super().__init__(config)
        self._markitdown: Any = None
        # Rendering paths may use openpyxl/python-pptx directly. Plain DOCX
        # uses neither; only its generic fallback needs those imports.
        if FileFormat.XLSX in self.supported_formats:
            apply_openpyxl_patches()
        elif FileFormat.PPTX in self.supported_formats:
            apply_pptx_patches()
        elif FileFormat.DOCX not in self.supported_formats:
            apply_all_patches()

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert Office document to Markdown using MarkItDown.

        Args:
            input_path: Path to the Office document
            output_dir: Unused (present for BaseConverter interface compatibility)

        Returns:
            ConvertResult with markdown content
        """
        return self._convert_with_markitdown(Path(input_path))

    def _convert_with_markitdown(self, input_path: Path) -> ConvertResult:
        """Convert using MarkItDown library."""
        if self._markitdown is None:
            from markitdown import MarkItDown

            apply_all_patches()
            self._markitdown = MarkItDown()
        result = self._markitdown.convert(input_path, keep_data_uris=True)

        metadata = {
            "source": str(input_path),
            "format": input_path.suffix.lstrip(".").upper(),
            "converter": "markitdown",
        }

        if result.title:
            metadata["title"] = result.title

        return ConvertResult(
            markdown=result.markdown,
            images=[],
            metadata=metadata,
        )


@register_converter(FileFormat.DOCX)
class DocxConverter(OfficeConverter):
    """Converter for DOCX (Word) documents.

    Uses a plain-document reader, Mammoth for rich content, and MarkItDown's
    preprocessing for OMML equations.
    """

    supported_formats = [FileFormat.DOCX]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        if self._markitdown is not None:
            return self._convert_with_markitdown(Path(input_path))
        from markitai.converter.docx import convert_docx_without_math

        path = Path(input_path)
        result = convert_docx_without_math(path)
        return result if result is not None else self._convert_with_markitdown(path)


@register_converter(FileFormat.PPTX)
class PptxConverter(OfficeConverter):
    """Converter for PPTX (PowerPoint) documents.

    Text extraction uses python-pptx directly, with generic format fallback.
    Slide rendering uses COM (Windows) or LibreOffice (Linux/macOS),
    falling back to PowerPoint AppleScript on macOS without LibreOffice.

    Modes:
    - Default: Text extraction only
    - --screenshot: Text + slide screenshots
    - --ocr: Text + commented slide images
    - --ocr --llm: Text + slides for LLM Vision
    """

    supported_formats = [FileFormat.PPTX]

    def _convert_with_markitdown(self, input_path: Path) -> ConvertResult:
        # Keep an explicitly supplied adapter and the generic fallback intact.
        if self._markitdown is not None:
            return super()._convert_with_markitdown(input_path)
        from zipfile import BadZipFile

        from pptx.exc import PackageNotFoundError

        from markitai.converter.pptx import convert_pptx

        try:
            return convert_pptx(input_path)
        except (BadZipFile, ValueError, KeyError, OSError, PackageNotFoundError):
            return super()._convert_with_markitdown(input_path)

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert PPTX document to Markdown.

        Modes:
        - --ocr --llm: Extract text + render slides for LLM Vision
        - --ocr only: Extract text + commented slide images
        - Default: Standard text extraction
        """
        input_path = Path(input_path)

        use_ocr = self.config and self.config.ocr.enabled
        use_llm = self.config and self.config.llm.enabled

        if use_ocr and use_llm:
            # --ocr --llm: Extract text + render slides for LLM Vision.
            # With MARKITAI_NO_VLM_OCR set, never send slide images to a
            # remote model: degrade to local OCR (the PDF path's rule).
            if not vlm_ocr_allowed():
                return self._degrade_vlm_ocr(input_path, output_dir)
            logger.info("PPTX OCR+LLM mode: extracting text and rendering slides")
            return self._render_slides_for_llm(input_path, output_dir)
        elif use_ocr:
            # --ocr only: text layer + local OCR of the pictures on the
            # slides + commented slide images
            logger.info("PPTX OCR mode: extracting text with slide images (commented)")
            return self._convert_with_slide_images(input_path, output_dir)

        # Standard text conversion; the dedicated reader preserves slide content.
        # COM is only needed for slide screenshots, not text extraction
        result = self._convert_with_markitdown(input_path)

        # Render slide screenshots if enabled (independent of OCR)
        enable_screenshot = self.config and self.config.screenshot.enabled
        if enable_screenshot and output_dir:
            screenshots_dir = ensure_screenshots_dir(output_dir)

            # Get image format from config
            image_format = "jpg"
            if self.config:
                image_format = normalize_image_extension(self.config.image.format)

            _slides, slide_images = self._render_slides_to_images(
                input_path, screenshots_dir, image_format
            )

            # Update metadata with page_images for LLM processing.
            # Screenshots are counted from page_images, never as images:
            # ConvertResult.images holds embedded pictures only.
            result.metadata["page_images"] = slide_images
            result.metadata["pages"] = len(slide_images)
            result.metadata["extracted_text"] = result.markdown
            if not use_llm:
                # Without the LLM the .md is the whole output: reference each
                # slide's screenshot in a comment after its content, as the
                # PDF path does per page. With it, the vision step feeds on
                # the plain text and adds the page comments to .llm.md itself.
                result.markdown = append_screenshot_comments(
                    result.markdown, slide_images, _SLIDE_MARKER_RE, "Slide"
                )

            logger.debug(f"Rendered {len(slide_images)} slide screenshots")

        return result

    def _degrade_vlm_ocr(
        self, input_path: Path, output_dir: Path | None
    ) -> ConvertResult:
        """Fall back to local OCR when MARKITAI_NO_VLM_OCR blocks the VLM path.

        Privacy-preserving degrade, identical to the PDF path: slide images
        never reach a remote model. Uses RapidOCR when installed, otherwise
        fails with an error naming both ways out.
        """
        if is_ocr_available():
            logger.warning(
                "[VLM OCR] Disabled by MARKITAI_NO_VLM_OCR; "
                "falling back to local RapidOCR for {}",
                input_path.name,
            )
            return self._convert_with_slide_images(input_path, output_dir)
        raise OCRBackendMissing(
            "VLM OCR is disabled by MARKITAI_NO_VLM_OCR=1 and the local "
            "RapidOCR backend is not installed. Either unset "
            "MARKITAI_NO_VLM_OCR to use the vision LLM, or install "
            f"RapidOCR ({OCR_INSTALL_HINT})."
        )

    def _ocr_pictures(self, markdown: str, input_path: Path) -> tuple[str, int]:
        """OCR the raster pictures embedded in the slides, in place.

        The slide text layer is already exact; what it cannot carry is text
        inside pictures (a pasted screenshot, a scanned page, a chart
        exported as an image). Each sizable picture is read locally and the
        recognized text goes right under its reference. Tiny pictures
        (icons, logos) and vector (SVG) ones are skipped; one that cannot
        be read keeps its bare reference and raises a notice.

        Returns:
            Tuple of (markdown, number of pictures OCR was run on)
        """
        import base64
        import io

        from PIL import Image

        from markitai.ocr import OCRProcessor

        processor: OCRProcessor | None = None
        read = 0
        unreadable = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal processor, read, unreadable
            ref = match.group(0)
            if match.group("mime") == "image/svg+xml":
                return ref
            try:
                data = base64.b64decode(match.group("data"), validate=False)
                with Image.open(io.BytesIO(data)) as picture:
                    width, height = picture.size
            except Exception:
                return ref
            if width * height < _PICTURE_OCR_MIN_PIXELS:
                return ref
            if processor is None:
                processor = OCRProcessor(self.config.ocr if self.config else None)
            read += 1
            try:
                text = processor.recognize_bytes(data).text.strip()
            except (OCRBackendMissing, OCRLanguageError):
                raise
            except Exception as e:
                logger.debug("[PPTX] Picture OCR failed in {}: {}", input_path.name, e)
                unreadable += 1
                return ref
            return f"{ref}\n\n{text}" if text else ref

        markdown = _DATA_URI_IMAGE_RE.sub(replace, markdown)
        if unreadable:
            user_notice(
                "[OCR] Could not read {} picture(s) in {}",
                unreadable,
                input_path.name,
            )
        return markdown, read

    def _convert_with_slide_images(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert PPTX with text extraction + commented slide images.

        OCR mode always renders slides — the ``screenshot`` flag only controls
        extra screenshots in the default (non-OCR) path. Pictures embedded in
        the slides are read with local OCR (see :meth:`_ocr_pictures`);
        ``ocr_used`` reports whether any were.

        Args:
            input_path: Path to the PPTX file
            output_dir: Output directory for slide images

        Returns:
            ConvertResult with text content and commented image references
        """
        # First, extract text using MarkItDown
        text_result = self._convert_with_markitdown(input_path)
        extracted_text, pictures_read = self._ocr_pictures(
            text_result.markdown, input_path
        )
        if not pictures_read:
            logger.debug(
                "[PPTX] {}: no pictures to OCR; text comes from the text layer",
                input_path.name,
            )

        # Setup screenshots directory for slide images
        if output_dir:
            screenshots_dir = ensure_screenshots_dir(output_dir)
        else:
            screenshots_dir = create_tracked_temp_dir()

        # Get image format from config
        image_format = "jpg"
        if self.config:
            image_format = normalize_image_extension(self.config.image.format)

        # OCR path: always render slides (independent of screenshot flag).
        # The slide images are screenshots (page_images), not embedded images.
        _slides, slide_images = self._render_slides_to_images(
            input_path, screenshots_dir, image_format
        )

        # Build markdown with extracted text and commented slide images
        markdown_parts = [extracted_text]
        if slide_images:
            markdown_parts.append("\n\n<!-- Slide images for reference -->")
            for slide_info in slide_images:
                markdown_parts.append(
                    f"<!-- ![Slide {slide_info['page']}]({SCREENSHOTS_REL_PATH}/{slide_info['name']}) -->"
                )

        markdown = "\n".join(markdown_parts)

        return ConvertResult(
            markdown=markdown,
            images=text_result.images,
            metadata={
                "source": str(input_path),
                "format": "PPTX",
                "ocr_used": pictures_read > 0,
                "ocr_path": "rapidocr" if pictures_read else "none",
                "slides": len(slide_images),
                # Counted as screenshots, not images. Deliberately not
                # page_images: this path also serves the MARKITAI_NO_VLM_OCR
                # degrade, whose slides must never reach a vision model.
                "screenshot_count": len(slide_images),
            },
        )

    def _render_slides_to_images(
        self, input_path: Path, screenshots_dir: Path, image_format: str
    ) -> tuple[list[ExtractedImage], list[dict]]:
        """Render slides to images using the best available method.

        Args:
            input_path: Path to the PPTX file
            screenshots_dir: Directory to save screenshot images
            image_format: Image format (jpg, png, etc.)

        Returns:
            Tuple of (ExtractedImage list, slide info list for metadata)
        """

        # Try Windows COM first
        if has_ms_office():
            try:
                return self._render_slides_with_com(
                    input_path, screenshots_dir, image_format
                )
            except Exception as e:
                logger.warning(f"COM rendering failed, trying PDF fallback: {e}")

        # Fallback: Convert to PDF and render pages
        # Log a hint for Windows users without MS Office
        import platform

        if platform.system() == "Windows":
            logger.warning(
                "[PPTX] MS Office not available. "
                "Install Microsoft Office for faster slide rendering. "
                "Falling back to LibreOffice PDF conversion..."
            )

        return self._render_slides_via_pdf(input_path, screenshots_dir, image_format)

    def _render_slides_with_com(
        self, input_path: Path, screenshots_dir: Path, image_format: str
    ) -> tuple[list[ExtractedImage], list[dict]]:
        """Render slides using PowerPoint COM automation."""
        import pythoncom  # type: ignore[import-not-found]
        import win32com.client  # type: ignore[import-not-found]

        logger.debug(f"Rendering slides with PowerPoint COM: {input_path.name}")

        ppt = None
        presentation = None
        images: list[ExtractedImage] = []
        slide_images: list[dict] = []

        # Create ImageProcessor for compression with config
        from markitai.image import ImageProcessor

        img_processor = ImageProcessor(self.config.image if self.config else None)

        # Initialize COM for this thread (required for asyncio thread pool)
        pythoncom.CoInitialize()
        try:
            ppt = win32com.client.Dispatch("PowerPoint.Application")
            presentation = ppt.Presentations.Open(
                str(input_path.resolve()),
                ReadOnly=True,
                Untitled=False,
                WithWindow=False,
            )

            export_format = "JPG" if image_format == "jpg" else image_format.upper()

            for i, slide in enumerate(presentation.Slides, 1):
                prefix = self.asset_prefix or input_path.name
                image_name = f"{prefix}.slide{i:04d}.{image_format}"
                image_path = screenshots_dir / image_name

                slide.Export(str(image_path.resolve()), export_format)

                # Apply compression with configured quality
                from PIL import Image

                with Image.open(image_path) as img:
                    original_width, original_height = img.size

                    # Compress if enabled in config
                    if self.config and self.config.image.compress:
                        format_map = {
                            "jpg": "JPEG",
                            "jpeg": "JPEG",
                            "png": "PNG",
                            "webp": "WEBP",
                        }
                        output_format = format_map.get(image_format, "JPEG")
                        compressed_img, compressed_data = img_processor.compress(
                            img.copy(),
                            quality=self.config.image.quality,
                            max_size=(
                                self.config.image.max_width,
                                self.config.image.max_height,
                            ),
                            output_format=output_format,
                        )
                        image_path.write_bytes(compressed_data)
                        width, height = compressed_img.size
                    else:
                        width, height = original_width, original_height

                images.append(
                    ExtractedImage(
                        path=image_path,
                        index=i,
                        original_name=image_name,
                        mime_type=f"image/{image_format}",
                        width=width,
                        height=height,
                    )
                )
                slide_images.append(
                    {
                        "page": i,
                        "path": str(image_path),
                        "name": image_name,
                    }
                )
                logger.debug(f"Rendered slide {i}/{len(presentation.Slides)}")

            presentation.Close()
            presentation = None

        finally:
            if presentation:
                try:
                    presentation.Close()
                except Exception as e:
                    logger.debug("[PPTX] COM presentation.Close() failed: {}", e)
            if ppt:
                try:
                    ppt.Quit()
                except Exception as e:
                    logger.debug("[PPTX] COM ppt.Quit() failed: {}", e)
            pythoncom.CoUninitialize()

        return images, slide_images

    def _render_slides_via_pdf(
        self, input_path: Path, screenshots_dir: Path, image_format: str
    ) -> tuple[list[ExtractedImage], list[dict]]:
        """Render slides by converting to PDF first."""
        import subprocess
        import time

        logger.info(f"[PPTX] Rendering slides via PDF: {input_path.name}")

        import platform

        soffice_cmd = find_libreoffice()
        if soffice_cmd is None:
            # macOS fallback: export PDF via PowerPoint AppleScript
            if (
                platform.system() == "Darwin"
                and (self.config is None or self.config.office.macos_fallback)
                and office_mac.powerpoint_available()
            ):
                pass  # fall through to the PowerPoint branch below
            elif platform.system() == "Windows":
                user_notice(
                    f"[PPTX] Cannot render slides of {input_path.name}: "
                    "Neither MS Office nor LibreOffice found. "
                    "Install Microsoft Office (recommended) or LibreOffice "
                    "(winget install TheDocumentFoundation.LibreOffice) "
                    "to enable slide rendering."
                )
                return [], []
            elif platform.system() == "Darwin":
                if self.config is not None and not self.config.office.macos_fallback:
                    user_notice(
                        f"[PPTX] Cannot render slides of {input_path.name}: "
                        "LibreOffice not found and "
                        "office.macos_fallback is disabled. Install LibreOffice "
                        "(brew install --cask libreoffice), or enable "
                        "office.macos_fallback to use Microsoft PowerPoint."
                    )
                else:
                    user_notice(
                        f"[PPTX] Cannot render slides of {input_path.name}: "
                        "Neither LibreOffice nor "
                        "Microsoft PowerPoint found. Install LibreOffice "
                        "(brew install --cask libreoffice) or Microsoft Office "
                        "to enable slide rendering."
                    )
                return [], []
            else:
                user_notice(
                    f"[PPTX] Cannot render slides of {input_path.name}: "
                    "LibreOffice not found. Install "
                    "LibreOffice (e.g. apt-get install libreoffice / dnf install "
                    "libreoffice) to enable slide rendering."
                )
                return [], []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / f"{input_path.stem}.pdf"

            if soffice_cmd is None:
                try:
                    pp_start = time.perf_counter()
                    office_mac.pptx_to_pdf(input_path, temp_path)
                    pp_time = time.perf_counter() - pp_start
                    logger.info(f"[PPTX] PowerPoint PDF export: {pp_time:.2f}s")
                except (RuntimeError, OSError) as e:
                    # OSError: the Office staging folder is not writable (macOS
                    # App Data protection); the text layer still converts.
                    if office_mac.is_container_permission_error(e):
                        user_notice(
                            "[PPTX] Cannot render slides of {}: {}",
                            input_path.name,
                            office_mac.CONTAINER_BLOCKED_HINT,
                        )
                        return [], []
                    user_notice(
                        "[PPTX] Cannot render slides of {}: PowerPoint PDF "
                        "export failed: {}",
                        input_path.name,
                        e,
                    )
                    return [], []
                if not pdf_path.exists():
                    user_notice(
                        "[PPTX] Cannot render slides of {}: PowerPoint did not "
                        "produce a PDF",
                        input_path.name,
                    )
                    return [], []
            else:
                # Create isolated user profile for concurrent LibreOffice execution
                profile_path = temp_path / "lo_profile"
                profile_path.mkdir()
                profile_url = profile_path.as_uri()

                try:
                    lo_start = time.perf_counter()
                    result = subprocess.run(
                        [
                            soffice_cmd,
                            "--headless",
                            f"-env:UserInstallation={profile_url}",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(input_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    lo_time = time.perf_counter() - lo_start
                    logger.info(f"[PPTX] LibreOffice conversion: {lo_time:.2f}s")
                    if result.returncode != 0 or not pdf_path.exists():
                        user_notice(
                            "[PPTX] Cannot render slides of {}: LibreOffice failed: {}",
                            input_path.name,
                            result.stderr,
                        )
                        return [], []
                except subprocess.TimeoutExpired:
                    logger.error("[PPTX] LibreOffice timeout (>600s)")
                    return [], []
                except Exception as e:
                    logger.error(f"[PPTX] LibreOffice error: {e}")
                    return [], []

            try:
                import pymupdf
            except ImportError:
                return [], []

            render_start = time.perf_counter()
            # Create ImageProcessor for compression
            from markitai.image import ImageProcessor

            img_processor = ImageProcessor(self.config.image if self.config else None)

            doc = pymupdf.open(pdf_path)
            try:
                images: list[ExtractedImage] = []
                slide_images: list[dict] = []
                dpi = DEFAULT_RENDER_DPI

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)

                    # Named after the resolved output (BaseConverter.asset_prefix)
                    # so a renamed re-run keeps the older output's slides.
                    prefix = self.asset_prefix or input_path.name
                    image_name = f"{prefix}.slide{page_num + 1:04d}.{image_format}"
                    image_path = screenshots_dir / image_name
                    # Save with compression (ensures < 5MB for LLM)
                    final_size, actual_path = img_processor.save_screenshot(
                        pix.samples, pix.width, pix.height, image_path
                    )

                    # Use the actual path returned by save_screenshot, which may
                    # differ from image_path when the fallback changes the extension
                    actual_name = actual_path.name
                    actual_mime = get_mime_type(
                        actual_path.suffix, default=f"image/{image_format}"
                    )

                    images.append(
                        ExtractedImage(
                            path=actual_path,
                            index=page_num + 1,
                            original_name=actual_name,
                            mime_type=actual_mime,
                            width=final_size[0],
                            height=final_size[1],
                        )
                    )
                    slide_images.append(
                        {
                            "page": page_num + 1,
                            "path": str(actual_path),
                            "name": actual_name,
                        }
                    )

                render_time = time.perf_counter() - render_start
                logger.info(f"[PPTX] Rendered {len(doc)} slides: {render_time:.2f}s")
                return images, slide_images
            finally:
                doc.close()

    def _render_slides_for_llm(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Extract text and render slides for LLM Vision analysis.

        This method:
        1. Extracts text using MarkItDown (fast, preserves structure)
        2. Always renders each slide as an image for LLM Vision

        The ``screenshot`` flag only controls extra screenshots in the default
        (non-OCR) path — OCR+LLM always needs slide images.

        Args:
            input_path: Path to the PPTX file
            output_dir: Optional output directory for slide images

        Returns:
            ConvertResult with extracted text and slide images
        """
        # Step 1: Extract text using MarkItDown
        text_result = self._convert_with_markitdown(input_path)
        extracted_text = text_result.markdown

        # Determine output path for slide images
        if output_dir:
            screenshots_dir = ensure_screenshots_dir(output_dir)
        else:
            screenshots_dir = create_tracked_temp_dir()

        # Get image format from config
        image_format = "jpg"
        if self.config:
            image_format = normalize_image_extension(self.config.image.format)

        # OCR+LLM path: always render slides (independent of screenshot flag).
        # They travel as page_images; images holds embedded pictures only.
        _slides, slide_images = self._render_slides_to_images(
            input_path, screenshots_dir, image_format
        )

        # One-time privacy disclosure: the rendered slide images are about to
        # be handed to the vision LLM for OCR reading. Nothing rendered means
        # nothing is sent, and nothing to disclose.
        if slide_images:
            ensure_vlm_ocr_disclosed(self.config, page_count=len(slide_images))

        return ConvertResult(
            markdown=extracted_text,
            images=text_result.images,
            metadata={
                "source": str(input_path),
                "format": "PPTX",
                "ocr_path": "vlm",
                "slides": len(slide_images),
                "extracted_text": extracted_text,
                "page_images": slide_images,
            },
        )


@register_converter(FileFormat.XLSX)
class XlsxConverter(OfficeConverter):
    """Converter for XLSX (Excel) documents.

    Reads each sheet with openpyxl; ambiguous values retain pandas formatting.
    """

    supported_formats = [FileFormat.XLSX]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        if self._markitdown is not None:
            return self._convert_with_markitdown(Path(input_path))
        from zipfile import BadZipFile

        from markitai.converter.xlsx import convert_xlsx

        path = Path(input_path)
        try:
            return convert_xlsx(path)
        except (BadZipFile, ValueError, KeyError, OSError):
            # Preserve generic format detection for mislabeled/non-XLSX files.
            return self._convert_with_markitdown(path)


@register_converter(FileFormat.XLS)
class XlsConverter(OfficeConverter):
    """Converter for legacy XLS (Excel 97-2003) documents.

    Uses MarkItDown directly (via xlrd) - cross-platform, no Office
    application or LibreOffice involved. Cell content is identical to
    upgrading through Excel first (verified live); embedded images and
    charts are not extracted, matching the XLSX path.
    """

    supported_formats = [FileFormat.XLS]
