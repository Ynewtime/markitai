"""Office document converters (DOCX, PPTX, XLSX)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from markitdown import MarkItDown

from markit.constants import DEFAULT_RENDER_DPI
from markit.converter.base import (
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    register_converter,
)
from markit.image import ImageProcessor
from markit.utils.office import find_libreoffice, has_ms_office

if TYPE_CHECKING:
    from markit.config import MarkitConfig


class OfficeConverter(BaseConverter):
    """Base converter for Office documents.

    Uses MarkItDown for text extraction (cross-platform).
    COM is only used for slide/page rendering when needed.
    """

    def __init__(self, config: MarkitConfig | None = None) -> None:
        super().__init__(config)
        self._markitdown = MarkItDown()

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert Office document to Markdown using MarkItDown."""
        return self._convert_with_markitdown(Path(input_path))

    def _convert_with_markitdown(self, input_path: Path) -> ConvertResult:
        """Convert using MarkItDown library."""
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

    Uses MarkItDown directly (via python-docx) - cross-platform.
    """

    supported_formats = [FileFormat.DOCX]


@register_converter(FileFormat.PPTX)
class PptxConverter(OfficeConverter):
    """Converter for PPTX (PowerPoint) documents.

    Text extraction uses MarkItDown (via python-pptx) - cross-platform.
    Slide rendering uses COM (Windows) or LibreOffice (Linux/macOS).

    Modes:
    - Default: Text extraction only
    - --screenshot: Text + slide screenshots
    - --ocr: Text + commented slide images
    - --ocr --llm: Text + slides for LLM Vision
    """

    supported_formats = [FileFormat.PPTX]

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
            # --ocr --llm: Extract text + render slides for LLM
            logger.info("PPTX OCR+LLM mode: extracting text and rendering slides")
            return self._render_slides_for_llm(input_path, output_dir)
        elif use_ocr:
            # --ocr only: Extract text + commented slide images
            logger.info("PPTX OCR mode: extracting text with slide images (commented)")
            return self._convert_with_ocr(input_path, output_dir)

        # Standard conversion - use MarkItDown directly (cross-platform)
        # COM is only needed for slide screenshots, not text extraction
        result = self._convert_with_markitdown(input_path)

        # Render slide screenshots if enabled (independent of OCR)
        enable_screenshot = self.config and self.config.screenshot.enabled
        if enable_screenshot and output_dir:
            screenshots_dir = output_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            # Get image format from config
            image_format = "jpg"
            if self.config:
                fmt = self.config.image.format
                image_format = "jpg" if fmt == "jpeg" else fmt

            images, slide_images = self._render_slides_to_images(
                input_path, screenshots_dir, image_format
            )

            # Update metadata with page_images for LLM processing
            result.metadata["page_images"] = slide_images
            result.metadata["pages"] = len(slide_images)
            result.metadata["pptx_llm_mode"] = True  # Enable LLM mode for screenshots
            result.metadata["extracted_text"] = result.markdown
            result.images = images

            logger.debug(f"Rendered {len(slide_images)} slide screenshots")

        return result

    def _convert_with_ocr(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert PPTX with text extraction + commented slide images.

        Args:
            input_path: Path to the PPTX file
            output_dir: Output directory for slide images

        Returns:
            ConvertResult with text content and commented image references
        """
        # First, extract text using MarkItDown
        text_result = self._convert_with_markitdown(input_path)
        extracted_text = text_result.markdown

        # Check if screenshot is enabled
        enable_screenshot = self.config and self.config.screenshot.enabled

        images: list[ExtractedImage] = []
        slide_images: list[dict] = []

        # Render slides as images (only if screenshot enabled)
        if enable_screenshot:
            # Setup screenshots directory for slide images
            if output_dir:
                screenshots_dir = output_dir / "screenshots"
                screenshots_dir.mkdir(parents=True, exist_ok=True)
            else:
                screenshots_dir = Path(tempfile.mkdtemp())

            # Get image format from config
            image_format = "jpg"
            if self.config:
                fmt = self.config.image.format
                image_format = "jpg" if fmt == "jpeg" else fmt

            images, slide_images = self._render_slides_to_images(
                input_path, screenshots_dir, image_format
            )

        # Build markdown with extracted text and commented slide images
        markdown_parts = [extracted_text]
        if enable_screenshot and slide_images:
            markdown_parts.append("\n\n<!-- Slide images for reference -->")
            for slide_info in slide_images:
                markdown_parts.append(
                    f"<!-- ![Slide {slide_info['page']}](screenshots/{slide_info['name']}) -->"
                )

        markdown = "\n".join(markdown_parts)

        return ConvertResult(
            markdown=markdown,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PPTX",
                "ocr_used": True,
                "slides": len(images),
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
        return self._render_slides_via_pdf(input_path, screenshots_dir, image_format)

    def _render_slides_with_com(
        self, input_path: Path, screenshots_dir: Path, image_format: str
    ) -> tuple[list[ExtractedImage], list[dict]]:
        """Render slides using PowerPoint COM automation."""
        import pythoncom
        import win32com.client

        logger.debug(f"Rendering slides with PowerPoint COM: {input_path.name}")

        ppt = None
        presentation = None
        images: list[ExtractedImage] = []
        slide_images: list[dict] = []

        # Create ImageProcessor for compression with config
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
                image_name = f"{input_path.name}.slide{i:04d}.{image_format}"
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
                except Exception:
                    pass
            if ppt:
                try:
                    ppt.Quit()
                except Exception:
                    pass
            pythoncom.CoUninitialize()

        return images, slide_images

    def _render_slides_via_pdf(
        self, input_path: Path, screenshots_dir: Path, image_format: str
    ) -> tuple[list[ExtractedImage], list[dict]]:
        """Render slides by converting to PDF first."""
        import subprocess
        import time

        logger.info(f"[PPTX] Rendering slides via PDF: {input_path.name}")

        soffice_cmd = find_libreoffice()
        if not soffice_cmd:
            logger.warning("LibreOffice not found")
            return [], []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / f"{input_path.stem}.pdf"

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
                    logger.warning(f"[PPTX] LibreOffice failed: {result.stderr}")
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

                    image_name = (
                        f"{input_path.name}.slide{page_num + 1:04d}.{image_format}"
                    )
                    image_path = screenshots_dir / image_name
                    # Save with compression (ensures < 5MB for LLM)
                    final_size = img_processor.save_screenshot(
                        pix.samples, pix.width, pix.height, image_path
                    )

                    images.append(
                        ExtractedImage(
                            path=image_path,
                            index=page_num + 1,
                            original_name=image_name,
                            mime_type=f"image/{image_format}",
                            width=final_size[0],
                            height=final_size[1],
                        )
                    )
                    slide_images.append(
                        {
                            "page": page_num + 1,
                            "path": str(image_path),
                            "name": image_name,
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
        2. Renders each slide as an image (if enable_screenshot is True)

        The CLI will send both text + images to LLM for enhanced analysis.

        Args:
            input_path: Path to the PPTX file
            output_dir: Optional output directory for slide images

        Returns:
            ConvertResult with extracted text and slide images
        """
        # Step 1: Extract text using MarkItDown
        text_result = self._convert_with_markitdown(input_path)
        extracted_text = text_result.markdown

        # Check if screenshot is enabled
        enable_screenshot = self.config and self.config.screenshot.enabled

        images: list[ExtractedImage] = []
        slide_images: list[dict] = []

        # Step 2: Render slides to images (only if screenshot enabled)
        if enable_screenshot:
            # Determine output path for slide images
            if output_dir:
                screenshots_dir = output_dir / "screenshots"
                screenshots_dir.mkdir(parents=True, exist_ok=True)
            else:
                screenshots_dir = Path(tempfile.mkdtemp())

            # Get image format from config
            image_format = "jpg"
            if self.config:
                fmt = self.config.image.format
                image_format = "jpg" if fmt == "jpeg" else fmt

            images, slide_images = self._render_slides_to_images(
                input_path, screenshots_dir, image_format
            )

        return ConvertResult(
            markdown=extracted_text,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PPTX",
                "pptx_llm_mode": True,
                "slides": len(images),
                "extracted_text": extracted_text,
                "page_images": slide_images,
            },
        )


@register_converter(FileFormat.XLSX)
class XlsxConverter(OfficeConverter):
    """Converter for XLSX (Excel) documents.

    Uses MarkItDown directly (via openpyxl) - cross-platform.
    """

    supported_formats = [FileFormat.XLSX]
