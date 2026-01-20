"""PDF document converter."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pymupdf4llm
from loguru import logger

from markit.constants import DEFAULT_RENDER_DPI
from markit.converter.base import (
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    register_converter,
)
from markit.image import ImageProcessor

if TYPE_CHECKING:
    from markit.config import MarkitConfig


@register_converter(FileFormat.PDF)
class PdfConverter(BaseConverter):
    """Converter for PDF documents using pymupdf4llm.

    Supports OCR mode for scanned PDFs when --ocr flag is enabled.
    """

    supported_formats = [FileFormat.PDF]

    def __init__(self, config: MarkitConfig | None = None) -> None:
        super().__init__(config)

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """
        Convert PDF document to Markdown.

        Args:
            input_path: Path to the input file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing markdown and extracted images
        """
        input_path = Path(input_path)
        images: list[ExtractedImage] = []

        # Check if OCR mode is enabled
        use_ocr = self.config and self.config.ocr.enabled
        use_llm = self.config and self.config.llm.enabled

        if use_ocr:
            if use_llm:
                # --ocr --llm: Render pages as images for LLM Vision analysis
                return self._render_pages_for_llm(input_path, output_dir)
            # --ocr only: Use RapidOCR for text extraction
            return self._convert_with_ocr(input_path, output_dir)

        # Determine image output path
        temp_dir: Path | None = None
        if output_dir:
            image_path = output_dir / "assets"
            image_path.mkdir(parents=True, exist_ok=True)
            write_images = True
        else:
            # Use temp directory if no output dir specified
            temp_dir = Path(tempfile.mkdtemp())
            image_path = temp_dir
            write_images = True

        # Get image format from config
        image_format = "png"
        dpi = DEFAULT_RENDER_DPI
        if self.config:
            image_format = self.config.image.format
            if image_format == "jpeg":
                image_format = "jpg"

        # Convert using pymupdf4llm
        markdown = cast(
            str,
            pymupdf4llm.to_markdown(
                str(input_path),
                write_images=write_images,
                image_path=str(image_path),
                image_format=image_format,
                dpi=dpi,
                force_text=True,
            ),
        )

        # Fix image paths in markdown: pymupdf4llm uses absolute/full paths,
        # we need relative paths (assets/xxx.jpg)
        markdown = self._fix_image_paths(markdown, image_path)

        # Collect extracted images (only for current file)
        if write_images and image_path.exists():
            # Use input filename as prefix to filter images from this file only
            file_prefix = input_path.name
            image_processor = ImageProcessor(self.config.image if self.config else None)
            for idx, img_file in enumerate(
                sorted(image_path.glob(f"{file_prefix}*.{image_format}"))
            ):
                suffix = img_file.suffix.lower().lstrip(".")
                width = 0
                height = 0

                # Optionally compress and overwrite to keep sizes consistent
                if self.config and self.config.image.compress:
                    format_map = {
                        "jpg": "JPEG",
                        "jpeg": "JPEG",
                        "png": "PNG",
                        "webp": "WEBP",
                    }
                    output_format = format_map.get(suffix, "PNG")
                    try:
                        from PIL import Image

                        with Image.open(img_file) as img:
                            compressed_img, compressed_data = image_processor.compress(
                                img.copy(),
                                quality=self.config.image.quality,
                                max_size=(
                                    self.config.image.max_width,
                                    self.config.image.max_height,
                                ),
                                output_format=output_format,
                            )
                            img_file.write_bytes(compressed_data)
                            width, height = compressed_img.size
                    except Exception:
                        pass

                if width == 0 or height == 0:
                    try:
                        from PIL import Image

                        with Image.open(img_file) as img:
                            width, height = img.size
                    except Exception:
                        width, height = 0, 0

                # Determine MIME type
                mime_map = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "webp": "image/webp",
                }
                mime_type = mime_map.get(suffix, "image/png")

                images.append(
                    ExtractedImage(
                        path=img_file,
                        index=idx + 1,
                        original_name=img_file.name,
                        mime_type=mime_type,
                        width=width,
                        height=height,
                    )
                )

        metadata: dict[str, Any] = {
            "source": str(input_path),
            "format": "PDF",
            "images_extracted": len(images),
        }

        # Render page screenshots if enabled (independent of OCR)
        enable_screenshot = self.config and self.config.screenshot.enabled
        if enable_screenshot and output_dir:
            page_images: list[dict] = []
            screenshots_dir = output_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            import pymupdf

            # Create ImageProcessor for compression
            img_processor = ImageProcessor(self.config.image if self.config else None)

            doc = pymupdf.open(input_path)
            try:
                screenshot_dpi = DEFAULT_RENDER_DPI
                screenshot_format = image_format if image_format != "png" else "jpg"
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # Render page to image
                    mat = pymupdf.Matrix(screenshot_dpi / 72, screenshot_dpi / 72)
                    pix = page.get_pixmap(matrix=mat)

                    # Save page image with compression (ensures < 5MB for LLM)
                    image_name = (
                        f"{input_path.name}.page{page_num + 1:04d}.{screenshot_format}"
                    )
                    screenshot_path = screenshots_dir / image_name
                    img_processor.save_screenshot(
                        pix.samples, pix.width, pix.height, screenshot_path
                    )

                    page_images.append(
                        {
                            "page": page_num + 1,
                            "path": str(screenshot_path),
                            "name": image_name,
                        }
                    )

                    logger.debug(f"Screenshot page {page_num + 1}/{len(doc)}")
            finally:
                doc.close()

            metadata["page_images"] = page_images
            metadata["pages"] = len(page_images)
            metadata["extracted_text"] = markdown

        # Clean up temporary directory if used
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        return ConvertResult(
            markdown=markdown,
            images=images,
            metadata=metadata,
        )

    def _fix_image_paths(self, markdown: str, image_path: Path) -> str:
        """Fix image paths to be relative to output directory.

        pymupdf4llm generates paths like: ![](full/path/to/assets/image.jpg)
        We need: ![](assets/image.jpg)
        """
        # Escape special regex characters in the path
        escaped_path = re.escape(str(image_path))
        # Match image references with the full path and replace with assets/filename
        # Preserve alt text if present
        pattern = rf"!\[([^\]]*)\]\({escaped_path}/([^)]+)\)"
        replacement = r"![\1](assets/\2)"
        return re.sub(pattern, replacement, markdown)

    def _collect_embedded_images(
        self, assets_dir: Path, input_name: str
    ) -> list[ExtractedImage]:
        """Collect embedded images extracted by pymupdf4llm.

        pymupdf4llm extracts embedded images with names like: filename.pdf-0-0.png
        (page index - image index on that page)

        Args:
            assets_dir: Directory where images were extracted
            input_name: Original PDF filename

        Returns:
            List of ExtractedImage for embedded images
        """
        embedded_images: list[ExtractedImage] = []
        # Pattern: filename.pdf-{page}-{index}.{ext}
        pattern = re.compile(rf"^{re.escape(input_name)}-(\d+)-(\d+)\.(png|jpg|jpeg)$")

        for image_file in assets_dir.iterdir():
            match = pattern.match(image_file.name)
            if match:
                page_idx = int(match.group(1))
                img_idx = int(match.group(2))
                ext = match.group(3)

                # Get image dimensions
                try:
                    import pymupdf

                    pix = pymupdf.Pixmap(str(image_file))
                    width, height = pix.width, pix.height
                except Exception:
                    width, height = 0, 0

                embedded_images.append(
                    ExtractedImage(
                        path=image_file,
                        index=page_idx * 100 + img_idx,  # Unique index
                        original_name=image_file.name,
                        mime_type=f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}",
                        width=width,
                        height=height,
                    )
                )

        return embedded_images

    def _convert_with_ocr(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert PDF using OCR for scanned documents.

        Also renders each page as an image (if enable_screenshot) for reference.

        Args:
            input_path: Path to the PDF file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing OCR-extracted markdown with commented page images
        """
        try:
            import pymupdf
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is not installed. Install with: pip install pymupdf"
            ) from e

        from markit.ocr import OCRProcessor

        ocr_config = self.config.ocr if self.config else None
        ocr = OCRProcessor(ocr_config)

        logger.info(f"Converting PDF with OCR: {input_path.name}")

        # Setup screenshots directory for page images
        if output_dir:
            screenshots_dir = output_dir / "screenshots"
        else:
            screenshots_dir = Path(tempfile.mkdtemp())

        # Get image format from config
        image_format = "jpg"
        if self.config:
            fmt = self.config.image.format
            image_format = "jpg" if fmt == "jpeg" else fmt

        # Check if screenshot is enabled
        enable_screenshot = self.config and self.config.screenshot.enabled

        images: list[ExtractedImage] = []
        page_images: list[dict] = []
        markdown_parts = []
        dpi = DEFAULT_RENDER_DPI

        # Step 2: Render each page as image (only if screenshot enabled)
        # Use parallel processing for better performance
        doc = pymupdf.open(input_path)
        total_pages = len(doc)
        doc.close()

        # Determine optimal worker count based on file size and system resources
        # Each worker opens its own PDF copy, so memory usage scales with workers × file_size
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        cpu_count = os.cpu_count() or 4

        # Adaptive worker count:
        # - Small files (<10MB): use up to cpu_count/2 workers
        # - Medium files (10-50MB): use up to 4 workers
        # - Large files (>50MB): use up to 2 workers to limit memory
        if file_size_mb < 10:
            max_workers = min(cpu_count // 2 or 2, total_pages, 6)
        elif file_size_mb < 50:
            max_workers = min(4, total_pages)
        else:
            max_workers = min(2, total_pages)

        # Ensure at least 1 worker
        max_workers = max(1, max_workers)

        if enable_screenshot:
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            def process_page_with_screenshot(page_num: int) -> dict:
                """Process a single page: render + OCR (thread-safe)."""
                # Each thread opens its own document (PyMuPDF not thread-safe)
                thread_doc = pymupdf.open(input_path)
                img_processor = ImageProcessor(
                    self.config.image if self.config else None
                )
                try:
                    page = thread_doc[page_num]

                    # Render page to image
                    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)

                    # Save page image with compression
                    image_name = (
                        f"{input_path.name}.page{page_num + 1:04d}.{image_format}"
                    )
                    image_path = screenshots_dir / image_name
                    final_size = img_processor.save_screenshot(
                        pix.samples, pix.width, pix.height, image_path
                    )

                    # OCR
                    try:
                        result = ocr.recognize_pdf_page(input_path, page_num, dpi=dpi)
                        text_content = (
                            result.text.strip()
                            if result.text.strip()
                            else "*(No text detected)*"
                        )
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                        text_content = f"*(OCR failed: {e})*"

                    page_content = f"{text_content}\n\n<!-- ![Page {page_num + 1}](screenshots/{image_name}) -->"

                    return {
                        "page_num": page_num,
                        "image": ExtractedImage(
                            path=image_path,
                            index=page_num + 1,
                            original_name=image_name,
                            mime_type=f"image/{image_format}",
                            width=final_size[0],
                            height=final_size[1],
                        ),
                        "page_image": {
                            "page": page_num + 1,
                            "path": str(image_path),
                            "name": image_name,
                        },
                        "markdown": page_content,
                    }
                finally:
                    thread_doc.close()

            # Process pages in parallel
            results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_page_with_screenshot, i): i
                    for i in range(total_pages)
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        result = future.result()
                        results[page_num] = result
                        logger.debug(
                            f"OCR processed page {page_num + 1}/{total_pages}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to process page {page_num + 1}: {e}")
                        results[page_num] = {
                            "page_num": page_num,
                            "image": None,
                            "page_image": None,
                            "markdown": f"*(Page processing failed: {e})*",
                        }

            # Collect results in order
            for i in range(total_pages):
                r = results[i]
                if r["image"]:
                    images.append(r["image"])
                if r["page_image"]:
                    page_images.append(r["page_image"])
                markdown_parts.append(r["markdown"])
        else:

            def process_page_ocr_only(page_num: int) -> dict:
                """Process a single page: OCR only (thread-safe)."""
                try:
                    result = ocr.recognize_pdf_page(input_path, page_num, dpi=dpi)
                    text_content = (
                        result.text.strip()
                        if result.text.strip()
                        else "*(No text detected)*"
                    )
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    text_content = f"*(OCR failed: {e})*"
                return {"page_num": page_num, "markdown": text_content}

            # Process pages in parallel
            results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_page_ocr_only, i): i
                    for i in range(total_pages)
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        result = future.result()
                        results[page_num] = result
                        logger.debug(
                            f"OCR processed page {page_num + 1}/{total_pages}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to process page {page_num + 1}: {e}")
                        results[page_num] = {
                            "page_num": page_num,
                            "markdown": f"*(OCR failed: {e})*",
                        }

            # Collect results in order
            for i in range(total_pages):
                markdown_parts.append(results[i]["markdown"])

        extracted_text = f"# {input_path.stem}\n\n" + "\n\n".join(markdown_parts)

        return ConvertResult(
            markdown=extracted_text,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PDF",
                "ocr_used": True,
                "pages": len(markdown_parts),
                "extracted_text": extracted_text,
                "page_images": page_images,
            },
        )

    def _render_pages_for_llm(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Extract text and render pages for LLM Vision analysis.

        This method:
        1. Extracts text using pymupdf4llm (fast, preserves links/tables)
        2. Renders each page as an image (if screenshot enabled)

        Returns:
            ConvertResult with extracted text and page images
        """
        try:
            import pymupdf
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is not installed. Install with: pip install pymupdf"
            ) from e

        logger.info(f"Extracting text and rendering pages for LLM: {input_path.name}")

        # Determine output paths
        if output_dir:
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            screenshots_dir = output_dir / "screenshots"
        else:
            assets_dir = Path(tempfile.mkdtemp())
            screenshots_dir = Path(tempfile.mkdtemp())

        # Get image format from config
        image_format = "jpg"
        if self.config:
            fmt = self.config.image.format
            image_format = "jpg" if fmt == "jpeg" else fmt

        # Step 1: Extract text using pymupdf4llm (fast, preserves structure)
        logger.debug("Extracting text with pymupdf4llm...")
        extracted_text = cast(
            str,
            pymupdf4llm.to_markdown(
                str(input_path),
                write_images=True,
                image_path=str(assets_dir),
                image_format=image_format,
                dpi=DEFAULT_RENDER_DPI,
                force_text=True,
            ),
        )
        extracted_text = self._fix_image_paths(extracted_text, assets_dir)

        # Collect embedded images extracted by pymupdf4llm
        embedded_images = self._collect_embedded_images(assets_dir, input_path.name)

        # Check if screenshot is enabled
        enable_screenshot = self.config and self.config.screenshot.enabled

        images: list[ExtractedImage] = list(embedded_images)
        page_images: list[dict] = []

        if enable_screenshot:
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            # Create ImageProcessor for compression
            img_processor = ImageProcessor(self.config.image if self.config else None)

            doc = pymupdf.open(input_path)
            try:
                dpi = DEFAULT_RENDER_DPI
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # Render page to image
                    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)

                    # Save page image with compression (ensures < 5MB for LLM)
                    image_name = (
                        f"{input_path.name}.page{page_num + 1:04d}.{image_format}"
                    )
                    image_path = screenshots_dir / image_name
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

                    page_images.append(
                        {
                            "page": page_num + 1,
                            "path": str(image_path),
                            "name": image_name,
                        }
                    )

                    logger.debug(f"Rendered page {page_num + 1}/{len(doc)}")
            finally:
                doc.close()

        return ConvertResult(
            markdown=extracted_text,
            images=images,
            metadata={
                "source": str(input_path),
                "format": "PDF",
                "pages": len(page_images) if page_images else 0,
                "extracted_text": extracted_text,
                "page_images": page_images,
            },
        )
