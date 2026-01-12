"""PyMuPDF-based PDF converter."""

from pathlib import Path

import anyio

from markit.converters.base import BaseConverter, ConversionResult, ExtractedImage
from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

log = get_logger(__name__)


def _normalize_image_spacing(markdown: str) -> str:
    """Normalize spacing around images.

    Ensures:
    - One blank line before first image (if preceded by text)
    - One blank line after last image (if followed by text)
    - At most one blank line between consecutive images
    """
    lines = markdown.split("\n")
    result = []
    last_content_type = "blank"  # "blank", "image", "text"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        is_image = stripped.startswith("![") and "](" in stripped
        is_blank = stripped == ""

        if is_image:
            if last_content_type == "image" or last_content_type == "blank":
                result.append(line)
            else:
                result.append("")
                result.append(line)
            last_content_type = "image"
        elif is_blank:
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j < len(lines):
                next_stripped = lines[j].strip()
                next_is_image = next_stripped.startswith("![") and "](" in next_stripped

                if last_content_type == "image" and next_is_image:
                    result.append(line)
                    i = j - 1
                    last_content_type = "blank"
                elif last_content_type == "image":
                    result.append(line)
                    last_content_type = "blank"
                else:
                    result.append(line)
                    last_content_type = "blank"
            else:
                if last_content_type != "image":
                    result.append(line)
                last_content_type = "blank"
        else:
            if last_content_type == "image":
                result.append("")
            result.append(line)
            last_content_type = "text"

        i += 1

    return "\n".join(result)


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename by replacing special characters."""
    replacements = {
        " ": "_",
        ":": "_",
        "：": "_",
        "/": "_",
        "\\": "_",
        "?": "_",
        "*": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }
    result = filename
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    while "__" in result:
        result = result.replace("__", "_")
    return result


class PyMuPDFConverter(BaseConverter):
    """PDF converter using PyMuPDF (fitz).

    PyMuPDF is fast and reliable for PDF processing.
    It excels at:
    - Fast text extraction
    - Image extraction
    - Handling complex PDFs
    """

    name = "pymupdf"
    supported_extensions = {".pdf"}

    def __init__(
        self,
        extract_images: bool = True,
        image_dpi: int = 150,
        ocr_enabled: bool = False,
        filter_small_images: bool = True,
        min_image_dimension: int = 50,
        min_image_area: int = 2500,
        min_image_size: int = 3072,
    ) -> None:
        """Initialize the PyMuPDF converter.

        Args:
            extract_images: Whether to extract images from PDF
            image_dpi: DPI for rasterizing images
            ocr_enabled: Enable OCR for scanned PDFs (requires Tesseract)
            filter_small_images: Whether to filter out small decorative images
            min_image_dimension: Minimum width or height in pixels
            min_image_area: Minimum area in pixels²
            min_image_size: Minimum file size in bytes
        """
        self.extract_images = extract_images
        self.image_dpi = image_dpi
        self.ocr_enabled = ocr_enabled
        self.filter_small_images = filter_small_images
        self.min_image_dimension = min_image_dimension
        self.min_image_area = min_image_area
        self.min_image_size = min_image_size

    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a PDF file to Markdown.

        Args:
            file_path: Path to the PDF file

        Returns:
            ConversionResult with markdown content and extracted images
        """
        if not await self.validate(file_path):
            raise ConversionError(
                file_path,
                f"Invalid PDF file: {file_path}",
            )

        log.info("Converting PDF with PyMuPDF", file=str(file_path))

        try:
            # Run blocking PDF processing in a thread
            result = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
                self._convert_sync,
                file_path,
            )
            return result
        except Exception as e:
            log.error("PyMuPDF conversion failed", file=str(file_path), error=str(e))
            raise ConversionError(file_path, str(e), cause=e) from e

    def _convert_sync(self, file_path: Path) -> ConversionResult:
        """Synchronous conversion implementation."""
        import fitz  # PyMuPDF

        doc: fitz.Document = fitz.open(file_path)  # type: ignore[assignment]
        markdown_parts = []
        images: list[ExtractedImage] = []
        image_counter = 0
        seen_xrefs: set[int] = set()  # Track extracted image xrefs to avoid duplicates

        try:
            for page_num, page in enumerate(doc, start=1):  # type: ignore[arg-type]
                # Extract text
                text = page.get_text("text")
                if text.strip():
                    markdown_parts.append(f"<!-- Page {page_num} -->\n")
                    markdown_parts.append(self._format_page_text(text))
                    markdown_parts.append("\n")

                # Extract images
                if self.extract_images:
                    page_images = self._extract_page_images(
                        doc, page, page_num, file_path, image_counter, seen_xrefs
                    )
                    for img, _ref in page_images:
                        images.append(img)
                        markdown_parts.append(f"![](assets/{img.filename})")
                        image_counter += 1

            # Combine markdown
            markdown = "\n".join(markdown_parts)

            # Clean up markdown
            markdown = self._cleanup_markdown(markdown)

            # Normalize image spacing
            markdown = _normalize_image_spacing(markdown)

            return ConversionResult(
                markdown=markdown,
                images=images,
                metadata={
                    "page_count": len(doc),
                    "converter": self.name,
                },
            )
        finally:
            doc.close()

    def _format_page_text(self, text: str) -> str:
        """Format extracted text as markdown."""
        lines = text.split("\n")
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect potential headings (ALL CAPS or short lines)
            if line.isupper() and len(line) < 100:
                formatted_lines.append(f"\n## {line.title()}\n")
            else:
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _extract_page_images(
        self,
        doc,
        page,
        page_num: int,
        source_file: Path,
        start_index: int,
        seen_xrefs: set[int] | None = None,
    ) -> list[tuple[ExtractedImage, dict]]:
        """Extract images that are actually rendered on a page.

        Uses get_image_info() to detect images that are visually present on the page,
        not just referenced in the page resources. This prevents extracting the same
        image multiple times when it's referenced but only displayed on specific pages.

        Args:
            doc: PyMuPDF document
            page: Page object
            page_num: Page number (1-based)
            source_file: Source PDF file path
            start_index: Starting index for image numbering (number of images already extracted)
            seen_xrefs: Set of already extracted image xrefs to avoid duplicates

        Returns:
            List of (ExtractedImage, image_info) tuples
        """
        images = []
        file_stem = source_file.stem
        skipped_count = 0
        duplicate_count = 0

        # Use get_image_info() to get images that are actually rendered on this page
        # This returns only images with visual presence, not just resource references
        try:
            image_info_list = page.get_image_info()
        except Exception as e:
            log.warning(
                "Failed to get image info",
                page=page_num,
                error=str(e),
            )
            return images

        for img_info in image_info_list:
            try:
                xref = img_info.get("xref", 0)
                if xref == 0:
                    continue

                # Skip duplicate images (same xref already extracted)
                if seen_xrefs is not None:
                    if xref in seen_xrefs:
                        duplicate_count += 1
                        continue
                    seen_xrefs.add(xref)

                base_image = doc.extract_image(xref)

                if base_image:
                    image_data = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Filter small images if enabled
                    if self.filter_small_images:
                        if not self._should_keep_image(image_data, width, height):
                            skipped_count += 1
                            continue

                    # Generate filename using actual kept image count
                    img_num = start_index + len(images) + 1
                    safe_stem = _sanitize_filename(file_stem)
                    filename = f"{safe_stem}_{img_num:03d}.{image_ext}"

                    extracted = ExtractedImage(
                        data=image_data,
                        format=image_ext,
                        filename=filename,
                        source_document=source_file,
                        position=page_num,
                        width=width,
                        height=height,
                    )

                    images.append((extracted, img_info))

            except Exception as e:
                log.warning(
                    "Failed to extract image",
                    page=page_num,
                    error=str(e),
                )

        if skipped_count > 0:
            log.debug(
                "Filtered small images",
                page=page_num,
                skipped=skipped_count,
            )

        if duplicate_count > 0:
            log.debug(
                "Skipped duplicate images",
                page=page_num,
                duplicates=duplicate_count,
            )

        return images

    def _should_keep_image(self, data: bytes, width: int, height: int) -> bool:
        """Check if an image should be kept based on filtering criteria.

        Args:
            data: Image data bytes
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            True if image should be kept, False if it should be filtered out
        """
        # Check file size
        if len(data) < self.min_image_size:
            return False

        # Check dimensions
        if width > 0 and height > 0:
            # Check minimum dimension
            if width < self.min_image_dimension or height < self.min_image_dimension:
                return False

            # Check minimum area
            if width * height < self.min_image_area:
                return False

        return True

    def _cleanup_markdown(self, markdown: str) -> str:
        """Clean up the generated markdown."""
        import re

        # Remove excessive blank lines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        # Remove page comment markers if desired
        # markdown = re.sub(r'<!-- Page \d+ -->\n', '', markdown)

        return markdown.strip()


class PyMuPDFTextExtractor:
    """Extract structured text from PDF using PyMuPDF."""

    def __init__(self, preserve_layout: bool = False):
        """Initialize the text extractor.

        Args:
            preserve_layout: Try to preserve original layout
        """
        self.preserve_layout = preserve_layout

    def extract(self, file_path: Path) -> str:
        """Extract text from PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text
        """
        import fitz

        doc = fitz.open(file_path)
        texts = []

        try:
            for page in doc:
                if self.preserve_layout:
                    # Use dict mode for layout preservation
                    text_dict = page.get_text("dict")
                    texts.append(self._format_text_dict(text_dict))  # type: ignore[arg-type]
                else:
                    texts.append(page.get_text("text"))

            return "\n\n".join(texts)
        finally:
            doc.close()

    def _format_text_dict(self, text_dict: dict) -> str:
        """Format text dictionary to preserve layout."""
        blocks = text_dict.get("blocks", [])
        lines = []

        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    text = " ".join(span.get("text", "") for span in line.get("spans", []))
                    lines.append(text)

        return "\n".join(lines)
