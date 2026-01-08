"""pdfplumber-based PDF converter."""

from pathlib import Path

import anyio

from markit.converters.base import BaseConverter, ConversionResult, ExtractedImage
from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

log = get_logger(__name__)


class PDFPlumberConverter(BaseConverter):
    """PDF converter using pdfplumber.

    pdfplumber excels at:
    - Table extraction
    - Detailed text positioning
    - Complex layout analysis
    """

    name = "pdfplumber"
    supported_extensions = {".pdf"}

    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        table_settings: dict | None = None,
    ) -> None:
        """Initialize the pdfplumber converter.

        Args:
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            table_settings: Custom table extraction settings
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.table_settings = table_settings or {}

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

        log.info("Converting PDF with pdfplumber", file=str(file_path))

        try:
            result = await anyio.to_thread.run_sync(
                self._convert_sync,
                file_path,
            )
            return result
        except Exception as e:
            log.error("pdfplumber conversion failed", file=str(file_path), error=str(e))
            raise ConversionError(file_path, str(e), cause=e) from e

    def _convert_sync(self, file_path: Path) -> ConversionResult:
        """Synchronous conversion implementation."""
        import pdfplumber

        markdown_parts = []
        images: list[ExtractedImage] = []
        image_counter = 0

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_content = []
                page_content.append(f"<!-- Page {page_num} -->\n")

                # Extract tables first
                if self.extract_tables:
                    tables = page.extract_tables(self.table_settings)
                    for _table_idx, table in enumerate(tables):
                        if table:
                            markdown_table = self._table_to_markdown(table)
                            page_content.append(f"\n{markdown_table}\n")

                # Extract text (excluding table areas if tables were extracted)
                text = page.extract_text() or ""
                if text.strip():
                    formatted_text = self._format_text(text)
                    page_content.append(formatted_text)

                # Extract images
                if self.extract_images:
                    page_images = self._extract_images(page, page_num, file_path, image_counter)
                    for img in page_images:
                        images.append(img)
                        page_content.append(
                            f"\n![Image {image_counter + 1}](assets/{img.filename})\n"
                        )
                        image_counter += 1

                markdown_parts.append("\n".join(page_content))

        markdown = "\n\n".join(markdown_parts)
        markdown = self._cleanup_markdown(markdown)

        return ConversionResult(
            markdown=markdown,
            images=images,
            metadata={
                "page_count": len(pdf.pages) if "pdf" in dir() else 0,
                "converter": self.name,
            },
        )

    def _table_to_markdown(self, table: list[list[str]]) -> str:
        """Convert a table to Markdown format."""
        if not table or not table[0]:
            return ""

        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = [(cell or "").replace("\n", " ").strip() for cell in row]
            cleaned_table.append(cleaned_row)

        # Build markdown table
        lines = []

        # Header row
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # Data rows
        for row in cleaned_table[1:]:
            # Pad row if necessary
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[: len(header)]) + " |")

        return "\n".join(lines)

    def _format_text(self, text: str) -> str:
        """Format extracted text."""
        lines = text.split("\n")
        formatted = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            formatted.append(line)

        return "\n".join(formatted)

    def _extract_images(
        self,
        page,
        page_num: int,
        source_file: Path,
        start_index: int,
    ) -> list[ExtractedImage]:
        """Extract images from a page."""
        images = []

        try:
            # pdfplumber provides image metadata
            for img_idx, img in enumerate(page.images):
                try:
                    # Get image from page
                    # Note: pdfplumber doesn't directly extract image bytes
                    # We need to crop the image region
                    x0, top, x1, bottom = (
                        img["x0"],
                        img["top"],
                        img["x1"],
                        img["bottom"],
                    )

                    # Crop and convert to image
                    cropped = page.crop((x0, top, x1, bottom))
                    pil_image = cropped.to_image(resolution=150)

                    # Convert to bytes
                    from io import BytesIO

                    buffer = BytesIO()
                    pil_image.original.save(buffer, format="PNG")
                    image_data = buffer.getvalue()

                    filename = f"image_{start_index + img_idx + 1:03d}.png"

                    extracted = ExtractedImage(
                        data=image_data,
                        format="png",
                        filename=filename,
                        source_document=source_file,
                        position=page_num,
                        width=int(x1 - x0),
                        height=int(bottom - top),
                    )
                    images.append(extracted)

                except Exception as e:
                    log.warning(
                        "Failed to extract image from page",
                        page=page_num,
                        index=img_idx,
                        error=str(e),
                    )

        except Exception as e:
            log.warning(
                "Failed to process images on page",
                page=page_num,
                error=str(e),
            )

        return images

    def _cleanup_markdown(self, markdown: str) -> str:
        """Clean up generated markdown."""
        import re

        # Remove excessive blank lines
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        return markdown.strip()


class TableExtractor:
    """Specialized table extractor using pdfplumber."""

    def __init__(self, settings: dict | None = None):
        """Initialize table extractor.

        Args:
            settings: pdfplumber table extraction settings
        """
        self.settings = settings or {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
        }

    def extract_tables(self, file_path: Path) -> list[list[list[str]]]:
        """Extract all tables from PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            List of tables, each table is a list of rows
        """
        import pdfplumber

        all_tables = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(self.settings)
                all_tables.extend(tables)

        return all_tables
