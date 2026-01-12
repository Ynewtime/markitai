"""PyMuPDF4LLM-based PDF converter.

Uses pymupdf4llm library for high-quality PDF to Markdown conversion
with support for tables, headings, and images.
"""

import re
import tempfile
from pathlib import Path

import anyio

from markit.converters.base import BaseConverter, ConversionResult, ExtractedImage
from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

log = get_logger(__name__)


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


class PyMuPDF4LLMConverter(BaseConverter):
    """PDF converter using PyMuPDF4LLM.

    PyMuPDF4LLM is specifically designed for LLM/RAG applications and provides:
    - Automatic table detection and Markdown table conversion
    - Heading detection based on font size
    - Image extraction with configurable options
    - Bold, italic, and code block formatting
    - Optional OCR support for scanned documents
    """

    name = "pymupdf4llm"
    supported_extensions = {".pdf"}

    def __init__(
        self,
        extract_images: bool = True,
        image_dpi: int = 150,
        write_images: bool = True,
        embed_images: bool = False,
        table_strategy: str = "lines_strict",
        force_text: bool = False,
    ) -> None:
        """Initialize the PyMuPDF4LLM converter.

        Args:
            extract_images: Whether to extract images from PDF
            image_dpi: DPI for rendering images
            write_images: Write images to files (recommended)
            embed_images: Embed images as base64 in markdown (alternative to write_images)
            table_strategy: Table detection strategy ("lines", "lines_strict", "text")
            force_text: Include text from image areas in output
        """
        self.extract_images = extract_images
        self.image_dpi = image_dpi
        self.write_images = write_images
        self.embed_images = embed_images
        self.table_strategy = table_strategy
        self.force_text = force_text

    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a PDF file to Markdown using PyMuPDF4LLM.

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

        log.info("Converting PDF with PyMuPDF4LLM", file=str(file_path))

        try:
            # Run blocking PDF processing in a thread
            result = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
                self._convert_sync,
                file_path,
            )
            return result
        except Exception as e:
            log.error("PyMuPDF4LLM conversion failed", file=str(file_path), error=str(e))
            raise ConversionError(file_path, str(e), cause=e) from e

    def _convert_sync(self, file_path: Path) -> ConversionResult:
        """Synchronous conversion implementation."""
        import pymupdf4llm

        file_stem = file_path.stem
        safe_stem = _sanitize_filename(file_stem)
        images: list[ExtractedImage] = []

        # Create a temporary directory for image extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Configure pymupdf4llm options
            kwargs = {
                "dpi": self.image_dpi,
                "force_text": self.force_text,
            }

            if self.extract_images:
                if self.write_images:
                    kwargs["write_images"] = True
                    kwargs["image_path"] = str(temp_path)
                elif self.embed_images:
                    kwargs["embed_images"] = True

            # Convert PDF to Markdown
            log.debug(
                "Calling pymupdf4llm.to_markdown",
                file=str(file_path),
                options=kwargs,
            )

            markdown_result = pymupdf4llm.to_markdown(str(file_path), **kwargs)
            # pymupdf4llm.to_markdown returns str with default options
            assert isinstance(markdown_result, str)
            markdown: str = markdown_result

            # Collect extracted images
            if self.extract_images and self.write_images:
                images = self._collect_images(temp_path, file_path, safe_stem)

                # Update markdown to use assets/ path
                markdown = self._update_image_paths(markdown, images)

        # Get page count for metadata
        import fitz

        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()

        return ConversionResult(
            markdown=markdown,
            images=images,
            metadata={
                "page_count": page_count,
                "converter": self.name,
            },
        )

    def _collect_images(
        self,
        temp_path: Path,
        source_file: Path,
        safe_stem: str,
    ) -> list[ExtractedImage]:
        """Collect extracted images from temporary directory.

        Args:
            temp_path: Path to temporary directory with images
            source_file: Source PDF file
            safe_stem: Sanitized filename stem

        Returns:
            List of ExtractedImage objects
        """
        images = []
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}

        # Find all image files in temp directory
        for img_file in sorted(temp_path.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue

            try:
                # Read image data
                with open(img_file, "rb") as f:
                    image_data = f.read()

                # Determine format
                img_format = img_file.suffix.lower().lstrip(".")
                if img_format == "jpg":
                    img_format = "jpeg"

                # Generate new filename with our naming convention
                img_num = len(images) + 1
                new_filename = f"{safe_stem}_{img_num:03d}.{img_format}"

                # Try to get image dimensions
                try:
                    import io

                    from PIL import Image

                    with Image.open(io.BytesIO(image_data)) as pil_img:
                        width, height = pil_img.size
                except Exception:
                    width, height = None, None

                extracted = ExtractedImage(
                    data=image_data,
                    format=img_format,
                    filename=new_filename,
                    source_document=source_file,
                    position=img_num,
                    width=width,
                    height=height,
                    original_path=img_file.name,
                )

                images.append(extracted)
                log.debug(
                    "Collected image",
                    original=img_file.name,
                    new_name=new_filename,
                )

            except Exception as e:
                log.warning(
                    "Failed to collect image",
                    file=img_file.name,
                    error=str(e),
                )

        return images

    def _update_image_paths(
        self,
        markdown: str,
        images: list[ExtractedImage],
    ) -> str:
        """Update image paths in markdown to use assets/ directory.

        pymupdf4llm generates paths like ![](filename.pdf-0-0.png)
        We need to update these to ![](assets/safe_stem_001.png)

        Args:
            markdown: Original markdown content
            images: List of extracted images

        Returns:
            Updated markdown with corrected image paths
        """
        # Create mapping from original filename to new filename
        path_mapping = {}
        for img in images:
            if img.original_path:
                path_mapping[img.original_path] = f"assets/{img.filename}"

        # Replace image paths in markdown
        # Pattern: ![...](path) or ![](path)
        def replace_path(match: re.Match) -> str:
            alt_text = match.group(1)
            original_path = match.group(2)

            # Check if this path needs replacement
            for orig, new in path_mapping.items():
                if orig in original_path or original_path.endswith(orig):
                    return f"![{alt_text}]({new})"

            return match.group(0)

        # Match markdown image syntax
        pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        updated_markdown = re.sub(pattern, replace_path, markdown)

        return updated_markdown
