"""Image extraction from documents."""

import zipfile
from pathlib import Path

import anyio

from markit.converters.base import ExtractedImage
from markit.utils.logging import get_logger

log = get_logger(__name__)


class ImageExtractor:
    """Extract images from various document formats."""

    # Image extensions to look for
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".emf",
        ".wmf",
    }

    async def extract_from_docx(self, docx_path: Path) -> list[ExtractedImage]:
        """Extract images from a Word document (.docx).

        DOCX files are ZIP archives containing media files in word/media/.

        Args:
            docx_path: Path to the .docx file

        Returns:
            List of extracted images
        """
        return await anyio.to_thread.run_sync(self._extract_from_docx_sync, docx_path)

    def _extract_from_docx_sync(self, docx_path: Path) -> list[ExtractedImage]:
        """Synchronous extraction from DOCX."""
        images = []

        try:
            with zipfile.ZipFile(docx_path, "r") as zf:
                for i, name in enumerate(zf.namelist()):
                    if name.startswith("word/media/"):
                        ext = Path(name).suffix.lower()
                        if ext in self.IMAGE_EXTENSIONS:
                            data = zf.read(name)
                            filename = Path(name).name
                            fmt = ext.lstrip(".")
                            if fmt == "jpg":
                                fmt = "jpeg"

                            images.append(
                                ExtractedImage(
                                    data=data,
                                    format=fmt,
                                    filename=filename,
                                    source_document=docx_path,
                                    position=i,
                                    original_path=name,
                                )
                            )
                            log.debug("Extracted image", filename=filename, format=fmt)
        except zipfile.BadZipFile as e:
            log.error("Invalid DOCX file", file=str(docx_path), error=str(e))

        log.info("Extracted images from DOCX", count=len(images), file=str(docx_path))
        return images

    async def extract_from_pptx(self, pptx_path: Path) -> list[ExtractedImage]:
        """Extract images from a PowerPoint presentation (.pptx).

        PPTX files are ZIP archives containing media files in ppt/media/.

        Args:
            pptx_path: Path to the .pptx file

        Returns:
            List of extracted images
        """
        return await anyio.to_thread.run_sync(self._extract_from_pptx_sync, pptx_path)

    def _extract_from_pptx_sync(self, pptx_path: Path) -> list[ExtractedImage]:
        """Synchronous extraction from PPTX."""
        images = []

        try:
            with zipfile.ZipFile(pptx_path, "r") as zf:
                for i, name in enumerate(zf.namelist()):
                    if name.startswith("ppt/media/"):
                        ext = Path(name).suffix.lower()
                        if ext in self.IMAGE_EXTENSIONS:
                            data = zf.read(name)
                            filename = Path(name).name
                            fmt = ext.lstrip(".")
                            if fmt == "jpg":
                                fmt = "jpeg"

                            images.append(
                                ExtractedImage(
                                    data=data,
                                    format=fmt,
                                    filename=filename,
                                    source_document=pptx_path,
                                    position=i,
                                    original_path=name,
                                )
                            )
                            log.debug("Extracted image", filename=filename, format=fmt)
        except zipfile.BadZipFile as e:
            log.error("Invalid PPTX file", file=str(pptx_path), error=str(e))

        log.info("Extracted images from PPTX", count=len(images), file=str(pptx_path))
        return images

    async def extract_from_xlsx(self, xlsx_path: Path) -> list[ExtractedImage]:
        """Extract images from an Excel spreadsheet (.xlsx).

        XLSX files are ZIP archives containing media files in xl/media/.

        Args:
            xlsx_path: Path to the .xlsx file

        Returns:
            List of extracted images
        """
        return await anyio.to_thread.run_sync(self._extract_from_xlsx_sync, xlsx_path)

    def _extract_from_xlsx_sync(self, xlsx_path: Path) -> list[ExtractedImage]:
        """Synchronous extraction from XLSX."""
        images = []

        try:
            with zipfile.ZipFile(xlsx_path, "r") as zf:
                for i, name in enumerate(zf.namelist()):
                    if name.startswith("xl/media/"):
                        ext = Path(name).suffix.lower()
                        if ext in self.IMAGE_EXTENSIONS:
                            data = zf.read(name)
                            filename = Path(name).name
                            fmt = ext.lstrip(".")
                            if fmt == "jpg":
                                fmt = "jpeg"

                            images.append(
                                ExtractedImage(
                                    data=data,
                                    format=fmt,
                                    filename=filename,
                                    source_document=xlsx_path,
                                    position=i,
                                    original_path=name,
                                )
                            )
                            log.debug("Extracted image", filename=filename, format=fmt)
        except zipfile.BadZipFile as e:
            log.error("Invalid XLSX file", file=str(xlsx_path), error=str(e))

        log.info("Extracted images from XLSX", count=len(images), file=str(xlsx_path))
        return images

    async def extract_from_pdf(self, pdf_path: Path) -> list[ExtractedImage]:
        """Extract images from a PDF document.

        Uses PyMuPDF for image extraction.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of extracted images
        """
        return await anyio.to_thread.run_sync(self._extract_from_pdf_sync, pdf_path)

    def _extract_from_pdf_sync(self, pdf_path: Path) -> list[ExtractedImage]:
        """Synchronous extraction from PDF using PyMuPDF."""
        images = []

        try:
            import pymupdf

            doc = pymupdf.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        image_ext = base_image["ext"]

                        filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"

                        images.append(
                            ExtractedImage(
                                data=image_data,
                                format=image_ext,
                                filename=filename,
                                source_document=pdf_path,
                                position=page_num * 1000 + img_index,
                                width=base_image.get("width"),
                                height=base_image.get("height"),
                            )
                        )
                        log.debug("Extracted image", filename=filename, page=page_num + 1)
                    except Exception as e:
                        log.warning("Failed to extract image", xref=xref, error=str(e))

            doc.close()
        except Exception as e:
            log.error("Failed to extract images from PDF", file=str(pdf_path), error=str(e))

        log.info("Extracted images from PDF", count=len(images), file=str(pdf_path))
        return images

    async def extract(self, file_path: Path) -> list[ExtractedImage]:
        """Extract images from any supported document format.

        Args:
            file_path: Path to the document

        Returns:
            List of extracted images
        """
        ext = file_path.suffix.lower()

        if ext == ".docx":
            return await self.extract_from_docx(file_path)
        elif ext == ".pptx":
            return await self.extract_from_pptx(file_path)
        elif ext == ".xlsx":
            return await self.extract_from_xlsx(file_path)
        elif ext == ".pdf":
            return await self.extract_from_pdf(file_path)
        else:
            log.warning("Unsupported format for image extraction", extension=ext)
            return []
