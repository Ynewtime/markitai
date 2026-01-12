"""Tests for image extractor module."""

import zipfile
from unittest.mock import MagicMock, patch

import pytest

from markit.image.extractor import ImageExtractor


class TestImageExtractorConstants:
    """Tests for ImageExtractor constants."""

    def test_image_extensions_defined(self):
        """Test that IMAGE_EXTENSIONS is defined."""
        extractor = ImageExtractor()
        assert extractor.IMAGE_EXTENSIONS is not None
        assert len(extractor.IMAGE_EXTENSIONS) > 0

    def test_image_extensions_includes_common_formats(self):
        """Test that common image formats are included."""
        extractor = ImageExtractor()
        assert ".png" in extractor.IMAGE_EXTENSIONS
        assert ".jpg" in extractor.IMAGE_EXTENSIONS
        assert ".jpeg" in extractor.IMAGE_EXTENSIONS
        assert ".gif" in extractor.IMAGE_EXTENSIONS
        assert ".bmp" in extractor.IMAGE_EXTENSIONS

    def test_image_extensions_includes_office_formats(self):
        """Test that Office-specific formats are included."""
        extractor = ImageExtractor()
        assert ".emf" in extractor.IMAGE_EXTENSIONS
        assert ".wmf" in extractor.IMAGE_EXTENSIONS


class TestExtractFromDocx:
    """Tests for DOCX image extraction."""

    def test_extract_from_valid_docx(self, tmp_path):
        """Test extracting images from a valid DOCX file."""
        # Create a mock DOCX file (ZIP archive)
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            # Add a fake image
            zf.writestr("word/media/image1.png", b"fake png data")
            zf.writestr("word/media/image2.jpg", b"fake jpg data")
            zf.writestr("word/document.xml", "<document/>")

        extractor = ImageExtractor()
        images = extractor._extract_from_docx_sync(docx_path)

        assert len(images) == 2
        assert images[0].format == "png"
        assert images[0].filename == "image1.png"
        assert images[1].format == "jpeg"  # jpg -> jpeg
        assert images[1].filename == "image2.jpg"

    def test_extract_from_docx_no_images(self, tmp_path):
        """Test extracting from DOCX with no images."""
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/document.xml", "<document/>")

        extractor = ImageExtractor()
        images = extractor._extract_from_docx_sync(docx_path)

        assert len(images) == 0

    def test_extract_from_invalid_docx(self, tmp_path):
        """Test extracting from invalid DOCX file."""
        docx_path = tmp_path / "test.docx"
        docx_path.write_text("not a zip file")

        extractor = ImageExtractor()
        images = extractor._extract_from_docx_sync(docx_path)

        assert len(images) == 0

    def test_extract_from_docx_filters_non_images(self, tmp_path):
        """Test that non-image files are filtered out."""
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/image1.png", b"png data")
            zf.writestr("word/media/video.mp4", b"video data")
            zf.writestr("word/media/document.xml", b"xml data")

        extractor = ImageExtractor()
        images = extractor._extract_from_docx_sync(docx_path)

        assert len(images) == 1
        assert images[0].filename == "image1.png"

    @pytest.mark.asyncio
    async def test_extract_from_docx_async(self, tmp_path):
        """Test async DOCX extraction."""
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/image1.png", b"fake png data")

        extractor = ImageExtractor()
        images = await extractor.extract_from_docx(docx_path)

        assert len(images) == 1


class TestExtractFromPptx:
    """Tests for PPTX image extraction."""

    def test_extract_from_valid_pptx(self, tmp_path):
        """Test extracting images from a valid PPTX file."""
        pptx_path = tmp_path / "test.pptx"
        with zipfile.ZipFile(pptx_path, "w") as zf:
            zf.writestr("ppt/media/image1.png", b"fake png data")
            zf.writestr("ppt/media/image2.gif", b"fake gif data")
            zf.writestr("ppt/slides/slide1.xml", "<slide/>")

        extractor = ImageExtractor()
        images = extractor._extract_from_pptx_sync(pptx_path)

        assert len(images) == 2
        assert images[0].format == "png"
        assert images[1].format == "gif"

    def test_extract_from_pptx_no_images(self, tmp_path):
        """Test extracting from PPTX with no images."""
        pptx_path = tmp_path / "test.pptx"
        with zipfile.ZipFile(pptx_path, "w") as zf:
            zf.writestr("ppt/slides/slide1.xml", "<slide/>")

        extractor = ImageExtractor()
        images = extractor._extract_from_pptx_sync(pptx_path)

        assert len(images) == 0

    def test_extract_from_invalid_pptx(self, tmp_path):
        """Test extracting from invalid PPTX file."""
        pptx_path = tmp_path / "test.pptx"
        pptx_path.write_text("not a zip file")

        extractor = ImageExtractor()
        images = extractor._extract_from_pptx_sync(pptx_path)

        assert len(images) == 0

    @pytest.mark.asyncio
    async def test_extract_from_pptx_async(self, tmp_path):
        """Test async PPTX extraction."""
        pptx_path = tmp_path / "test.pptx"
        with zipfile.ZipFile(pptx_path, "w") as zf:
            zf.writestr("ppt/media/image1.png", b"fake png data")

        extractor = ImageExtractor()
        images = await extractor.extract_from_pptx(pptx_path)

        assert len(images) == 1


class TestExtractFromXlsx:
    """Tests for XLSX image extraction."""

    def test_extract_from_valid_xlsx(self, tmp_path):
        """Test extracting images from a valid XLSX file."""
        xlsx_path = tmp_path / "test.xlsx"
        with zipfile.ZipFile(xlsx_path, "w") as zf:
            zf.writestr("xl/media/image1.png", b"fake png data")
            zf.writestr("xl/media/image2.bmp", b"fake bmp data")
            zf.writestr("xl/workbook.xml", "<workbook/>")

        extractor = ImageExtractor()
        images = extractor._extract_from_xlsx_sync(xlsx_path)

        assert len(images) == 2
        assert images[0].format == "png"
        assert images[1].format == "bmp"

    def test_extract_from_xlsx_no_images(self, tmp_path):
        """Test extracting from XLSX with no images."""
        xlsx_path = tmp_path / "test.xlsx"
        with zipfile.ZipFile(xlsx_path, "w") as zf:
            zf.writestr("xl/workbook.xml", "<workbook/>")

        extractor = ImageExtractor()
        images = extractor._extract_from_xlsx_sync(xlsx_path)

        assert len(images) == 0

    def test_extract_from_invalid_xlsx(self, tmp_path):
        """Test extracting from invalid XLSX file."""
        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.write_text("not a zip file")

        extractor = ImageExtractor()
        images = extractor._extract_from_xlsx_sync(xlsx_path)

        assert len(images) == 0

    @pytest.mark.asyncio
    async def test_extract_from_xlsx_async(self, tmp_path):
        """Test async XLSX extraction."""
        xlsx_path = tmp_path / "test.xlsx"
        with zipfile.ZipFile(xlsx_path, "w") as zf:
            zf.writestr("xl/media/image1.png", b"fake png data")

        extractor = ImageExtractor()
        images = await extractor.extract_from_xlsx(xlsx_path)

        assert len(images) == 1


class TestExtractFromPdf:
    """Tests for PDF image extraction."""

    def test_extract_from_pdf_sync(self, tmp_path):
        """Test sync PDF extraction with mocked pymupdf."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        # Mock pymupdf document
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(1, 0, 0, 0, 0, "Image1", "", "")]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.extract_image.return_value = {
            "image": b"image data",
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        with patch("pymupdf.open", return_value=mock_doc):
            extractor = ImageExtractor()
            images = extractor._extract_from_pdf_sync(pdf_path)

            assert len(images) == 1
            assert images[0].format == "png"
            assert images[0].width == 100
            assert images[0].height == 100

    def test_extract_from_pdf_handles_error(self, tmp_path):
        """Test PDF extraction handles errors gracefully."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        with patch("pymupdf.open", side_effect=Exception("Failed to open PDF")):
            extractor = ImageExtractor()
            images = extractor._extract_from_pdf_sync(pdf_path)

            assert len(images) == 0

    def test_extract_from_pdf_handles_image_error(self, tmp_path):
        """Test PDF extraction handles individual image errors."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_images.return_value = [(1, 0, 0, 0, 0, "Image1", "", "")]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.extract_image.side_effect = Exception("Failed to extract")

        with patch("pymupdf.open", return_value=mock_doc):
            extractor = ImageExtractor()
            images = extractor._extract_from_pdf_sync(pdf_path)

            # Should return empty list as the one image failed
            assert len(images) == 0

    @pytest.mark.asyncio
    async def test_extract_from_pdf_async(self, tmp_path):
        """Test async PDF extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_images.return_value = [(1, 0, 0, 0, 0, "Image1", "", "")]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.extract_image.return_value = {
            "image": b"image data",
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        with patch("pymupdf.open", return_value=mock_doc):
            extractor = ImageExtractor()
            images = await extractor.extract_from_pdf(pdf_path)

            assert len(images) == 1


class TestExtract:
    """Tests for the generic extract method."""

    @pytest.mark.asyncio
    async def test_extract_docx(self, tmp_path):
        """Test extract routes to DOCX handler."""
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/image1.png", b"png data")

        extractor = ImageExtractor()
        images = await extractor.extract(docx_path)

        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_extract_pptx(self, tmp_path):
        """Test extract routes to PPTX handler."""
        pptx_path = tmp_path / "test.pptx"
        with zipfile.ZipFile(pptx_path, "w") as zf:
            zf.writestr("ppt/media/image1.png", b"png data")

        extractor = ImageExtractor()
        images = await extractor.extract(pptx_path)

        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_extract_xlsx(self, tmp_path):
        """Test extract routes to XLSX handler."""
        xlsx_path = tmp_path / "test.xlsx"
        with zipfile.ZipFile(xlsx_path, "w") as zf:
            zf.writestr("xl/media/image1.png", b"png data")

        extractor = ImageExtractor()
        images = await extractor.extract(xlsx_path)

        assert len(images) == 1

    @pytest.mark.asyncio
    async def test_extract_pdf(self, tmp_path):
        """Test extract routes to PDF handler."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)

        with patch("pymupdf.open", return_value=mock_doc):
            extractor = ImageExtractor()
            images = await extractor.extract(pdf_path)

            assert len(images) == 0

    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self, tmp_path):
        """Test extract returns empty for unsupported format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("text content")

        extractor = ImageExtractor()
        images = await extractor.extract(txt_path)

        assert len(images) == 0


class TestExtractedImageData:
    """Tests for ExtractedImage data structure."""

    def test_extracted_image_has_correct_fields(self, tmp_path):
        """Test that extracted images have correct fields."""
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/image1.png", b"fake png data")

        extractor = ImageExtractor()
        images = extractor._extract_from_docx_sync(docx_path)

        assert len(images) == 1
        img = images[0]

        assert img.data == b"fake png data"
        assert img.format == "png"
        assert img.filename == "image1.png"
        assert img.source_document == docx_path
        assert img.position is not None
        assert img.original_path == "word/media/image1.png"

    def test_jpg_converted_to_jpeg(self, tmp_path):
        """Test that .jpg extension is converted to jpeg format."""
        docx_path = tmp_path / "test.docx"
        with zipfile.ZipFile(docx_path, "w") as zf:
            zf.writestr("word/media/photo.jpg", b"fake jpg data")

        extractor = ImageExtractor()
        images = extractor._extract_from_docx_sync(docx_path)

        assert len(images) == 1
        assert images[0].format == "jpeg"
        assert images[0].filename == "photo.jpg"
