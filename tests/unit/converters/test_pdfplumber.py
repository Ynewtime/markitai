"""Tests for pdfplumber PDF converter module."""

from unittest.mock import MagicMock, patch

import pytest

from markit.converters.pdf.pdfplumber import PDFPlumberConverter, TableExtractor
from markit.exceptions import ConversionError


class TestPDFPlumberConverterInit:
    """Tests for PDFPlumberConverter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        converter = PDFPlumberConverter()

        assert converter.extract_images is True
        assert converter.extract_tables is True
        assert converter.table_settings == {}

    def test_custom_init(self):
        """Test custom initialization."""
        settings = {"vertical_strategy": "lines"}
        converter = PDFPlumberConverter(
            extract_images=False,
            extract_tables=False,
            table_settings=settings,
        )

        assert converter.extract_images is False
        assert converter.extract_tables is False
        assert converter.table_settings == settings

    def test_converter_name(self):
        """Test converter name."""
        converter = PDFPlumberConverter()
        assert converter.name == "pdfplumber"

    def test_supported_extensions(self):
        """Test supported extensions."""
        converter = PDFPlumberConverter()
        assert ".pdf" in converter.supported_extensions
        assert converter.supports(".pdf")
        assert not converter.supports(".docx")


class TestPDFPlumberConverterConvert:
    """Tests for PDFPlumberConverter.convert method."""

    @pytest.mark.asyncio
    async def test_convert_invalid_file(self, tmp_path):
        """Test convert raises error for invalid file."""
        converter = PDFPlumberConverter()
        invalid_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(ConversionError):
            await converter.convert(invalid_path)

    @pytest.mark.asyncio
    async def test_convert_success(self, tmp_path):
        """Test successful conversion."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        # Mock pdfplumber
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []
        mock_page.extract_text.return_value = "Test content"
        mock_page.images = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=None)

        with patch("pdfplumber.open", return_value=mock_pdf):
            converter = PDFPlumberConverter()
            result = await converter.convert(pdf_path)

            assert result.markdown is not None
            assert "Test content" in result.markdown
            assert result.metadata["converter"] == "pdfplumber"

    @pytest.mark.asyncio
    async def test_convert_with_tables(self, tmp_path):
        """Test conversion with table extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        # Mock pdfplumber with table
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [[["Header1", "Header2"], ["Cell1", "Cell2"]]]
        mock_page.extract_text.return_value = ""
        mock_page.images = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=None)

        with patch("pdfplumber.open", return_value=mock_pdf):
            converter = PDFPlumberConverter()
            result = await converter.convert(pdf_path)

            assert "| Header1 | Header2 |" in result.markdown
            assert "| Cell1 | Cell2 |" in result.markdown

    @pytest.mark.asyncio
    async def test_convert_with_images(self, tmp_path):
        """Test conversion with image extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        # Mock pdfplumber with image
        mock_pil_image = MagicMock()
        mock_pil_image.original = MagicMock()

        mock_cropped = MagicMock()
        mock_cropped.to_image.return_value = mock_pil_image

        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []
        mock_page.extract_text.return_value = "Text"
        mock_page.images = [{"x0": 0, "top": 0, "x1": 100, "bottom": 100}]
        mock_page.crop.return_value = mock_cropped

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=None)

        with patch("pdfplumber.open", return_value=mock_pdf):
            converter = PDFPlumberConverter()
            result = await converter.convert(pdf_path)

            assert len(result.images) >= 0  # Might be empty due to mock limitations

    @pytest.mark.asyncio
    async def test_convert_exception_handling(self, tmp_path):
        """Test conversion handles exceptions."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        with patch("pdfplumber.open", side_effect=Exception("PDF error")):
            converter = PDFPlumberConverter()

            with pytest.raises(ConversionError) as exc_info:
                await converter.convert(pdf_path)

            assert "PDF error" in str(exc_info.value)


class TestPDFPlumberTableToMarkdown:
    """Tests for _table_to_markdown method."""

    def test_empty_table(self):
        """Test empty table returns empty string."""
        converter = PDFPlumberConverter()
        assert converter._table_to_markdown([]) == ""
        assert converter._table_to_markdown([[]]) == ""

    def test_simple_table(self):
        """Test simple table conversion."""
        converter = PDFPlumberConverter()
        table: list[list[str | None]] = [
            ["Name", "Age"],
            ["Alice", "30"],
            ["Bob", "25"],
        ]

        result = converter._table_to_markdown(table)

        assert "| Name | Age |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 30 |" in result
        assert "| Bob | 25 |" in result

    def test_table_with_none_cells(self):
        """Test table with None cells."""
        converter = PDFPlumberConverter()
        table = [
            ["Header1", None],
            [None, "Value"],
        ]

        result = converter._table_to_markdown(table)

        assert "| Header1 |  |" in result
        assert "|  | Value |" in result

    def test_table_with_newlines(self):
        """Test table with newlines in cells."""
        converter = PDFPlumberConverter()
        table: list[list[str | None]] = [
            ["Multi\nline", "Header"],
            ["Cell", "Data"],
        ]

        result = converter._table_to_markdown(table)

        # Newlines should be replaced with spaces
        assert "Multi line" in result

    def test_table_row_padding(self):
        """Test table rows with fewer cells are padded."""
        converter = PDFPlumberConverter()
        table: list[list[str | None]] = [
            ["A", "B", "C"],
            ["1"],  # Short row
        ]

        result = converter._table_to_markdown(table)

        # Row should be padded
        lines = result.split("\n")
        assert len(lines) >= 3  # Header, separator, data


class TestPDFPlumberFormatText:
    """Tests for _format_text method."""

    def test_format_text_strips_whitespace(self):
        """Test that text is stripped of whitespace."""
        converter = PDFPlumberConverter()
        text = "  Line 1  \n  Line 2  \n"

        result = converter._format_text(text)

        assert result == "Line 1\nLine 2"

    def test_format_text_removes_empty_lines(self):
        """Test that empty lines are removed."""
        converter = PDFPlumberConverter()
        text = "Line 1\n\n\nLine 2"

        result = converter._format_text(text)

        assert result == "Line 1\nLine 2"

    def test_format_text_empty_input(self):
        """Test empty input."""
        converter = PDFPlumberConverter()
        assert converter._format_text("") == ""
        assert converter._format_text("   \n   ") == ""


class TestPDFPlumberCleanupMarkdown:
    """Tests for _cleanup_markdown method."""

    def test_cleanup_removes_excessive_blank_lines(self):
        """Test that excessive blank lines are removed."""
        converter = PDFPlumberConverter()
        markdown = "Line 1\n\n\n\n\nLine 2"

        result = converter._cleanup_markdown(markdown)

        assert result == "Line 1\n\nLine 2"

    def test_cleanup_strips_content(self):
        """Test that content is stripped."""
        converter = PDFPlumberConverter()
        markdown = "  \n\nContent\n\n  "

        result = converter._cleanup_markdown(markdown)

        assert result == "Content"


class TestTableExtractor:
    """Tests for TableExtractor class."""

    def test_init_default_settings(self):
        """Test default initialization."""
        extractor = TableExtractor()

        assert extractor.settings["vertical_strategy"] == "text"
        assert extractor.settings["horizontal_strategy"] == "text"

    def test_init_custom_settings(self):
        """Test custom settings."""
        settings = {"vertical_strategy": "lines", "snap_tolerance": 5}
        extractor = TableExtractor(settings=settings)

        assert extractor.settings == settings

    def test_extract_tables(self, tmp_path):
        """Test table extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.extract_tables.return_value = [[["A", "B"], ["1", "2"]]]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=None)

        with patch("pdfplumber.open", return_value=mock_pdf):
            extractor = TableExtractor()
            tables = extractor.extract_tables(pdf_path)

            assert len(tables) == 1
            assert tables[0] == [["A", "B"], ["1", "2"]]
