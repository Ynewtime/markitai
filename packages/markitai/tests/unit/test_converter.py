"""Tests for converter modules."""

from pathlib import Path

from markitai.converter import (
    ConvertResult,
    ExtractedImage,
    FileFormat,
    detect_format,
    get_converter,
)
from markitai.converter.base import EXTENSION_MAP
from markitai.converter.text import MarkdownConverter, TxtConverter


class TestFileFormat:
    """Tests for FileFormat enum and detection."""

    def test_detect_common_formats(self) -> None:
        """Test detection of common file formats."""
        assert detect_format("document.docx") == FileFormat.DOCX
        assert detect_format("presentation.pptx") == FileFormat.PPTX
        assert detect_format("spreadsheet.xlsx") == FileFormat.XLSX
        assert detect_format("document.pdf") == FileFormat.PDF
        assert detect_format("readme.txt") == FileFormat.TXT
        assert detect_format("README.md") == FileFormat.MD

    def test_detect_case_insensitive(self) -> None:
        """Test case-insensitive format detection."""
        assert detect_format("DOC.DOCX") == FileFormat.DOCX
        assert detect_format("doc.PDF") == FileFormat.PDF

    def test_detect_unknown_format(self) -> None:
        """Test detection of unknown formats."""
        assert detect_format("file.xyz") == FileFormat.UNKNOWN
        assert detect_format("file") == FileFormat.UNKNOWN

    def test_detect_from_path(self, tmp_path: Path) -> None:
        """Test format detection from Path objects."""
        test_file = tmp_path / "test.docx"
        assert detect_format(test_file) == FileFormat.DOCX

    def test_extension_map_completeness(self) -> None:
        """Test that all expected extensions are mapped."""
        expected_extensions = [
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".pdf",
            ".txt",
            ".md",
            ".markdown",
            ".jpeg",
            ".jpg",
            ".png",
            ".webp",
        ]
        for ext in expected_extensions:
            assert ext in EXTENSION_MAP, f"Missing extension: {ext}"


class TestConvertResult:
    """Tests for ConvertResult dataclass."""

    def test_empty_result(self) -> None:
        """Test creating empty result."""
        result = ConvertResult(markdown="")
        assert result.markdown == ""
        assert result.images == []
        assert result.metadata == {}
        assert result.has_images is False

    def test_result_with_content(self) -> None:
        """Test creating result with content."""
        result = ConvertResult(
            markdown="# Test",
            images=[],
            metadata={"source": "test.docx"},
        )
        assert result.markdown == "# Test"
        assert result.has_images is False

    def test_result_with_images(self, tmp_path: Path) -> None:
        """Test creating result with images."""
        image = ExtractedImage(
            path=tmp_path / "image.png",
            index=1,
            original_name="image.png",
            mime_type="image/png",
            width=100,
            height=100,
        )
        result = ConvertResult(
            markdown="# Test\n![](image.png)",
            images=[image],
        )
        assert result.has_images is True
        assert len(result.images) == 1


class TestGetConverter:
    """Tests for get_converter function."""

    def test_get_txt_converter(self) -> None:
        """Test getting TXT converter."""
        converter = get_converter("test.txt")
        assert converter is not None
        assert isinstance(converter, TxtConverter)

    def test_get_md_converter(self) -> None:
        """Test getting Markdown converter."""
        converter = get_converter("test.md")
        assert converter is not None
        assert isinstance(converter, MarkdownConverter)

    def test_get_unknown_converter(self) -> None:
        """Test getting converter for unknown format."""
        converter = get_converter("test.xyz")
        assert converter is None


class TestTextConverters:
    """Tests for text file converters."""

    def test_txt_converter(self, tmp_path: Path) -> None:
        """Test TXT converter."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        converter = TxtConverter()
        result = converter.convert(test_file)

        assert result.markdown == "Hello, World!"
        assert result.metadata["format"] == "TXT"

    def test_md_converter(self, tmp_path: Path) -> None:
        """Test Markdown converter."""
        test_file = tmp_path / "test.md"
        content = "# Title\n\nSome content."
        test_file.write_text(content)

        converter = MarkdownConverter()
        result = converter.convert(test_file)

        assert result.markdown == content
        assert result.metadata["format"] == "MD"

    def test_converter_with_unicode(self, tmp_path: Path) -> None:
        """Test converter with Unicode content."""
        test_file = tmp_path / "test.txt"
        content = "你好世界！🌍"
        test_file.write_text(content, encoding="utf-8")

        converter = TxtConverter()
        result = converter.convert(test_file)

        assert result.markdown == content
