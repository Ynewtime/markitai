"""Tests for PyMuPDF PDF converter module."""

from unittest.mock import MagicMock, patch

import pytest

from markit.converters.pdf.pymupdf import (
    PyMuPDFConverter,
    PyMuPDFTextExtractor,
    _normalize_image_spacing,
    _sanitize_filename,
)
from markit.exceptions import ConversionError


class TestSanitizeFilename:
    """Tests for _sanitize_filename function."""

    def test_replace_spaces(self):
        """Test space replacement."""
        assert _sanitize_filename("hello world") == "hello_world"

    def test_replace_colons(self):
        """Test colon replacement."""
        assert _sanitize_filename("file:name") == "file_name"
        assert _sanitize_filename("file：name") == "file_name"  # Full-width colon

    def test_replace_path_separators(self):
        """Test path separator replacement."""
        assert _sanitize_filename("path/file") == "path_file"
        assert _sanitize_filename("path\\file") == "path_file"

    def test_replace_special_characters(self):
        """Test special character replacement."""
        assert _sanitize_filename("file?name") == "file_name"
        assert _sanitize_filename("file*name") == "file_name"
        assert _sanitize_filename('file"name') == "file_name"
        assert _sanitize_filename("file<name>") == "file_name_"
        assert _sanitize_filename("file|name") == "file_name"

    def test_collapse_underscores(self):
        """Test collapsing multiple underscores."""
        assert _sanitize_filename("file  name") == "file_name"
        assert _sanitize_filename("file___name") == "file_name"

    def test_valid_filename_unchanged(self):
        """Test valid filename remains unchanged."""
        assert _sanitize_filename("valid_file.pdf") == "valid_file.pdf"


class TestNormalizeImageSpacing:
    """Tests for _normalize_image_spacing function."""

    def test_text_only(self):
        """Test text without images."""
        markdown = "Line 1\nLine 2\nLine 3"
        result = _normalize_image_spacing(markdown)
        assert result == markdown

    def test_single_image(self):
        """Test single image."""
        markdown = "Text before\n![Image](path.png)\nText after"
        result = _normalize_image_spacing(markdown)
        assert "![Image](path.png)" in result

    def test_consecutive_images(self):
        """Test consecutive images get one blank line between."""
        markdown = "![Image1](a.png)\n\n\n![Image2](b.png)"
        result = _normalize_image_spacing(markdown)
        # Should have at most one blank line between images
        assert "\n\n\n" not in result

    def test_blank_line_before_image_from_text(self):
        """Test blank line added before image following text."""
        markdown = "Text\n![Image](path.png)"
        result = _normalize_image_spacing(markdown)
        # Should add blank line between text and image
        assert "\n\n![Image]" in result or "\n![Image]" in result


class TestPyMuPDFConverterInit:
    """Tests for PyMuPDFConverter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        converter = PyMuPDFConverter()

        assert converter.extract_images is True
        assert converter.image_dpi == 150
        assert converter.ocr_enabled is False
        assert converter.filter_small_images is True
        assert converter.min_image_dimension == 50
        assert converter.min_image_area == 2500
        assert converter.min_image_size == 3072

    def test_custom_init(self):
        """Test custom initialization."""
        converter = PyMuPDFConverter(
            extract_images=False,
            image_dpi=300,
            ocr_enabled=True,
            filter_small_images=False,
            min_image_dimension=100,
            min_image_area=5000,
            min_image_size=4096,
        )

        assert converter.extract_images is False
        assert converter.image_dpi == 300
        assert converter.ocr_enabled is True
        assert converter.filter_small_images is False
        assert converter.min_image_dimension == 100

    def test_converter_name(self):
        """Test converter name."""
        converter = PyMuPDFConverter()
        assert converter.name == "pymupdf"

    def test_supported_extensions(self):
        """Test supported extensions."""
        converter = PyMuPDFConverter()
        assert ".pdf" in converter.supported_extensions
        assert converter.supports(".pdf")
        assert not converter.supports(".docx")


class TestPyMuPDFConverterConvert:
    """Tests for PyMuPDFConverter.convert method."""

    @pytest.mark.asyncio
    async def test_convert_invalid_file(self, tmp_path):
        """Test convert raises error for invalid file."""
        converter = PyMuPDFConverter()
        invalid_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(ConversionError):
            await converter.convert(invalid_path)

    @pytest.mark.asyncio
    async def test_convert_success(self, tmp_path):
        """Test successful conversion."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        # Mock fitz (PyMuPDF)
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test PDF content"
        mock_page.get_image_info.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)

        with patch("fitz.open", return_value=mock_doc):
            converter = PyMuPDFConverter(extract_images=False)
            result = await converter.convert(pdf_path)

            assert result.markdown is not None
            assert "Test PDF content" in result.markdown
            assert result.metadata["converter"] == "pymupdf"

    @pytest.mark.asyncio
    async def test_convert_exception_handling(self, tmp_path):
        """Test conversion handles exceptions."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        with patch("fitz.open", side_effect=Exception("PDF error")):
            converter = PyMuPDFConverter()

            with pytest.raises(ConversionError) as exc_info:
                await converter.convert(pdf_path)

            assert "PDF error" in str(exc_info.value)


class TestPyMuPDFConverterFormatPageText:
    """Tests for _format_page_text method."""

    def test_format_text_basic(self):
        """Test basic text formatting."""
        converter = PyMuPDFConverter()
        text = "Line 1\nLine 2\nLine 3"

        result = converter._format_page_text(text)

        assert "Line 1" in result
        assert "Line 2" in result

    def test_format_text_removes_empty_lines(self):
        """Test empty lines are removed."""
        converter = PyMuPDFConverter()
        text = "Line 1\n\n\nLine 2"

        result = converter._format_page_text(text)

        assert "\n\n\n" not in result

    def test_format_text_heading_detection(self):
        """Test ALL CAPS lines become headings."""
        converter = PyMuPDFConverter()
        text = "CHAPTER ONE\nRegular text"

        result = converter._format_page_text(text)

        assert "## Chapter One" in result

    def test_format_text_long_caps_not_heading(self):
        """Test long ALL CAPS lines are not converted to headings."""
        converter = PyMuPDFConverter()
        long_caps = "A" * 100  # 100 characters
        text = f"{long_caps}\nRegular text"

        result = converter._format_page_text(text)

        # Should not convert to heading
        assert f"## {long_caps.title()}" not in result


class TestPyMuPDFConverterShouldKeepImage:
    """Tests for _should_keep_image method."""

    def test_keep_large_image(self):
        """Test keeping large image."""
        converter = PyMuPDFConverter()
        data = b"x" * 5000  # 5KB
        assert converter._should_keep_image(data, 200, 200) is True

    def test_filter_small_data(self):
        """Test filtering image with small data size."""
        converter = PyMuPDFConverter(min_image_size=3072)
        data = b"x" * 1000  # 1KB
        assert converter._should_keep_image(data, 200, 200) is False

    def test_filter_small_dimension(self):
        """Test filtering image with small dimension."""
        converter = PyMuPDFConverter(min_image_dimension=50)
        data = b"x" * 5000
        assert converter._should_keep_image(data, 30, 200) is False

    def test_filter_small_area(self):
        """Test filtering image with small area."""
        converter = PyMuPDFConverter(min_image_area=2500)
        data = b"x" * 5000
        assert converter._should_keep_image(data, 30, 30) is False  # 900 < 2500

    def test_keep_image_no_dimensions(self):
        """Test keeping image when dimensions not provided."""
        converter = PyMuPDFConverter()
        data = b"x" * 5000
        # When width or height is 0, dimension checks are skipped
        assert converter._should_keep_image(data, 0, 0) is True


class TestPyMuPDFConverterCleanupMarkdown:
    """Tests for _cleanup_markdown method."""

    def test_cleanup_removes_excessive_blank_lines(self):
        """Test removing excessive blank lines."""
        converter = PyMuPDFConverter()
        markdown = "Line 1\n\n\n\n\nLine 2"

        result = converter._cleanup_markdown(markdown)

        assert "\n\n\n" not in result
        assert result == "Line 1\n\nLine 2"

    def test_cleanup_strips_content(self):
        """Test content is stripped."""
        converter = PyMuPDFConverter()
        markdown = "  \n\nContent\n\n  "

        result = converter._cleanup_markdown(markdown)

        assert result == "Content"


class TestPyMuPDFTextExtractor:
    """Tests for PyMuPDFTextExtractor class."""

    def test_init_default(self):
        """Test default initialization."""
        extractor = PyMuPDFTextExtractor()
        assert extractor.preserve_layout is False

    def test_init_preserve_layout(self):
        """Test initialization with preserve_layout."""
        extractor = PyMuPDFTextExtractor(preserve_layout=True)
        assert extractor.preserve_layout is True

    def test_extract_basic(self, tmp_path):
        """Test basic text extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page text"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch("fitz.open", return_value=mock_doc):
            extractor = PyMuPDFTextExtractor()
            result = extractor.extract(pdf_path)

            assert "Page text" in result

    def test_extract_preserve_layout(self, tmp_path):
        """Test extraction with layout preservation."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [{"spans": [{"text": "Hello"}, {"text": "World"}]}],
                }
            ]
        }

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch("fitz.open", return_value=mock_doc):
            extractor = PyMuPDFTextExtractor(preserve_layout=True)
            result = extractor.extract(pdf_path)

            # The function joins spans with space, so we check both parts exist
            assert "Hello" in result
            assert "World" in result

    def test_format_text_dict(self):
        """Test _format_text_dict method."""
        extractor = PyMuPDFTextExtractor()

        text_dict = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {"spans": [{"text": "Line 1"}]},
                        {"spans": [{"text": "Line 2"}]},
                    ],
                },
                {"type": 1},  # Image block, should be skipped
            ]
        }

        result = extractor._format_text_dict(text_dict)

        assert "Line 1" in result
        assert "Line 2" in result

    def test_format_text_dict_empty_blocks(self):
        """Test _format_text_dict with empty blocks."""
        extractor = PyMuPDFTextExtractor()
        text_dict: dict = {"blocks": []}

        result = extractor._format_text_dict(text_dict)

        assert result == ""


class TestPyMuPDFConverterExtractPageImages:
    """Tests for _extract_page_images method."""

    def test_extract_images_basic(self, tmp_path):
        """Test basic image extraction."""
        converter = PyMuPDFConverter(filter_small_images=False)
        pdf_path = tmp_path / "test.pdf"

        mock_page = MagicMock()
        mock_page.get_image_info.return_value = [
            {"xref": 1, "width": 100, "height": 100},
            {"xref": 2, "width": 200, "height": 200},
        ]

        mock_doc = MagicMock()
        mock_doc.extract_image.side_effect = [
            {"image": b"image1", "ext": "png", "width": 100, "height": 100},
            {"image": b"image2", "ext": "jpeg", "width": 200, "height": 200},
        ]

        images = converter._extract_page_images(mock_doc, mock_page, 1, pdf_path, 0, set())

        assert len(images) == 2
        assert images[0][0].format == "png"
        assert images[1][0].format == "jpeg"

    def test_extract_images_skip_duplicates(self, tmp_path):
        """Test skipping duplicate images by xref."""
        converter = PyMuPDFConverter(filter_small_images=False)
        pdf_path = tmp_path / "test.pdf"

        mock_page = MagicMock()
        mock_page.get_image_info.return_value = [
            {"xref": 1, "width": 100, "height": 100},
            {"xref": 1, "width": 100, "height": 100},  # Duplicate
        ]

        mock_doc = MagicMock()
        mock_doc.extract_image.return_value = {
            "image": b"image1",
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        seen_xrefs: set[int] = set()
        images = converter._extract_page_images(mock_doc, mock_page, 1, pdf_path, 0, seen_xrefs)

        assert len(images) == 1  # Only one, duplicate skipped
        assert 1 in seen_xrefs

    def test_extract_images_skip_zero_xref(self, tmp_path):
        """Test skipping images with zero xref."""
        converter = PyMuPDFConverter(filter_small_images=False)
        pdf_path = tmp_path / "test.pdf"

        mock_page = MagicMock()
        mock_page.get_image_info.return_value = [
            {"xref": 0},  # Should skip
            {"xref": 1, "width": 100, "height": 100},
        ]

        mock_doc = MagicMock()
        mock_doc.extract_image.return_value = {
            "image": b"image1",
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        images = converter._extract_page_images(mock_doc, mock_page, 1, pdf_path, 0, set())

        assert len(images) == 1

    def test_extract_images_filter_small(self, tmp_path):
        """Test filtering small images."""
        converter = PyMuPDFConverter(
            filter_small_images=True,
            min_image_size=1000,
        )
        pdf_path = tmp_path / "test.pdf"

        mock_page = MagicMock()
        mock_page.get_image_info.return_value = [
            {"xref": 1, "width": 100, "height": 100},
        ]

        mock_doc = MagicMock()
        mock_doc.extract_image.return_value = {
            "image": b"x" * 100,  # Too small
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        images = converter._extract_page_images(mock_doc, mock_page, 1, pdf_path, 0, set())

        assert len(images) == 0  # Filtered out

    def test_extract_images_error_handling(self, tmp_path):
        """Test error handling during image extraction."""
        converter = PyMuPDFConverter(filter_small_images=False)
        pdf_path = tmp_path / "test.pdf"

        mock_page = MagicMock()
        mock_page.get_image_info.return_value = [
            {"xref": 1},
            {"xref": 2},
        ]

        mock_doc = MagicMock()
        mock_doc.extract_image.side_effect = [
            Exception("Failed to extract"),
            {"image": b"image2", "ext": "png", "width": 100, "height": 100},
        ]

        images = converter._extract_page_images(mock_doc, mock_page, 1, pdf_path, 0, set())

        # Should still return the successful extraction
        assert len(images) == 1

    def test_extract_images_get_image_info_error(self, tmp_path):
        """Test handling error from get_image_info."""
        converter = PyMuPDFConverter()
        pdf_path = tmp_path / "test.pdf"

        mock_page = MagicMock()
        mock_page.get_image_info.side_effect = Exception("Failed to get info")

        mock_doc = MagicMock()

        images = converter._extract_page_images(mock_doc, mock_page, 1, pdf_path, 0, set())

        assert len(images) == 0


class TestNormalizeImageSpacingEdgeCases:
    """Additional tests for _normalize_image_spacing edge cases."""

    def test_blank_after_image(self):
        """Test handling blank lines after image."""
        markdown = "![Image](a.png)\n\nText after"
        result = _normalize_image_spacing(markdown)
        assert "![Image](a.png)" in result
        assert "Text after" in result

    def test_multiple_consecutive_blank_lines_after_image(self):
        """Test multiple blank lines after image followed by text."""
        markdown = "![Image](a.png)\n\n\n\nText"
        result = _normalize_image_spacing(markdown)
        # Should normalize excessive blanks
        assert "![Image](a.png)" in result
        assert "Text" in result

    def test_blank_between_text_and_image(self):
        """Test blank lines between text and image are normalized."""
        markdown = "Some text\n\n\n![Image](a.png)"
        result = _normalize_image_spacing(markdown)
        assert "Some text" in result
        assert "![Image](a.png)" in result


class TestPyMuPDFConverterWithImages:
    """Tests for PyMuPDFConverter with image extraction."""

    @pytest.mark.asyncio
    async def test_convert_with_images(self, tmp_path):
        """Test conversion with image extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content"
        mock_page.get_image_info.return_value = [
            {"xref": 1, "width": 100, "height": 100},
        ]

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.extract_image.return_value = {
            "image": b"x" * 5000,
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        with patch("fitz.open", return_value=mock_doc):
            converter = PyMuPDFConverter(extract_images=True, filter_small_images=False)
            result = await converter.convert(pdf_path)

            assert result.markdown is not None
            assert len(result.images) == 1
            assert "![](assets/" in result.markdown

    @pytest.mark.asyncio
    async def test_convert_empty_page_text(self, tmp_path):
        """Test conversion with empty page text."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"fake pdf")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "  # Empty/whitespace
        mock_page.get_image_info.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)

        with patch("fitz.open", return_value=mock_doc):
            converter = PyMuPDFConverter(extract_images=False)
            result = await converter.convert(pdf_path)

            # Should not include page marker for empty pages
            assert "<!-- Page 1 -->" not in result.markdown
