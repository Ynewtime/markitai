"""Tests for OCR processor module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markitai.config import OCRConfig
from markitai.ocr import OCRProcessor, OCRResult


@pytest.fixture
def ocr_config() -> OCRConfig:
    """Return a test OCR configuration."""
    return OCRConfig(enabled=True, lang="zh")


class TestOCRProcessor:
    """Tests for OCRProcessor class."""

    def test_init(self, ocr_config: OCRConfig):
        """Test processor initialization."""
        processor = OCRProcessor(ocr_config)
        assert processor.config == ocr_config
        assert processor._engine is None  # Lazy initialization

    def test_init_no_config(self):
        """Test processor initialization without config."""
        processor = OCRProcessor()
        assert processor.config is None
        assert processor._engine is None

    def test_language_mapping(self):
        """Test language code mapping."""
        from rapidocr import LangRec

        processor = OCRProcessor()
        assert processor._get_lang_enum("zh") == LangRec.CH
        assert processor._get_lang_enum("en") == LangRec.EN
        assert processor._get_lang_enum("ch") == LangRec.CH
        assert processor._get_lang_enum("ja") == LangRec.JAPAN
        # Test unknown language defaults to CH
        assert processor._get_lang_enum("unknown") == LangRec.CH


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_create_result(self):
        """Test creating OCR result."""
        result = OCRResult(
            text="Test text",
            confidence=0.95,
            boxes=[[0, 0, 100, 20]],
        )
        assert result.text == "Test text"
        assert result.confidence == 0.95
        assert len(result.boxes) == 1


class TestOCRProcessorMocked:
    """Tests for OCRProcessor with mocked RapidOCR."""

    def test_recognize(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test OCR recognition."""
        processor = OCRProcessor(ocr_config)

        # Create a test image file
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        # Mock the engine
        mock_engine = MagicMock()
        mock_engine.return_value = MagicMock(
            txts=["Line 1", "Line 2"],
            scores=[0.9, 0.85],
            boxes=[[0, 0, 100, 20], [0, 30, 100, 50]],
        )

        with patch.object(processor, "_engine", mock_engine):
            result = processor.recognize(test_image)

            assert result.text == "Line 1\nLine 2"
            assert result.confidence == pytest.approx(0.875)
            assert len(result.boxes) == 2

    def test_recognize_file_not_found(self, ocr_config: OCRConfig):
        """Test OCR with non-existent file."""
        processor = OCRProcessor(ocr_config)

        with pytest.raises(FileNotFoundError):
            processor.recognize(Path("/non/existent/file.png"))

    def test_recognize_empty_result(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test OCR with empty result."""
        processor = OCRProcessor(ocr_config)

        test_image = tmp_path / "empty.png"
        test_image.write_bytes(b"fake image data")

        mock_engine = MagicMock()
        mock_engine.return_value = MagicMock(
            txts=[],
            scores=[],
            boxes=[],
        )

        with patch.object(processor, "_engine", mock_engine):
            result = processor.recognize(test_image)

            assert result.text == ""
            assert result.confidence == 0.0
            assert len(result.boxes) == 0


class TestOCRPDFMethods:
    """Tests for PDF-related OCR methods."""

    def test_is_scanned_pdf_with_text(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test detecting non-scanned PDF (has text)."""
        processor = OCRProcessor(ocr_config)

        # Create a mock PDF with text
        test_pdf = tmp_path / "text.pdf"
        test_pdf.write_bytes(b"fake pdf data")

        # Patch fitz at the point where it's imported
        mock_fitz = MagicMock()
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "A" * 200  # Lots of text
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = processor.is_scanned_pdf(test_pdf)

            assert result is False  # Not scanned, has text

    def test_is_scanned_pdf_no_text(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test detecting scanned PDF (no text)."""
        processor = OCRProcessor(ocr_config)

        test_pdf = tmp_path / "scanned.pdf"
        test_pdf.write_bytes(b"fake pdf data")

        mock_fitz = MagicMock()
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # No text
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = processor.is_scanned_pdf(test_pdf)

            assert result is True  # Is scanned, no text


class TestOCRRecognizeToMarkdown:
    """Tests for recognize_to_markdown method."""

    def test_recognize_to_markdown_with_method(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """Test markdown conversion when result has to_markdown method."""
        processor = OCRProcessor(ocr_config)

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        mock_result = MagicMock()
        mock_result.to_markdown.return_value = "# Heading\n\nParagraph"

        mock_engine = MagicMock()
        mock_engine.return_value = mock_result

        with patch.object(processor, "_engine", mock_engine):
            result = processor.recognize_to_markdown(test_image)

            assert result == "# Heading\n\nParagraph"
            mock_result.to_markdown.assert_called_once()

    def test_recognize_to_markdown_fallback(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """Test markdown conversion fallback when no to_markdown method."""
        processor = OCRProcessor(ocr_config)

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        mock_result = MagicMock(spec=[])  # No to_markdown method
        mock_result.txts = ["Line 1", "Line 2"]

        mock_engine = MagicMock()
        mock_engine.return_value = mock_result

        with patch.object(processor, "_engine", mock_engine):
            result = processor.recognize_to_markdown(test_image)

            assert result == "Line 1\n\nLine 2"
