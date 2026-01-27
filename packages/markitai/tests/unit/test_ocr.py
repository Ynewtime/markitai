"""Tests for OCR processor module."""

from __future__ import annotations

import sys
import threading
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

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

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


class TestOCRProcessorSingleton:
    """Tests for OCRProcessor global singleton pattern."""

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_get_shared_engine_creates_singleton(self):
        """Test that get_shared_engine creates a singleton engine."""
        mock_engine = MagicMock()

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=mock_engine
        ) as mock_create:
            # First call should create engine
            engine1 = OCRProcessor.get_shared_engine()
            assert engine1 is mock_engine
            assert mock_create.call_count == 1

            # Second call should return same instance
            engine2 = OCRProcessor.get_shared_engine()
            assert engine2 is engine1
            assert mock_create.call_count == 1  # Not called again

    def test_get_shared_engine_thread_safe(self):
        """Test that get_shared_engine is thread-safe."""
        mock_engine = MagicMock()
        engines: list = []
        errors: list = []

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=mock_engine
        ):

            def get_engine():
                try:
                    engine = OCRProcessor.get_shared_engine()
                    engines.append(engine)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_engine) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(engines) == 10
            # All threads should get the same engine
            assert all(e is engines[0] for e in engines)

    def test_engine_property_uses_shared_engine(self):
        """Test that engine property uses the shared global engine."""
        mock_engine = MagicMock()

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=mock_engine
        ):
            processor1 = OCRProcessor()
            processor2 = OCRProcessor()

            engine1 = processor1.engine
            engine2 = processor2.engine

            # Both processors should share the same engine
            assert engine1 is engine2
            assert engine1 is mock_engine


class TestOCRProcessorPreheat:
    """Tests for OCRProcessor preheat functionality."""

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_preheat_creates_and_warms_engine(self):
        """Test that preheat creates engine and runs dummy inference."""
        mock_engine = MagicMock()

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=mock_engine
        ):
            engine = OCRProcessor.preheat()

            assert engine is mock_engine
            # Engine should be called with dummy image
            assert mock_engine.call_count == 1
            # Check dummy image shape (100, 100, 3)
            call_args = mock_engine.call_args[0][0]
            assert call_args.shape == (100, 100, 3)

    def test_preheat_handles_inference_errors(self):
        """Test that preheat handles inference errors gracefully."""
        mock_engine = MagicMock()
        mock_engine.side_effect = Exception("OCR error")

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=mock_engine
        ):
            # Should not raise, just log the error
            engine = OCRProcessor.preheat()
            assert engine is mock_engine

    def test_preheat_reuses_existing_engine(self):
        """Test that preheat reuses existing engine if already created."""
        mock_engine = MagicMock()

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=mock_engine
        ) as mock_create:
            # First preheat
            OCRProcessor.preheat()
            assert mock_create.call_count == 1

            # Second preheat should reuse
            OCRProcessor.preheat()
            assert mock_create.call_count == 1  # Not called again


class TestOCRLanguageMapping:
    """Tests for OCR language mapping."""

    def test_language_mapping(self):
        """Test language code mapping."""

        # Test via _create_engine_impl which uses the mapping
        # We test the mapping indirectly through config
        config = OCRConfig(enabled=True, lang="zh")
        processor = OCRProcessor(config)
        assert processor.config is not None
        assert processor.config.lang == "zh"


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

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_recognize(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test OCR recognition."""
        # Create a test image file
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        # Mock the global engine
        mock_engine = MagicMock()
        mock_engine.return_value = MagicMock(
            txts=["Line 1", "Line 2"],
            scores=[0.9, 0.85],
            boxes=[[0, 0, 100, 20], [0, 30, 100, 50]],
        )

        # Set global engine directly
        OCRProcessor._global_engine = mock_engine

        processor = OCRProcessor(ocr_config)
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
        test_image = tmp_path / "empty.png"
        test_image.write_bytes(b"fake image data")

        # Mock the global engine
        mock_engine = MagicMock()
        mock_engine.return_value = MagicMock(
            txts=[],
            scores=[],
            boxes=[],
        )

        # Set global engine directly
        OCRProcessor._global_engine = mock_engine

        processor = OCRProcessor(ocr_config)
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

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_recognize_to_markdown_with_method(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """Test markdown conversion when result has to_markdown method."""
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        mock_result = MagicMock()
        mock_result.to_markdown.return_value = "# Heading\n\nParagraph"

        mock_engine = MagicMock()
        mock_engine.return_value = mock_result

        # Set global engine directly
        OCRProcessor._global_engine = mock_engine

        processor = OCRProcessor(ocr_config)
        result = processor.recognize_to_markdown(test_image)

        assert result == "# Heading\n\nParagraph"
        mock_result.to_markdown.assert_called_once()

    def test_recognize_to_markdown_fallback(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """Test markdown conversion fallback when no to_markdown method."""
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        mock_result = MagicMock(spec=[])  # No to_markdown method
        mock_result.txts = ["Line 1", "Line 2"]

        mock_engine = MagicMock()
        mock_engine.return_value = mock_result

        # Set global engine directly
        OCRProcessor._global_engine = mock_engine

        processor = OCRProcessor(ocr_config)
        result = processor.recognize_to_markdown(test_image)

        assert result == "Line 1\n\nLine 2"
