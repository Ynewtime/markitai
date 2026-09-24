"""Tests for OCR processor module."""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from markitai.config import OCRConfig
from markitai.ocr import OCRProcessor, OCRResult, is_likely_garbled


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

        # Set global engine directly (config must match to avoid rebuild)
        OCRProcessor._global_engine = mock_engine
        OCRProcessor._global_config = ocr_config

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
        """Test OCR with an empty result too small to tile."""
        from PIL import Image

        test_image = tmp_path / "empty.png"
        Image.new("RGB", (100, 100), "white").save(test_image)

        mock_engine = MagicMock()
        mock_engine.return_value = MagicMock(txts=[], scores=[], boxes=[])

        OCRProcessor._global_engine = mock_engine
        OCRProcessor._global_config = ocr_config

        processor = OCRProcessor(ocr_config)
        result = processor.recognize(test_image)

        assert result.text == ""
        assert result.confidence == 0.0
        assert len(result.boxes) == 0
        assert mock_engine.call_count == 1

    def test_sparse_layout_retries_overlapping_tiles(self, ocr_config: OCRConfig):
        """Small text in a mostly empty image gets a tiled second pass."""
        import numpy as np

        empty = MagicMock(txts=None, scores=None, boxes=None)
        found = MagicMock(
            txts=("markitai test fixture sample.jpg",),
            scores=(0.99,),
            boxes=(np.array([[10, 10], [200, 10], [200, 30], [10, 30]]),),
        )
        mock_engine = MagicMock(side_effect=[empty, found, empty, empty, empty])
        OCRProcessor._global_engine = mock_engine
        OCRProcessor._global_config = ocr_config

        result = OCRProcessor(ocr_config).recognize_numpy(
            np.zeros((768, 1024, 3), dtype=np.uint8)
        )

        assert result.text == "markitai test fixture sample.jpg"
        assert result.confidence == pytest.approx(0.99)
        assert mock_engine.call_count == 5


class TestOCRPDFMethods:
    """Tests for PDF-related OCR methods."""

    def test_is_scanned_pdf_with_text(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test detecting non-scanned PDF (has text)."""
        processor = OCRProcessor(ocr_config)

        # Create a mock PDF with text
        test_pdf = tmp_path / "text.pdf"
        test_pdf.write_bytes(b"fake pdf data")

        # Patch pymupdf at the point where it's imported (never the
        # legacy `fitz` alias: since 1.28.2 it prints to stdout)
        mock_pymupdf = MagicMock()
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "A" * 200  # Lots of text
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            result = processor.is_scanned_pdf(test_pdf)

            assert result is False  # Not scanned, has text

    def test_is_scanned_pdf_no_text(self, ocr_config: OCRConfig, tmp_path: Path):
        """Test detecting scanned PDF (no text)."""
        processor = OCRProcessor(ocr_config)

        test_pdf = tmp_path / "scanned.pdf"
        test_pdf.write_bytes(b"fake pdf data")

        mock_pymupdf = MagicMock()
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # No text
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.close = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            result = processor.is_scanned_pdf(test_pdf)

            assert result is True  # Is scanned, no text


class TestIsLikelyGarbled:
    """Tests for the is_likely_garbled heuristic."""

    def test_garbled_english_consonant_soup(self):
        """All-consonant text (vowel ratio 0%) is flagged as garbled."""
        text = "bcdfghjklm npqrstvwxz " * 3  # 60 letters, 0 vowels
        assert is_likely_garbled(text) is True

    def test_garbled_english_caesar_shifted(self):
        """Caesar-shifted text with collapsed vowel ratio is garbled."""
        # "+3" shift of "The quick brown fox jumps over the lazy dog"
        text = "Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj"
        assert is_likely_garbled(text) is True

    def test_normal_english_not_garbled(self):
        """Natural English has ~35-40% vowels and is never flagged."""
        text = "The quick brown fox jumps over the lazy dog"
        assert is_likely_garbled(text) is False

    def test_cjk_only_not_garbled(self):
        """Pure CJK text has no Latin letters to judge -- never garbled."""
        text = "这是一份完全由中文组成的测试文档，不包含任何拉丁字母内容。" * 3
        assert is_likely_garbled(text) is False

    def test_mostly_cjk_with_garbled_latin_not_garbled(self):
        """A mostly-CJK page is exempt even if its few Latin runs look off."""
        cjk = "中文内容占据页面绝大部分，拉丁字母只是少数点缀而已。" * 2
        latin = "bcdfghjklm npqrstvwxz bcdfghjklm"  # 30 letters, 0 vowels
        assert is_likely_garbled(cjk + latin) is False

    def test_latin_dominant_garbled_with_some_cjk(self):
        """Garbled Latin-dominant text stays flagged despite a few CJK chars."""
        text = "bcdfghjklm npqrstvwxz " * 3 + "图表"
        assert is_likely_garbled(text) is True

    def test_short_text_not_flagged(self):
        """Below 30 Latin letters the ratio is meaningless -- never flagged."""
        assert is_likely_garbled("xyz qrst") is False
        assert is_likely_garbled("") is False

    def test_mixed_normal_english_and_cjk_not_garbled(self):
        """Normal bilingual text is not flagged."""
        text = "The annual report 年度报告 shows revenue growth 收入增长 this year."
        assert is_likely_garbled(text) is False


class TestIsScannedPdfZeroPages:
    """Tests for the zero-page guard in is_scanned_pdf."""

    def test_zero_page_pdf_returns_false(self, ocr_config: OCRConfig, tmp_path: Path):
        """A zero-page PDF must not divide by zero and is not scanned."""
        processor = OCRProcessor(ocr_config)

        test_pdf = tmp_path / "empty.pdf"
        test_pdf.write_bytes(b"fake pdf data")

        mock_pymupdf = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_doc.close = MagicMock()
        mock_pymupdf.open.return_value = mock_doc

        with patch.dict(sys.modules, {"pymupdf": mock_pymupdf}):
            assert processor.is_scanned_pdf(test_pdf) is False


class TestOCRRecognizeToMarkdown:
    """Tests for recognize_to_markdown method."""

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_recognize_to_markdown_lays_boxes_out_in_reading_order(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """Image OCR is laid out by markitai, not RapidOCR's to_markdown(),
        which interleaved a two-column page line by line."""
        import numpy as np

        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        def box(x0: float, y0: float) -> np.ndarray:
            return np.array(
                [[x0, y0], [x0 + 300, y0], [x0 + 300, y0 + 20], [x0, y0 + 20]]
            )

        texts, boxes = [], []
        for n in range(1, 4):  # recognition order: row by row across columns
            texts += [f"Left column sentence {n}.", f"Right column sentence {n}."]
            boxes += [box(50, n * 25), box(450, n * 25)]
        mock_result = MagicMock(
            txts=texts, scores=[0.99] * len(texts), boxes=boxes, img=None
        )

        OCRProcessor._global_engine = MagicMock(return_value=mock_result)
        OCRProcessor._global_config = ocr_config

        result = OCRProcessor(ocr_config).recognize_to_markdown(test_image)

        assert result.splitlines()[:3] == [
            "Left column sentence 1.",
            "Left column sentence 2.",
            "Left column sentence 3.",
        ]
        mock_result.to_markdown.assert_not_called()

    def test_recognize_to_markdown_without_geometry(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """Boxes without usable geometry keep their recognition order."""
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake image data")

        mock_result = MagicMock(spec=[])
        mock_result.txts = ["Line 1", "Line 2"]
        mock_result.scores = [0.9, 0.9]
        mock_result.boxes = None

        OCRProcessor._global_engine = MagicMock(return_value=mock_result)
        OCRProcessor._global_config = ocr_config

        result = OCRProcessor(ocr_config).recognize_to_markdown(test_image)

        assert result == "Line 1\nLine 2"


class TestOCRConfigChangedRebuildsEngine:
    """Tests that shared engine is rebuilt when OCR config changes."""

    def teardown_method(self):
        """Reset global engine after each test."""
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_shared_engine_rebuilds_on_config_change(self):
        """When config changes (e.g., lang zh -> en), engine must be rebuilt."""
        engine_zh = MagicMock(name="engine_zh")
        engine_en = MagicMock(name="engine_en")

        call_count = 0

        def fake_create(config=None):
            nonlocal call_count
            call_count += 1
            if config and config.lang == "en":
                return engine_en
            return engine_zh

        with patch.object(OCRProcessor, "_create_engine_impl", side_effect=fake_create):
            config_zh = OCRConfig(enabled=True, lang="zh")
            config_en = OCRConfig(enabled=True, lang="en")

            # First call with zh config
            result1 = OCRProcessor.get_shared_engine(config_zh)
            assert result1 is engine_zh
            assert call_count == 1

            # Second call with different config (en) should rebuild
            result2 = OCRProcessor.get_shared_engine(config_en)
            assert result2 is engine_en
            assert call_count == 2  # Engine was recreated

    def test_shared_engine_reuses_on_same_config(self):
        """When config is the same, engine should be reused."""
        engine_zh = MagicMock(name="engine_zh")

        with patch.object(
            OCRProcessor, "_create_engine_impl", return_value=engine_zh
        ) as mock_create:
            config1 = OCRConfig(enabled=True, lang="zh")
            config2 = OCRConfig(enabled=True, lang="zh")

            result1 = OCRProcessor.get_shared_engine(config1)
            result2 = OCRProcessor.get_shared_engine(config2)

            assert result1 is result2
            assert mock_create.call_count == 1

    def test_shared_engine_none_config_to_real_config(self):
        """Going from None config to a real config should rebuild."""
        engine_default = MagicMock(name="engine_default")
        engine_zh = MagicMock(name="engine_zh")

        call_count = 0

        def fake_create(config=None):
            nonlocal call_count
            call_count += 1
            if config and config.lang == "zh":
                return engine_zh
            return engine_default

        with patch.object(OCRProcessor, "_create_engine_impl", side_effect=fake_create):
            # First call with no config
            result1 = OCRProcessor.get_shared_engine(None)
            assert result1 is engine_default
            assert call_count == 1

            # Second call with real config should rebuild
            config_zh = OCRConfig(enabled=True, lang="zh")
            result2 = OCRProcessor.get_shared_engine(config_zh)
            assert result2 is engine_zh
            assert call_count == 2


class TestOCRLanguageResolution:
    """ocr.lang must select a recognizer that exists, or fail up front.

    Under RapidOCR 3.9 (default PP-OCRv6) the documented ko/ar/th/latin
    values raised "Unsupported rec.lang_type" on every page, and unknown
    values were silently mapped to the Chinese model.
    """

    @pytest.mark.parametrize(
        ("lang", "expected"),
        [
            ("en", ("en", False)),
            ("zh", ("ch", False)),
            ("ja", ("japan", False)),
            ("zh_tw", ("chinese_cht", False)),
            ("ko", ("korean", True)),
            ("ar", ("arabic", True)),
            ("th", ("th", True)),
            ("latin", ("latin", True)),
            ("KO", ("korean", True)),
        ],
    )
    def test_documented_languages(self, lang: str, expected: tuple[str, bool]):
        from markitai.ocr import resolve_ocr_language

        assert resolve_ocr_language(lang) == expected

    def test_multilingual_iso_code_is_accepted(self):
        from markitai.ocr import resolve_ocr_language

        assert resolve_ocr_language("fr") == ("fr", False)

    def test_ppocrv6_table_matches_the_installed_rapidocr(self):
        """The copied PP-OCRv6 language table must track RapidOCR's own."""
        pytest.importorskip("rapidocr")
        from rapidocr.utils.model_resolver import PP_OCRV6_LANGS

        from markitai.ocr import _PPOCRV6_REC_LANGS

        assert frozenset(PP_OCRV6_LANGS) == _PPOCRV6_REC_LANGS

    def test_unknown_language_is_an_error_not_chinese(self):
        from markitai.ocr import OCRLanguageError, resolve_ocr_language

        with pytest.raises(OCRLanguageError, match="Unsupported ocr.lang 'klingon'"):
            resolve_ocr_language("klingon")

    def test_engine_creation_rejects_unknown_language_before_rapidocr(self):
        pytest.importorskip("rapidocr")
        from markitai.ocr import OCRLanguageError

        with (
            patch("rapidocr.RapidOCR") as rapid,
            pytest.raises(OCRLanguageError),
        ):
            OCRProcessor._create_engine_impl(OCRConfig(enabled=True, lang="xx"))
        rapid.assert_not_called()

    @pytest.mark.parametrize(
        "lang",
        [
            "en",
            "zh",
            "zh_tw",
            "ja",
            "ko",
            "ar",
            "th",
            "latin",
            "fr",
            "cyrillic",
            "eslav",
            "el",
            "devanagari",
            "ta",
            "te",
        ],
    )
    def test_params_resolve_to_a_model_rapidocr_ships(self, lang: str):
        """Resolve the model exactly as RapidOCR's engine init does (no download)."""
        pytest.importorskip("rapidocr")
        from rapidocr.inference_engine.base import FileInfo, InferSession
        from rapidocr.main import DEFAULT_CFG_PATH
        from rapidocr.utils.parse_parameters import ParseParams
        from rapidocr.utils.typings import TaskType

        params = OCRProcessor._language_params(lang)
        cfg = ParseParams.update_batch(ParseParams.load(DEFAULT_CFG_PATH), params)
        rec = cfg.Rec
        rec.lang_type = ParseParams.LangType(TaskType.REC, rec.lang_type)
        info = InferSession.get_model_url(
            FileInfo(
                engine_type=rec.engine_type,
                ocr_version=rec.ocr_version,
                task_type=rec.task_type,
                lang_type=rec.lang_type,
                model_type=rec.model_type,
            )
        )
        assert info["model_dir"].endswith(".onnx")


class _ThresholdRecordingEngine:
    """Stand-in for RapidOCR that mutates thresholds in place like update_params."""

    def __init__(self) -> None:
        import types

        self.text_score = 0.5
        self.text_det = types.SimpleNamespace(
            postprocess_op=types.SimpleNamespace(box_thresh=0.5)
        )
        self.seen_by_full_passes: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    def __call__(self, image, box_thresh=None, text_score=None):
        import time

        if text_score is not None:
            self.text_score = text_score
        if box_thresh is not None:
            self.text_det.postprocess_op.box_thresh = box_thresh
        time.sleep(0.002)  # widen the race window
        if box_thresh is None:
            with self._lock:
                self.seen_by_full_passes.append(
                    (self.text_score, self.text_det.postprocess_op.box_thresh)
                )
        return MagicMock(txts=None, scores=None, boxes=None)


class TestSparseTileThreadSafety:
    """Concurrent pages must never run with (or keep) the tile thresholds."""

    def teardown_method(self):
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_concurrent_tile_fallback_leaves_shared_thresholds_intact(
        self, ocr_config: OCRConfig
    ):
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np

        engine = _ThresholdRecordingEngine()
        OCRProcessor._global_engine = engine
        OCRProcessor._global_config = ocr_config
        processor = OCRProcessor(ocr_config)
        # Every full pass is empty, so each call also runs the tiled retry
        image = np.zeros((1200, 1600, 3), dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: processor.recognize_numpy(image), range(24)))

        assert engine.text_score == 0.5
        assert engine.text_det.postprocess_op.box_thresh == 0.5
        assert engine.seen_by_full_passes
        assert set(engine.seen_by_full_passes) == {(0.5, 0.5)}


class TestOCRParagraphLayout:
    """OCR lines separated by a large vertical gap start a new paragraph."""

    def teardown_method(self):
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_large_gap_becomes_a_blank_line(self, ocr_config: OCRConfig):
        import numpy as np

        def box(top: int) -> np.ndarray:
            return np.array([[10, top], [300, top], [300, top + 20], [10, top + 20]])

        mock_engine = MagicMock(
            return_value=MagicMock(
                txts=("First line", "same paragraph", "New paragraph"),
                scores=(0.9, 0.9, 0.9),
                boxes=(box(0), box(26), box(90)),
            )
        )
        OCRProcessor._global_engine = mock_engine
        OCRProcessor._global_config = ocr_config

        result = OCRProcessor(ocr_config).recognize_numpy(
            np.zeros((200, 400, 3), dtype=np.uint8)
        )

        assert result.text == "First line\nsame paragraph\n\nNew paragraph"

    def test_markdown_pass_does_not_mutate_engine_flags(
        self, ocr_config: OCRConfig, tmp_path: Path
    ):
        """return_word_box/return_single_char_box would stick to the shared engine."""
        test_image = tmp_path / "t.png"
        test_image.write_bytes(b"x")
        mock_result = MagicMock(txts=["Heading"], scores=[0.99], boxes=[])
        mock_result.to_markdown.return_value = "Heading"
        mock_engine = MagicMock(return_value=mock_result)
        OCRProcessor._global_engine = mock_engine
        OCRProcessor._global_config = ocr_config

        OCRProcessor(ocr_config).recognize_to_markdown(test_image)

        assert mock_engine.call_args.kwargs == {}


def _scanned_page_png(lines: list[str]) -> bytes:
    """A white page image with black text; ``""`` entries leave a paragraph gap."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    try:
        font: Any = ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", 36
            )
        except OSError:
            font = ImageFont.load_default(size=36)
    y = 80
    for line in lines:
        if not line:
            y += 90
            continue
        draw.text((80, y), line, fill="black", font=font)
        y += 50
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.slow
class TestRealEngineOnScannedPdf:
    """End to end with the real RapidOCR engine (slow: loads ONNX models)."""

    def teardown_method(self):
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_mixed_pdf_keeps_structure_paragraphs_and_thresholds(self, tmp_path: Path):
        pytest.importorskip("rapidocr")
        import pymupdf

        from markitai.config import MarkitaiConfig
        from markitai.converter.pdf import PdfConverter

        doc = pymupdf.open()
        native = doc.new_page()
        native.insert_text((72, 90), "Native Chapter Heading", fontsize=24)
        native.insert_text(
            (72, 130),
            "This page has a real text layer with enough words to stay native.",
            fontsize=11,
        )
        for number in (2, 3):
            page = doc.new_page()
            page.insert_image(
                page.rect,
                stream=_scanned_page_png(
                    [
                        f"Scanned page number {number}",
                        "The quick brown fox jumps over the lazy dog.",
                        "",
                        "Second paragraph after a gap.",
                    ]
                ),
            )
        pdf_file = tmp_path / "scan.pdf"
        doc.save(pdf_file)
        doc.close()

        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.ocr.lang = "en"
        result = PdfConverter(config).convert(pdf_file, tmp_path / "out")

        markdown = result.markdown
        assert "# Native Chapter Heading" in markdown
        assert "Scanned page number 2" in markdown
        assert "Scanned page number 3" in markdown
        assert "lazy dog.\n\nSecond paragraph" in markdown

        engine = OCRProcessor._global_engine
        assert engine.text_score == 0.5
        assert engine.text_det.postprocess_op.box_thresh == 0.5


class TestUpsideDownDetection:
    """A page is turned over only when both of its halves read upside down.

    The eight widest lines were asked, wherever they sat: a page pasted up
    from two halves facing opposite ways (or a wide flipped banner) could
    turn the whole page over at a 60% vote.
    """

    def teardown_method(self):
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    @staticmethod
    def _detect(
        ocr_config: OCRConfig,
        flipped_rows: set[int],
        widths: list[int] | None = None,
        unsure_rows: frozenset[int] = frozenset(),
    ) -> bool:
        """Lines at rows 0, 1, ... (eight by default); *flipped_rows* read as 180°."""
        import types

        import numpy as np

        boxes = [
            np.array([[0, r * 50], [w, r * 50], [w, r * 50 + 20], [0, r * 50 + 20]])
            for r, w in enumerate(widths or [400] * 8)
        ]

        def classify(crops: list[Any]) -> Any:
            rows = [int(crop[0][1]) // 50 for crop in crops]
            return types.SimpleNamespace(
                cls_res=[
                    (
                        "180" if r in flipped_rows else "0",
                        0.6 if r in unsure_rows else 0.99,
                    )
                    for r in rows
                ]
            )

        engine = MagicMock()
        engine.text_cls = classify
        OCRProcessor._global_engine = engine
        OCRProcessor._global_config = ocr_config

        crop_module = types.ModuleType("rapidocr.utils.process_img")
        crop_module.get_rotate_crop_image = lambda _image, poly: poly  # type: ignore[attr-defined]
        raw = MagicMock(boxes=boxes, img=np.zeros((400, 400, 3), dtype=np.uint8))
        with patch.dict(sys.modules, {"rapidocr.utils.process_img": crop_module}):
            return OCRProcessor(ocr_config)._is_upside_down(raw)

    def test_whole_page_flipped(self, ocr_config: OCRConfig):
        assert self._detect(ocr_config, set(range(8))) is True

    def test_one_line_misread_still_flips(self, ocr_config: OCRConfig):
        assert self._detect(ocr_config, set(range(8)) - {6}) is True

    def test_halves_facing_opposite_ways_are_not_flipped(self, ocr_config: OCRConfig):
        assert self._detect(ocr_config, {0, 1, 2, 3}) is False
        assert self._detect(ocr_config, {4, 5, 6, 7}) is False
        # The widest lines all sit in the flipped top half: sampling only
        # the widest lines of the page asked five of them and three others
        top_wide = [900] * 5 + [300] * 5
        assert self._detect(ocr_config, {0, 1, 2, 3, 4}, top_wide) is False

    def test_answers_weigh_as_much_as_the_classifier_is_sure(
        self, ocr_config: OCRConfig
    ):
        """The real classifier is often unsure on a turned-over scan."""
        unsure = frozenset({1, 6})
        assert self._detect(ocr_config, {0, 2, 3, 4, 5, 7}, None, unsure) is True
        # Unanimous but unsure (wide sans-serif faces such as Verdana or
        # DejaVu Sans): counting only sure answers left these unturned
        unsure = frozenset({0, 1, 2, 4, 5})
        assert self._detect(ocr_config, set(range(8)), None, unsure) is True
        # ... while two sure 0° answers outweigh six unsure 180° ones
        unsure = frozenset({0, 1, 2, 4, 5, 6})
        assert self._detect(ocr_config, set(range(8)) - {3, 7}, None, unsure) is False

    def test_a_sixty_percent_vote_no_longer_flips(self, ocr_config: OCRConfig):
        assert self._detect(ocr_config, {0, 1, 2, 4, 5}) is False

    def test_the_sample_covers_both_halves(self):
        import numpy as np

        from markitai.ocr import _orientation_sample

        # The widest lines all sit in the top half
        polygons = [
            np.array([[0, y], [w, y], [w, y + 20], [0, y + 20]], dtype=np.float32)
            for y, w in [(0, 900), (30, 900), (60, 900), (90, 900), (120, 900)]
            + [(600, 100), (630, 100), (660, 100), (690, 100)]
        ]

        top, bottom = _orientation_sample(polygons)

        assert len(top) == 4
        assert len(bottom) == 4
        assert all(float(poly[:, 1].mean()) > 400 for poly in bottom)


class TestEngineGateWriterPreference:
    """A waiting exclusive holder keeps new shared holders out.

    Otherwise a steady stream of page recognitions could starve the
    sparse-tile retry, which needs the engine to itself.
    """

    def test_new_readers_wait_behind_a_waiting_writer(self):
        from markitai.ocr import _EngineGate

        gate = _EngineGate()
        order: list[str] = []
        reader_in = threading.Event()
        release_reader = threading.Event()

        def first_reader() -> None:
            with gate.shared():
                reader_in.set()
                release_reader.wait(5)
            order.append("first reader out")

        def writer() -> None:
            with gate.exclusive():
                order.append("writer")

        def late_reader() -> None:
            with gate.shared():
                order.append("late reader")

        threads = [threading.Thread(target=first_reader)]
        threads[0].start()
        assert reader_in.wait(5)
        threads.append(threading.Thread(target=writer))
        threads[1].start()
        # Wait until the writer is queued before the late reader arrives
        for _ in range(500):
            if gate._writers_waiting:
                break
            threading.Event().wait(0.01)
        assert gate._writers_waiting == 1
        threads.append(threading.Thread(target=late_reader))
        threads[2].start()
        threading.Event().wait(0.05)
        assert order == []  # the late reader is held back, not let in

        release_reader.set()
        for thread in threads:
            thread.join(5)

        assert order.index("writer") < order.index("late reader")


@pytest.mark.slow
class TestRealEngineOrientation:
    """The real line classifier on pasted-up and turned-over pages (slow)."""

    def teardown_method(self):
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    @staticmethod
    def _half(lines: list[str], height: int = 877) -> Any:
        from PIL import Image, ImageDraw, ImageFont

        try:
            font: Any = ImageFont.truetype("DejaVuSans.ttf", 36)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial.ttf", 36
                )
            except OSError:
                font = ImageFont.load_default(size=36)
        image = Image.new("RGB", (1240, height), "white")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text((60, 80 + index * 60), line, fill="black", font=font)
        return image

    def test_halves_facing_opposite_ways_keep_the_upright_half_first(self):
        pytest.importorskip("rapidocr")
        import numpy as np
        from PIL import Image

        top = self._half([f"Upright line {n} of the top half." for n in range(1, 7)])
        # The flipped half has the wider lines: sampling only the widest
        # lines of the page asked mostly flipped ones and turned it over
        bottom = self._half(
            [
                f"Flipped line {n} of the bottom half, set much wider."
                for n in range(1, 7)
            ]
        ).rotate(180)
        page = Image.new("RGB", (1240, 1754), "white")
        page.paste(top, (0, 0))
        page.paste(bottom, (0, 877))

        processor = OCRProcessor(OCRConfig(enabled=True, lang="en"))
        raw = processor._run_engine(np.asarray(page))

        assert processor._is_upside_down(raw) is False
        text = processor._build_ocr_result(raw).text
        assert text.startswith("Upright line 1 of the top half.")

    def test_a_page_scanned_upside_down_is_still_turned_over(self):
        pytest.importorskip("rapidocr")
        import numpy as np

        lines = [f"Whole page line {n} was scanned upside down." for n in range(1, 9)]
        page = self._half(lines, height=1754).rotate(180)

        processor = OCRProcessor(OCRConfig(enabled=True, lang="en"))
        raw = processor._run_engine(np.asarray(page))

        assert processor._is_upside_down(raw) is True
        # Top line first: the box lowest in the scan leads the text.
        # RapidOCR's own line classifier leaves some of these lines unturned
        # and misreads them (with DejaVu Sans, all of them), so the order is
        # checked on the geometry, and on whichever lines stayed legible.
        text = processor._build_ocr_result(raw).text
        placed = [
            (float(np.asarray(box)[:, 1].mean()), str(txt).strip())
            for box, txt in zip(raw.boxes, raw.txts)
            if str(txt).strip()
        ]
        assert text.splitlines()[0] == max(placed)[1]
        numbers = [int(n) for n in re.findall(r"line (\d) was scanned", text)]
        assert numbers == sorted(numbers)


@pytest.mark.slow
class TestRealEngineMarginStamp:
    """A vertical arXiv stamp in the margin leaves the body lines alone (slow)."""

    def teardown_method(self):
        OCRProcessor._global_engine = None
        OCRProcessor._global_config = None

    def test_stamp_is_set_apart_and_paragraphs_survive(self):
        pytest.importorskip("rapidocr")
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        try:
            font: Any = ImageFont.truetype("DejaVuSans.ttf", 30)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial.ttf", 30
                )
            except OSError:
                font = ImageFont.load_default(size=30)
        page = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(page)
        y = 200
        for para in range(1, 4):
            for n in range(1, 5):
                draw.text(
                    (160, y),
                    f"Paragraph {para} line {n} of the body text.",
                    fill="black",
                    font=font,
                )
                y += 45
            y += 60
        stamp = Image.new("RGB", (1100, 50), "white")
        ImageDraw.Draw(stamp).text(
            (0, 5), "arXiv:2401.01234v1 [cs.CL] 2 Jan 2024", fill="black", font=font
        )
        page.paste(stamp.rotate(90, expand=True), (40, 250))

        result = OCRProcessor(OCRConfig(enabled=True, lang="en")).recognize_numpy(
            np.asarray(page)
        )

        body = result.text.split("\n\n")
        assert body[:3] == [
            "\n".join(
                f"Paragraph {para} line {n} of the body text." for n in range(1, 5)
            )
            for para in range(1, 4)
        ]
        assert "arXiv" in "".join(body[3:])
