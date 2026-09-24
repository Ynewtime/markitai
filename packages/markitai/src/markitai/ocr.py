"""OCR module using RapidOCR.

RapidOCR is an *optional* backend (the ``ocr`` extra), not a core dependency:
it pulls opencv-python and its own ONNX models for a feature only scanned
documents need. Every path that can hit the missing backend routes its
message through :data:`OCR_INSTALL_HINT` so the user is told exactly one
command, in exactly one wording, wherever they hit the wall.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import DEFAULT_OCR_SAMPLE_PAGES, DEFAULT_RENDER_DPI
from markitai.ocr_layout import layout_ocr_text
from markitai.utils.errors import MissingDependencyError, extra_install_command

if TYPE_CHECKING:
    from markitai.config import OCRConfig

#: The single actionable command for enabling OCR. Deliberately not
#: "uv add rapidocr": markitai is normally installed as an isolated tool, where
#: `uv add` would put the wheel in a project venv the tool cannot import from.
OCR_INSTALL_HINT = extra_install_command("ocr")


class OCRBackendMissing(MissingDependencyError):
    """The optional OCR backend is not installed.

    Distinct from a runtime OCR failure: callers may degrade gracefully when
    the engine runs and fails, but must not swallow this one. The user asked
    for OCR and can get it with a single command, so telling them the
    conversion succeeded would be a lie.

    Inherits :class:`MissingDependencyError` (still an ``ImportError``): the
    message is self-explanatory, so user-facing rendering drops the class
    name.
    """


def is_ocr_available() -> bool:
    """Return whether the optional OCR backend can be imported.

    Metadata-only on purpose: importing rapidocr drags in opencv, whose large
    dylib costs seconds of first-run signature validation. Callers that only
    need to *phrase a message* must never pay that.
    """
    import importlib.util
    import sys

    if "rapidocr" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("rapidocr") is not None
    except (ImportError, ValueError):
        return False


# Minimum Latin letters before the vowel-ratio test is meaningful; short
# labels and non-Latin text below this floor are never flagged.
_GARBLED_MIN_LATIN_LETTERS = 30

# CJK character ranges (Han, Kana, Hangul). A page dominated by CJK text
# needs no vowel check -- the vowel-ratio heuristic only applies to Latin.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
)


def _is_cjk_char(ch: str) -> bool:
    """Return True if the character falls in a CJK Unicode range."""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def is_likely_garbled(text: str) -> bool:
    """Detect substitution-cipher / broken-cmap garbling in extracted text.

    Real Latin-script text has a vowel ratio of roughly 30-45%, but a
    broken ToUnicode cmap (a real pymupdf failure mode) almost always maps
    the original A/E/I/O/U onto non-vowel letters, driving the apparent
    vowel ratio to near zero. Only Latin (ASCII alphabetic) characters are
    tested: text without enough Latin letters to judge, or dominated by
    CJK characters, is treated as fine.

    Args:
        text: Extracted text (typically a full page) to check

    Returns:
        True if the text looks garbled (unreadable despite being present)
    """
    letters = 0
    vowels = 0
    cjk = 0
    for ch in text:
        if ch.isascii() and ch.isalpha():
            letters += 1
            if ch.lower() in "aeiou":
                vowels += 1
        elif _is_cjk_char(ch):
            cjk += 1
    if letters < _GARBLED_MIN_LATIN_LETTERS:
        return False
    if cjk >= letters:
        # Mostly-CJK page: the Latin letters are a minority (codes,
        # abbreviations) and the vowel test would be meaningless.
        return False
    # Vowel ratio < 20% is well outside any natural Latin-script language.
    return vowels * 5 < letters


@dataclass
class OCRResult:
    """Result of OCR processing."""

    text: str
    confidence: float
    boxes: list[list[float]]


class OCRError(RuntimeError):
    """OCR ran and could not read the input (undecodable image, engine error).

    Converters raise it instead of writing the failure into the Markdown:
    an error message in the body is not content, and reporting the item as
    converted would hide that nothing was read.
    """


class OCRLanguageError(ValueError):
    """``ocr.lang`` names a language no installed RapidOCR model reads."""


# markitai's short codes (the documented ``ocr.lang`` values) and common
# spellings -> RapidOCR recognition language.
_LANG_ALIASES = {
    "zh": "ch",
    "zh_cn": "ch",
    "zh-cn": "ch",
    "cn": "ch",
    "zh_tw": "chinese_cht",
    "zh-tw": "chinese_cht",
    "cht": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
    "ar": "arabic",
}

# Languages the default PP-OCRv6 multilingual model does not read. RapidOCR
# 3.9 ships a PP-OCRv5 mobile recognizer for each, selected explicitly;
# without that, every page failed with "Unsupported rec.lang_type".
_PPOCRV5_REC_LANGS = frozenset(
    {
        "korean",
        "arabic",
        "th",
        "latin",
        "cyrillic",
        "eslav",
        "el",
        "devanagari",
        "ta",
        "te",
    }
)

# Languages the default PP-OCRv6 multilingual recognizer reads: RapidOCR
# 3.9's ``rapidocr.utils.model_resolver.PP_OCRV6_LANGS`` (a unit test keeps
# the two in step). Copied rather than imported so validating ocr.lang --
# `markitai doctor` does it -- never imports rapidocr, which drags in cv2.
_PPOCRV6_REC_LANGS = frozenset(
    {
        "ch", "chinese_cht", "en", "japan", "af", "az", "bs", "ca", "cs",
        "cy", "da", "de", "es", "et", "eu", "fi", "fr", "ga", "gl", "hr",
        "hu", "id", "is", "it", "ku", "la", "lb", "lt", "lv", "mi", "ms",
        "mt", "nl", "no", "oc", "pl", "pt", "qu", "rm", "ro", "rs_latin",
        "sk", "sl", "sq", "sv", "sw", "tl", "tr", "uz", "vi", "french",
        "german",
    }
)  # fmt: skip


def resolve_ocr_language(lang: str) -> tuple[str, bool]:
    """Map an ``ocr.lang`` value to a RapidOCR recognition language.

    Args:
        lang: The configured language (``en``, ``zh``, ``ko``, ``fr``, ...)

    Returns:
        Tuple of (RapidOCR ``Rec.lang_type`` value, whether it needs the
        PP-OCRv5 recognizer instead of the default PP-OCRv6 one)

    Raises:
        OCRLanguageError: No RapidOCR model reads this language. Silently
            falling back to the Chinese model, as before, turned a typo or
            an unsupported script into confidently wrong text.
    """
    key = lang.strip().lower()
    rec_lang = _LANG_ALIASES.get(key, key)
    if rec_lang in _PPOCRV5_REC_LANGS:
        return rec_lang, True
    if rec_lang in _PPOCRV6_REC_LANGS:
        return rec_lang, False
    # Short enough to survive the error truncation in reports and --json
    raise OCRLanguageError(
        f"Unsupported ocr.lang {lang!r}: use en, zh, zh_tw, ja, ko, ar, th, "
        "latin, or a code such as fr/de/es (see the OCR configuration docs)"
    )


class _EngineGate:
    """Shared/exclusive gate around the process-wide RapidOCR engine.

    Plain recognition calls only read the engine's thresholds and run
    concurrently (shared). The sparse-tile retry must lower them, and
    RapidOCR's ``__call__`` writes them into the engine in place; it holds
    the gate exclusively so no concurrent page is recognized with the
    lowered thresholds, and restores them before releasing. Waiting
    exclusive holders block new shared ones, so a retry cannot starve.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def shared(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if not self._readers:
                    self._cond.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


# Page orientation: the widest lines of each half of the page are asked. Only
# confident answers count, and at least this share of the sample must give
# one; the page counts as upside down when this share of those say 180°, and
# they outvote 0° in each half. A pasted-up page whose two halves face
# opposite ways must not be turned over as a whole.
_ORIENTATION_SAMPLE = 8
_ORIENTATION_MIN_CONFIDENT = 0.5
_ORIENTATION_SHARE = 0.8
_ORIENTATION_MIN_SCORE = 0.8


def _orientation_vote(label: Any) -> bool | None:
    """A classifier answer: True for 180°, False for 0°, None when unsure."""
    try:
        angle, score = str(label[0]), float(label[1])
    except (TypeError, ValueError, IndexError):
        return None
    if score < _ORIENTATION_MIN_SCORE or angle not in ("0", "180"):
        return None
    return angle == "180"


def _orientation_sample(polygons: list[Any]) -> list[list[Any]]:
    """The widest line polygons of the top and of the bottom half of the text.

    Up to half of ``_ORIENTATION_SAMPLE`` from each half; a half with fewer
    lines leaves its share to the other.
    """

    def width(poly: Any) -> float:
        return float(poly[:, 0].max() - poly[:, 0].min())

    centers = [float(poly[:, 1].mean()) for poly in polygons]
    middle = (min(centers) + max(centers)) / 2
    top = sorted(
        (p for p, c in zip(polygons, centers) if c < middle), key=width, reverse=True
    )
    bottom = sorted(
        (p for p, c in zip(polygons, centers) if c >= middle), key=width, reverse=True
    )
    share = _ORIENTATION_SAMPLE // 2
    top_count = min(len(top), max(share, _ORIENTATION_SAMPLE - len(bottom)))
    bottom_count = min(len(bottom), _ORIENTATION_SAMPLE - top_count)
    return [top[:top_count], bottom[:bottom_count]]


def _polygon_bounds(box: Any) -> tuple[float, float, float, float] | None:
    """Axis-aligned bound of one RapidOCR polygon, or None when malformed."""
    try:
        points = box.tolist() if hasattr(box, "tolist") else list(box)
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _capture_engine_thresholds(engine: Any) -> dict[str, Any]:
    """Snapshot RapidOCR's mutable detection thresholds before a scoped call.

    RapidOCR.__call__ writes ``text_score`` and the detector's ``box_thresh``
    in place and never restores them; snapshotting lets a tuned per-tile pass
    leave the shared singleton exactly as it found it. Attribute paths are
    probed defensively so a RapidOCR layout change degrades to a no-op.
    """
    snapshot: dict[str, Any] = {}
    if hasattr(engine, "text_score"):
        snapshot["text_score"] = engine.text_score
    postprocess = getattr(getattr(engine, "text_det", None), "postprocess_op", None)
    if postprocess is not None and hasattr(postprocess, "box_thresh"):
        snapshot["box_thresh"] = postprocess.box_thresh
    return snapshot


def _restore_engine_thresholds(engine: Any, snapshot: dict[str, Any]) -> None:
    """Restore thresholds captured by :func:`_capture_engine_thresholds`."""
    if "text_score" in snapshot:
        engine.text_score = snapshot["text_score"]
    if "box_thresh" in snapshot:
        engine.text_det.postprocess_op.box_thresh = snapshot["box_thresh"]


class OCRProcessor:
    """OCR processor using RapidOCR.

    Implements global singleton pattern for engine to avoid cold start
    delays on subsequent calls. ONNX Runtime initialization is expensive
    (1-8s depending on backend), so sharing the engine across instances
    significantly improves batch processing performance.
    """

    # Global singleton engine with thread-safe initialization
    _global_engine: Any = None
    _global_config: OCRConfig | None = None
    _init_lock = threading.Lock()
    # Guards the shared engine's mutable thresholds (see _EngineGate)
    _gate = _EngineGate()

    def __init__(self, config: OCRConfig | None = None) -> None:
        """
        Initialize OCR processor.

        Args:
            config: Optional OCR configuration
        """
        self.config = config
        self._engine = None

    @classmethod
    def _config_fingerprint(cls, config: OCRConfig | None) -> str:
        """Return a hashable fingerprint for the given OCR config.

        Used to detect when the config has changed so that the shared
        engine can be rebuilt.
        """
        if config is None:
            return ""
        return config.model_dump_json()

    @classmethod
    def get_shared_engine(cls, config: OCRConfig | None = None) -> Any:
        """Get or create global singleton engine (thread-safe).

        Uses double-checked locking for thread-safe lazy initialization.
        The engine is shared across all OCRProcessor instances to avoid
        repeated ONNX Runtime cold starts. When the config changes
        (e.g., language switch), the engine is rebuilt automatically.

        Args:
            config: Optional OCR configuration for engine creation

        Returns:
            Shared RapidOCR engine instance
        """
        new_fp = cls._config_fingerprint(config)
        if (
            cls._global_engine is None
            or cls._config_fingerprint(cls._global_config) != new_fp
        ):
            with cls._init_lock:
                if (
                    cls._global_engine is None
                    or cls._config_fingerprint(cls._global_config) != new_fp
                ):
                    logger.debug("Creating global shared OCR engine")
                    # Create first, assign after: if creation raises, the old
                    # engine/config pair stays consistent instead of serving a
                    # stale engine under the new config's fingerprint
                    cls._global_engine = cls._create_engine_impl(config)
                    cls._global_config = config
        return cls._global_engine

    @classmethod
    def preheat(cls, config: OCRConfig | None = None) -> Any:
        """Preheat OCR engine at application startup.

        Call this during batch processing initialization to eliminate
        cold start delay from the first actual OCR call. Performs a
        dummy inference to complete GPU compilation (DirectML/CUDA).

        Args:
            config: Optional OCR configuration

        Returns:
            Preheated RapidOCR engine instance
        """
        import numpy as np

        logger.info("Preheating OCR engine...")
        engine = cls.get_shared_engine(config)

        # Execute dummy inference to complete GPU compilation
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            engine(dummy_image)
            logger.debug("OCR engine preheat completed")
        except Exception as e:
            # Ignore errors from dummy image (empty image may not be recognized)
            logger.debug(f"OCR preheat inference ignored: {e}")

        return engine

    @classmethod
    def _create_engine_impl(cls, config: OCRConfig | None = None) -> Any:
        """Create RapidOCR engine with configuration (implementation).

        Args:
            config: Optional OCR configuration

        Returns:
            New RapidOCR engine instance
        """
        if not is_ocr_available():
            raise OCRBackendMissing(
                f"--ocr requires the optional OCR backend (RapidOCR), "
                f"which is not installed. {OCR_INSTALL_HINT}"
            )

        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise OCRBackendMissing(
                f"--ocr requires the optional OCR backend (RapidOCR), "
                f"which failed to import: {e}. {OCR_INSTALL_HINT}"
            ) from e

        # Build params
        params: dict[str, Any] = {
            # Empty first-pass detections are handled by the tiled fallback
            # below, so RapidOCR's warning would be noisy and misleading.
            "Global.log_level": "error",
        }

        # Set language if configured. Validated before RapidOCR sees it: an
        # unknown value used to fall back to the Chinese model silently.
        if config and config.lang:
            params.update(cls._language_params(config.lang))

        try:
            return RapidOCR(params=params)
        except (ValueError, TypeError) as e:
            # RapidOCR rejects a language/model combination at construction
            # ("Unsupported rec.lang_type"); say which setting caused it.
            lang = config.lang if config else None
            raise OCRLanguageError(
                f"RapidOCR cannot build a recognizer for ocr.lang={lang!r}: {e}"
            ) from e

    @staticmethod
    def _language_params(lang: str) -> dict[str, Any]:
        """RapidOCR params selecting the recognizer for ``lang``.

        Raises:
            OCRLanguageError: ``lang`` is not a language RapidOCR reads.
        """
        from rapidocr import LangRec

        rec_lang, needs_v5 = resolve_ocr_language(lang)
        try:
            lang_value: Any = LangRec(rec_lang)
        except ValueError:
            # PP-OCRv6 reads many ISO codes LangRec has no member for (fr,
            # de, ...); RapidOCR accepts those as plain strings.
            lang_value = rec_lang
        params: dict[str, Any] = {"Rec.lang_type": lang_value}
        if needs_v5:
            from rapidocr import ModelType, OCRVersion

            params["Rec.ocr_version"] = OCRVersion.PPOCRV5
            params["Rec.model_type"] = ModelType.MOBILE
        return params

    @property
    def engine(self) -> Any:
        """Get or create the RapidOCR engine.

        Uses the global shared engine by default to avoid cold start delays.
        Falls back to instance-specific engine only if configs differ.
        """
        # Use global shared engine for better performance
        return self.get_shared_engine(self.config)

    def _run_engine(self, image: Any) -> Any:
        """One full-image recognition pass with the engine's own thresholds."""
        engine = self.engine
        with self._gate.shared():
            return engine(image)

    def _build_ocr_result(self, raw_result: Any) -> OCRResult:
        """Build OCRResult from raw RapidOCR engine output.

        Extracts texts, scores, and boxes from the engine result, lays the
        text out in reading order, and calculates average confidence.

        Args:
            raw_result: Raw result from RapidOCR engine call

        Returns:
            OCRResult with recognized text and metadata
        """
        # Extract text from result (RapidOCR returns union type with incomplete stubs)
        # Use 'is not None' to avoid numpy array boolean ambiguity
        texts = list(raw_result.txts) if raw_result.txts is not None else []
        scores = list(raw_result.scores) if raw_result.scores is not None else []
        boxes = list(raw_result.boxes) if raw_result.boxes is not None else []

        full_text = layout_ocr_text(
            [str(text) for text in texts],
            [
                _polygon_bounds(boxes[index]) if index < len(boxes) else None
                for index in range(len(texts))
            ],
            upside_down=self._is_upside_down(raw_result),
        )
        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        logger.debug(
            f"OCR completed: {len(texts)} text blocks, "
            f"avg confidence: {avg_confidence:.2f}"
        )

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=[
                box.tolist() if hasattr(box, "tolist") else list(box) for box in boxes
            ],
        )

    def _is_upside_down(self, raw_result: Any) -> bool:
        """Whether the page was scanned rotated by 180 degrees.

        RapidOCR's line classifier turns each upside-down line the right way
        up before recognition, so the text reads fine, but the boxes keep
        the scan's coordinates and come out bottom line first. The same
        classifier, asked about the widest lines of the top and the bottom
        half, tells the page's orientation apart from the odd flipped label
        and from a page pasted up from two halves facing opposite ways.
        """
        image = getattr(raw_result, "img", None)
        boxes = raw_result.boxes
        classify = getattr(self.engine, "text_cls", None)
        if image is None or boxes is None or classify is None or len(boxes) < 2:
            return False
        try:
            import numpy as np
            from rapidocr.utils.process_img import get_rotate_crop_image

            polygons = [np.asarray(box, dtype=np.float32) for box in boxes]
            halves = _orientation_sample(polygons)
            sample = [poly for half in halves for poly in half]
            crops = [get_rotate_crop_image(image, poly.copy()) for poly in sample]
            with self._gate.shared():
                labels = list(classify(crops).cls_res or [])
        except Exception as e:  # orientation is a refinement; never fail OCR
            logger.debug(f"OCR orientation check skipped: {e}")
            return False
        if not labels or len(labels) != len(sample):
            return False
        votes = [_orientation_vote(label) for label in labels]
        confident = [vote for vote in votes if vote is not None]
        if len(confident) < max(2, len(votes) * _ORIENTATION_MIN_CONFIDENT):
            return False
        if sum(confident) < len(confident) * _ORIENTATION_SHARE:
            return False
        for half in (votes[: len(halves[0])], votes[len(halves[0]) :]):
            flipped = sum(vote is True for vote in half)
            upright = sum(vote is False for vote in half)
            if (flipped or upright) and flipped <= upright:
                return False
        return True

    @staticmethod
    def _load_rgb_array(image_path: Path) -> Any:
        """Decode an image with EXIF orientation applied for fallback OCR."""
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(image_path) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))

    @staticmethod
    def _box_bounds(
        box: Any, fallback: tuple[int, int, int, int]
    ) -> tuple[float, float, float, float]:
        """Return a tolerant axis-aligned bound for one RapidOCR polygon."""
        try:
            points = box.tolist() if hasattr(box, "tolist") else list(box)
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            if xs and ys:
                return min(xs), min(ys), max(xs), max(ys)
        except (TypeError, ValueError, IndexError):
            pass
        return (
            float(fallback[0]),
            float(fallback[1]),
            float(fallback[2]),
            float(fallback[3]),
        )

    @staticmethod
    def _boxes_overlap(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> bool:
        """Detect duplicate text boxes produced in overlapping image tiles."""
        ix = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
        iy = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
        intersection = ix * iy
        if intersection <= 0:
            return False
        left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
        right_area = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
        return intersection / min(left_area, right_area) >= 0.35

    def _recognize_sparse_tiles(self, image_array: Any) -> OCRResult:
        """Retry an empty detection using overlapping document-image tiles.

        RapidOCR scales the shorter image side for detection. On a mostly
        empty screenshot this can make a small caption only a few pixels high,
        even though it is perfectly legible. Overlapping tiles are the common
        document-OCR remedy: each tile gives small text enough effective
        resolution without blindly enlarging an already large full image.
        """
        import numpy as np

        if not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_array)}")
        if image_array.ndim < 2:
            return OCRResult(text="", confidence=0.0, boxes=[])

        height, width = image_array.shape[:2]
        if width < 320 and height < 320:
            return OCRResult(text="", confidence=0.0, boxes=[])

        # Keep tiles near a document-friendly 500-900 px, capped to avoid a
        # pathological number of inferences for very large scans.
        columns = min(4, max(1, math.ceil(width / 800)))
        rows = min(4, max(1, math.ceil(height / 600)))
        if columns == 1 and rows == 1:
            return OCRResult(text="", confidence=0.0, boxes=[])

        overlap_x = max(16, round(math.ceil(width / columns) * 0.06))
        overlap_y = max(16, round(math.ceil(height / rows) * 0.06))

        # RapidOCR's engine is a process-wide singleton and __call__ mutates
        # its box_thresh/text_score in place (update_params), never restoring
        # them. Passing the low tile thresholds would otherwise leak into every
        # later full-image pass across the whole batch. Snapshot and restore,
        # holding the engine exclusively: PDF pages are recognized in a
        # thread pool, and a concurrent page must neither run with the tile
        # thresholds nor have its own snapshot restored over ours.
        engine = self.engine
        with self._gate.exclusive():
            blocks = self._recognize_tiles_locked(
                engine, image_array, rows, columns, overlap_x, overlap_y
            )

        blocks.sort(key=lambda block: (block["bounds"][1], block["bounds"][0]))
        if not blocks:
            return OCRResult(text="", confidence=0.0, boxes=[])
        return OCRResult(
            text=layout_ocr_text(
                [block["text"] for block in blocks],
                [block["bounds"] for block in blocks],
            ),
            confidence=sum(block["score"] for block in blocks) / len(blocks),
            boxes=[list(block["bounds"]) for block in blocks],
        )

    def _recognize_tiles_locked(
        self,
        engine: Any,
        image_array: Any,
        rows: int,
        columns: int,
        overlap_x: int,
        overlap_y: int,
    ) -> list[dict[str, Any]]:
        """Run the tiled pass; the caller holds the engine gate exclusively."""
        height, width = image_array.shape[:2]
        tile_width = math.ceil(width / columns)
        tile_height = math.ceil(height / rows)
        blocks: list[dict[str, Any]] = []
        saved_thresholds = _capture_engine_thresholds(engine)
        try:
            for row in range(rows):
                for column in range(columns):
                    x0 = max(0, column * tile_width - overlap_x)
                    y0 = max(0, row * tile_height - overlap_y)
                    x1 = min(width, (column + 1) * tile_width + overlap_x)
                    y1 = min(height, (row + 1) * tile_height + overlap_y)
                    tile = image_array[y0:y1, x0:x1]
                    raw: Any = engine(tile, box_thresh=0.35, text_score=0.45)
                    texts = list(raw.txts) if raw.txts is not None else []
                    scores = list(raw.scores) if raw.scores is not None else []
                    boxes = list(raw.boxes) if raw.boxes is not None else []

                    for index, raw_text in enumerate(texts):
                        text = str(raw_text).strip()
                        if not text:
                            continue
                        score = float(scores[index]) if index < len(scores) else 0.0
                        local = self._box_bounds(
                            boxes[index] if index < len(boxes) else None,
                            (0, 0, x1 - x0, y1 - y0),
                        )
                        bounds = (
                            local[0] + x0,
                            local[1] + y0,
                            local[2] + x0,
                            local[3] + y0,
                        )
                        normalized = " ".join(text.casefold().split())
                        duplicate = next(
                            (
                                block
                                for block in blocks
                                if block["normalized"] == normalized
                                and self._boxes_overlap(block["bounds"], bounds)
                            ),
                            None,
                        )
                        candidate = {
                            "text": text,
                            "score": score,
                            "bounds": bounds,
                            "normalized": normalized,
                        }
                        if duplicate is None:
                            blocks.append(candidate)
                        elif score > duplicate["score"]:
                            duplicate.update(candidate)
        finally:
            _restore_engine_thresholds(engine, saved_thresholds)
        return blocks

    def recognize(self, image_path: Path | str) -> OCRResult:
        """Perform OCR on an image file, retrying sparse layouts by tile."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug(f"Running OCR on: {image_path.name}")
        result = self._build_ocr_result(self._run_engine(str(image_path)))
        if result.text.strip():
            return result
        logger.debug("OCR full-image pass was empty; retrying overlapping tiles")
        return self._recognize_sparse_tiles(self._load_rgb_array(image_path))

    def recognize_numpy(self, image_array: Any) -> OCRResult:
        """Perform OCR on an RGB(A) array, with a sparse-layout fallback."""
        import numpy as np

        if not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_array)}")

        logger.debug(f"Running OCR on numpy array: shape={image_array.shape}")
        result = self._build_ocr_result(self._run_engine(image_array))
        if result.text.strip():
            return result
        return self._recognize_sparse_tiles(image_array)

    def recognize_bytes(self, image_data: bytes) -> OCRResult:
        """
        Perform OCR on image bytes.

        Args:
            image_data: Raw image bytes

        Returns:
            OCRResult with recognized text and metadata
        """
        import io

        import numpy as np
        from PIL import Image

        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed (RapidOCR works best with RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array and use recognize_numpy directly
        # This avoids temporary file I/O
        image_array = np.array(image)
        return self.recognize_numpy(image_array)

    def recognize_pdf_page(
        self,
        pdf_path: Path,
        page_num: int,
        dpi: int = DEFAULT_RENDER_DPI,
    ) -> OCRResult:
        """
        Perform OCR on a specific PDF page.

        Args:
            pdf_path: Path to the PDF file
            page_num: Page number (0-indexed)
            dpi: Resolution for rendering

        Returns:
            OCRResult with recognized text
        """
        try:
            import pymupdf
        except ImportError as e:
            raise MissingDependencyError(
                "PyMuPDF is not installed. Install with: uv add pymupdf"
            ) from e

        doc = pymupdf.open(pdf_path)
        try:
            if page_num >= len(doc):
                raise ValueError(
                    f"Page {page_num} out of range. PDF has {len(doc)} pages."
                )

            page = doc[page_num]

            # Render page to image
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            # Use recognize_pixmap for direct processing
            return self.recognize_pixmap(pix.samples, pix.width, pix.height, pix.n)

        finally:
            doc.close()

    def recognize_pixmap(
        self,
        samples: bytes,
        width: int,
        height: int,
        n_channels: int,
    ) -> OCRResult:
        """
        Perform OCR on raw pixel data (e.g., from pymupdf pixmap).

        This method is optimized for use with pymupdf's pixmap.samples,
        avoiding redundant image encoding/decoding.

        Args:
            samples: Raw pixel data bytes
            width: Image width in pixels
            height: Image height in pixels
            n_channels: Number of color channels (3 for RGB, 4 for RGBA)

        Returns:
            OCRResult with recognized text
        """
        import numpy as np

        # Convert raw bytes to numpy array
        image_array = np.frombuffer(samples, dtype=np.uint8).reshape(
            (height, width, n_channels)
        )

        # If RGBA, convert to RGB
        if n_channels == 4:
            image_array = image_array[:, :, :3]

        return self.recognize_numpy(image_array)

    def is_scanned_pdf(
        self, pdf_path: Path, sample_pages: int = DEFAULT_OCR_SAMPLE_PAGES
    ) -> bool:
        """
        Check if a PDF is likely scanned (image-based).

        Args:
            pdf_path: Path to the PDF file
            sample_pages: Number of pages to sample

        Returns:
            True if PDF appears to be scanned
        """
        try:
            import pymupdf
        except ImportError:
            return False

        doc = pymupdf.open(pdf_path)
        try:
            total_text_length = 0
            pages_to_check = min(sample_pages, len(doc))
            if pages_to_check == 0:
                # Zero-page PDF: nothing to scan, avoid division by zero
                return False

            for i in range(pages_to_check):
                page = doc[i]
                # Note: pymupdf get_text() returns str but type stubs say Any
                text: str = page.get_text()  # type: ignore[assignment]
                total_text_length += len(text.strip())

            # If very little text extracted, likely scanned
            avg_text_per_page = total_text_length / pages_to_check
            return avg_text_per_page < 100  # Threshold: less than 100 chars per page

        finally:
            doc.close()

    def recognize_to_markdown(self, image_path: Path | str) -> str:
        """Perform OCR and lay the result out as Markdown in reading order.

        Args:
            image_path: Path to the image file

        Returns:
            Markdown text (see :func:`markitai.ocr_layout.layout_ocr_text`)
        """
        image_path = Path(image_path)
        return self._result_to_markdown(
            self._run_engine(str(image_path)),
            lambda: self._load_rgb_array(image_path),
        )

    def recognize_array_to_markdown(self, image_array: Any) -> str:
        """Like :meth:`recognize_to_markdown`, for an already decoded RGB array.

        Used for inputs RapidOCR cannot open from a path itself: the frames
        of a multi-page TIFF and rasterized SVGs.
        """
        return self._result_to_markdown(
            self._run_engine(image_array), lambda: image_array
        )

    def _result_to_markdown(self, result: Any, load_array: Any) -> str:
        """Markdown for one full-image pass, retrying empty ones by tile."""
        texts = list(result.txts) if result.txts is not None else []

        # Laid out by markitai rather than RapidOCR's to_markdown(), which
        # has no notion of columns, vertical text or page orientation. An
        # empty pass goes on to a sparse tiled retry, which may still recover
        # small text. No return_word_box/return_single_char_box on the pass:
        # only line boxes are read, and RapidOCR would store those flags in
        # the shared engine for every later call.
        if texts:
            return self._build_ocr_result(result).text

        logger.debug("OCR Markdown pass was empty; retrying overlapping tiles")
        # The tiled result is already laid out in paragraphs by line gaps
        return self._recognize_sparse_tiles(load_array()).text
