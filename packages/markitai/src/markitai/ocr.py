"""OCR module using RapidOCR."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import DEFAULT_OCR_SAMPLE_PAGES, DEFAULT_RENDER_DPI

if TYPE_CHECKING:
    from markitai.config import OCRConfig

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
        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise ImportError(
                "RapidOCR is not installed. Install with: uv add rapidocr"
            ) from e

        # Build params
        params: dict[str, Any] = {
            # Empty first-pass detections are handled by the tiled fallback
            # below, so RapidOCR's warning would be noisy and misleading.
            "Global.log_level": "error",
        }

        # Set language if configured (must use LangRec enum)
        if config and config.lang:
            from rapidocr import LangRec

            lang_map = {
                "zh": LangRec.CH,
                "ch": LangRec.CH,
                "en": LangRec.EN,
                "ja": LangRec.JAPAN,
                "japan": LangRec.JAPAN,
                "ko": LangRec.KOREAN,
                "korean": LangRec.KOREAN,
                "ar": LangRec.ARABIC,
                "arabic": LangRec.ARABIC,
                "th": LangRec.TH,
                "latin": LangRec.LATIN,
            }
            lang_enum = lang_map.get(config.lang.lower(), LangRec.CH)
            params["Rec.lang_type"] = lang_enum  # type: ignore[assignment]

        return RapidOCR(params=params)

    @property
    def engine(self) -> Any:
        """Get or create the RapidOCR engine.

        Uses the global shared engine by default to avoid cold start delays.
        Falls back to instance-specific engine only if configs differ.
        """
        # Use global shared engine for better performance
        return self.get_shared_engine(self.config)

    def _build_ocr_result(self, raw_result: Any) -> OCRResult:
        """Build OCRResult from raw RapidOCR engine output.

        Extracts texts, scores, and boxes from the engine result,
        joins text blocks, and calculates average confidence.

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

        full_text = "\n".join(texts)
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

        tile_width = math.ceil(width / columns)
        tile_height = math.ceil(height / rows)
        overlap_x = max(16, round(tile_width * 0.06))
        overlap_y = max(16, round(tile_height * 0.06))
        blocks: list[dict[str, Any]] = []

        for row in range(rows):
            for column in range(columns):
                x0 = max(0, column * tile_width - overlap_x)
                y0 = max(0, row * tile_height - overlap_y)
                x1 = min(width, (column + 1) * tile_width + overlap_x)
                y1 = min(height, (row + 1) * tile_height + overlap_y)
                tile = image_array[y0:y1, x0:x1]
                raw: Any = self.engine(tile, box_thresh=0.35, text_score=0.45)
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

        blocks.sort(key=lambda block: (block["bounds"][1], block["bounds"][0]))
        if not blocks:
            return OCRResult(text="", confidence=0.0, boxes=[])
        return OCRResult(
            text="\n".join(block["text"] for block in blocks),
            confidence=sum(block["score"] for block in blocks) / len(blocks),
            boxes=[list(block["bounds"]) for block in blocks],
        )

    def recognize(self, image_path: Path | str) -> OCRResult:
        """Perform OCR on an image file, retrying sparse layouts by tile."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug(f"Running OCR on: {image_path.name}")
        result = self._build_ocr_result(self.engine(str(image_path)))
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
        result = self._build_ocr_result(self.engine(image_array))
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
            import fitz  # pymupdf
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is not installed. Install with: uv add pymupdf"
            ) from e

        doc = fitz.open(pdf_path)
        try:
            if page_num >= len(doc):
                raise ValueError(
                    f"Page {page_num} out of range. PDF has {len(doc)} pages."
                )

            page = doc[page_num]

            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
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
            import fitz
        except ImportError:
            return False

        doc = fitz.open(pdf_path)
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
        """
        Perform OCR and format result as markdown.

        Uses RapidOCR's built-in to_markdown() method if available.

        Args:
            image_path: Path to the image file

        Returns:
            Markdown formatted text if RapidOCR supports to_markdown(),
            otherwise plain text joined by double newlines
        """
        image_path = Path(image_path)

        result: Any = self.engine(
            str(image_path),
            return_word_box=True,
            return_single_char_box=True,
        )
        texts = list(result.txts) if result.txts is not None else []

        # Preserve RapidOCR's table/layout Markdown when the full-image pass
        # succeeded. Its empty-result Markdown is intentionally not returned:
        # a sparse tiled retry may still recover small text.
        if texts and hasattr(result, "to_markdown"):
            return result.to_markdown()
        if texts:
            return "\n\n".join(str(text) for text in texts)

        logger.debug("OCR Markdown pass was empty; retrying overlapping tiles")
        fallback = self._recognize_sparse_tiles(self._load_rgb_array(image_path))
        return "\n\n".join(fallback.text.splitlines())
