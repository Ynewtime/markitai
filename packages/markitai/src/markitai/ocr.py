"""OCR module using RapidOCR."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import DEFAULT_OCR_SAMPLE_PAGES, DEFAULT_RENDER_DPI

if TYPE_CHECKING:
    from markitai.config import OCRConfig


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
    def get_shared_engine(cls, config: OCRConfig | None = None) -> Any:
        """Get or create global singleton engine (thread-safe).

        Uses double-checked locking for thread-safe lazy initialization.
        The engine is shared across all OCRProcessor instances to avoid
        repeated ONNX Runtime cold starts.

        Args:
            config: Optional OCR configuration for engine creation

        Returns:
            Shared RapidOCR engine instance
        """
        if cls._global_engine is None:
            with cls._init_lock:
                if cls._global_engine is None:
                    logger.debug("Creating global shared OCR engine")
                    cls._global_config = config
                    cls._global_engine = cls._create_engine_impl(config)
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
            "Global.log_level": "warning",  # Reduce logging noise
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

    def recognize(self, image_path: Path | str) -> OCRResult:
        """
        Perform OCR on an image file.

        Args:
            image_path: Path to the image file

        Returns:
            OCRResult with recognized text and metadata
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug(f"Running OCR on: {image_path.name}")

        result: Any = self.engine(str(image_path))
        return self._build_ocr_result(result)

    def recognize_numpy(self, image_array: Any) -> OCRResult:
        """
        Perform OCR on a numpy array (RGB image data).

        This is more efficient than recognize_bytes as it avoids
        intermediate file I/O when the image is already in memory.

        Args:
            image_array: numpy array of shape (H, W, 3) or (H, W, 4) in RGB(A) format

        Returns:
            OCRResult with recognized text and metadata
        """
        import numpy as np

        # Ensure we have a proper numpy array
        if not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_array)}")

        logger.debug(f"Running OCR on numpy array: shape={image_array.shape}")

        result: Any = self.engine(image_array)
        return self._build_ocr_result(result)

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

        # Try to use built-in markdown conversion
        if hasattr(result, "to_markdown"):
            return result.to_markdown()

        # Fallback: simple text extraction
        texts = result.txts if result.txts else []
        return "\n\n".join(texts)
