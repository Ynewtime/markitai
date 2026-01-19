"""OCR module using RapidOCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from markit.config import OCRConfig


@dataclass
class OCRResult:
    """Result of OCR processing."""

    text: str
    confidence: float
    boxes: list[list[float]]


class OCRProcessor:
    """OCR processor using RapidOCR."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        """
        Initialize OCR processor.

        Args:
            config: Optional OCR configuration
        """
        self.config = config
        self._engine = None

    @property
    def engine(self):
        """Get or create the RapidOCR engine (lazy loading)."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _get_lang_enum(self, lang_code: str):
        """Map language code to RapidOCR LangRec enum."""
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
        return lang_map.get(lang_code.lower(), LangRec.CH)

    def _create_engine(self):
        """Create RapidOCR engine with configuration."""
        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise ImportError(
                "RapidOCR is not installed. "
                "Install with: pip install rapidocr onnxruntime"
            ) from e

        # Build params
        params = {
            "Global.log_level": "warning",  # Reduce logging noise
        }

        # Set language if configured (must use LangRec enum)
        if self.config and self.config.lang:
            params["Rec.lang_type"] = self._get_lang_enum(self.config.lang)  # type: ignore[assignment]

        return RapidOCR(params=params)

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

        # Extract text from result (RapidOCR returns union type with incomplete stubs)
        texts = result.txts if result.txts else []
        scores = result.scores if result.scores else []
        boxes = result.boxes if result.boxes else []

        # Join all recognized text
        full_text = "\n".join(texts)

        # Calculate average confidence
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

    def recognize_bytes(self, image_data: bytes) -> OCRResult:
        """
        Perform OCR on image bytes.

        Args:
            image_data: Raw image bytes

        Returns:
            OCRResult with recognized text and metadata
        """
        import io

        from PIL import Image

        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if needed (RapidOCR works best with RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to temporary file (RapidOCR works with file paths)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG", quality=95)
            tmp_path = Path(tmp.name)

        try:
            return self.recognize(tmp_path)
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    def recognize_pdf_page(
        self,
        pdf_path: Path,
        page_num: int,
        dpi: int = 150,
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
                "PyMuPDF is not installed. Install with: pip install pymupdf"
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

            # Convert to bytes
            image_data = pix.tobytes("jpeg")

            return self.recognize_bytes(image_data)

        finally:
            doc.close()

    def is_scanned_pdf(self, pdf_path: Path, sample_pages: int = 3) -> bool:
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
            Markdown formatted text
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
