"""Format router for selecting appropriate converters."""

from pathlib import Path

from markit.config.constants import (
    LEGACY_FORMATS,
    MARKITDOWN_FORMATS,
    SUPPORTED_EXTENSIONS,
)
from markit.converters.base import BaseConverter, BaseProcessor, ConversionPlan
from markit.converters.markitdown import MarkItDownConverter
from markit.exceptions import ConverterNotFoundError
from markit.utils.logging import get_logger

log = get_logger(__name__)


class FormatRouter:
    """Routes files to appropriate converters based on format.

    The router determines:
    - Primary converter to use
    - Fallback converter if primary fails
    - Pre-processors needed (e.g., Office conversion for legacy formats)
    - Post-processors (e.g., LLM enhancement)
    """

    def __init__(
        self,
        pdf_engine: str = "pymupdf",
        enable_markitdown_plugins: bool = True,
        filter_small_images: bool = True,
        min_image_dimension: int = 50,
        min_image_area: int = 2500,
        min_image_size: int = 3072,
    ) -> None:
        """Initialize the format router.

        Args:
            pdf_engine: PDF processing engine (pymupdf, pdfplumber, markitdown)
            enable_markitdown_plugins: Enable MarkItDown plugins
            filter_small_images: Whether to filter small decorative images
            min_image_dimension: Minimum width or height for images
            min_image_area: Minimum area for images
            min_image_size: Minimum file size for images
        """
        self.pdf_engine = pdf_engine
        self.enable_markitdown_plugins = enable_markitdown_plugins
        self.filter_small_images = filter_small_images
        self.min_image_dimension = min_image_dimension
        self.min_image_area = min_image_area
        self.min_image_size = min_image_size

        # Initialize converters
        self._converters: dict[str, BaseConverter] = {}
        self._preprocessors: dict[str, BaseProcessor] = {}
        self._init_converters()

    def _init_converters(self) -> None:
        """Initialize available converters."""
        # MarkItDown is the primary converter
        self._converters["markitdown"] = MarkItDownConverter(
            enable_plugins=self.enable_markitdown_plugins
        )

        # PDF converters
        try:
            from markit.converters.pdf import (
                PDFPlumberConverter,
                PyMuPDF4LLMConverter,
                PyMuPDFConverter,
            )

            self._converters["pymupdf4llm"] = PyMuPDF4LLMConverter()
            self._converters["pymupdf"] = PyMuPDFConverter(
                filter_small_images=self.filter_small_images,
                min_image_dimension=self.min_image_dimension,
                min_image_area=self.min_image_area,
                min_image_size=self.min_image_size,
            )
            self._converters["pdfplumber"] = PDFPlumberConverter()
        except ImportError as e:
            log.warning("PDF converters not available", error=str(e))

        # Pandoc converter
        try:
            from markit.converters.pandoc import PandocConverter, check_pandoc_available

            if check_pandoc_available():
                self._converters["pandoc"] = PandocConverter()
            else:
                log.warning("Pandoc not installed, fallback converter unavailable")
        except ImportError as e:
            log.warning("Pandoc converter not available", error=str(e))

        # Office preprocessor
        try:
            from markit.converters.office import OfficePreprocessor

            self._preprocessors["office"] = OfficePreprocessor()
        except Exception as e:
            log.warning("Office preprocessor not available", error=str(e))

    def route(self, file_path: Path) -> ConversionPlan:
        """Route a file to the appropriate converter(s).

        Args:
            file_path: Path to the file to convert

        Returns:
            ConversionPlan with primary/fallback converters and processors

        Raises:
            ConverterNotFoundError: If no suitable converter is found
        """
        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ConverterNotFoundError(file_path, extension)

        log.debug(
            "Routing file",
            file=str(file_path),
            extension=extension,
        )

        # Handle legacy formats (need pre-processing)
        if extension in LEGACY_FORMATS:
            return self._route_legacy_format(file_path, extension)

        # Handle PDF based on configured engine
        if extension == ".pdf":
            return self._route_pdf()

        # Handle MarkItDown supported formats
        if extension in MARKITDOWN_FORMATS:
            return self._route_markitdown(extension)

        # Default to MarkItDown for other supported formats
        return self._route_markitdown(extension)

    def _route_markitdown(self, extension: str) -> ConversionPlan:
        """Create plan for MarkItDown conversion."""
        primary = self._converters["markitdown"]

        # Pandoc as fallback for document formats
        fallback = None
        if extension in {".docx", ".pptx", ".xlsx", ".html", ".htm"}:
            fallback = self._converters.get("pandoc")

        return ConversionPlan(
            primary_converter=primary,
            fallback_converter=fallback,
            pre_processors=[],
            post_processors=[],
        )

    def _route_pdf(self) -> ConversionPlan:
        """Create plan for PDF conversion."""
        # Use configured PDF engine
        if self.pdf_engine == "pymupdf4llm":
            primary = self._converters.get("pymupdf4llm") or self._converters["markitdown"]
            fallback = self._converters.get("pymupdf")
        elif self.pdf_engine == "pymupdf":
            primary = self._converters.get("pymupdf") or self._converters["markitdown"]
            fallback = self._converters.get("pdfplumber")
        elif self.pdf_engine == "pdfplumber":
            primary = self._converters.get("pdfplumber") or self._converters["markitdown"]
            fallback = self._converters.get("pymupdf")
        elif self.pdf_engine == "markitdown":
            primary = self._converters["markitdown"]
            fallback = self._converters.get("pymupdf4llm")
        else:
            primary = self._converters["markitdown"]
            fallback = None

        return ConversionPlan(
            primary_converter=primary,
            fallback_converter=fallback,
            pre_processors=[],
            post_processors=[],
        )

    def _route_legacy_format(self, file_path: Path, extension: str) -> ConversionPlan:
        """Create plan for legacy format conversion.

        Legacy formats (.doc, .ppt, .xls) need to be converted
        to modern formats first using MS Office or LibreOffice.
        """
        pre_processors = []

        # Add Office preprocessor if available
        office_preprocessor = self._preprocessors.get("office")
        if office_preprocessor:
            pre_processors.append(office_preprocessor)
            log.debug(
                "Using Office preprocessor for legacy format",
                extension=extension,
                file=str(file_path),
            )
        else:
            log.warning(
                "Office preprocessor not available for legacy format",
                extension=extension,
                file=str(file_path),
            )

        # Primary converter is MarkItDown (will process the converted modern format)
        primary = self._converters["markitdown"]
        fallback = self._converters.get("pandoc")

        return ConversionPlan(
            primary_converter=primary,
            fallback_converter=fallback,
            pre_processors=pre_processors,
            post_processors=[],
        )

    def get_supported_extensions(self) -> set[str]:
        """Return all supported file extensions."""
        return set(SUPPORTED_EXTENSIONS.keys())

    def is_supported(self, extension: str) -> bool:
        """Check if a file extension is supported.

        Args:
            extension: File extension including dot (e.g., '.docx')

        Returns:
            True if supported
        """
        return extension.lower() in SUPPORTED_EXTENSIONS
