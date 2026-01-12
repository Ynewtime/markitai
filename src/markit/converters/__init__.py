"""Document converters module for MarkIt."""

from markit.converters.base import (
    BaseConverter,
    BaseProcessor,
    ConversionPlan,
    ConversionResult,
    ExtractedImage,
)
from markit.converters.markitdown import MarkItDownConverter
from markit.converters.office import (
    LibreOfficeConverter,
    OfficePreprocessor,
    check_office_available,
)
from markit.converters.pandoc import (
    PandocConverter,
    PandocTableConverter,
    check_pandoc_available,
    get_pandoc_version,
)
from markit.converters.pdf import (
    PDFPlumberConverter,
    PyMuPDFConverter,
    PyMuPDFTextExtractor,
    TableExtractor,
)

__all__ = [
    # Base classes
    "BaseConverter",
    "BaseProcessor",
    "ConversionPlan",
    "ConversionResult",
    "ExtractedImage",
    # Converters
    "MarkItDownConverter",
    "PandocConverter",
    "PandocTableConverter",
    "PyMuPDFConverter",
    "PyMuPDFTextExtractor",
    "PDFPlumberConverter",
    "TableExtractor",
    # Preprocessors
    "OfficePreprocessor",
    "LibreOfficeConverter",
    # Utility functions
    "check_pandoc_available",
    "get_pandoc_version",
    "check_office_available",
]
