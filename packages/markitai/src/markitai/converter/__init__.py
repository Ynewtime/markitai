"""Converter modules for various document formats."""

# Apply compatibility patches before importing converters
from markitai.converter._patches import apply_all_patches

apply_all_patches()

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    detect_format,
    get_converter,
)
from markitai.converter.image import ImageConverter
from markitai.converter.legacy import DocConverter, PptConverter, XlsConverter

# Import converters to register them
from markitai.converter.office import DocxConverter, PptxConverter, XlsxConverter
from markitai.converter.pdf import PdfConverter
from markitai.converter.text import MarkdownConverter, TxtConverter

# Cloudflare converter is NOT auto-registered with @register_converter.
# It shouldn't override local converters by default. Instead, it's explicitly
# selected in the workflow layer when cloudflare.convert_enabled is True.

__all__ = [
    "BaseConverter",
    "ConvertResult",
    "ExtractedImage",
    "FileFormat",
    "get_converter",
    "detect_format",
    "DocxConverter",
    "PptxConverter",
    "XlsxConverter",
    "PdfConverter",
    "TxtConverter",
    "MarkdownConverter",
    "DocConverter",
    "PptConverter",
    "XlsConverter",
    "ImageConverter",
]
