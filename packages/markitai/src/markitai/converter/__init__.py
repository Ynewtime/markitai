"""Converter modules for various document formats.

Converters are imported on demand, not here: ``get_converter`` (in ``base``)
maps a format to its module and imports that one. Importing them all up
front cost every markitai process ~200ms and, through
``markitdown`` -> ``magika`` -> ``onnxruntime``, a native teardown that can
abort a finished run (see ``markitai.utils.shutdown``) — for converting a
``.txt``, which needs none of it.

The class names below stay importable from this package for callers that
want one directly; each resolves through ``__getattr__``, which imports the
same single module ``get_converter`` would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from markitai.converter.base import (
    IMAGE_ONLY_FORMATS,
    BaseConverter,
    ConvertResult,
    ExtractedImage,
    FileFormat,
    detect_format,
    get_converter,
    unsupported_format_message,
)

if TYPE_CHECKING:
    from markitai.converter.delimited import TsvConverter
    from markitai.converter.eml import EmlConverter
    from markitai.converter.image import ImageConverter
    from markitai.converter.latex import TexConverter
    from markitai.converter.legacy import DocConverter, PptConverter
    from markitai.converter.markitdown_ext import (
        CsvConverter,
        EpubConverter,
        HtmConverter,
        HtmlConverter,
        IpynbConverter,
        MsgConverter,
        NumbersConverter,
        XhtmlConverter,
    )
    from markitai.converter.markup import OrgConverter, RstConverter
    from markitai.converter.office import (
        DocxConverter,
        PptxConverter,
        XlsConverter,
        XlsxConverter,
    )
    from markitai.converter.opendocument import OdsConverter, OdtConverter
    from markitai.converter.pdf import PdfConverter
    from markitai.converter.text import MarkdownConverter, TxtConverter
    from markitai.converter.xml_doc import XmlConverter

# Cloudflare converter is NOT registered for any format: it must not override
# local converters by default. The workflow layer selects it explicitly when
# cloudflare.convert_enabled is True.

_LAZY_EXPORTS: dict[str, str] = {
    "CsvConverter": "markitai.converter.markitdown_ext",
    "DocConverter": "markitai.converter.legacy",
    "DocxConverter": "markitai.converter.office",
    "EmlConverter": "markitai.converter.eml",
    "EpubConverter": "markitai.converter.markitdown_ext",
    "HtmConverter": "markitai.converter.markitdown_ext",
    "HtmlConverter": "markitai.converter.markitdown_ext",
    "ImageConverter": "markitai.converter.image",
    "IpynbConverter": "markitai.converter.markitdown_ext",
    "MarkdownConverter": "markitai.converter.text",
    "MsgConverter": "markitai.converter.markitdown_ext",
    "NumbersConverter": "markitai.converter.markitdown_ext",
    "OdsConverter": "markitai.converter.opendocument",
    "OdtConverter": "markitai.converter.opendocument",
    "OrgConverter": "markitai.converter.markup",
    "PdfConverter": "markitai.converter.pdf",
    "PptConverter": "markitai.converter.legacy",
    "PptxConverter": "markitai.converter.office",
    "RstConverter": "markitai.converter.markup",
    "TexConverter": "markitai.converter.latex",
    "TsvConverter": "markitai.converter.delimited",
    "TxtConverter": "markitai.converter.text",
    "XhtmlConverter": "markitai.converter.markitdown_ext",
    "XlsConverter": "markitai.converter.office",
    "XlsxConverter": "markitai.converter.office",
    "XmlConverter": "markitai.converter.xml_doc",
}

__all__ = [
    "IMAGE_ONLY_FORMATS",
    "BaseConverter",
    "ConvertResult",
    "CsvConverter",
    "DocConverter",
    "DocxConverter",
    "EmlConverter",
    "EpubConverter",
    "ExtractedImage",
    "FileFormat",
    "HtmConverter",
    "HtmlConverter",
    "ImageConverter",
    "IpynbConverter",
    "MarkdownConverter",
    "MsgConverter",
    "NumbersConverter",
    "OdsConverter",
    "OdtConverter",
    "OrgConverter",
    "PdfConverter",
    "PptConverter",
    "PptxConverter",
    "RstConverter",
    "TexConverter",
    "TsvConverter",
    "TxtConverter",
    "XhtmlConverter",
    "XlsConverter",
    "XlsxConverter",
    "XmlConverter",
    "detect_format",
    "get_converter",
    "unsupported_format_message",
]


def __getattr__(name: str) -> Any:
    """Import the one module that defines ``name``."""
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    import importlib

    return getattr(importlib.import_module(module), name)
