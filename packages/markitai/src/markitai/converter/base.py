"""Base converter classes and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig


class FileFormat(Enum):
    """Supported file formats."""

    # Office Open XML formats (2007+)
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"

    # Legacy Office formats (97-2003); doc/ppt need MS Office or LibreOffice,
    # xls converts in pure Python (xlrd via markitdown)
    DOC = "doc"
    PPT = "ppt"
    XLS = "xls"

    # PDF
    PDF = "pdf"

    # Text
    TXT = "txt"
    MD = "md"

    # Images
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    WEBP = "webp"
    SVG = "svg"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"
    HEIC = "heic"
    HEIF = "heif"
    AVIF = "avif"

    # Structured data
    CSV = "csv"
    XML = "xml"
    TSV = "tsv"

    # OpenDocument
    ODS = "ods"
    ODT = "odt"

    # Apple
    NUMBERS = "numbers"

    # E-book
    EPUB = "epub"

    # Rich text / markup
    RTF = "rtf"
    RST = "rst"
    ORG = "org"
    TEX = "tex"

    # Web
    HTML = "html"
    HTM = "htm"
    XHTML = "xhtml"

    # Email
    EML = "eml"
    MSG = "msg"

    # Notebook
    IPYNB = "ipynb"

    # Unknown
    UNKNOWN = "unknown"


# Image formats that produce no useful text content without LLM/OCR.
# These are skipped in non-LLM mode (Rule A) and routed to Vision
# analysis in --llm --pure mode.
IMAGE_ONLY_FORMATS: frozenset[FileFormat] = frozenset(
    {
        FileFormat.JPEG,
        FileFormat.JPG,
        FileFormat.PNG,
        FileFormat.WEBP,
        FileFormat.GIF,
        FileFormat.BMP,
        FileFormat.TIFF,
        FileFormat.SVG,
        FileFormat.HEIC,
        FileFormat.HEIF,
        FileFormat.AVIF,
    }
)


# Mapping of file extensions to formats
EXTENSION_MAP: dict[str, FileFormat] = {
    ".docx": FileFormat.DOCX,
    ".doc": FileFormat.DOC,
    ".pptx": FileFormat.PPTX,
    ".ppt": FileFormat.PPT,
    ".xlsx": FileFormat.XLSX,
    ".xls": FileFormat.XLS,
    ".pdf": FileFormat.PDF,
    ".txt": FileFormat.TXT,
    ".md": FileFormat.MD,
    ".markdown": FileFormat.MD,
    ".jpeg": FileFormat.JPEG,
    ".jpg": FileFormat.JPG,
    ".png": FileFormat.PNG,
    ".webp": FileFormat.WEBP,
    ".svg": FileFormat.SVG,
    ".csv": FileFormat.CSV,
    ".xml": FileFormat.XML,
    ".ods": FileFormat.ODS,
    ".odt": FileFormat.ODT,
    ".numbers": FileFormat.NUMBERS,
    ".gif": FileFormat.GIF,
    ".bmp": FileFormat.BMP,
    ".tiff": FileFormat.TIFF,
    ".tif": FileFormat.TIFF,
    ".heic": FileFormat.HEIC,
    ".heif": FileFormat.HEIF,
    ".avif": FileFormat.AVIF,
    ".tsv": FileFormat.TSV,
    ".epub": FileFormat.EPUB,
    ".rtf": FileFormat.RTF,
    ".rst": FileFormat.RST,
    ".org": FileFormat.ORG,
    ".tex": FileFormat.TEX,
    ".html": FileFormat.HTML,
    ".htm": FileFormat.HTM,
    ".xhtml": FileFormat.XHTML,
    ".eml": FileFormat.EML,
    ".msg": FileFormat.MSG,
    ".ipynb": FileFormat.IPYNB,
}


def detect_format(path: Path | str) -> FileFormat:
    """Detect file format from extension."""
    path = Path(path)
    ext = path.suffix.lower()
    return EXTENSION_MAP.get(ext, FileFormat.UNKNOWN)


@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""

    path: Path
    index: int
    original_name: str
    mime_type: str
    width: int
    height: int
    data: bytes | None = None  # Raw image data before saving


@dataclass
class ConvertResult:
    """Result of a document conversion."""

    markdown: str
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def has_images(self) -> bool:
        """Check if any images were extracted."""
        return len(self.images) > 0


class BaseConverter(ABC):
    """Abstract base class for document converters."""

    # Formats this converter can handle
    supported_formats: list[FileFormat] = []

    def __init__(self, config: MarkitaiConfig | None = None) -> None:
        """Initialize converter with optional configuration."""
        self.config = config

    @abstractmethod
    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """
        Convert a document to Markdown.

        Args:
            input_path: Path to the input file
            output_dir: Optional output directory for extracted images

        Returns:
            ConvertResult containing markdown and extracted images
        """
        pass

    def can_convert(self, path: Path | str) -> bool:
        """Check if this converter can handle the given file."""
        fmt = detect_format(path)
        return fmt in self.supported_formats


# Registry of converters by format, filled by @register_converter as each
# converter module is imported.
_converter_registry: dict[FileFormat, type[BaseConverter]] = {}


# Which module defines the converter for each format, so that importing one
# converter never drags in the rest. The cost of getting this wrong is not
# just startup time: markitdown imports Magika, Magika imports onnxruntime,
# and onnxruntime's static destructors can abort an already-finished process
# (see markitai.utils.shutdown). Converting a .txt must not load any of it.
_CONVERTER_MODULES: dict[FileFormat, str] = {
    FileFormat.PDF: "markitai.converter.pdf",
    FileFormat.TXT: "markitai.converter.text",
    FileFormat.MD: "markitai.converter.text",
    FileFormat.DOCX: "markitai.converter.office",
    FileFormat.PPTX: "markitai.converter.office",
    FileFormat.XLSX: "markitai.converter.office",
    FileFormat.XLS: "markitai.converter.office",
    FileFormat.DOC: "markitai.converter.legacy",
    FileFormat.PPT: "markitai.converter.legacy",
    FileFormat.EML: "markitai.converter.eml",
    FileFormat.JPEG: "markitai.converter.image",
    FileFormat.JPG: "markitai.converter.image",
    FileFormat.PNG: "markitai.converter.image",
    FileFormat.WEBP: "markitai.converter.image",
    FileFormat.SVG: "markitai.converter.image",
    FileFormat.GIF: "markitai.converter.image",
    FileFormat.BMP: "markitai.converter.image",
    FileFormat.TIFF: "markitai.converter.image",
    FileFormat.HEIC: "markitai.converter.image",
    FileFormat.HEIF: "markitai.converter.image",
    FileFormat.AVIF: "markitai.converter.image",
    FileFormat.HTML: "markitai.converter.markitdown_ext",
    FileFormat.HTM: "markitai.converter.markitdown_ext",
    FileFormat.XHTML: "markitai.converter.markitdown_ext",
    FileFormat.CSV: "markitai.converter.markitdown_ext",
    FileFormat.EPUB: "markitai.converter.markitdown_ext",
    FileFormat.MSG: "markitai.converter.markitdown_ext",
    FileFormat.IPYNB: "markitai.converter.markitdown_ext",
    FileFormat.NUMBERS: "markitai.converter.markitdown_ext",
}


_ConverterT = TypeVar("_ConverterT", bound=BaseConverter)


def register_converter(
    fmt: FileFormat,
) -> Callable[[type[_ConverterT]], type[_ConverterT]]:
    """Register a converter without erasing its concrete class type."""

    def decorator(cls: type[_ConverterT]) -> type[_ConverterT]:
        _converter_registry[fmt] = cls
        return cls

    return decorator


def load_converter_class(fmt: FileFormat) -> type[BaseConverter] | None:
    """Return the converter class for one format, importing it if needed.

    A format with no native module falls through to kreuzberg, which
    registers itself for the formats markitai does not cover natively (and
    only when the optional dependency is installed).
    """
    import importlib

    registered = _converter_registry.get(fmt)
    if registered is not None:
        return registered

    module = _CONVERTER_MODULES.get(fmt)
    if module is not None:
        importlib.import_module(module)
        return _converter_registry.get(fmt)

    from markitai.converter.kreuzberg import register_kreuzberg_converters

    register_kreuzberg_converters()
    return _converter_registry.get(fmt)


def get_converter(
    path: Path | str,
    config: MarkitaiConfig | None = None,
) -> BaseConverter | None:
    """
    Get an appropriate converter for the given file.

    Args:
        path: Path to the file to convert
        config: Optional configuration

    Returns:
        A converter instance or None if no converter found
    """
    fmt = detect_format(path)
    converter_cls = load_converter_class(fmt)

    if converter_cls is None:
        return None

    return converter_cls(config=config)
