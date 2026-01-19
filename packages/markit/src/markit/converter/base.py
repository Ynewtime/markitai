"""Base converter classes and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markit.config import MarkitConfig


class FileFormat(Enum):
    """Supported file formats."""

    # Office formats (new)
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"

    # Office formats (legacy) - P5
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

    # Unknown
    UNKNOWN = "unknown"


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

    def __init__(self, config: MarkitConfig | None = None) -> None:
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


# Registry of converters by format
_converter_registry: dict[FileFormat, type[BaseConverter]] = {}


def register_converter(fmt: FileFormat):
    """Decorator to register a converter for a file format."""

    def decorator(cls: type[BaseConverter]):
        _converter_registry[fmt] = cls
        return cls

    return decorator


def get_converter(
    path: Path | str,
    config: MarkitConfig | None = None,
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
    converter_cls = _converter_registry.get(fmt)

    if converter_cls is None:
        return None

    return converter_cls(config=config)
