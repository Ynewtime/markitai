"""Base converter interface and data classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""

    data: bytes
    format: str  # png, jpg, etc.
    filename: str
    source_document: Path
    position: int  # Position in document
    width: int | None = None
    height: int | None = None
    original_path: str | None = None  # Original path/reference in document


@dataclass
class ConversionResult:
    """Result of a document conversion."""

    markdown: str
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def images_count(self) -> int:
        """Return the number of extracted images."""
        return len(self.images)


@dataclass
class ConversionPlan:
    """Plan for converting a document."""

    primary_converter: "BaseConverter"
    fallback_converter: "BaseConverter | None" = None
    pre_processors: list["BaseProcessor"] = field(default_factory=list)
    post_processors: list["BaseProcessor"] = field(default_factory=list)


class BaseConverter(ABC):
    """Abstract base class for document converters."""

    name: str = "base"
    supported_extensions: set[str] = set()

    @abstractmethod
    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a document file to Markdown.

        Args:
            file_path: Path to the document file

        Returns:
            ConversionResult with markdown content and extracted images
        """
        pass

    def supports(self, extension: str) -> bool:
        """Check if this converter supports the given file extension.

        Args:
            extension: File extension including the dot (e.g., '.docx')

        Returns:
            True if supported, False otherwise
        """
        return extension.lower() in self.supported_extensions

    async def validate(self, file_path: Path) -> bool:
        """Validate that the file can be converted.

        Args:
            file_path: Path to the document file

        Returns:
            True if file is valid for conversion
        """
        if not file_path.exists():
            return False
        if not file_path.is_file():
            return False
        return self.supports(file_path.suffix)


class BaseProcessor(ABC):
    """Abstract base class for pre/post processors."""

    name: str = "base"

    @abstractmethod
    async def process(self, file_path: Path) -> Path:
        """Process a file and return the path to the processed result.

        Args:
            file_path: Path to the file to process

        Returns:
            Path to the processed file (may be the same or a new file)
        """
        pass
