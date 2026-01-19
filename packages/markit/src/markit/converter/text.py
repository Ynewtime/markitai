"""Text file converters (TXT, MD)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from markit.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)

if TYPE_CHECKING:
    pass


class TextConverter(BaseConverter):
    """Base converter for plain text files."""

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """
        Read text file content directly.

        Args:
            input_path: Path to the input file
            output_dir: Unused for text files

        Returns:
            ConvertResult containing the file content as markdown
        """
        input_path = Path(input_path)

        # Read file content
        content = input_path.read_text(encoding="utf-8")

        metadata = {
            "source": str(input_path),
            "format": input_path.suffix.lstrip(".").upper() or "TXT",
        }

        return ConvertResult(
            markdown=content,
            images=[],
            metadata=metadata,
        )


@register_converter(FileFormat.TXT)
class TxtConverter(TextConverter):
    """Converter for plain text files."""

    supported_formats = [FileFormat.TXT]


@register_converter(FileFormat.MD)
class MarkdownConverter(TextConverter):
    """Converter for Markdown files."""

    supported_formats = [FileFormat.MD]
