"""Markitdown-based converters for non-Office formats.

Registers markitdown converters for formats that markitdown supports natively
but markitai didn't previously handle (HTML, CSV, EPUB, MSG, IPYNB).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from markitdown import MarkItDown

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)

_markitdown: MarkItDown | None = None


def _convert(input_path: Path) -> ConvertResult:
    """Convert a file to Markdown using markitdown.

    Args:
        input_path: Path to the input file.

    Returns:
        ConvertResult with markdown content and metadata.
    """
    global _markitdown
    if _markitdown is None:
        _markitdown = MarkItDown()

    input_path = Path(input_path)
    logger.debug("[Markitdown] Converting: {}", input_path.name)

    result = _markitdown.convert(input_path, keep_data_uris=True)

    metadata: dict = {
        "source": str(input_path),
        "format": input_path.suffix.lstrip(".").upper(),
        "converter": "markitdown",
    }
    if result.title:
        metadata["title"] = result.title

    return ConvertResult(
        markdown=result.markdown,
        images=[],
        metadata=metadata,
    )


@register_converter(FileFormat.HTML)
class HtmlConverter(BaseConverter):
    """Converter for HTML files using markitdown."""

    supported_formats = [FileFormat.HTML, FileFormat.HTM, FileFormat.XHTML]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)


# Also register for .htm and .xhtml extensions
@register_converter(FileFormat.HTM)
class HtmConverter(HtmlConverter):
    """Converter for .htm files (delegates to HtmlConverter)."""

    supported_formats = [FileFormat.HTM]


@register_converter(FileFormat.XHTML)
class XhtmlConverter(HtmlConverter):
    """Converter for .xhtml files (delegates to HtmlConverter)."""

    supported_formats = [FileFormat.XHTML]


@register_converter(FileFormat.CSV)
class CsvConverter(BaseConverter):
    """Converter for CSV files using markitdown."""

    supported_formats = [FileFormat.CSV]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)


@register_converter(FileFormat.EPUB)
class EpubConverter(BaseConverter):
    """Converter for EPUB e-book files using markitdown."""

    supported_formats = [FileFormat.EPUB]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)


@register_converter(FileFormat.MSG)
class MsgConverter(BaseConverter):
    """Converter for Outlook MSG email files using markitdown."""

    supported_formats = [FileFormat.MSG]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)


@register_converter(FileFormat.IPYNB)
class IpynbConverter(BaseConverter):
    """Converter for Jupyter Notebook files using markitdown."""

    supported_formats = [FileFormat.IPYNB]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)


@register_converter(FileFormat.NUMBERS)
class NumbersConverter(BaseConverter):
    """Converter for Apple Numbers spreadsheet files using markitdown."""

    supported_formats = [FileFormat.NUMBERS]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        return _convert(input_path)
