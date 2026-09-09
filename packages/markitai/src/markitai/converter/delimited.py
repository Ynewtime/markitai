"""Native TSV converter (Python stdlib ``csv``).

Tab-separated data is a Markdown table and nothing else, so it needs no
extraction engine: ``csv`` reads it, and the table renderer here is shared
with the OpenDocument converter.
"""

from __future__ import annotations

import csv
from pathlib import Path

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    conversion_failed,
    register_converter,
)

# csv defaults to 128 KB per field; a pathological file would otherwise raise
# instead of converting.
_MAX_FIELD_BYTES = 4 * 1024 * 1024


def escape_cell(value: str) -> str:
    """Make one cell safe inside a Markdown table row."""
    return value.replace("|", "\\|").replace("\n", "<br>").replace("\r", "").strip()


def render_markdown_table(rows: list[list[str]]) -> str:
    """Render rows as a Markdown table, the first row being the header."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    lines = []
    for index, row in enumerate(rows):
        padded = [escape_cell(cell) for cell in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if index == 0:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _delimiter_for(first_line: str) -> str:
    """Tab, unless the first line has none — then let csv guess."""
    if "\t" in first_line or not first_line:
        return "\t"
    try:
        return csv.Sniffer().sniff(first_line, delimiters="\t,;|").delimiter
    except csv.Error:
        return "\t"


@register_converter(FileFormat.TSV)
class TsvConverter(BaseConverter):
    """Converter for tab-separated values files."""

    supported_formats = [FileFormat.TSV]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[TsvConverter] Converting: {}", input_path.name)

        metadata: dict = {
            "source": str(input_path),
            "format": "TSV",
            "converter": "delimited",
        }

        text = input_path.read_text(encoding="utf-8", errors="replace")
        first_line = text.split("\n", 1)[0]

        previous_limit = csv.field_size_limit(_MAX_FIELD_BYTES)
        try:
            rows = [
                list(row)
                for row in csv.reader(
                    text.splitlines(), delimiter=_delimiter_for(first_line)
                )
            ]
        except csv.Error as exc:
            logger.warning("[TsvConverter] {}: {}", input_path.name, exc)
            conversion_failed(f"malformed TSV: {exc}")
        finally:
            csv.field_size_limit(previous_limit)

        rows = [row for row in rows if any(cell.strip() for cell in row)]
        return ConvertResult(
            markdown=render_markdown_table(rows),
            images=[],
            metadata=metadata,
        )
