"""Native OpenDocument converters for .odt and .ods (stdlib only).

An OpenDocument file is a zip whose ``content.xml`` already holds the whole
document tree, so ``zipfile`` plus ``ElementTree`` is the entire dependency
list. As in the XML converter, a ``content.xml`` carrying a DTD or entity
declarations is refused rather than parsed.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree  # noqa: S405 - DOCTYPE/ENTITY refused below

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    conversion_failed,
    register_converter,
)
from markitai.converter.delimited import render_markdown_table

_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_OFFICE_NS = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"

_DECLARATION_RE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)\b", re.IGNORECASE)

# A spreadsheet's last row/column is routinely "repeated" a million times to
# fill the sheet; materialising that is how a 10 KB file becomes a 2 GB list.
_MAX_REPEAT = 1000
_MAX_CONTENT_BYTES = 100 * 1024 * 1024
_MAX_HEADING_LEVEL = 6


class _OdfError(Exception):
    """Unreadable OpenDocument file (bad zip, no content, bad XML)."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def _repeat_count(elem: ElementTree.Element, attribute: str) -> int:
    raw = elem.get(f"{{{_TABLE_NS}}}{attribute}")
    if raw is None:
        return 1
    try:
        return max(1, min(int(raw), _MAX_REPEAT))
    except ValueError:
        return 1


def _text_of(elem: ElementTree.Element) -> str:
    """Flatten an element's text, honouring the ODF whitespace elements."""
    parts: list[str] = [elem.text or ""]
    for child in elem:
        tag = _local(child.tag)
        if tag == "s":
            try:
                count = int(child.get(f"{{{_TEXT_NS}}}c", "1"))
            except ValueError:
                count = 1
            parts.append(" " * max(1, min(count, _MAX_REPEAT)))
        elif tag == "tab":
            parts.append("\t")
        elif tag == "line-break":
            parts.append("\n")
        else:
            parts.append(_text_of(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _content_root(input_path: Path) -> ElementTree.Element:
    try:
        with zipfile.ZipFile(input_path) as archive:
            try:
                info = archive.getinfo("content.xml")
            except KeyError:
                raise _OdfError("no content.xml in the archive") from None
            if info.file_size > _MAX_CONTENT_BYTES:
                raise _OdfError(
                    f"content.xml is {info.file_size} bytes, over the "
                    f"{_MAX_CONTENT_BYTES} byte limit"
                )
            data = archive.read("content.xml")
    except zipfile.BadZipFile as exc:
        raise _OdfError(f"not a readable zip archive: {exc}") from exc
    except OSError as exc:
        raise _OdfError(f"cannot read the archive: {exc}") from exc

    if _DECLARATION_RE.search(data):
        raise _OdfError(
            "content.xml carries a DOCTYPE or ENTITY declaration, which the "
            "stdlib parser would expand (billion laughs)"
        )
    try:
        return ElementTree.fromstring(data)  # noqa: S314 - refused above
    except ElementTree.ParseError as exc:
        raise _OdfError(f"malformed content.xml: {exc}") from exc


def _cell_text(cell: ElementTree.Element) -> str:
    blocks = [_text_of(child).strip() for child in cell]
    return " ".join(block for block in blocks if block)


def _table_rows(table: ElementTree.Element) -> list[list[str]]:
    """Read a table into a rectangular matrix, trailing blanks trimmed."""
    rows: list[list[str]] = []
    for element in table.iter():
        if _local(element.tag) != "table-row":
            continue
        cells: list[str] = []
        for cell in element:
            if _local(cell.tag) not in {"table-cell", "covered-table-cell"}:
                continue
            text = _cell_text(cell)
            repeat = _repeat_count(cell, "number-columns-repeated")
            cells.extend([text] * (1 if not text else repeat))
        repeat_rows = _repeat_count(element, "number-rows-repeated")
        rows.extend([cells] * (1 if not any(cells) else repeat_rows))

    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    width = max((len(row) for row in rows), default=0)
    while width and not any(
        row[width - 1].strip() for row in rows if len(row) >= width
    ):
        width -= 1
    return [row[:width] for row in rows]


def _render_list(elem: ElementTree.Element, depth: int, out: list[str]) -> None:
    for item in elem:
        if _local(item.tag) not in {"list-item", "list-header"}:
            continue
        for child in item:
            tag = _local(child.tag)
            if tag == "list":
                _render_list(child, depth + 1, out)
                continue
            text = _text_of(child).strip()
            if text:
                out.append(f"{'  ' * depth}- {text}")


def _render_text_element(elem: ElementTree.Element, out: list[str]) -> None:
    tag = _local(elem.tag)
    if tag == "h":
        text = _text_of(elem).strip()
        if text:
            try:
                level = int(elem.get(f"{{{_TEXT_NS}}}outline-level", "1"))
            except ValueError:
                level = 1
            out.append(f"{'#' * max(1, min(level, _MAX_HEADING_LEVEL))} {text}")
    elif tag == "p":
        text = _text_of(elem).strip()
        if text:
            out.append(text)
    elif tag == "list":
        items: list[str] = []
        _render_list(elem, 0, items)
        if items:
            out.append("\n".join(items))
    elif tag == "table":
        table = render_markdown_table(_table_rows(elem))
        if table:
            out.append(table)
    else:
        # Sections, frames, change-tracking wrappers: keep walking.
        for child in elem:
            _render_text_element(child, out)


@register_converter(FileFormat.ODT)
class OdtConverter(BaseConverter):
    """Converter for OpenDocument text documents."""

    supported_formats = [FileFormat.ODT]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[OdtConverter] Converting: {}", input_path.name)

        metadata: dict = {
            "source": str(input_path),
            "format": "ODT",
            "converter": "opendocument",
        }
        try:
            root = _content_root(input_path)
        except _OdfError as exc:
            logger.warning("[OdtConverter] {}: {}", input_path.name, exc)
            conversion_failed(str(exc))

        blocks: list[str] = []
        for body in root.iter(f"{{{_OFFICE_NS}}}text"):
            for child in body:
                _render_text_element(child, blocks)
            break

        for block in blocks:
            if block.startswith("# "):
                metadata["title"] = block[2:].strip()
                break

        return ConvertResult(
            markdown="\n\n".join(blocks) + "\n" if blocks else "",
            images=[],
            metadata=metadata,
        )


@register_converter(FileFormat.ODS)
class OdsConverter(BaseConverter):
    """Converter for OpenDocument spreadsheets."""

    supported_formats = [FileFormat.ODS]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[OdsConverter] Converting: {}", input_path.name)

        metadata: dict = {
            "source": str(input_path),
            "format": "ODS",
            "converter": "opendocument",
        }
        try:
            root = _content_root(input_path)
        except _OdfError as exc:
            logger.warning("[OdsConverter] {}: {}", input_path.name, exc)
            conversion_failed(str(exc))

        blocks: list[str] = []
        for element in root.iter(f"{{{_TABLE_NS}}}table"):
            name = element.get(f"{{{_TABLE_NS}}}name", "").strip()
            table = render_markdown_table(_table_rows(element))
            if not table:
                continue
            if name:
                metadata.setdefault("title", name)
                blocks.append(f"## {name}")
            blocks.append(table)

        return ConvertResult(
            markdown="\n\n".join(blocks) + "\n" if blocks else "",
            images=[],
            metadata=metadata,
        )
