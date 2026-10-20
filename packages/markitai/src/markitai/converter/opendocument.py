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
from xml.etree import ElementTree  # nosec B405 - DOCTYPE/ENTITY refused before parsing

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    conversion_failed,
    not_a_package,
    register_converter,
)
from markitai.converter.delimited import render_markdown_table
from markitai.converter.sheet_tables import Spans, sheet_markdown

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


_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
_LINK_SCHEMES = ("http://", "https://", "mailto:", "ftp://")
# Not part of the text: a comment (its author, date and body), the
# end-of-range mark a ranged comment leaves, and the record of tracked
# changes, which holds the text a reviewer deleted
_SKIPPED_TAGS = frozenset({"annotation", "annotation-end", "tracked-changes"})


def _link(text: str, href: str) -> str:
    """``[text](href)`` for a web or mail address, the bare text otherwise
    (a bookmark inside the document, a relative file)."""
    href = href.strip()
    if not text.strip() or not href.lower().startswith(_LINK_SCHEMES):
        return text
    if any(ch.isspace() or ch in "()<>" for ch in href):
        href = f"<{href.replace('<', '%3C').replace('>', '%3E')}>"
    return f"[{text}]({href})"


def _text_of(elem: ElementTree.Element, notes: list[str] | None = None) -> str:
    """Flatten an element's text, honouring the ODF whitespace elements.

    A footnote or endnote becomes a ``[^n]`` marker with its body added to
    *notes* (left out where there is no list to add it to), a hyperlink a
    Markdown link, and comments are not text.
    """
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
        elif tag in _SKIPPED_TAGS:
            pass
        elif tag == "note":
            if notes is not None:
                body = next((c for c in child if _local(c.tag) == "note-body"), None)
                paragraphs = (
                    [_text_of(p, notes).strip() for p in body]
                    if body is not None
                    else []
                )
                notes.append(" ".join(p for p in paragraphs if p))
                parts.append(f"[^{len(notes)}]")
        elif tag == "a":
            parts.append(_link(_text_of(child, notes), child.get(_XLINK_HREF, "")))
        else:
            parts.append(_text_of(child, notes))
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
        reason = not_a_package(input_path, "OpenDocument")
        raise _OdfError(reason or f"not a readable zip archive: {exc}") from exc
    except OSError as exc:
        raise _OdfError(f"cannot read the archive: {exc}") from exc

    if _DECLARATION_RE.search(data):
        raise _OdfError(
            "content.xml carries a DOCTYPE or ENTITY declaration, which the "
            "stdlib parser would expand (billion laughs)"
        )
    try:
        return ElementTree.fromstring(data)  # nosec B314 - refused above
    except ElementTree.ParseError as exc:
        raise _OdfError(f"malformed content.xml: {exc}") from exc


def _cell_text(cell: ElementTree.Element, notes: list[str] | None = None) -> str:
    blocks = [
        _text_of(child, notes).strip()
        for child in cell
        if _local(child.tag) not in _SKIPPED_TAGS
    ]
    return "\n".join(block for block in blocks if block)


def _table_rows(
    table: ElementTree.Element,
    spans: Spans | None = None,
    notes: list[str] | None = None,
) -> list[list[str]]:
    """Read a table into a rectangular matrix, trailing blanks trimmed.

    ``spans`` collects (row, column) -> column span for merged cells, and
    ``notes`` the bodies of footnotes in the cells.
    """
    # Runs of blank cells/rows keep their full width, but are only emitted
    # once a value follows: the trailing filler runs spreadsheets write
    # (thousands wide) never get expanded.
    rows: list[list[str]] = []
    blank_rows = 0
    for element in table.iter():
        if _local(element.tag) != "table-row":
            continue
        cells: list[str] = []
        merged: dict[int, int] = {}
        blank_cells = 0
        for cell in element:
            if _local(cell.tag) not in {"table-cell", "covered-table-cell"}:
                continue
            text = _cell_text(cell, notes)
            repeat = _repeat_count(cell, "number-columns-repeated")
            if not text:
                blank_cells += repeat
                continue
            cells.extend([""] * blank_cells)
            blank_cells = 0
            span = _repeat_count(cell, "number-columns-spanned")
            if span > 1:
                merged[len(cells)] = span
            cells.extend([text] * repeat)
        repeat_rows = _repeat_count(element, "number-rows-repeated")
        if not any(cells):
            blank_rows += repeat_rows
            continue
        if spans is not None:
            for row in range(repeat_rows):
                for column, span in merged.items():
                    spans[(len(rows) + blank_rows + row, column)] = span
        rows.extend([[] for _ in range(blank_rows)])
        blank_rows = 0
        rows.extend([cells] * repeat_rows)

    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    width = max((len(row) for row in rows), default=0)
    while width and not any(
        row[width - 1].strip() for row in rows if len(row) >= width
    ):
        width -= 1
    return [row[:width] for row in rows]


def _render_list(
    elem: ElementTree.Element, depth: int, out: list[str], notes: list[str]
) -> None:
    for item in elem:
        if _local(item.tag) not in {"list-item", "list-header"}:
            continue
        for child in item:
            tag = _local(child.tag)
            if tag == "list":
                _render_list(child, depth + 1, out, notes)
                continue
            text = _text_of(child, notes).strip()
            if text:
                out.append(f"{'  ' * depth}- {text}")


def _render_text_element(
    elem: ElementTree.Element, out: list[str], notes: list[str]
) -> None:
    tag = _local(elem.tag)
    if tag in _SKIPPED_TAGS:
        return
    if tag == "h":
        text = _text_of(elem, notes).strip()
        if text:
            try:
                level = int(elem.get(f"{{{_TEXT_NS}}}outline-level", "1"))
            except ValueError:
                level = 1
            out.append(f"{'#' * max(1, min(level, _MAX_HEADING_LEVEL))} {text}")
    elif tag == "p":
        text = _text_of(elem, notes).strip()
        if text:
            out.append(text)
    elif tag == "list":
        items: list[str] = []
        _render_list(elem, 0, items, notes)
        if items:
            out.append("\n".join(items))
    elif tag == "table":
        table = render_markdown_table(_table_rows(elem, notes=notes))
        if table:
            out.append(table)
    else:
        # Sections, frames, change-tracking wrappers: keep walking.
        for child in elem:
            _render_text_element(child, out, notes)


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
        notes: list[str] = []
        for body in root.iter(f"{{{_OFFICE_NS}}}text"):
            for child in body:
                _render_text_element(child, blocks, notes)
            break

        for block in blocks:
            if block.startswith("# "):
                metadata["title"] = block[2:].strip()
                break
        if notes:
            blocks.append(
                "\n".join(f"[^{n}]: {body}" for n, body in enumerate(notes, 1))
            )

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
            spans: Spans = {}
            table = sheet_markdown(_table_rows(element, spans), spans)
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
