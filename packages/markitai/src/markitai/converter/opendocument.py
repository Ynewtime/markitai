"""Native OpenDocument converters for .odt and .ods (stdlib only).

An OpenDocument file is a zip whose ``content.xml`` already holds the whole
document tree, so ``zipfile`` plus ``ElementTree`` is the entire dependency
list. As in the XML converter, a ``content.xml`` carrying a DTD or entity
declarations is refused rather than parsed.

A text document keeps its bold and italic (from the automatic styles in
``content.xml`` and the named ones in ``styles.xml``), its numbered lists,
and its pictures, which become ``data:`` URIs the image pipeline saves as
the item's assets, like a DOCX picture.
"""

from __future__ import annotations

import base64
import re
import zipfile
from dataclasses import dataclass, field
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
_STYLE_NS = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
_FO_NS = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
_DRAW_NS = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"

_DECLARATION_RE = re.compile(rb"<!\s*(DOCTYPE|ENTITY)\b", re.IGNORECASE)

# A spreadsheet's last row/column is routinely "repeated" a million times to
# fill the sheet; materialising that is how a 10 KB file becomes a 2 GB list.
_MAX_REPEAT = 1000
_MAX_CONTENT_BYTES = 100 * 1024 * 1024
_MAX_HEADING_LEVEL = 6
# Pictures embedded as data: URIs: one at most 32 MiB, a document 128 MiB
# (the EPUB reader's limits); a larger one is left out
_PICTURE_MAX_BYTES = 32 * 1024 * 1024
_PICTURES_MAX_BYTES = 128 * 1024 * 1024
_PICTURE_TYPES = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
    ".bmp": "bmp",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".emf": "x-emf",
    ".wmf": "x-wmf",
}
_BOLD_WEIGHTS = frozenset({"bold", "600", "700", "800", "900"})


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


# (bold, italic): what a run of text looks like
Emphasis = tuple[bool, bool]
_PLAIN: Emphasis = (False, False)


@dataclass
class _Style:
    parent: str | None
    bold: bool | None
    italic: bool | None


@dataclass
class _Document:
    """What a text document's styles and archive say about its content."""

    # (family, name) -> the style's own bold/italic and its parent
    styles: dict[tuple[str, str], _Style] = field(default_factory=dict)
    # list style -> level (1-based) -> start number, None for a bullet level
    lists: dict[str, dict[int, int | None]] = field(default_factory=dict)
    archive: zipfile.ZipFile | None = None
    notes: list[str] = field(default_factory=list)
    picture_budget: int = _PICTURES_MAX_BYTES
    pictures: dict[str, str | None] = field(default_factory=dict)
    # list style -> the number its last top-level list would show next, for
    # a list that continues it (LibreOffice splits a list a paragraph
    # interrupts into two, the second with continue-numbering)
    next_number: dict[str, int] = field(default_factory=dict)

    def emphasis(self, family: str, name: str | None, inherited: Emphasis) -> Emphasis:
        """*inherited* with what style *name* (and its parents) set."""
        bold = italic = None
        seen: set[str] = set()
        while name and name not in seen and (bold is None or italic is None):
            seen.add(name)
            style = self.styles.get((family, name))
            if style is None:
                break
            bold = style.bold if bold is None else bold
            italic = style.italic if italic is None else italic
            name = style.parent
        return (
            inherited[0] if bold is None else bold,
            inherited[1] if italic is None else italic,
        )

    def picture(self, href: str) -> str | None:
        """The ``data:`` URI of the picture the archive holds at *href*."""
        if href in self.pictures:
            return self.pictures[href]
        uri = None
        kind = _PICTURE_TYPES.get(Path(href).suffix.lower())
        if self.archive is not None and kind and "://" not in href:
            name = href.removeprefix("./")
            try:
                info = self.archive.getinfo(name)
                limit = min(_PICTURE_MAX_BYTES, self.picture_budget)
                data = self.archive.read(info) if info.file_size <= limit else b""
            except (KeyError, OSError, zipfile.BadZipFile, RuntimeError):
                data = b""
            if data and len(data) <= self.picture_budget:
                self.picture_budget -= len(data)
                encoded = base64.b64encode(data).decode("ascii")
                uri = f"data:image/{kind};base64,{encoded}"
        self.pictures[href] = uri
        return uri


def _read_styles(
    root: ElementTree.Element, document: _Document, containers: frozenset[str]
) -> None:
    """Add the text, paragraph and list styles in *root*'s *containers*
    (``styles``, ``automatic-styles``) to *document*."""
    for container in root:
        if _local(container.tag) not in containers:
            continue
        for style in container:
            tag = _local(style.tag)
            name = style.get(f"{{{_STYLE_NS}}}name")
            if not name:
                continue
            if tag == "style":
                family = style.get(f"{{{_STYLE_NS}}}family", "")
                bold = italic = None
                for props in style:
                    if _local(props.tag) != "text-properties":
                        continue
                    weight = props.get(f"{{{_FO_NS}}}font-weight")
                    if weight is not None:
                        bold = weight.lower() in _BOLD_WEIGHTS
                    slant = props.get(f"{{{_FO_NS}}}font-style")
                    if slant is not None:
                        italic = slant.lower() in {"italic", "oblique"}
                document.styles[(family, name)] = _Style(
                    style.get(f"{{{_STYLE_NS}}}parent-style-name"), bold, italic
                )
            elif tag == "list-style":
                levels: dict[int, int | None] = {}
                for level in style:
                    try:
                        depth = int(level.get(f"{{{_TEXT_NS}}}level", "0"))
                    except ValueError:
                        continue
                    # A number level with no format (num-format="") shows
                    # no number: a bullet as far as Markdown goes
                    if _local(level.tag) == "list-level-style-number" and level.get(
                        f"{{{_STYLE_NS}}}num-format"
                    ):
                        levels[depth] = _int_attribute(level, "start-value", 1)
                    else:
                        levels[depth] = None
                document.lists[name] = levels


def _int_attribute(elem: ElementTree.Element, name: str, default: int) -> int:
    try:
        return max(0, int(elem.get(f"{{{_TEXT_NS}}}{name}", str(default))))
    except ValueError:
        return default


def _frame_markdown(frame: ElementTree.Element, document: _Document) -> str | None:
    """A frame's picture as ``![alt](data:…)``, None when it holds none
    the archive has (a text box, a chart, a linked file)."""
    alt = ""
    uri = None
    for child in frame:
        tag = _local(child.tag)
        if tag in {"title", "desc"} and not alt:
            alt = " ".join(re.sub(r"[\[\]]", " ", "".join(child.itertext())).split())
        elif tag == "image" and uri is None:
            uri = document.picture(child.get(_XLINK_HREF, "").strip())
    return f"![{alt}]({uri})" if uri else None


# A piece of inline text: (text, emphasis, raw). Raw pieces (a picture, a
# footnote marker) are never inside emphasis markers.
_Run = tuple[str, Emphasis, bool]


def _runs(
    elem: ElementTree.Element,
    document: _Document | None,
    emphasis: Emphasis,
    out: list[_Run],
) -> None:
    def add(text: str | None, style: Emphasis = emphasis, raw: bool = False) -> None:
        if text:
            out.append((text, style, raw))

    add(elem.text)
    for child in elem:
        tag = _local(child.tag)
        if tag == "s":
            try:
                count = int(child.get(f"{{{_TEXT_NS}}}c", "1"))
            except ValueError:
                count = 1
            add(" " * max(1, min(count, _MAX_REPEAT)))
        elif tag == "tab":
            add("\t")
        elif tag == "line-break":
            add("\n")
        elif tag in _SKIPPED_TAGS:
            pass
        elif tag == "note":
            if document is not None:
                body = next((c for c in child if _local(c.tag) == "note-body"), None)
                paragraphs = (
                    [_text_of(p, document).strip() for p in body]
                    if body is not None
                    else []
                )
                document.notes.append(" ".join(p for p in paragraphs if p))
                add(f"[^{len(document.notes)}]", _PLAIN, True)
        elif tag == "a":
            add(_link(_text_of(child), child.get(_XLINK_HREF, "")))
        elif tag == "frame" and document is not None:
            picture = _frame_markdown(child, document)
            if picture is not None:
                add(picture, _PLAIN, True)
            else:
                _runs(child, document, emphasis, out)
        elif tag == "span" and document is not None:
            style = child.get(f"{{{_TEXT_NS}}}style-name")
            _runs(child, document, document.emphasis("text", style, emphasis), out)
        else:
            _runs(child, document, emphasis, out)
        add(child.tail)


def _emphasize(text: str, emphasis: Emphasis) -> str:
    """*text* between the markers of *emphasis*, line by line, with its
    surrounding whitespace outside them."""
    marker = "*" * (2 * emphasis[0] + emphasis[1])
    if not marker:
        return text
    lines = []
    for line in text.split("\n"):
        core = line.strip()
        if core:
            start = line.index(core)
            line = f"{line[:start]}{marker}{core}{marker}{line[start + len(core) :]}"
        lines.append(line)
    return "\n".join(lines)


def _render_runs(runs: list[_Run], base: Emphasis = _PLAIN) -> str:
    """The runs as Markdown; text in emphasis *base* gets no markers of its
    own, so a bold paragraph is one bold span."""
    parts: list[str] = []
    group: list[str] = []
    group_style: Emphasis | None = None

    def flush() -> None:
        if group and group_style is not None:
            text = "".join(group)
            if group_style == base:
                parts.append(text)
            else:
                # Only what differs from the base; an italic run in a bold
                # paragraph is marked italic inside the paragraph's bold
                parts.append(
                    _emphasize(
                        text,
                        (
                            group_style[0] and not base[0],
                            group_style[1] and not base[1],
                        ),
                    )
                )
        group.clear()

    for text, style, raw in runs:
        if raw:
            flush()
            group_style = None
            parts.append(text)
            continue
        if style != group_style:
            flush()
            group_style = style
        group.append(text)
    flush()
    text = "".join(parts)
    return text if base == _PLAIN else _emphasize(text.strip(), base)


def _text_of(
    elem: ElementTree.Element,
    document: _Document | None = None,
    emphasis: Emphasis = _PLAIN,
) -> str:
    """Flatten an element's text, honouring the ODF whitespace elements.

    A footnote or endnote becomes a ``[^n]`` marker with its body added to
    the document's notes, a picture a ``data:`` URI image, a hyperlink a
    Markdown link, and comments are not text. Without a *document* (a
    spreadsheet cell) there are no notes, pictures or emphasis. *emphasis*
    is the paragraph's own: the whole text is marked with it once.
    """
    runs: list[_Run] = []
    _runs(elem, document, emphasis, runs)
    return _render_runs(runs, emphasis)


def _open_archive(input_path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(input_path)
    except zipfile.BadZipFile as exc:
        reason = not_a_package(input_path, "OpenDocument")
        raise _OdfError(reason or f"not a readable zip archive: {exc}") from exc
    except OSError as exc:
        raise _OdfError(f"cannot read the archive: {exc}") from exc


def _member_root(
    archive: zipfile.ZipFile, name: str, *, required: bool = True
) -> ElementTree.Element | None:
    """The parsed XML of archive member *name* (None when it is optional
    and missing)."""
    try:
        info = archive.getinfo(name)
    except KeyError:
        if not required:
            return None
        raise _OdfError(f"no {name} in the archive") from None
    if info.file_size > _MAX_CONTENT_BYTES:
        raise _OdfError(
            f"{name} is {info.file_size} bytes, over the "
            f"{_MAX_CONTENT_BYTES} byte limit"
        )
    try:
        data = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise _OdfError(f"cannot read {name}: {exc}") from exc
    if _DECLARATION_RE.search(data):
        raise _OdfError(
            f"{name} carries a DOCTYPE or ENTITY declaration, which the "
            "stdlib parser would expand (billion laughs)"
        )
    try:
        return ElementTree.fromstring(data)  # nosec B314 - refused above
    except ElementTree.ParseError as exc:
        raise _OdfError(f"malformed {name}: {exc}") from exc


def _content_root(input_path: Path) -> ElementTree.Element:
    with _open_archive(input_path) as archive:
        root = _member_root(archive, "content.xml")
    assert root is not None
    return root


def _cell_text(cell: ElementTree.Element, document: _Document | None = None) -> str:
    blocks = [
        _text_of(child, document).strip()
        for child in cell
        if _local(child.tag) not in _SKIPPED_TAGS
    ]
    return "\n".join(block for block in blocks if block)


def _table_rows(
    table: ElementTree.Element,
    spans: Spans | None = None,
    document: _Document | None = None,
) -> list[list[str]]:
    """Read a table into a rectangular matrix, trailing blanks trimmed.

    ``spans`` collects (row, column) -> column span for merged cells, and
    ``document`` the footnotes and pictures of a text document's cells.
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
            text = _cell_text(cell, document)
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


def _paragraph_text(elem: ElementTree.Element, document: _Document) -> str:
    """A paragraph's text, marked with its paragraph style's emphasis."""
    style = elem.get(f"{{{_TEXT_NS}}}style-name")
    return _text_of(elem, document, document.emphasis("paragraph", style, _PLAIN))


def _heading(elem: ElementTree.Element, document: _Document) -> str | None:
    text = _text_of(elem, document).strip()
    if not text:
        return None
    try:
        level = int(elem.get(f"{{{_TEXT_NS}}}outline-level", "1"))
    except ValueError:
        level = 1
    return f"{'#' * max(1, min(level, _MAX_HEADING_LEVEL))} {text}"


def _render_list(
    elem: ElementTree.Element,
    document: _Document,
    out: list[str],
    style: str | None,
    level: int = 1,
    indent: str = "",
) -> None:
    """A list's items as Markdown lines in *out*: numbered where the list
    style numbers that level, every further paragraph of an item, and a
    list header (an unnumbered entry), indented under the item. A heading
    in a list (outline numbering) is a heading block of its own."""
    style = elem.get(f"{{{_TEXT_NS}}}style-name") or style
    number = document.lists.get(style or "", {}).get(level)
    continues = elem.get(f"{{{_TEXT_NS}}}continue-numbering") == "true" or bool(
        elem.get(f"{{{_TEXT_NS}}}continue-list")
    )
    if number is not None and level == 1 and continues and style:
        number = document.next_number.get(style, number)
    for item in elem:
        item_tag = _local(item.tag)
        if item_tag not in {"list-item", "list-header"}:
            continue
        marker = None
        if item_tag == "list-item":
            if number is not None:
                number = _int_attribute(item, "start-value", number)
                marker = f"{number}. "
                number += 1
            else:
                marker = "- "
        child_indent = indent + " " * len(marker or "")
        for child in item:
            tag = _local(child.tag)
            if tag == "list":
                _render_list(child, document, out, style, level + 1, child_indent)
                continue
            if tag == "h":
                heading = _heading(child, document)
                if heading:
                    # out holds the lines of one list block: a heading
                    # closes it, and the block after it starts a new one
                    out.append(f"\n{heading}\n")
                marker = None
                continue
            text = _paragraph_text(child, document).strip()
            if not text:
                continue
            lines = text.split("\n")
            first = f"{indent}{marker}" if marker else child_indent
            out.append(first + lines[0])
            out.extend(child_indent + line if line else line for line in lines[1:])
            marker = None
    if number is not None and level == 1 and style:
        document.next_number[style] = number


def _render_text_element(
    elem: ElementTree.Element, out: list[str], document: _Document
) -> None:
    tag = _local(elem.tag)
    if tag in _SKIPPED_TAGS:
        return
    if tag == "h":
        heading = _heading(elem, document)
        if heading:
            out.append(heading)
    elif tag == "p":
        text = _paragraph_text(elem, document).strip()
        if text:
            out.append(text)
    elif tag == "list":
        items: list[str] = []
        _render_list(elem, document, items, None)
        out.extend(
            block.strip("\n") for block in "\n".join(items).split("\n\n") if block
        )
    elif tag == "table":
        table = render_markdown_table(_table_rows(elem, document=document))
        if table:
            out.append(table)
    else:
        # Sections, frames, change-tracking wrappers: keep walking.
        for child in elem:
            _render_text_element(child, out, document)


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
        blocks: list[str] = []
        document = _Document()
        try:
            with _open_archive(input_path) as archive:
                root = _member_root(archive, "content.xml")
                assert root is not None
                styles = _member_root(archive, "styles.xml", required=False)
                # The named styles; styles.xml's automatic ones belong to
                # page headers and footers, and reuse the body's names
                if styles is not None:
                    _read_styles(styles, document, frozenset({"styles"}))
                _read_styles(root, document, frozenset({"automatic-styles"}))
                document.archive = archive
                for body in root.iter(f"{{{_OFFICE_NS}}}text"):
                    for child in body:
                        _render_text_element(child, blocks, document)
                    break
        except _OdfError as exc:
            logger.warning("[OdtConverter] {}: {}", input_path.name, exc)
            conversion_failed(str(exc))
        notes = document.notes

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
