"""The document JSON: one conversion result as a versioned object with stable
fields (``-f json``; markitai-rust's ``docs/json-output.md``, schema
``document-v1``).

The same object markitai-rust writes, key for key: every key is always
present (a value that does not apply or is unknown is ``None``, an empty
list ``[]``), keys keep the order below, and within schema version 1 keys are
only added. Pages, pictures, tables and the plain text are read from the
Markdown with the CommonMark + GFM-table rules markitai-rust's pulldown-cmark
applies, so both backends describe the same Markdown the same way.

The Markdown files stay the published result: the document is derived from
them (and from what the conversion knows beside them: warnings, model usage,
picture analyses) once they are written.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Every document carries it. Minor versions add keys; a removed, renamed
#: or retyped key would be version 2.
SCHEMA_VERSION = "1.0"

#: The batch index beside the per-file documents.
INDEX_NAME = "markitai.index.json"

#: Front matter keys in the order the Markdown writes them (markitai-rust's
#: ``output::FIELD_ORDER``); other keys follow in name order.
FIELD_ORDER = (
    "title",
    "source",
    "author",
    "authors",
    "language",
    "publisher",
    "date",
    "site",
    "published",
    "canonical_url",
    "description",
    "tags",
    "markitai_processed",
    "fetch_strategy",
)

# Rust's ``char::is_whitespace`` (Unicode White_Space): Python's str.strip()
# also strips U+001C..U+001F, which are text there.
_WHITESPACE = "\t\n\x0b\x0c\r \x85\xa0                　"


def _trim(text: str) -> str:
    return text.strip(_WHITESPACE)


def _trim_end(text: str) -> str:
    return text.rstrip(_WHITESPACE)


def _lines(text: str) -> list[str]:
    """Rust's ``str::lines``: split at ``\\n``, drop one ``\\r`` before it and
    the empty piece after a final newline."""
    if not text:
        return []
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    return [part[:-1] if part.endswith("\r") else part for part in parts]


# ---------------------------------------------------------------------------
# Notice codes (markitai-rust's notices.rs)
# ---------------------------------------------------------------------------

_SUBJECT_NONE = 0
#: ``PDF page 3:`` / ``Office pages 2–3`` / ``TIFF page 4``: the page numbers
_SUBJECT_PAGE = 1
#: ``Image analysis failed for asset-1.emf:``: the image file name
_SUBJECT_IMAGE_AFTER_FOR = 2
#: ``PDF embedded image name:``: the image name
_SUBJECT_EMBEDDED_IMAGE = 3
#: ``Hidden worksheet "Notes"``: the quoted sheet name
_SUBJECT_QUOTED = 4

# (code, level, starts with, contains, subject); the first match wins, so
# specific families come before the general ones of the same prefix. The
# table is markitai-rust's, row for row; codes are only ever added.
_FAMILIES: tuple[tuple[str, str, str, str, int], ...] = (
    (
        "pdf.page_without_text",
        "warning",
        "PDF page ",
        ": native text was not recovered",
        _SUBJECT_PAGE,
    ),
    (
        "pdf.page_plain_text",
        "info",
        "PDF page ",
        ": read as plain text because ",
        _SUBJECT_PAGE,
    ),
    ("pdf.page", "warning", "PDF page ", "", _SUBJECT_PAGE),
    (
        "pdf.images_after_text",
        "info",
        "PDF images are placed after their page's text",
        "",
        _SUBJECT_NONE,
    ),
    (
        "pdf.images_after_text",
        "info",
        "Embedded PDF images are placed after their page's text",
        "",
        _SUBJECT_NONE,
    ),
    (
        "pdf.embedded_image",
        "warning",
        "PDF embedded image ",
        "",
        _SUBJECT_EMBEDDED_IMAGE,
    ),
    ("pdf.layout", "warning", "PDF layout refinement", "", _SUBJECT_NONE),
    ("pdf.hidden_text", "warning", "PDF hidden-text removal", "", _SUBJECT_NONE),
    ("pdf.hidden_text", "warning", "PDF hidden text: ", "", _SUBJECT_NONE),
    (
        "ocr.no_text",
        "warning",
        "Local OCR found no readable text on TIFF page ",
        "",
        _SUBJECT_PAGE,
    ),
    ("ocr.no_text", "warning", "Local OCR found no readable text", "", _SUBJECT_NONE),
    ("ocr.capped", "warning", "Local OCR read only the first ", "", _SUBJECT_PAGE),
    ("ocr.unread", "warning", "Local OCR could not read ", "", _SUBJECT_PAGE),
    ("ocr.unavailable", "warning", "Local OCR", "", _SUBJECT_NONE),
    (
        "llm.no_model_succeeded",
        "warning",
        "No model succeeded for this document",
        "",
        _SUBJECT_NONE,
    ),
    ("llm.enhancement_failed", "warning", "LLM enhancement failed", "", _SUBJECT_NONE),
    (
        "llm.answer_retried",
        "info",
        "The model's ",
        "and a later answer was used",
        _SUBJECT_NONE,
    ),
    (
        "llm.answer_retried",
        "info",
        "The provider withheld the ",
        "and a later answer was used",
        _SUBJECT_NONE,
    ),
    ("llm.answer_rejected", "warning", "LLM ", " answer was rejected", _SUBJECT_NONE),
    ("llm.answer_rejected", "warning", "The provider withheld the ", "", _SUBJECT_NONE),
    (
        "llm.answer_rejected",
        "warning",
        "A visual batch answer was rejected",
        "",
        _SUBJECT_NONE,
    ),
    (
        "llm.cache_unavailable",
        "warning",
        "Persistent LLM cache is unavailable",
        "",
        _SUBJECT_NONE,
    ),
    (
        "llm.cache_entry_ignored",
        "warning",
        "A malformed ",
        " cache entry was ignored",
        _SUBJECT_NONE,
    ),
    (
        "llm.text_fallback",
        "warning",
        "Rendered-page enhancement failed",
        "",
        _SUBJECT_NONE,
    ),
    ("llm.cost_incomplete", "warning", "Cost is incomplete", "", _SUBJECT_NONE),
    ("text_only.no_text", "warning", "No text was recognised", "", _SUBJECT_NONE),
    (
        "image.analysis_failed",
        "warning",
        "Image analysis failed for ",
        "",
        _SUBJECT_IMAGE_AFTER_FOR,
    ),
    ("image.analysis_failed", "warning", "Image analysis failed", "", _SUBJECT_NONE),
    ("image.not_localized", "warning", "Image resource ", "", _SUBJECT_NONE),
    (
        "office.renderer_missing",
        "warning",
        "Page screenshots of Office documents need",
        "",
        _SUBJECT_NONE,
    ),
    (
        "office.renderer_missing",
        "warning",
        "Page screenshots and OCR of Office pages need",
        "",
        _SUBJECT_NONE,
    ),
    (
        "office.renderer_missing",
        "warning",
        "OCR of Office pages needs",
        "",
        _SUBJECT_NONE,
    ),
    ("office.page", "warning", "Office page", "", _SUBJECT_PAGE),
    (
        "office.screenshot_format",
        "info",
        "Office screenshots use lossless WebP",
        "",
        _SUBJECT_NONE,
    ),
    (
        "office.count_verified",
        "info",
        "",
        " count was verified against LibreOffice",
        _SUBJECT_NONE,
    ),
    ("spreadsheet.hidden_sheet", "info", "Hidden worksheet ", "", _SUBJECT_QUOTED),
    (
        "spreadsheet.screenshot_scope",
        "info",
        "Workbook screenshots use complete-sheet export",
        "",
        _SUBJECT_NONE,
    ),
    (
        "spreadsheet.font_color",
        "info",
        "Workbook fonts with no declared color",
        "",
        _SUBJECT_NONE,
    ),
    ("spreadsheet.rendering", "warning", "Workbook ", "", _SUBJECT_NONE),
    (
        "spreadsheet.numbers_formatting",
        "info",
        "Numbers display formatting is normalized",
        "",
        _SUBJECT_NONE,
    ),
    (
        "spreadsheet.numbers_objects",
        "warning",
        "Numbers non-table canvas objects",
        "",
        _SUBJECT_NONE,
    ),
    ("word.picture_records", "warning", "Word picture metadata", "", _SUBJECT_NONE),
    (
        "table.merged_cells",
        "warning",
        "Merged table cells are represented",
        "",
        _SUBJECT_NONE,
    ),
    ("table.malformed", "warning", "pipe table at line ", "", _SUBJECT_NONE),
    ("mail.eml", "warning", "EML ", "", _SUBJECT_NONE),
    ("mail.msg", "warning", "MSG ", "", _SUBJECT_NONE),
    (
        "browser.screenshot_failed",
        "warning",
        "Browser screenshot failed",
        "",
        _SUBJECT_NONE,
    ),
    (
        "cloudflare.not_used",
        "warning",
        "The Cloudflare backend was not used",
        "",
        _SUBJECT_NONE,
    ),
)

# Python's own sentences for the same families (its notices name their
# stage and file first: "[PDF] report.pdf: ..."). After markitai-rust's rows,
# so a sentence both write keeps its code; no new code is made here.
_FAMILIES += (
    (
        "pdf.page_without_text",
        "warning",
        "[PDF] ",
        " look scanned",
        _SUBJECT_PAGE,
    ),
    (
        "pdf.hidden_text",
        "warning",
        "[PDF] ",
        " hidden text span(s) detected",
        _SUBJECT_NONE,
    ),
    ("pdf.page", "warning", "[PDF] ", "", _SUBJECT_PAGE),
    ("ocr.unread", "warning", "[OCR] Could not read ", "", _SUBJECT_NONE),
    ("ocr.no_text", "warning", "[OCR] No text found in ", "", _SUBJECT_NONE),
    ("ocr.unavailable", "warning", "[OCR] ", "", _SUBJECT_NONE),
    (
        "office.renderer_missing",
        "warning",
        "[PPTX] Cannot render slides of ",
        " found",
        _SUBJECT_NONE,
    ),
    ("office.page", "warning", "[PPTX] Cannot render slides of ", "", _SUBJECT_NONE),
    ("mail.msg", "warning", "[MSG] ", "", _SUBJECT_NONE),
    (
        "llm.cost_incomplete",
        "warning",
        "No price is known for LLM model ",
        "",
        _SUBJECT_NONE,
    ),
    (
        "llm.cost_incomplete",
        "warning",
        "LLM request to ",
        "the cost is incomplete",
        _SUBJECT_NONE,
    ),
)

_PAGE_SUBJECT = re.compile(r"\bpages? (\d+)(?:\s*[-–]\s*(\d+))?")
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Classified:
    """A warning read as a family: its stable code, level and subjects."""

    code: str
    level: str
    items: list[str] = field(default_factory=list)


def classify_notice(message: str) -> Classified:
    """The family of one warning sentence; ``other`` when none claims it."""
    family = next(
        (
            row
            for row in _FAMILIES
            if message.startswith(row[2]) and (not row[3] or row[3] in message)
        ),
        None,
    )
    if family is None:
        return Classified("other", "warning", [])
    code, level, prefix, _, subject = family
    rest = message[len(prefix) :]
    items: list[str] = []
    if subject == _SUBJECT_PAGE:
        found = _PAGE_SUBJECT.search(message)
        if found:
            first = _ascii_int(found.group(1))
            last = _ascii_int(found.group(2)) if found.group(2) else first
            if last < first or last - first >= 10_000:
                last = first
            items = [str(page) for page in range(first, last + 1)]
    elif subject == _SUBJECT_IMAGE_AFTER_FOR:
        name, separator, _ = rest.partition(": ")
        items = [name] if separator else []
    elif subject == _SUBJECT_EMBEDDED_IMAGE:
        name, separator, _ = rest.partition(":")
        items = [name] if separator else []
    elif subject == _SUBJECT_QUOTED:
        found = _QUOTED.search(rest)
        if found:
            items = [found.group(1).replace('\\"', '"').replace("\\\\", "\\")]
    return Classified(code, level, items)


def _ascii_int(text: str) -> int:
    """*text* as a number when it is ASCII digits, else 0 (Rust's
    ``parse().unwrap_or(0)``)."""
    return int(text) if text.isascii() and text.isdigit() else 0


# ---------------------------------------------------------------------------
# Reading the Markdown
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _parser() -> Any:
    """CommonMark with GFM tables and strikethrough, as pulldown-cmark reads
    it for markitai-rust: link targets as written (no percent-encoding) and
    every scheme (stdout output links pictures as ``file://`` URIs)."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    md.normalizeLink = lambda url: url  # type: ignore[method-assign]
    md.validateLink = lambda _url: True  # type: ignore[method-assign]
    for rule in md.block.ruler.__rules__:
        if rule.name in ("paragraph", "lheading", "heading"):
            md.block.ruler.at(
                rule.name, _spaces_and_tabs_stripped(rule.fn), {"alt": rule.alt}
            )
        elif rule.name == "table":
            md.block.ruler.at(
                rule.name, _table_cells_stripped(rule.fn), {"alt": rule.alt}
            )
    return md


def _table_cells_stripped(rule: Any) -> Any:
    """The table rule with its cells stripped of spaces and tabs only (see
    :func:`_spaces_and_tabs_stripped`)."""
    from markdown_it.rules_block.table import escapedSplit, getLine

    def wrapped(state: Any, start: int, end: int, silent: bool) -> bool:
        count = len(state.tokens)
        if not rule(state, start, end, silent):
            return False
        if silent:
            return True
        cells: list[str] = []
        for token in state.tokens[count:]:
            if token.type == "tr_open" and token.map:
                cells = escapedSplit(getLine(state, token.map[0]).strip(" \t"))
                if cells and cells[0] == "":
                    cells.pop(0)
                if cells and cells[-1] == "":
                    cells.pop()
            elif token.type == "inline":
                token.content = cells.pop(0).strip(" \t") if cells else ""
        return True

    return wrapped


def _spaces_and_tabs_stripped(rule: Any) -> Any:
    """*rule* with its text stripped of spaces and tabs only, as CommonMark
    says: markdown-it strips every Unicode space, so a Chinese paragraph
    lost the ideographic spaces (U+3000) that indent it."""

    def wrapped(state: Any, start: int, end: int, silent: bool) -> bool:
        count = len(state.tokens)
        if not rule(state, start, end, silent):
            return False
        if silent or len(state.tokens) < count + 2:
            return True
        inline = state.tokens[count + 1]
        if inline.type != "inline" or not inline.map:
            return True
        first, last = inline.map
        if state.tokens[count].markup.startswith("#"):
            # ATX: widen the stripped text over the Unicode spaces beside it
            line = state.src[
                state.bMarks[first] + state.tShift[first] : state.eMarks[first]
            ]
            content = inline.content
            at = (
                line.find(content, len(line) - len(line.lstrip("#"))) if content else -1
            )
            if at > 0:
                begin, finish = at, at + len(content)
                while (
                    begin > 0
                    and line[begin - 1] not in " \t#"
                    and line[begin - 1].isspace()
                ):
                    begin -= 1
                while (
                    finish < len(line)
                    and line[finish] not in " \t"
                    and line[finish].isspace()
                ):
                    finish += 1
                inline.content = line[begin:finish]
        else:
            inline.content = state.getLines(first, last, state.blkIndent, False).strip(
                " \t\n"
            )
        return True

    return wrapped


_TASK_MARKER = re.compile(r"^\[[ xX]\][ \t]+")


def _block_tokens(markdown: str) -> list[Any]:
    tokens = _parser().parse(markdown)
    # pulldown-cmark reads a GFM task marker (`- [x] done`) as its own event,
    # which neither reader below uses: the item's text starts after it
    for index, token in enumerate(tokens):
        if (
            token.type == "inline"
            and index >= 2
            and tokens[index - 1].type == "paragraph_open"
            and tokens[index - 2].type == "list_item_open"
            and token.children
            and token.children[0].type == "text"
        ):
            first = token.children[0]
            marker = _TASK_MARKER.match(first.content)
            if marker is None and first.content in ("[ ]", "[x]", "[X]"):
                rest = token.children[1:]
                if rest and rest[0].type in ("softbreak", "hardbreak"):
                    first.content = ""
            elif marker is not None:
                first.content = first.content[marker.end() :]
    return tokens


def _picture_alt(token: Any) -> str:
    """An image's alt text: its text and code spans, as pulldown-cmark's
    events give them."""
    parts: list[str] = []
    for child in token.children or []:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "image":
            parts.append(_picture_alt(child))
    return "".join(parts)


@dataclass
class _Scan:
    #: ``(target, alt)`` of each picture, screenshots excluded
    images: list[tuple[str, str]] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    tables: list[tuple[list[str], list[list[str]]]] = field(default_factory=list)


def _is_screenshot(target: str, names: set[str]) -> bool:
    return (
        ".markitai/screenshots/" in target
        or target.startswith("screenshots/")
        or target.rsplit("/", 1)[-1] in names
    )


def _scan(markdown: str, names: set[str]) -> _Scan:
    """Pictures and tables of a run of Markdown."""
    found = _Scan()
    table: tuple[list[str], list[list[str]]] | None = None
    in_head = False
    row: list[str] = []

    def picture(token: Any, cell: list[str] | None) -> None:
        target = str(token.attrGet("src") or "")
        alt = _picture_alt(token)
        if cell is not None:
            cell.append(alt)
        if _is_screenshot(target, names):
            found.screenshots.append(target)
        else:
            found.images.append((target, alt))

    for token in _block_tokens(markdown):
        kind = token.type
        if kind == "table_open":
            table = ([], [])
        elif kind == "table_close":
            if table is not None:
                found.tables.append(table)
            table = None
        elif kind == "thead_open":
            in_head = True
        elif kind == "thead_close":
            in_head = False
        elif kind == "tr_close":
            if table is not None:
                if in_head:
                    table[0][:] = row
                else:
                    table[1].append(row)
            row = []
        elif kind == "inline":
            cell: list[str] | None = [] if table is not None else None
            for child in token.children or []:
                if child.type == "image":
                    picture(child, cell)
                elif cell is not None:
                    if child.type in ("text", "code_inline"):
                        cell.append(child.content)
                    elif child.type in ("softbreak", "hardbreak"):
                        cell.append(" ")
            if cell is not None:
                row.append(_trim("".join(cell)))
    return found


_TAG = re.compile(r"<[^>]*>", re.DOTALL)


def plain_text(markdown: str) -> str:
    """The text of Markdown without its markup: no pictures, links reduced to
    their text, table cells separated by tabs, blocks by a blank line."""
    out: list[str] = []

    def text() -> str:
        return "".join(out)

    def block_end() -> None:
        current = text().rstrip(" \t")
        out[:] = [current]
        if current and not current.endswith("\n\n"):
            out.append("\n" if current.endswith("\n") else "\n\n")

    def html(content: str) -> None:
        content = content.lstrip(_WHITESPACE)
        if not content.startswith("<!--"):
            stripped = _trim(_TAG.sub("", content))
            if stripped:
                out.append(stripped)
                out.append(" ")

    first_cell = True
    for token in _block_tokens(markdown):
        kind = token.type
        if kind == "inline":
            for child in token.children or []:
                if child.type in ("text", "code_inline"):
                    out.append(child.content)
                elif child.type in ("softbreak", "hardbreak"):
                    out.append("\n")
                elif child.type == "html_inline":
                    html(child.content)
        elif kind in ("fence", "code_block"):
            out.append(token.content)
            block_end()
        elif kind == "html_block":
            # pulldown-cmark gives an HTML block line by line
            for line in _split_lines_keepends(token.content):
                html(line)
            block_end()
        elif kind == "tr_open":
            first_cell = True
        elif kind in ("th_open", "td_open"):
            if not first_cell:
                out.append("\t")
            first_cell = False
        elif kind == "tr_close":
            out.append("\n")
        elif kind == "list_item_close":
            if not text().endswith("\n"):
                out.append("\n")
        elif kind == "paragraph_close" and token.hidden:
            # A tight list's paragraphs are no blocks of their own
            continue
        elif kind in (
            "paragraph_close",
            "heading_close",
            "blockquote_close",
            "bullet_list_close",
            "ordered_list_close",
            "table_close",
            "hr",
        ):
            block_end()
    lines: list[str] = []
    for line in _lines(text()):
        line = _trim_end(line)
        if not (not line and (not lines or not lines[-1])):
            lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

# The converters' page and slide markers, and the rag profile's
_MARKER = re.compile(r"^\s*<!--\s*(?:(?:Page|Slide) number|page):\s*(\d+)\s*-->\s*$")
_COMMENT_PICTURE = re.compile(r"<!--\s*!\[[^\]]*\]\(([^)\s]+)\)\s*-->")
_SHOT_PAGE = re.compile(r"\.page(\d+)\.[A-Za-z0-9]+$")
_SHOT_FILE = re.compile(r"\.(?:page|slide)(\d+)\.[A-Za-z0-9]+$")


@dataclass
class _Segment:
    page: int | None
    text: str


def _fence_run(trimmed: str) -> tuple[str, int] | None:
    ticks = len(trimmed) - len(trimmed.lstrip("`"))
    if ticks >= 3:
        return "`", ticks
    tildes = len(trimmed) - len(trimmed.lstrip("~"))
    if tildes >= 3:
        return "~", tildes
    return None


def _segments(markdown: str) -> list[_Segment]:
    """Split at page and slide markers outside fenced code."""
    segments: list[_Segment] = []
    start = 0
    offset = 0
    page: int | None = None
    fence: tuple[str, int] | None = None
    for line in _split_lines_keepends(markdown):
        trimmed = line.lstrip(_WHITESPACE)
        found_run = _fence_run(trimmed)
        if fence is None and found_run is not None:
            fence = found_run
        elif (
            fence is not None
            and found_run is not None
            and fence[0] == found_run[0]
            and found_run[1] >= fence[1]
            and not _trim(trimmed[found_run[1] :])
        ):
            fence = None
        marker = _MARKER.match(line.rstrip("\r\n")) if fence is None else None
        if marker is not None and marker.group(1).isascii():
            if offset > start or page is not None:
                segments.append(_Segment(page, markdown[start:offset]))
            page = int(marker.group(1))
            start = offset + len(line)
        offset += len(line)
    if start < len(markdown) or page is not None:
        segments.append(_Segment(page, markdown[start:]))
    return segments


def _split_lines_keepends(markdown: str) -> list[str]:
    """Rust's ``split_inclusive('\\n')``."""
    if not markdown:
        return []
    parts = markdown.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _screenshot_targets(text: str, names: set[str]) -> list[str]:
    """Screenshots a page links, in comments (``<!-- ![Page 1](…) -->``) or
    visibly."""
    targets = [
        found.group(1)
        for found in _COMMENT_PICTURE.finditer(text)
        if _is_screenshot(found.group(1), names)
    ]
    for target in _scan(text, names).screenshots:
        if target not in targets:
            targets.append(target)
    return targets


def _page_of_screenshot(target: str) -> int | None:
    found = _SHOT_PAGE.search(target)
    return int(found.group(1)) if found and found.group(1).isascii() else None


def _without_screenshot_comments(text: str) -> str:
    """A page's Markdown without the comments that only link its screenshot."""
    kept = []
    for line in _lines(text):
        trimmed = _trim(line)
        if trimmed == "<!-- Page images for reference -->":
            continue
        found = _COMMENT_PICTURE.search(trimmed)
        if found and found.start() == 0 and found.end() == len(trimmed):
            continue
        kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


@dataclass
class DocumentParts:
    """What one finished conversion knows, for its document JSON.

    ``base`` and ``llm`` are the published Markdown bodies (front matter
    removed); ``paged`` the final body with its page markers when the
    published one has none. ``analysis`` holds the picture analyses
    (``{"asset", "alt", "desc", "text"}``) and ``llm_usage`` the per-model
    usage the conversion tallied.
    """

    source: str
    base: str = ""
    llm: str | None = None
    paged: str | None = None
    frontmatter: Mapping[str, Any] = field(default_factory=dict)
    source_label: str | None = None
    source_name: str | None = None
    skip_reason: str | None = None
    warnings: Sequence[str] = ()
    llm_usage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    analysis: Sequence[Mapping[str, Any]] = ()
    assets: Sequence[Path] = ()
    screenshots: Sequence[Path] = ()
    output_path: Path | None = None
    llm_output_path: Path | None = None
    duration_s: float = 0.0


def _round(value: float, factor: float) -> float:
    """Rust's ``f64::round`` (half away from zero) at ``1/factor``."""
    scaled = value * factor
    rounded = math.copysign(math.floor(abs(scaled) + 0.5), scaled) / factor
    return 0.0 if rounded == 0 else rounded


def _shown_path(path: Path, base: Path | None) -> str:
    """*path* relative to *base* with ``/`` separators when inside it (also
    when only their resolved forms agree: macOS's ``/var`` is
    ``/private/var``), else as it is."""
    if base is not None:
        for inner, outer in ((path, base), (path.resolve(), base.resolve())):
            try:
                relative = inner.relative_to(outer)
            except (ValueError, OSError):
                continue
            if relative.parts:
                return "/".join(relative.parts)
    return str(path)


def is_url(source: str) -> bool:
    lowered = source[:8].lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _secret_parameter(name: str) -> bool:
    from markitai.utils.url_redaction import secret_parameter

    return secret_parameter(name)


def document_url(url: str) -> str:
    """The URL without credentials, a secret query or fragment value
    reading ``REDACTED`` (markitai-rust's ``output::redact_url``)."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return url
    if not host:
        return url
    netloc = parts.netloc.rpartition("@")[2]

    def redacted(text: str) -> str:
        pairs = parse_qsl(text, keep_blank_values=True)
        if not any(_secret_parameter(key) for key, _ in pairs):
            return text
        return urlencode(
            [
                (key, "REDACTED" if _secret_parameter(key) else value)
                for key, value in pairs
            ]
        )

    return urlunsplit(
        (
            parts.scheme,
            netloc,
            parts.path,
            redacted(parts.query),
            redacted(parts.fragment),
        )
    )


def _measure(path: Path) -> tuple[int, str] | None:
    """Size and SHA-256 of a regular file."""
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(64 * 1024), b""):
                size += len(block)
                digest.update(block)
    except OSError:
        return None
    return size, digest.hexdigest()


def _source(parts: DocumentParts) -> dict[str, Any]:
    source = parts.source
    if is_url(source):
        name = parts.source_name
        if not name:
            from markitai.utils.cli_helpers import url_to_filename

            name = url_to_filename(source).removesuffix(".md")
        return {
            "kind": "url",
            "path": None,
            "url": document_url(source),
            "name": name,
            "type": None,
            "size": None,
            "sha256": None,
        }
    shown = parts.source_label or source
    name = PurePosixPath(shown.replace("\\", "/")).name or shown
    suffix = PurePosixPath(name).suffix
    measured = _measure(Path(source).expanduser())
    return {
        "kind": "file",
        "path": shown,
        "url": None,
        "name": name,
        # Python reads a file by its extension
        "type": suffix[1:].lower() if len(suffix) > 1 else None,
        "size": measured[0] if measured else None,
        "sha256": measured[1] if measured else None,
    }


def _properties(frontmatter: Mapping[str, Any]) -> dict[str, Any]:
    ordered = {key: frontmatter[key] for key in FIELD_ORDER if key in frontmatter}
    for key in sorted(frontmatter):
        if key not in ordered:
            ordered[key] = frontmatter[key]
    return ordered


def _analysis_for(
    analysis: Sequence[Mapping[str, Any]], target: str
) -> Mapping[str, Any] | None:
    """The analysis record of the picture the Markdown links as *target*."""
    wanted = PurePosixPath(target.removeprefix("./")).parts
    if not wanted:
        return None
    for entry in analysis:
        asset = entry.get("asset")
        if isinstance(asset, str):
            parts = PurePosixPath(asset.replace("\\", "/")).parts
            if parts[len(parts) - len(wanted) :] == wanted and len(parts) >= len(
                wanted
            ):
                return entry
    return None


def _text_field(entry: Mapping[str, Any] | None, key: str) -> str | None:
    value = entry.get(key) if entry is not None else None
    if isinstance(value, str) and _trim(value):
        return _trim(value)
    return None


def _count(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _cost(value: Any) -> float:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0.0
    )


def build_document(
    parts: DocumentParts, base_dir: Path | None = None
) -> dict[str, Any]:
    """The document JSON of a finished conversion, as a dict in key order.

    *base_dir* is the folder the document is written to: output paths are
    given relative to it when they are inside it.
    """
    from markitai.types import usage_has_unpriced

    final_body = parts.llm if parts.llm is not None else parts.base
    paged = parts.paged if parts.paged is not None else final_body
    shot_names = {path.name for path in parts.screenshots}
    segments = _segments(paged)
    shots: dict[int, str] = {}
    for segment in segments:
        for target in _screenshot_targets(segment.text, shot_names):
            number = _page_of_screenshot(target)
            if number is None:
                number = segment.page
            if number is not None:
                shots.setdefault(number, target)
    # Python's Markdown does not link its page renders: they are found by
    # name (``report.pdf.page0002.jpg``, ``deck.pptx.slide0003.jpg``)
    for path in parts.screenshots:
        named = _SHOT_FILE.search(path.name)
        if named and named.group(1).isascii():
            shots.setdefault(int(named.group(1)), _shown_path(path, base_dir))
    pages: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for segment in segments:
        body = _without_screenshot_comments(segment.text)
        scan = _scan(body, shot_names)
        for path, alt in scan.images:
            if all(image["path"] != path for image in images):
                entry = _analysis_for(parts.analysis, path)
                images.append(
                    {
                        "path": path,
                        "alt": alt,
                        "description": _text_field(entry, "desc"),
                        "text": _text_field(entry, "text"),
                        "page": segment.page,
                    }
                )
        tables.extend(
            {"page": segment.page, "header": header, "rows": rows}
            for header, rows in scan.tables
        )
        if segment.page is not None:
            markdown = _trim(body)
            pages.append(
                {
                    "number": segment.page,
                    "text": plain_text(markdown),
                    "markdown": markdown,
                    "images": [path for path, _ in scan.images],
                    "screenshot": shots.get(segment.page),
                }
            )
    usage = parts.llm_usage
    models = [
        {
            "model": model,
            "requests": _count(row, "requests"),
            "input_tokens": _count(row, "input_tokens"),
            "output_tokens": _count(row, "output_tokens"),
            "cost_usd": _round(_cost(row.get("cost_usd")), 1e6),
        }
        for model, row in sorted(usage.items())
    ]
    title = parts.frontmatter.get("title")
    notices = []
    for message in parts.warnings:
        found = classify_notice(message)
        notices.append(
            {
                "code": found.code,
                "level": found.level,
                "message": message,
                "items": found.items,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_identity(),
        "source": _source(parts),
        "status": "skipped" if parts.skip_reason is not None else "completed",
        "skip_reason": parts.skip_reason,
        "title": title if isinstance(title, str) else None,
        "properties": _properties(parts.frontmatter),
        "markdown": {"base": parts.base, "llm": parts.llm},
        "text": plain_text(_without_screenshot_comments(final_body)),
        "pages": pages,
        "images": images,
        "tables": tables,
        "notices": notices,
        "usage": {
            "requests": sum(model["requests"] for model in models),
            "input_tokens": sum(model["input_tokens"] for model in models),
            "output_tokens": sum(model["output_tokens"] for model in models),
            "cost_usd": _round(
                sum(_cost(row.get("cost_usd")) for row in usage.values()), 1e6
            ),
            "cost_complete": not usage_has_unpriced(usage),
            "models": models,
        },
        "timings": {"duration_s": _round(max(parts.duration_s, 0.0), 1e3)},
        "outputs": {
            "markdown": (
                _shown_path(parts.output_path, base_dir) if parts.output_path else None
            ),
            "llm_markdown": (
                _shown_path(parts.llm_output_path, base_dir)
                if parts.llm_output_path
                else None
            ),
            "assets": [_shown_path(path, base_dir) for path in parts.assets],
            "screenshots": [_shown_path(path, base_dir) for path in parts.screenshots],
        },
    }


def tool_identity() -> dict[str, str]:
    """``tool``: this build's name, version and short commit."""
    from markitai import __version__
    from markitai.build_info import build_identity

    return {
        "name": "markitai",
        "version": __version__,
        "commit": build_identity()["commit"] or "unknown",
    }


def render(document: Mapping[str, Any]) -> str:
    """The document as written to a file or stdout: indented, with a final
    newline (serde_json's pretty form)."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
