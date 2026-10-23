"""Reading order and Markdown layout for OCR text boxes.

RapidOCR returns one box per detected text line, sorted top to bottom. That
order is the reading order only for a single upright column. Joined as it
comes, a two-column page interleaves its columns line by line, vertical
Chinese comes out column-reversed, an upside-down scan comes out bottom line
first, and a table becomes one cell per line.

:func:`layout_ocr_text` rebuilds the reading order from the box geometry
alone — no model, no image — so every OCR path (PDF pages, images, the tiled
retry) lays text out the same way:

* vertical text (boxes much taller than wide, holding most of the text)
  reads right to left, column by column;
* on a horizontal page, a box many lines tall and upright (the arXiv stamp
  in the margin, a sidebar, a seal) is set aside and appended, so it cannot
  merge the body lines it overlaps;
* horizontal text splits into columns at a gutter between lines that sit
  side by side; lines reaching into the gutter (titles, footers), and lines
  above the point where every column has begun, are full-width blocks that
  separate the column runs;
* boxes on one visual line join into one line, and runs of rows with the
  same number of widely spaced, aligned cells become a Markdown table;
* a vertical gap taller than most of a line starts a new paragraph.

:func:`structure_ocr_page` adds what the page's pixels show
(:mod:`markitai.ocr_structure`) and the document structure the geometry
implies, with markitai-rust's rules so both give the same Markdown:

* **headings**: a one-piece line set at least 1.25 times the body text's
  size (or bold, 1.1 times, and opening a paragraph), at most 60
  characters and not ending in ``,`` or ``;``; the lines right below it of
  the same size (within 20 %) continue it, three lines at most. The size
  is measured on the ink from the top of the capitals to the baseline
  (``ocr_structure.size_ratios``), the box height only for a line too
  short to measure; the body text's is that of its long lines. Its level
  comes later, from all the document's OCR'd headings
  (:func:`resolve_heading_levels`): sizes within 8 % of a tier's largest
  share its level, the largest tier is ``#``;
* **list items**: a line starting with a bullet glyph the recognizer read,
  or with a bullet blob left of it; the lines right below indented like
  its text continue it; items are written tight (``- a`` then ``- b``);
* ruled **tables** and **figures** where the pixels show them, written at
  their top; a figure's labels are not text;
* **bold and italic** per word, in paragraphs and list items (never in
  headings or tables); a run that goes on across a line break is one run.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

Bounds = tuple[float, float, float, float]

# Blank-line paragraph break when the vertical gap between two lines exceeds
# this fraction of the taller line's height (RapidOCR's to_markdown rule).
PARAGRAPH_GAP_RATIO = 0.8

# A box this much taller than wide is a vertical text column.
_VERTICAL_ASPECT = 1.5
# Vertical columns sit further apart than horizontal lines: a new paragraph
# only past a gap of this many column widths.
_VERTICAL_PARAGRAPH_GAP_RATIO = 2.0
# Share of the recognized characters vertical boxes must hold for the page
# to read vertically...
_VERTICAL_SHARE = 0.6
# ...and the vertical boxes' median height as a share of the text area:
# upright tags beside horizontal text are short, real columns run long.
_VERTICAL_MIN_HEIGHT_SHARE = 0.25
# On a horizontal page, an upright box taller than this many median line
# heights is margin furniture, not part of any line.
_MARGIN_HEIGHT_RATIO = 3.0
# Boxes narrower than this share of the text width may sit in a column.
_NARROW_SHARE = 0.55
# Gutters are only looked for away from the page edges.
_GUTTER_EDGE_SHARE = 0.1
# A gutter is at least this share of the text width (and one line tall).
_GUTTER_MIN_SHARE = 0.02
# Each column needs this many lines, with this median length (CJK counts
# double), before the page is read as columns rather than as a table.
_COLUMN_MIN_LINES = 3
_COLUMN_MIN_TEXT = 8
# Two boxes sit side by side when their heights overlap by this share.
_SIDE_BY_SIDE_OVERLAP = 0.3
# Cells of a table row are separated by more than this many line heights.
_CELL_GAP_RATIO = 1.0
# Two cells a row look just like "label  value" and a hanging-indent list:
# such a run is a table only with this many rows whose cells line up.
_TWO_CELL_TABLE_MIN_ROWS = 3
# An enumerator or bullet in front of a text this long is a list item.
_LIST_ITEM_MIN_TEXT = 16
_LIST_MARKER_RE = re.compile(
    r"^(?:\(?\d+(?:\.\d+)*[.)、]?|\(?[A-Za-z][.)]|[ivxIVX]+[.)]"
    r"|[一二三四五六七八九十]+[、.．]|[•·●○■□◆◇▪▫*\-–—])$"
)


@dataclass(frozen=True)
class _Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF  # kana
        or 0x3400 <= code <= 0x9FFF  # CJK ideographs
        or 0xAC00 <= code <= 0xD7AF  # hangul
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF  # full-width forms
    )


def _display_length(text: str) -> int:
    return sum(2 if _is_cjk(ch) else 1 for ch in text)


# Two CJK boxes closer than this share of the line height are one run of
# text (the detector split it); wider apart, the scan had a space there.
_CJK_JOIN_GAP_RATIO = 0.3


def _join_line(line: list[_Box]) -> str:
    """Join the boxes of one visual line, left to right.

    A space goes between boxes, except between two CJK characters that sit
    next to each other: CJK text has no spaces, and a detector split inside
    it must not add one. A visible gap keeps its space either way.
    """
    out = line[0].text
    for previous, box in zip(line, line[1:]):
        gap = box.x0 - previous.x1
        tight = gap < max(previous.height, box.height) * _CJK_JOIN_GAP_RATIO
        if not (tight and _is_cjk(out[-1]) and _is_cjk(box.text[0])):
            out += " "
        out += box.text
    return out


def layout_ocr_text(
    texts: list[str],
    bounds: list[Bounds | None],
    *,
    upside_down: bool = False,
) -> str:
    """Lay recognized text boxes out as Markdown in reading order.

    Args:
        texts: Recognized text per box.
        bounds: Axis-aligned ``(x0, y0, x1, y1)`` per box, or None when the
            box geometry is unusable.
        upside_down: The page was scanned rotated by 180 degrees; box
            coordinates are flipped before ordering.

    Returns:
        Markdown text. Boxes without geometry are appended, in their
        original order, after the laid-out ones.
    """
    boxes: list[_Box] = []
    loose: list[str] = []
    for text, bound in zip(texts, bounds):
        text = str(text).strip()
        if not text:
            continue
        if bound is None:
            loose.append(text)
        else:
            boxes.append(_Box(text, *bound))

    if upside_down and boxes:
        right = max(box.x1 for box in boxes)
        bottom = max(box.y1 for box in boxes)
        boxes = [
            _Box(
                box.text,
                right - box.x1,
                bottom - box.y1,
                right - box.x0,
                bottom - box.y0,
            )
            for box in boxes
        ]

    margin: list[_Box] = []
    if not boxes:
        body = ""
    elif _is_vertical(boxes):
        body = _render_vertical(boxes)
    else:
        boxes, margin = _split_margin(boxes)
        body = _render_columns(boxes)
    parts = [body, *(box.text for box in sorted(margin, key=lambda b: b.x0))]
    return "\n\n".join(part for part in (*parts, "\n".join(loose)) if part)


# --- structure ---------------------------------------------------------------------

#: A heading line is this much taller than the body text...
_HEADING_RATIO = 1.25
#: ...or bold, this much taller, and opening a paragraph.
_BOLD_HEADING_RATIO = 1.1
#: A heading's next line continues it when this much taller (or bold)...
_HEADING_NEXT_RATIO = 1.15
#: ...and within this share of its first line's height.
_HEADING_NEXT_SIZE = 0.2
_HEADING_MAX_ROWS = 3
_HEADING_MAX_CHARS = 60
_HEADING_OPEN_END = ",;，；、"
#: Heading sizes within this share of a tier's largest share its level
#: (sizes from the ink are steady to a pixel: 18 and 16 pt headings, 27
#: and 24 px at 150 dpi, are two levels, as in the text layer).
_HEADING_TIER = 0.92
_BULLET_GLYPHS = "•●▪■◦‣∙○□➢►▶-–*"
#: A list item's next line continues it when indented at least to the
#: item's text minus this many line heights.
_ITEM_INDENT = 0.5
_HEADING_MARK = re.compile(r"^\x00H(\d+(?:\.\d+)?)\x00 ", re.MULTILINE)


@dataclass
class OcrPage:
    """An OCR'd page laid out with its structure.

    ``markdown`` holds two kinds of placeholders: ``\\x00H<size>\\x00 `` in
    front of a heading (:func:`resolve_heading_levels` turns it into
    ``#``s) and ``\\x00F<n>\\x00`` on its own line where figure ``n`` goes
    (:func:`place_figures`).
    """

    markdown: str
    #: The recognized lines in reading order: (text, (x0, y0, x1, y1)).
    lines: list[tuple[str, Bounds]] = field(default_factory=list)
    #: The figures' crop boxes, in the order of their placeholders.
    figures: list[Bounds] = field(default_factory=list)
    #: The reading's mean score (set by ``OCRProcessor.recognize_page``).
    confidence: float = 0.0


@dataclass
class _Page:
    """What :func:`_render_flow` knows of the page beyond the boxes."""

    body: float
    bullets: set[int] = field(default_factory=set)
    words: dict[int, list[int | None]] = field(default_factory=dict)
    bold_lines: set[int] = field(default_factory=set)
    #: Lines' type size over the body's, where the ink shows it.
    sizes: dict[int, float] = field(default_factory=dict)
    blocks: dict[str, str] = field(default_factory=dict)
    #: The lines a table block took, read where the table is written.
    block_lines: dict[str, list[_Box]] = field(default_factory=dict)
    order: list[_Box] = field(default_factory=list)


def structure_ocr_page(
    texts: Sequence[str],
    bounds: Sequence[Bounds | None],
    image: Any = None,
    *,
    upside_down: bool = False,
    figures: bool = False,
) -> OcrPage:
    """Lay a page's boxes out with headings, lists, tables, figures, emphasis.

    Args:
        texts: Recognized text per box.
        bounds: ``(x0, y0, x1, y1)`` per box in the pixels of *image*.
        image: The page as read (RGB array); without it only the geometry
            speaks (headings, read bullet glyphs).
        upside_down: The page was read rotated by 180 degrees (no pixel
            steps then: the boxes are flipped, the pixels are not).
        figures: Cut figures out (a PDF page; an image input is its own
            picture).
    """
    from markitai import ocr_structure as px

    boxes: list[_Box] = []
    loose: list[str] = []
    for text, bound in zip(texts, bounds):
        text = str(text).strip()
        if not text:
            continue
        if bound is None:
            loose.append(text)
        else:
            boxes.append(_Box(text, *bound))
    if not boxes or upside_down or _is_vertical(boxes):
        markdown = layout_ocr_text(list(texts), list(bounds), upside_down=upside_down)
        return OcrPage(markdown, [(b.text, (b.x0, b.y0, b.x1, b.y1)) for b in boxes])

    boxes, margin = _split_margin(boxes)
    body = _body_height(boxes)
    page = _Page(body=body)
    pixels = px.page_pixels(image) if image is not None else None
    figure_boxes: list[Bounds] = []
    if pixels is not None:
        lines = [(b.text, (b.x0, b.y0, b.x1, b.y1)) for b in boxes]
        taken: set[int] = set()
        placed: list[_Box] = []
        grids = px.find_grids(pixels, body)
        for grid in grids:
            cells = px.table_cells(
                pixels, grid, [line for i, line in enumerate(lines) if i not in taken]
            )
            if cells is None:
                continue
            rows, local = cells
            free = [i for i in range(len(lines)) if i not in taken]
            taken.update(free[i] for i in local)
            token = f"\x00T{len(page.blocks)}\x00"
            page.blocks[token] = px.markdown_table(rows)
            x0, y0, x1, _y1 = grid.box
            placed.append(_Box(token, x0, y0, x1, y0 + body))
            page.block_lines[token] = [boxes[i] for i in sorted(free[j] for j in local)]
        bullets = px.find_bullets(pixels, [b for _t, b in lines], body)
        page.bullets = {id(boxes[i]) for i in bullets if i not in taken}
        if figures:
            cleared = [g.box for g in grids]
            cleared += [
                (x - body / 2, lines[i][1][1], x + body / 2, lines[i][1][3])
                for i, x in bullets.items()
            ]
            free = [i for i in range(len(lines)) if i not in taken]
            found = px.find_figures(pixels, [lines[i] for i in free], body, cleared)
            for crop, labels in found:
                token = f"\x00F{len(figure_boxes)}\x00"
                figure_boxes.append(crop)
                page.blocks[token] = token
                taken.update(free[i] for i in labels)
                placed.append(_Box(token, crop[0], crop[1], crop[2], crop[1] + body))
        emphasis = px.measure_emphasis(pixels, lines, body)
        for index, flags in emphasis.words.items():
            page.words[id(boxes[index])] = flags
            # Bold line: every measured word of four letters or more bold
            measured = [
                f
                for f, word in zip(flags, lines[index][0].split(), strict=False)
                if f is not None and not px.short_word(word)
            ]
            if measured and all(f & px.BOLD for f in measured):
                page.bold_lines.add(id(boxes[index]))
        for index, ratio in px.size_ratios(pixels, lines).items():
            page.sizes[id(boxes[index])] = ratio
        laid = [b for i, b in enumerate(boxes) if i not in taken] + placed
    else:
        laid = list(boxes)

    body_text = _render_columns(laid, page)
    for token, block in page.blocks.items():
        if token.startswith("\x00T"):
            body_text = body_text.replace(token, block)
    parts = [body_text, *(box.text for box in sorted(margin, key=lambda b: b.x0))]
    markdown = "\n\n".join(part for part in (*parts, "\n".join(loose)) if part)
    emitted = {id(b) for b in page.order}
    rest = sorted(
        (b for b in boxes + margin if id(b) not in emitted), key=lambda b: (b.y0, b.x0)
    )
    ordered = [
        (b.text, (b.x0, b.y0, b.x1, b.y1))
        for b in page.order + rest
        if not b.text.startswith("\x00")
    ]
    return OcrPage(markdown, ordered, figure_boxes)


#: Lines this long are body text, whose height the page's sizes compare to.
_BODY_LINE_CHARS = 12


def _body_height(boxes: list[_Box]) -> float:
    """The body text's line height: the median of the lines of twelve
    characters or more (a photo's or a chart's many short labels, a
    keyboard's keys, would pull a plain median down and make every body
    line a heading); of all lines when fewer than three are that long."""
    long = [b.height for b in boxes if len(b.text.replace(" ", "")) >= _BODY_LINE_CHARS]
    return median(long) if len(long) >= 3 else median(b.height for b in boxes)


def place_figures(markdown: str, references: list[str | None]) -> str:
    """Put figure ``n``'s Markdown (``![](…)``) where its placeholder is;
    a figure without one (not saved) leaves nothing."""
    for index, reference in enumerate(references):
        markdown = markdown.replace(f"\x00F{index}\x00", reference or "")
    markdown = re.sub(r"\x00F\d+\x00", "", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip("\n")


def resolve_heading_levels(pages: list[str]) -> list[str]:
    """Turn heading placeholders into levels across all *pages* at once.

    Sizes (line height over the page's body height) sorted from the
    largest: a size at least 0.92 times its tier's largest joins that tier,
    else it starts the next one; tier n is n ``#``s (six at most).
    """
    sizes = sorted(
        {float(m.group(1)) for page in pages for m in _HEADING_MARK.finditer(page)},
        reverse=True,
    )
    level: dict[float, int] = {}
    tier, top = 0, 0.0
    for size in sizes:
        if not tier or size < _HEADING_TIER * top:
            tier, top = tier + 1, size
        level[size] = min(tier, 6)
    return [
        _HEADING_MARK.sub(lambda m: "#" * level[float(m.group(1))] + " ", page)
        for page in pages
    ]


def _split_margin(boxes: list[_Box]) -> tuple[list[_Box], list[_Box]]:
    """Set apart upright boxes many lines tall: (body boxes, margin boxes)."""
    line_height = median(box.height for box in boxes)
    body: list[_Box] = []
    margin: list[_Box] = []
    for box in boxes:
        upright = box.height > box.width * _VERTICAL_ASPECT
        tall = box.height > line_height * _MARGIN_HEIGHT_RATIO
        (margin if upright and tall else body).append(box)
    if not body:
        return boxes, []
    return body, margin


# --- vertical text -------------------------------------------------------------


def _is_vertical(boxes: list[_Box]) -> bool:
    """Whether the page reads in vertical columns.

    Weighed by characters, not boxes: a row of short upright tags beside
    horizontal lines outnumbers the lines but holds little of the text. The
    columns must also run a fair share of the text area's height.
    """
    vertical = [
        box
        for box in boxes
        if len(box.text) >= 2 and box.height > box.width * _VERTICAL_ASPECT
    ]
    if len(vertical) < 2:
        return False
    total = sum(len(box.text) for box in boxes)
    if sum(len(box.text) for box in vertical) < total * _VERTICAL_SHARE:
        return False
    area = max(box.y1 for box in boxes) - min(box.y0 for box in boxes)
    return median(box.height for box in vertical) >= area * _VERTICAL_MIN_HEIGHT_SHARE


def _join_column(column: list[_Box]) -> str:
    """Join the boxes of one vertical column, top to bottom.

    The rule of :func:`_join_line`, turned by 90 degrees: only two CJK
    characters that touch join without a space, so the cells of a table
    scanned on its side do not run together.
    """
    out = column[0].text
    for previous, box in zip(column, column[1:]):
        gap = box.y0 - previous.y1
        tight = gap < max(previous.width, box.width) * _CJK_JOIN_GAP_RATIO
        if not (tight and _is_cjk(out[-1]) and _is_cjk(box.text[0])):
            out += " "
        out += box.text
    return out


def _render_vertical(boxes: list[_Box]) -> str:
    """Columns right to left; each column top to bottom is one line."""
    columns: list[list[_Box]] = []
    for box in sorted(boxes, key=lambda b: -b.cx):
        for column in columns:
            anchor = column[0]
            overlap = min(anchor.x1, box.x1) - max(anchor.x0, box.x0)
            if overlap > min(anchor.width, box.width) * 0.5:
                column.append(box)
                break
        else:
            columns.append([box])

    lines: list[str] = []
    previous: list[_Box] | None = None
    for column in columns:
        column.sort(key=lambda b: b.y0)
        if previous is not None:
            gap = min(b.x0 for b in previous) - max(b.x1 for b in column)
            width = max(max(b.width for b in previous), max(b.width for b in column))
            if width > 0 and gap > width * _VERTICAL_PARAGRAPH_GAP_RATIO:
                lines.append("")
        lines.append(_join_column(column))
        previous = column
    return "\n".join(lines)


# --- columns ---------------------------------------------------------------------


Gutter = tuple[float, float]


def _y_overlap(a: _Box, b: _Box) -> float:
    return min(a.y1, b.y1) - max(a.y0, b.y0)


def _side_by_side(a: _Box, b: _Box) -> bool:
    """Two boxes on one row, apart from each other horizontally."""
    if a.x1 > b.x0 and b.x1 > a.x0:
        return False
    return _y_overlap(a, b) > min(a.height, b.height) * _SIDE_BY_SIDE_OVERLAP


def _find_gutters(boxes: list[_Box]) -> list[Gutter]:
    """``(left, right)`` spans of the column gutters.

    Only narrow lines that have a neighbour on their own row are looked at:
    column lines sit side by side, while a title, an address block or a
    caption alone on its row would otherwise either bridge the gutter or
    open one where there are no columns at all.
    """
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    width = right - left
    if width <= 0:
        return []
    narrow = [box for box in boxes if box.width <= width * _NARROW_SHARE]
    paired = [
        box
        for box in narrow
        if any(other is not box and _side_by_side(box, other) for other in narrow)
    ]
    if len(paired) < _COLUMN_MIN_LINES * 2:
        return []

    min_gap = max(width * _GUTTER_MIN_SHARE, median(box.height for box in paired))
    lo = left + width * _GUTTER_EDGE_SHARE
    hi = right - width * _GUTTER_EDGE_SHARE
    gutters: list[Gutter] = []
    reach = left
    for box in sorted(paired, key=lambda b: b.x0):
        if box.x0 - reach >= min_gap and lo <= (reach + box.x0) / 2 <= hi:
            gutters.append((reach, box.x0))
        reach = max(reach, box.x1)
    return gutters


def _column_of(box: _Box, gutters: list[Gutter]) -> int:
    return sum(1 for gl, gr in gutters if box.cx > (gl + gr) / 2)


def _crosses(box: _Box, gutters: list[Gutter]) -> bool:
    """Whether the box reaches into a gutter (so it belongs to no column)."""
    return any(box.x0 < gr and box.x1 > gl for gl, gr in gutters)


def _columns_side_by_side(columns: dict[int, list[_Box]]) -> bool:
    """Neighbouring columns share enough rows to be read side by side."""
    for index in range(len(columns) - 1):
        a, b = columns[index], columns[index + 1]
        paired_a = sum(1 for box in a if any(_y_overlap(box, o) > 0 for o in b))
        paired_b = sum(1 for box in b if any(_y_overlap(box, o) > 0 for o in a))
        if min(paired_a, paired_b) < _COLUMN_MIN_LINES:
            return False
    return True


def _render_columns(boxes: list[_Box], page: _Page | None = None) -> str:
    gutters = _find_gutters(boxes)
    wide: set[int] = set()
    if gutters:
        columns: dict[int, list[_Box]] = {}
        for index, box in enumerate(boxes):
            if _crosses(box, gutters):
                wide.add(index)
            else:
                columns.setdefault(_column_of(box, gutters), []).append(box)
        text_like = (
            len(columns) == len(gutters) + 1
            and all(
                len(members) >= _COLUMN_MIN_LINES
                and median(_display_length(b.text) for b in members) >= _COLUMN_MIN_TEXT
                for members in columns.values()
            )
            and _columns_side_by_side(columns)
        )
        if not text_like:
            gutters = []  # a table or scattered labels, not text columns
        else:
            # Above the point where every column has begun, a line has no
            # neighbour to be read beside (an address block at the top of a
            # letter): it reads as a full-width block, before the columns.
            start = max(min(b.y0 for b in members) for members in columns.values())
            wide.update(i for i, box in enumerate(boxes) if box.cy < start)
    if not gutters:
        return _render_flow(boxes, page)

    # Full-width blocks (a title, a footer) separate runs of columns: every
    # column of a run is read before the block that follows it.
    blocks: list[str] = []
    run: dict[int, list[_Box]] = {}
    block: list[_Box] = []

    def flush() -> None:
        for column in sorted(run):
            blocks.append(_render_flow(run[column], page))
        run.clear()
        if block:
            blocks.append(_render_flow(block, page))
            block.clear()

    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].y0, boxes[i].x0))
    for index in order:
        box = boxes[index]
        if index in wide:
            if run:
                flush()
            block.append(box)
        else:
            if block:
                flush()
            run.setdefault(_column_of(box, gutters), []).append(box)
    flush()
    return "\n\n".join(part for part in blocks if part)


# --- lines, paragraphs and tables ------------------------------------------------


def _same_line(box: _Box, line: list[_Box]) -> bool:
    """Whether *box* sits on *line*.

    The line is measured by its boxes' median height and center, not by
    the span of everything it collected: one tall box would otherwise
    stretch the line over the next ones, and every later box would join.
    """
    height = median(b.height for b in line)
    center = median(b.cy for b in line)
    top, bottom = center - height / 2, center + height / 2
    overlap = max(0.0, min(bottom, box.y1) - max(top, box.y0))
    smaller = max(1.0, min(box.height, height))
    return overlap / smaller > 0.5 or abs(box.cy - center) < smaller * 0.35


def _group_lines(boxes: list[_Box]) -> list[list[_Box]]:
    lines: list[list[_Box]] = []
    for box in sorted(boxes, key=lambda b: (b.cy, b.x0)):
        for line in lines:
            if _same_line(box, line):
                line.append(box)
                break
        else:
            lines.append([box])
    for line in lines:
        line.sort(key=lambda b: b.x0)
    lines.sort(key=lambda line: (min(b.y0 for b in line), line[0].x0))
    return lines


def _is_cell_row(line: list[_Box]) -> bool:
    if len(line) < 2:
        return False
    height = max(b.height for b in line)
    return all(
        right.x0 - left.x1 > height * _CELL_GAP_RATIO
        for left, right in zip(line, line[1:])
    )


def _cells_line_up(cells: list[_Box]) -> bool:
    """One column of cells shares a left edge, right edge or center.

    The tolerance is one character of the column's text.
    """
    char = median(c.width / max(1, len(c.text)) for c in cells)
    return any(
        max(edge(c) for c in cells) - min(edge(c) for c in cells) < char
        for edge in (
            lambda c: c.x0,
            lambda c: c.x1,
            lambda c: c.cx,
        )
    )


def _is_table(rows: list[list[_Box]]) -> bool:
    """Whether a run of cell rows with equal cell counts is a table.

    Two cells a row need more proof than wider rows: form labels with their
    values, and a numbered list with a hanging indent, look just the same.
    """
    if len(rows) < 2:
        return False
    if len(rows[0]) > 2:
        return True
    if len(rows) < _TWO_CELL_TABLE_MIN_ROWS:
        return False
    if not all(_cells_line_up([row[i] for row in rows]) for i in range(2)):
        return False
    list_like = all(_LIST_MARKER_RE.match(row[0].text) for row in rows) and (
        median(_display_length(row[1].text) for row in rows) >= _LIST_ITEM_MIN_TEXT
    )
    return not list_like


def _table(rows: list[list[_Box]]) -> str:
    def row(cells: list[_Box]) -> str:
        return "| " + " | ".join(c.text.replace("|", "\\|") for c in cells) + " |"

    header, *body = rows
    divider = "| " + " | ".join("---" for _ in header) + " |"
    return "\n".join([row(header), divider, *(row(cells) for cells in body)])


def _render_flow(boxes: list[_Box], page: _Page | None = None) -> str:
    """One column: visual lines, paragraph breaks, and cell-row tables.

    With *page* (:func:`structure_ocr_page`) also headings, list items,
    the page's table and figure blocks and emphasis.
    """
    lines = _group_lines(boxes)
    flow = _Flow(page)
    previous: list[_Box] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        # A run of cell rows with the same number of cells is a table
        run_end = index
        if _is_cell_row(line):
            while (
                run_end + 1 < len(lines)
                and _is_cell_row(lines[run_end + 1])
                and len(lines[run_end + 1]) == len(line)
            ):
                run_end += 1
        is_table = _is_table(lines[index : run_end + 1])
        gap_before = previous is not None and (
            is_table or _paragraph_gap(previous, line)
        )

        if page is not None and any(b.text in page.blocks for b in line):
            # A table or figure, with any text beside its top line before it
            beside = [b for b in line if b.text not in page.blocks]
            if beside:
                if gap_before:
                    flow.paragraph_break()
                flow.line(beside)
            for token in (b.text for b in line if b.text in page.blocks):
                flow.block(token)
                flow.emitted(*page.block_lines.get(token, []))
            previous = None
            index += 1
            continue
        if is_table:
            flow.block(_table(lines[index : run_end + 1]))
            flow.emitted(*(b for row in lines[index : run_end + 1] for b in row))
            previous = None
            index = run_end + 1
            continue
        if page is not None:
            heading = _heading_run(lines, index, page, previous is None or gap_before)
            if heading:
                rows = [lines[i] for i in heading]
                size = max(_size(b, page) for row in rows for b in row)
                text = _join_rows([_join_line(row) for row in rows])
                flow.block(f"\x00H{size:.3f}\x00 {text}")
                flow.emitted(*(b for row in rows for b in row))
                previous = None
                index = heading[-1] + 1
                continue
            item = _list_item(line, page)
            if item is not None:
                run = _item_run(lines, index, page)
                flow.item(item, [lines[i] for i in run[1:]], line)
                previous = lines[run[-1]]
                index = run[-1] + 1
                continue
        if gap_before:
            flow.paragraph_break()
        flow.line(line)
        previous = line
        index += 1
    return flow.text()


def _paragraph_gap(previous: list[_Box], line: list[_Box]) -> bool:
    gap = min(b.y0 for b in line) - max(b.y1 for b in previous)
    height = max(max(b.height for b in previous), max(b.height for b in line))
    return height > 0 and gap > height * PARAGRAPH_GAP_RATIO


def _join_rows(rows: list[str]) -> str:
    out = rows[0]
    for row in rows[1:]:
        out += row if _is_cjk(out[-1]) and _is_cjk(row[0]) else " " + row
    return out


def _heading_run(
    lines: list[list[_Box]], index: int, page: _Page, opens: bool
) -> list[int]:
    """*lines[index]* and the lines continuing it, when it is a heading."""

    def fits(line: list[_Box]) -> bool:
        text = line[0].text
        return (
            len(line) == 1
            and len(text.replace(" ", "")) <= _HEADING_MAX_CHARS
            and not text.endswith(tuple(_HEADING_OPEN_END))
            and id(line[0]) not in page.bullets
            and not text.startswith(tuple(_BULLET_GLYPHS))
        )

    line = lines[index]
    if not fits(line) or page.body <= 0:
        return []
    size = _size(line[0], page)
    bold = id(line[0]) in page.bold_lines
    if not (size >= _HEADING_RATIO or (bold and opens and size >= _BOLD_HEADING_RATIO)):
        return []
    run = [index]
    while len(run) < _HEADING_MAX_ROWS and run[-1] + 1 < len(lines):
        following = lines[run[-1] + 1]
        if not fits(following) or _paragraph_gap(lines[run[-1]], following):
            break
        other = following[0]
        # A short last line ("a.", "elit.") has no ink size: its box speaks
        other_size = _size(other, page)
        same_size = abs(other_size - size) <= size * _HEADING_NEXT_SIZE or (
            id(other) not in page.sizes
            and abs(other.height - line[0].height)
            <= line[0].height * _HEADING_NEXT_SIZE
        )
        like = other_size >= _HEADING_NEXT_RATIO or (
            bold and id(other) in page.bold_lines
        )
        if not (same_size and like):
            break
        run.append(run[-1] + 1)
    return run


def _size(box: _Box, page: _Page) -> float:
    """A line's type size over the body's: from its ink where the pixels
    show it (``ocr_structure.size_ratios``), else its box height's ratio."""
    measured = page.sizes.get(id(box))
    return measured if measured is not None else box.height / page.body


def _list_item(line: list[_Box], page: _Page) -> str | None:
    """The item's text when *line* starts a list item (its glyph removed)."""
    first = line[0]
    text = _join_line(line)
    if len(text) > 2 and text[0] in _BULLET_GLYPHS and text[1] == " ":
        return text[2:].strip()
    if id(first) in page.bullets:
        return text
    return None


def _item_run(lines: list[list[_Box]], index: int, page: _Page) -> list[int]:
    """The item at *lines[index]* and the lines right below that continue it."""
    left = lines[index][0].x0
    run = [index]
    while run[-1] + 1 < len(lines):
        following = lines[run[-1] + 1]
        height = max(b.height for b in following)
        if (
            _paragraph_gap(lines[run[-1]], following)
            or _list_item(following, page) is not None
            or any(b.text in page.blocks for b in following)
            or following[0].x0 < left - _ITEM_INDENT * height
        ):
            break
        run.append(run[-1] + 1)
    return run


class _Flow:
    """The Markdown of one column as it is written, line by line."""

    def __init__(self, page: _Page | None) -> None:
        self.page = page
        self.parts: list[str] = []
        self.words: list[str] = []
        self.flags: list[int | None] = []
        self.breaks: set[int] = set()
        self.in_list = False

    def emitted(self, *boxes: _Box) -> None:
        if self.page is not None:
            self.page.order.extend(boxes)

    def _line_words(self, line: list[_Box]) -> tuple[list[str], list[int | None]]:
        """The line's words and their emphasis (None: not measured)."""
        page = self.page
        text = _join_line(line)
        if page is None or any(_is_cjk(c) for c in text):
            return [text], [None]
        words: list[str] = []
        flags: list[int | None] = []
        for box in line:
            tokens = box.text.split()
            measured = page.words.get(id(box))
            words.extend(tokens)
            flags.extend(measured if measured else [None] * len(tokens))
        return words, flags

    def _flush(self) -> None:
        if not self.words:
            return
        from markitai.ocr_structure import emphasized, smooth_emphasis

        if all(f is None for f in self.flags):
            text = ""
            for i, word in enumerate(self.words):
                text += ("\n" if i in self.breaks else " ") if i else ""
                text += word
        else:
            text = emphasized(
                self.words, smooth_emphasis(self.words, self.flags), self.breaks
            )
        self.parts.append(text)
        self.words, self.flags, self.breaks = [], [], set()

    def _gap(self) -> None:
        self._flush()
        if self.parts and self.parts[-1] != "":
            self.parts.append("")

    def line(self, line: list[_Box]) -> None:
        if self.in_list:
            self._gap()
            self.in_list = False
        self.emitted(*line)
        words, flags = self._line_words(line)
        if self.words:
            self.breaks.add(len(self.words))
        self.words.extend(words)
        self.flags.extend(flags)

    def paragraph_break(self) -> None:
        self._gap()

    def block(self, markdown: str) -> None:
        self._gap()
        self.in_list = False
        self.parts.extend([markdown, ""])

    def item(self, text: str, more: list[list[_Box]], first: list[_Box]) -> None:
        if not self.in_list:
            self._gap()
            self.in_list = True
        else:
            self._flush()
        self.emitted(*first, *(b for row in more for b in row))
        words, flags = self._line_words(first)
        if text != " ".join(words):
            # The read glyph is gone from the item's text: a word of its
            # own, or inside the one word a CJK line is
            if len(words) > 1 and words[0] in _BULLET_GLYPHS:
                words, flags = words[1:], flags[1:]
            else:
                words, flags = [text], [None]
        rows = [(words, flags)] + [self._line_words(row) for row in more]
        for row, (row_words, row_flags) in enumerate(rows):
            if row:
                self.breaks.add(len(self.words))
            self.words.extend(row_words)
            self.flags.extend(row_flags)
        start = len(self.parts)
        self._flush()
        if len(self.parts) > start:
            self.parts[-1] = "- " + self.parts[-1].replace("\n", "\n  ")

    def text(self) -> str:
        self._flush()
        while self.parts and self.parts[-1] == "":
            self.parts.pop()
        while self.parts and self.parts[0] == "":
            self.parts.pop(0)
        return "\n".join(self.parts)
