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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

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


def _render_columns(boxes: list[_Box]) -> str:
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
        return _render_flow(boxes)

    # Full-width blocks (a title, a footer) separate runs of columns: every
    # column of a run is read before the block that follows it.
    blocks: list[str] = []
    run: dict[int, list[_Box]] = {}
    block: list[_Box] = []

    def flush() -> None:
        for column in sorted(run):
            blocks.append(_render_flow(run[column]))
        run.clear()
        if block:
            blocks.append(_render_flow(block))
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


def _render_flow(boxes: list[_Box]) -> str:
    """One column: visual lines, paragraph breaks, and cell-row tables."""
    lines = _group_lines(boxes)
    parts: list[str] = []
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

        if previous is not None:
            gap = min(b.y0 for b in line) - max(b.y1 for b in previous)
            height = max(max(b.height for b in previous), max(b.height for b in line))
            if is_table or (height > 0 and gap > height * PARAGRAPH_GAP_RATIO):
                parts.append("")

        if is_table:
            parts.append(_table(lines[index : run_end + 1]))
            previous = lines[run_end]
            index = run_end + 1
            if index < len(lines):
                parts.append("")
                previous = None  # the blank line after a table is already in
            continue

        parts.append(_join_line(line))
        previous = line
        index += 1
    return "\n".join(parts)
