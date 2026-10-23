"""What an OCR'd page's pixels add to its text boxes: tables, figures,
bullets and emphasis.

The recognizer returns one box of text per line. Its geometry gives the
reading order and headings (:mod:`markitai.ocr_layout`); the page's pixels
show what the boxes cannot:

* **ruled tables**: horizontal and vertical rules that cross form a grid;
  each line goes into the cell under its centre (a line running across a
  column rule is split at its word gaps), so a table comes out as a table
  instead of lines of text;
* **figures**: ink no text box covers (a chart's bars, a diagram, a photo)
  merged into regions large enough to be pictures; the short labels in or
  next to one (axis numbers, a legend) belong to it, and the region is cut
  out of the page as an image instead of its labels being read as text;
* **bullets**: recognizers drop a lone dot; a small solid blob left of a
  line, at the same position as another one on the page, is a bullet;
* **bold and italic**, per word: a word's stroke width against the body
  text's (bold), and whether shearing its ink makes its columns line up
  (italic). Only where the ink splits into as many words as the text has.

The rules and thresholds are markitai-rust's (``ocr/page.rs``) so both
implementations give the same Markdown for the same page; numbers are in
pixels of the page as rendered for OCR, scaled by the body text's line
height where they depend on the type size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any

Bounds = tuple[float, float, float, float]

#: Ink is darker than this share of the paper's luminance (text)...
_INK_SHARE = 0.6
#: ...and marked pixels darker than this one, or this colourful (light
#: grey rules, pale or yellow chart areas).
_MARK_SHARE = 0.88
_MARK_CHROMA = 48
#: Paper darker than this is no page to analyse (a photo, a dark scan).
_MIN_PAPER = 80

# Rules: a horizontal rule runs at least max(3 body lines, width / 16), a
# vertical one 1.2 body lines; neither is thicker than max(3 px, 0.15 body).
_H_RULE_BODIES = 3.0
_H_RULE_WIDTH_SHARE = 1 / 16
_V_RULE_BODIES = 1.2
_RULE_THICK_BODIES = 0.15
#: Rules within max(4 px, 0.3 body) of each other cross.
_CROSS_BODIES = 0.3
#: A grid has at least this many row edges and column edges.
_GRID_MIN_EDGES = 3
#: A line crossing a column rule by this many line heights is split.
_SPLIT_OVERHANG = 0.3

# Bullets: one blob in [left - 3.5 h, left - 0.1 h], each side 0.1-0.6 h.
_BULLET_REACH = 3.5
_BULLET_GAP = 0.1
_BULLET_MIN_SIDE = 0.1
_BULLET_MAX_SIDE = 0.6
_BULLET_MIN_FILL = 0.3
_BULLET_MIN_PIXELS = 4
#: Another bullet of the page lies within this many body heights in x.
_BULLET_ALIGN = 0.5

# Figures: cells of a third of a body line, marked when holding at least
# max(2, cell² / 20) marked pixels; parts of at least 12 cells, 2.5 body
# lines on each side; labels of at most 24 characters within one body line.
_FIGURE_CELL_BODIES = 1 / 3
_FIGURE_MIN_CELLS = 12
_FIGURE_MIN_SIDE = 2.5
_FIGURE_REACH_CELLS = 2
_FIGURE_TEXT_PAD = 0.15
_FIGURE_LABEL_CHARS = 24
_FIGURE_LABEL_REACH = 1.0
_FIGURE_LABEL_ROUNDS = 3
_FIGURE_FRAMED_CHARS = 30
_FIGURE_MERGE = 1.0
_FIGURE_CROP_PAD = 0.3
_CAPTION = re.compile(r"\b(?:fig|figure|table|chart|source)\b|[图圖表]", re.IGNORECASE)

# Emphasis: a word's stroke 1.5 times the body's is bold (1.35 next to a
# bold word); italic when a shear lines its columns up 12 % better.
_BOLD = 1.5
_BOLD_NEXT = 1.35
_ITALIC_GAIN = 1.12
_SHEARS = (0.1, 0.15, 0.2, 0.25, 0.3)
_WORD_GAP_SHARE = 0.2
_MIN_WORD_INK = 30
_BODY_LINE_RANGE = (0.8, 1.25)
_MIN_BODY_LINES = 3
#: A word of fewer letters or digits takes its neighbours' emphasis.
_SHORT_WORD = 4

BOLD = 1
ITALIC = 2


@dataclass
class Pixels:
    """A page's ink (text) and marked (anything drawn) pixels."""

    ink: Any  # bool array, rows x columns
    marked: Any

    @property
    def height(self) -> int:
        return int(self.ink.shape[0])

    @property
    def width(self) -> int:
        return int(self.ink.shape[1])


def page_pixels(image: Any) -> Pixels | None:
    """The ink and marked pixels of an RGB(A) page render, or None when
    the page is too dark to tell ink from paper."""
    import numpy as np

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        return None
    rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    # One byte a pixel (a 48 MP photo would take 0.5 GB as int32)
    sample = rgb[::4, ::4].astype(np.int32)
    paper = float(
        np.percentile(
            (299 * sample[..., 0] + 587 * sample[..., 1] + 114 * sample[..., 2])
            // 1000,
            90,
        )
    )
    if paper < _MIN_PAPER:
        return None
    try:
        import cv2
    except ImportError:  # every pixel step needs OpenCV (RapidOCR ships it)
        return None
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # 0.299 R + 0.587 G + 0.114 B
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return Pixels(
        ink=lum < _INK_SHARE * paper,
        marked=(lum < _MARK_SHARE * paper) | (chroma >= _MARK_CHROMA),
    )


def _clip(pixels: Pixels, box: Bounds) -> tuple[int, int, int, int]:
    x0 = max(0, int(box[0]))
    y0 = max(0, int(box[1]))
    x1 = min(pixels.width, int(round(box[2])))
    y1 = min(pixels.height, int(round(box[3])))
    return x0, y0, max(x0, x1), max(y0, y1)


# --- ruled tables ----------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Grid:
    """A ruled table: its box, row edges (y) and column edges (x)."""

    rows: list[float]
    columns: list[float]

    @property
    def box(self) -> Bounds:
        return (self.columns[0], self.rows[0], self.columns[-1], self.rows[-1])


def find_rules(pixels: Pixels, body: float) -> tuple[list[Rule], list[Rule]]:
    """Horizontal and vertical rules: long thin runs of marked pixels."""
    import cv2
    import numpy as np

    marked = pixels.marked.astype(np.uint8)
    thick = max(3, int(_RULE_THICK_BODIES * body))
    h_len = max(3, int(max(_H_RULE_BODIES * body, pixels.width * _H_RULE_WIDTH_SHARE)))
    v_len = max(3, int(_V_RULE_BODIES * body))
    found: list[list[Rule]] = []
    for kernel_gap, kernel_run, horizontal in (
        ((1, 3), (1, h_len), True),
        ((3, 1), (v_len, 1), False),
    ):
        # A dashed or anti-aliased rule has gaps of a pixel or two
        closed = cv2.morphologyEx(
            marked, cv2.MORPH_CLOSE, np.ones(kernel_gap, np.uint8)
        )
        runs = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones(kernel_run, np.uint8))
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(
            runs, connectivity=8
        )
        rules: list[Rule] = []
        for x, y, w, h, _area in stats[1:count]:
            if (h if horizontal else w) <= thick:
                rules.append(Rule(int(x), int(y), int(x + w), int(y + h)))
        found.append(rules)
    return found[0], found[1]


def _merge_edges(values: list[float], tolerance: float) -> list[float]:
    edges: list[list[float]] = []
    for value in sorted(values):
        if edges and value - edges[-1][-1] <= tolerance:
            edges[-1].append(value)
        else:
            edges.append([value])
    return [sum(group) / len(group) for group in edges]


def find_grids(pixels: Pixels, body: float) -> list[Grid]:
    """Ruled tables: clusters of crossing rules with 3+ row and column edges."""
    horizontal, vertical = find_rules(pixels, body)
    if not horizontal or not vertical:
        return []
    tol = max(4.0, _CROSS_BODIES * body)
    rules = horizontal + vertical
    parent = list(range(len(rules)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for hi, h in enumerate(horizontal):
        for vi, v in enumerate(vertical):
            if (
                v.x0 - tol <= h.x1
                and h.x0 - tol <= v.x1
                and h.y0 - tol <= v.y1
                and v.y0 - tol <= h.y1
            ):
                parent[find(hi)] = find(len(horizontal) + vi)
    clusters: dict[int, list[Rule]] = {}
    for index, rule in enumerate(rules):
        clusters.setdefault(find(index), []).append(rule)
    grids: list[Grid] = []
    for members in clusters.values():
        hs = [r for r in members if r in horizontal]
        vs = [r for r in members if r in vertical]
        if not hs or not vs:
            continue
        x0 = min(r.x0 for r in members)
        x1 = max(r.x1 for r in members)
        rows = _merge_edges([r.cy for r in hs if r.x1 - r.x0 >= (x1 - x0) / 2], tol)
        columns = _merge_edges([r.cx for r in vs] + [x0, x1], tol)
        if len(rows) >= _GRID_MIN_EDGES and len(columns) >= _GRID_MIN_EDGES:
            grids.append(Grid(rows=rows, columns=columns))
    return grids


def _index_of(edges: list[float], value: float) -> int | None:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return None


def word_spans(pixels: Pixels, box: Bounds, words: int) -> list[Bounds] | None:
    """The words of a line as runs of ink columns, when there are *words*.

    Columns with no ink split the line where the gap is at least
    max(2 px, a fifth of the line height); None when the runs do not
    match the words the text has.
    """
    import numpy as np

    x0, y0, x1, y1 = _clip(pixels, box)
    if x1 <= x0 or y1 <= y0 or words < 1:
        return None
    columns = pixels.ink[y0:y1, x0:x1].any(axis=0)
    filled = np.flatnonzero(columns)
    if filled.size == 0:
        return None
    gap = max(2, int(_WORD_GAP_SHARE * (y1 - y0)))
    breaks = np.flatnonzero(np.diff(filled) > gap)
    starts = np.concatenate(([filled[0]], filled[breaks + 1]))
    ends = np.concatenate((filled[breaks], [filled[-1]]))
    if len(starts) != words:
        return None
    return [
        (float(x0 + s), float(y0), float(x0 + e + 1), float(y1))
        for s, e in zip(starts, ends, strict=True)
    ]


def _cell_join(parts: list[str]) -> str:
    out = ""
    for part in parts:
        if not out:
            out = part
        elif _cjk(out[-1]) and _cjk(part[0]):
            out += part
        else:
            out += " " + part
    return out


def _cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xAC00 <= code <= 0xD7AF
    )


def table_cells(
    pixels: Pixels, grid: Grid, lines: list[tuple[str, Bounds]]
) -> tuple[list[list[str]], list[int]] | None:
    """The grid's rows of cell texts and the lines it took, or None when
    fewer than two cells hold text.

    A line goes into the cell under its centre; a line reaching more than
    0.3 line heights into two or more columns is split per word (at its
    ink gaps, else by its characters' share of its width).
    """
    gx0, gy0, gx1, gy1 = grid.box
    rows, columns = len(grid.rows) - 1, len(grid.columns) - 1
    cells: list[list[list[tuple[float, float, str]]]] = [
        [[] for _ in range(columns)] for _ in range(rows)
    ]
    taken: list[int] = []
    for index, (text, box) in enumerate(lines):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (gx0 <= cx <= gx1 and gy0 <= cy <= gy1):
            continue
        row = _index_of(grid.rows, cy)
        if row is None:
            continue
        taken.append(index)
        height = box[3] - box[1]
        reached = [
            c
            for c in range(columns)
            if min(box[2], grid.columns[c + 1]) - max(box[0], grid.columns[c])
            > _SPLIT_OVERHANG * height
        ]
        pieces: list[tuple[str, Bounds]] = [(text, box)]
        if len(reached) >= 2:
            words = text.split()
            spans = word_spans(pixels, box, len(words))
            if spans is None and words:
                # By the characters' share of the width
                total = sum(len(w) + 1 for w in words)
                spans, left = [], box[0]
                for word in words:
                    right = left + (box[2] - box[0]) * (len(word) + 1) / total
                    spans.append((left, box[1], right, box[3]))
                    left = right
            pieces = list(zip(words, spans or [], strict=False))
        for piece, span in pieces:
            # A word running past the outer rules goes to the edge column
            centre = (span[0] + span[2]) / 2
            column = _index_of(grid.columns, centre)
            if column is None:
                column = 0 if centre < grid.columns[0] else columns - 1
            cells[row][column].append((span[1], span[0], piece))
    texts = [
        [_cell_join([p for _y, _x, p in sorted(cell)]) for cell in row] for row in cells
    ]
    keep_rows = [r for r in range(rows) if any(texts[r])]
    keep_columns = [c for c in range(columns) if any(texts[r][c] for r in range(rows))]
    filled = sum(1 for r in range(rows) for c in range(columns) if texts[r][c])
    if filled < 2 or not keep_rows or not keep_columns:
        return None
    return [[texts[r][c] for c in keep_columns] for r in keep_rows], taken


def markdown_table(rows: list[list[str]]) -> str:
    def row(cells: list[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    header, *body = rows
    divider = "| " + " | ".join("---" for _ in header) + " |"
    return "\n".join([row(header), divider, *(row(cells) for cells in body)])


# --- bullets -----------------------------------------------------------------------


def bullet_at(pixels: Pixels, box: Bounds) -> float | None:
    """The x centre of a bullet blob left of the line *box*, if one is there."""
    import cv2
    import numpy as np

    height = box[3] - box[1]
    if height <= 0:
        return None
    x0, y0, x1, y1 = _clip(
        pixels,
        (
            box[0] - _BULLET_REACH * height,
            box[1],
            box[0] - _BULLET_GAP * height,
            box[3],
        ),
    )
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    region = pixels.ink[y0:y1, x0:x1].astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )
    blobs = [i for i in range(1, count) if stats[i][4] >= _BULLET_MIN_PIXELS]
    if len(blobs) != 1:
        return None
    _x, _y, w, h, area = stats[blobs[0]]
    if not (
        _BULLET_MIN_SIDE * height <= w <= _BULLET_MAX_SIDE * height
        and _BULLET_MIN_SIDE * height <= h <= _BULLET_MAX_SIDE * height
        and 0.5 <= w / h <= 2
        and area >= _BULLET_MIN_FILL * w * h
    ):
        return None
    cx, cy = centroids[blobs[0]]
    if abs((y0 + cy) - (box[1] + box[3]) / 2) > height / 4:
        return None
    return float(x0 + cx)


def find_bullets(pixels: Pixels, boxes: list[Bounds], body: float) -> dict[int, float]:
    """Lines (by index) with a bullet left of them: blobs of the same x as
    another bullet of the page (a lone blob is a stray mark)."""
    found = {
        i: x for i, box in enumerate(boxes) if (x := bullet_at(pixels, box)) is not None
    }
    return {
        i: x
        for i, x in found.items()
        if any(
            j != i and abs(x - other) <= _BULLET_ALIGN * body
            for j, other in found.items()
        )
    }


# --- figures -----------------------------------------------------------------------


def find_figures(
    pixels: Pixels,
    lines: list[tuple[str, Bounds]],
    body: float,
    cleared: list[Bounds],
) -> list[tuple[Bounds, list[int]]]:
    """Figure regions (padded crop boxes) and the label lines each holds.

    *cleared*: boxes that are no figure (table grids, bullets).
    """
    import cv2
    import numpy as np

    cell = max(1, round(body * _FIGURE_CELL_BODIES))
    rows, columns = pixels.height // cell, pixels.width // cell
    if rows < 1 or columns < 1:
        return []
    marked = pixels.marked[: rows * cell, : columns * cell]
    counts = marked.reshape(rows, cell, columns, cell).sum(axis=(1, 3))
    cells = counts >= max(2, cell * cell / 20)
    pad = _FIGURE_TEXT_PAD * body
    for x0, y0, x1, y1 in [b for _t, b in lines] + cleared:
        cells[
            max(0, int((y0 - pad) // cell)) : int((y1 + pad) // cell) + 1,
            max(0, int((x0 - pad) // cell)) : int((x1 + pad) // cell) + 1,
        ] = False
    if not cells.any():
        return []
    reach = 2 * _FIGURE_REACH_CELLS + 1
    joined = cv2.dilate(cells.astype(np.uint8), np.ones((reach, reach), np.uint8))
    count, labels = cv2.connectedComponents(joined, connectivity=8)
    parts: list[list[float]] = []
    for label in range(1, count):
        ys, xs = np.nonzero((labels == label) & cells)
        if len(ys) < _FIGURE_MIN_CELLS:
            continue
        box = [
            float(xs.min() * cell),
            float(ys.min() * cell),
            float((xs.max() + 1) * cell),
            float((ys.max() + 1) * cell),
        ]
        if (
            box[2] - box[0] < _FIGURE_MIN_SIDE * body
            or box[3] - box[1] < _FIGURE_MIN_SIDE * body
        ):
            continue
        framed = any(
            len(text) >= _FIGURE_FRAMED_CHARS
            and _inside((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, box)
            and b[2] - b[0] >= (box[2] - box[0]) / 2
            for text, b in lines
        )
        if not framed:
            parts.append(box)

    figures: list[tuple[list[float], set[int]]] = []
    reach_px = _FIGURE_LABEL_REACH * body
    for box in parts:
        labels_of: set[int] = set()
        for _round in range(_FIGURE_LABEL_ROUNDS):
            grown = False
            for index, (text, b) in enumerate(lines):
                if index in labels_of or not _label_like(text):
                    continue
                if _near(b, box, reach_px):
                    labels_of.add(index)
                    box[:] = [
                        min(box[0], b[0]),
                        min(box[1], b[1]),
                        max(box[2], b[2]),
                        max(box[3], b[3]),
                    ]
                    grown = True
            if not grown:
                break
        figures.append((box, labels_of))

    merged = True
    while merged:
        merged = False
        for i in range(len(figures)):
            for j in range(i + 1, len(figures)):
                a, b = figures[i][0], figures[j][0]
                if _near(b, a, _FIGURE_MERGE * body):
                    a[:] = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ]
                    figures[i][1].update(figures[j][1])
                    del figures[j]
                    merged = True
                    break
            if merged:
                break

    crop = _FIGURE_CROP_PAD * body
    out: list[tuple[Bounds, list[int]]] = []
    for box, labels_of in sorted(figures, key=lambda f: (f[0][1], f[0][0])):
        out.append(
            (
                (
                    max(0.0, box[0] - crop),
                    max(0.0, box[1] - crop),
                    min(float(pixels.width), box[2] + crop),
                    min(float(pixels.height), box[3] + crop),
                ),
                sorted(labels_of),
            )
        )
    return out


def _inside(x: float, y: float, box: Bounds | list[float]) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _near(a: Bounds | list[float], b: Bounds | list[float], reach: float) -> bool:
    """Box *a* overlaps box *b* grown by *reach* on each side."""
    return (
        a[0] <= b[2] + reach
        and a[2] >= b[0] - reach
        and a[1] <= b[3] + reach
        and a[3] >= b[1] - reach
    )


def _label_like(text: str) -> bool:
    return (
        len(text.replace(" ", "")) <= _FIGURE_LABEL_CHARS
        and _CAPTION.search(text) is None
    )


# --- type size ---------------------------------------------------------------------

#: A line needs this many letters or digits for its ink to give its size...
_SIZE_MIN_CHARS = 4
#: ...and the body's size comes from lines of at least this many.
_SIZE_BODY_CHARS = 20
#: The baseline is the bottom of the rows holding at least half the ink of
#: the fullest row; the top, the first row with a twentieth of it.
_CORE_SHARE = 0.5
_TOP_SHARE = 0.05


def cap_height(pixels: Pixels, box: Bounds) -> int | None:
    """Rows from the top of the line's ink (capitals, ascenders) down to its
    baseline: its type size without the descenders and the box padding."""
    import numpy as np

    x0, y0, x1, y1 = _clip(pixels, box)
    rows = pixels.ink[y0:y1, x0:x1].sum(axis=1)
    if rows.size == 0 or not rows.max():
        return None
    core = np.flatnonzero(rows >= _CORE_SHARE * rows.max())
    top = np.flatnonzero(rows >= _TOP_SHARE * rows.max())
    return int(core[-1] - top[0] + 1)


def size_ratios(pixels: Pixels, lines: list[tuple[str, Bounds]]) -> dict[int, float]:
    """Each line's type size over the body text's (lines by index).

    Detector boxes are padded and take in descenders: a 14 pt bold heading
    over 10.5 pt text measured 1.03 to 1.28 times the body's box height
    depending on whether it had a comma; from the top of its ink to its
    baseline it is 1.41 times every time.
    """

    def chars(text: str) -> int:
        return sum(1 for c in text if c.isalnum())

    heights = {
        i: h
        for i, (text, box) in enumerate(lines)
        if chars(text) >= _SIZE_MIN_CHARS and (h := cap_height(pixels, box))
    }
    body = [h for i, h in heights.items() if chars(lines[i][0]) >= _SIZE_BODY_CHARS]
    if len(body) < _MIN_BODY_LINES:
        return {}
    reference = median(body)
    return {i: h / reference for i, h in heights.items()}


# --- emphasis ------------------------------------------------------------------------


def stroke(pixels: Pixels, box: Bounds) -> float | None:
    """Mean stroke width of the ink in *box*: twice its ink over its edges."""
    import numpy as np

    x0, y0, x1, y1 = _clip(pixels, box)
    ink = pixels.ink[y0:y1, x0:x1]
    total = int(ink.sum())
    if total < _MIN_WORD_INK:
        return None
    edges = int(np.count_nonzero(np.diff(ink, axis=1))) + int(
        np.count_nonzero(np.diff(ink, axis=0))
    )
    return 2 * total / edges if edges else None


def _slanted(pixels: Pixels, box: Bounds) -> bool:
    """Whether shearing the word upright lines its ink columns up better."""
    import numpy as np

    x0, y0, x1, y1 = _clip(pixels, box)
    ink = pixels.ink[y0:y1, x0:x1]
    height = ink.shape[0]
    if height < 2 or not ink.any():
        return False
    rows, cols = np.nonzero(ink)
    upright = float((np.bincount(cols).astype(np.float64) ** 2).sum())
    best = 0.0
    for shear in _SHEARS:
        shifted = np.round(cols - shear * (height - 1 - rows)).astype(np.int64)
        shifted -= shifted.min()
        best = max(best, float((np.bincount(shifted).astype(np.float64) ** 2).sum()))
    return upright > 0 and best >= _ITALIC_GAIN * upright


@dataclass
class Emphasis:
    """Per line (by index): each word's measured emphasis, or None when the
    word was not measured (too little ink, or the line's ink did not split
    into its words)."""

    words: dict[int, list[int | None]] = field(default_factory=dict)


def body_stroke(pixels: Pixels, boxes: list[Bounds], body: float) -> float | None:
    """The body text's stroke: the median over lines of body height."""
    low, high = _BODY_LINE_RANGE
    strokes = [
        s
        for box in boxes
        if low * body <= box[3] - box[1] <= high * body
        and (s := stroke(pixels, box)) is not None
    ]
    return median(strokes) if len(strokes) >= _MIN_BODY_LINES else None


def measure_emphasis(
    pixels: Pixels, lines: list[tuple[str, Bounds]], body: float
) -> Emphasis:
    """Bold and italic per word of each line (see the module docstring)."""
    reference = body_stroke(pixels, [b for _t, b in lines], body)
    emphasis = Emphasis()
    if reference is None:
        return emphasis
    for index, (text, box) in enumerate(lines):
        words = text.split()
        spans = word_spans(pixels, box, len(words))
        if spans is None:
            continue
        strokes = [stroke(pixels, span) for span in spans]
        flags: list[int | None] = []
        for span, width in zip(spans, strokes, strict=True):
            if width is None:
                flags.append(None)
                continue
            flag = BOLD if width >= _BOLD * reference else 0
            if _slanted(pixels, span):
                flag |= ITALIC
            flags.append(flag)
        # Next to a bold word a slightly lighter stroke is bold too
        for i, width in enumerate(strokes):
            flag = flags[i]
            if width is None or flag is None or flag & BOLD:
                continue
            neighbours = [flags[j] for j in (i - 1, i + 1) if 0 <= j < len(flags)]
            if width >= _BOLD_NEXT * reference and any(
                n is not None and n & BOLD for n in neighbours
            ):
                flags[i] = flag | BOLD
        emphasis.words[index] = flags
    return emphasis


def short_word(word: str) -> bool:
    return sum(1 for c in word if c.isalnum()) < _SHORT_WORD


def smooth_emphasis(words: list[str], flags: list[int | None]) -> list[int]:
    """A short or unmeasured word between two emphasized ones takes their
    emphasis; an emphasized short word on its own loses it."""
    out = [f or 0 for f in flags]
    for i, word in enumerate(words):
        if not (short_word(word) or flags[i] is None):
            continue
        before = next(
            (out[j] for j in range(i - 1, -1, -1) if not short_word(words[j])), 0
        )
        after = next(
            (
                flags[j] or 0
                for j in range(i + 1, len(words))
                if not short_word(words[j])
            ),
            0,
        )
        out[i] = before & after
    return out


def emphasized(words: list[str], flags: list[int], breaks: set[int]) -> str:
    """The words joined with their emphasis as Markdown runs; a line break
    goes before each word index in *breaks* (inside a run too)."""
    marks = {0: "", BOLD: "**", ITALIC: "*", BOLD | ITALIC: "***"}
    out: list[str] = []
    current = 0
    for i, word in enumerate(words):
        flag = flags[i]
        if flag != current:
            if current:
                out.append(marks[current])
            separator = "\n" if i in breaks else " "
            if out:
                out.append(separator)
            if flag:
                out.append(marks[flag])
            current = flag
        elif out:
            out.append("\n" if i in breaks else " ")
        out.append(word)
    if current:
        out.append(marks[current])
    return "".join(out)
