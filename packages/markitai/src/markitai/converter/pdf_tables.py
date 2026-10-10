"""Tables without ruling lines that the PDF extraction left as a paragraph.

pymupdf4llm finds tables from their ruling lines, and its layout model also
recognises many tables without them, but not all: a browser's default table
style (no borders, cells a few points apart) came out as one paragraph,
``Language Year Typing Primary Use Python 1991 ...``. Here the page's words
are grouped into rows, and a run of evenly spaced rows whose words leave the
same vertical white gaps in every row is read as a table with one column per
gap. It replaces a paragraph of the page's Markdown only when that
paragraph's text is exactly the table's words in reading order, so a table
the extraction already rendered, or text it reflowed differently, is left as
it is.

What is not taken for a table: fewer than three rows or three columns, cells
longer than a few words (prose, even in columns), a header row with an empty
cell, rows that leave more than one cell empty, a column of punctuation only
(aligned grammar or code), and text set mostly in a monospaced font (plain
text documents align words by spaces). Converting 39 PDFs (papers, a
two-column paper, reports, forms, an RFC, Japanese statistics, and generated
tables and prose look-alikes), it changed the two tables the extraction had
flattened and nothing else, at about 3 ms per page.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from loguru import logger

MIN_ROWS = 3
MIN_COLUMNS = 3
#: White space (points) that must separate two columns in every row.
MIN_GAP = 2.0
MAX_CELL_WORDS = 6
MAX_MEDIAN_CELL_WORDS = 3
#: Rows further apart than this many line heights belong to another block.
MAX_ROW_PITCH = 2.6

_PUNCTUATION = re.compile(r"^[^\w]+$")
_SPACES = re.compile(r"\s+")
_NOT_PROSE = ("#", "<!--", "!", "```", ">", "-", "*", "|")

Word = tuple[float, float, float, float, str]


def _visual_rows(words: list[Any]) -> list[dict[str, Any]]:
    """Words grouped by vertical overlap, top to bottom, left to right."""
    rows: list[dict[str, Any]] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        center, height = (w[1] + w[3]) / 2, w[3] - w[1]
        if rows and abs(center - rows[-1]["center"]) < 0.4 * max(
            height, rows[-1]["height"]
        ):
            row = rows[-1]
            row["words"].append(w)
            row["top"], row["bottom"] = min(row["top"], w[1]), max(row["bottom"], w[3])
        else:
            rows.append(
                {
                    "center": center,
                    "height": height,
                    "words": [w],
                    "top": w[1],
                    "bottom": w[3],
                }
            )
    for row in rows:
        row["words"].sort(key=lambda w: w[0])
    return rows


def _columns(rows: list[dict[str, Any]]) -> list[list[float]]:
    """The x extents left between the white gaps shared by every row."""
    spans = sorted((w[0], w[2]) for row in rows for w in row["words"])
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + MIN_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _cells(rows: list[dict[str, Any]], columns: list[list[float]]) -> list[list[str]]:
    table: list[list[str]] = []
    for row in rows:
        line: list[list[str]] = [[] for _ in columns]
        for w in row["words"]:
            center = (w[0] + w[2]) / 2
            index = next(
                (i for i, (start, end) in enumerate(columns) if start <= center <= end),
                0,
            )
            line[index].append(w[4])
        table.append([" ".join(cell) for cell in line])
    return table


def _plausible(table: list[list[str]]) -> bool:
    filled = [cell for row in table for cell in row if cell]
    counts = [len(cell.split()) for cell in filled]
    if not counts or max(counts) > MAX_CELL_WORDS:
        return False
    if statistics.median(counts) > MAX_MEDIAN_CELL_WORDS:
        return False
    width = len(table[0])
    if not all(table[0]):
        return False
    return all(sum(1 for cell in row if cell) >= width - 1 for row in table)


def _punctuation_column(table: list[list[str]]) -> bool:
    for i in range(len(table[0])):
        column = [row[i] for row in table if row[i]]
        if column and all(_PUNCTUATION.match(cell) for cell in column):
            return True
    return False


def _monospaced_share(page: Any, top: float, bottom: float) -> float:
    """Share of the band's characters set in a monospaced font."""
    mono = total = 0
    text = page.get_text("dict", clip=(page.rect.x0, top, page.rect.x1, bottom))
    for block in text.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                count = len(span.get("text", "").strip())
                total += count
                font = span.get("font", "").lower()
                if span.get("flags", 0) & 8 or "mono" in font or "courier" in font:
                    mono += count
    return mono / total if total else 0.0


def find_unruled_tables(page: Any) -> list[list[list[str]]]:
    """Tables read from one page's word positions (rows of cell texts)."""
    rows = _visual_rows(page.get_text("words"))
    tables: list[list[list[str]]] = []
    i = 0
    while i < len(rows):
        best: tuple[int, list[list[str]]] | None = None
        for j in range(i + MIN_ROWS, len(rows) + 1):
            block = rows[i:j]
            pitches = [b["center"] - a["center"] for a, b in zip(block, block[1:])]
            line_height = statistics.median(row["height"] for row in block)
            if (
                max(pitches) > 1.6 * min(pitches)
                or max(pitches) > MAX_ROW_PITCH * line_height
            ):
                break
            columns = _columns(block)
            if len(columns) < MIN_COLUMNS:
                break
            table = _cells(block, columns)
            if _plausible(table):
                best = (j, table)
        if best is None:
            i += 1
            continue
        j, table = best
        top, bottom = rows[i]["top"], rows[j - 1]["bottom"]
        if (
            not _punctuation_column(table)
            and _monospaced_share(page, top, bottom) < 0.5
        ):
            tables.append(table)
        i = j
    return tables


def _markdown_table(table: list[list[str]]) -> str:
    def row(cells: list[str]) -> str:
        # No cell holds a "|": a paragraph containing one is never replaced
        return "|" + "|".join(cells) + "|"

    return "\n".join(
        [row(table[0]), "|" + "|".join("---" for _ in table[0]) + "|"]
        + [row(r) for r in table[1:]]
    )


def _normalized(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def replace_flattened_tables(
    markdown: str, tables: list[list[list[str]]]
) -> tuple[str, int]:
    """Swap each paragraph that is exactly a table's words for the table."""
    if not tables:
        return markdown, 0
    paragraphs = markdown.split("\n\n")
    replaced = 0
    for table in tables:
        words = _normalized(" ".join(cell for row in table for cell in row if cell))
        for k, paragraph in enumerate(paragraphs):
            stripped = paragraph.strip()
            if not stripped or stripped.startswith(_NOT_PROSE) or "|" in stripped:
                continue
            if _normalized(stripped) == words:
                paragraphs[k] = _markdown_table(table)
                replaced += 1
                break
    return "\n\n".join(paragraphs), replaced


def restore_flattened_tables(
    input_path: Path, page_numbers: list[int], page_texts: list[str]
) -> list[str]:
    """Per-page Markdown with flattened unruled tables turned back into tables.

    Never fails a conversion: any error leaves the texts as they were.
    """
    try:
        import pymupdf

        with pymupdf.open(input_path) as doc:
            if doc.needs_pass:
                return page_texts
            result = list(page_texts)
            restored = 0
            for index, (page_number, text) in enumerate(zip(page_numbers, page_texts)):
                if not 1 <= page_number <= doc.page_count:
                    continue
                tables = find_unruled_tables(doc[page_number - 1])
                result[index], count = replace_flattened_tables(text, tables)
                restored += count
            if restored:
                logger.debug(
                    "[PDF] {}: restored {} table(s) without ruling lines",
                    input_path.name,
                    restored,
                )
            return result
    except Exception as e:
        logger.debug("[PDF] Table detection skipped for {}: {}", input_path.name, e)
        return page_texts
