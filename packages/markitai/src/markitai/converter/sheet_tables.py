"""Spreadsheet grids as Markdown: banner rows, one table per block.

Shared by the XLSX, XLS and ODS converters. A sheet is a grid of cell texts
plus the horizontal merges it declares. The rules, in order:

1. Fully blank rows before a table or at its end are dropped; inside a table
   they stay, so later rows keep their place.
2. A *banner row* is a row whose only value sits in a cell merged across two
   or more columns (a sheet title, a "Contact: ..." line, a section label such
   as "Afternoon"). It becomes a paragraph, never a table header, and it ends
   the table above it.
3. The rows between banners form one table each. The block's first row is its
   header, exactly as written: an empty header cell stays empty (no
   ``Unnamed: 1``), and empty cells anywhere stay empty (no ``NaN``).
4. Columns that are empty in every row of a block are left out of that table.
"""

from __future__ import annotations

import re

from markitai.converter.delimited import render_markdown_table

_WHITESPACE = re.compile(r"\s+")
# A paragraph line that would otherwise open a heading, quote, list, table,
# code fence or thematic break.
_BLOCK_START = re.compile(r"^(?:[#>+|`~=]|-|\*)")
_ORDERED_START = re.compile(r"^(\d+)([.)])(?=\s|$)")

Spans = dict[tuple[int, int], int]


def banner_text(text: str) -> str:
    """One banner cell as a Markdown paragraph (literal, not markup)."""
    text = _WHITESPACE.sub(" ", text).strip()
    text = text.replace("*", r"\*").replace("_", r"\_")
    if _BLOCK_START.match(text):
        text = "\\" + text
    return _ORDERED_START.sub(r"\1\\\2", text)


def _banner(row: list[str], row_index: int, spans: Spans) -> str | None:
    filled = [col for col, text in enumerate(row) if text.strip()]
    if len(filled) != 1:
        return None
    if spans.get((row_index, filled[0]), 1) < 2:
        return None
    return row[filled[0]]


def _table(block: list[list[str]]) -> str:
    width = max(len(row) for row in block)
    rows = [row + [""] * (width - len(row)) for row in block]
    keep = [col for col in range(width) if any(row[col].strip() for row in rows)]
    return render_markdown_table([[row[col] for col in keep] for row in rows])


def sheet_markdown(rows: list[list[str]], spans: Spans | None = None) -> str:
    """Render one sheet's grid; ``spans`` maps (row, col) to its column span."""
    spans = spans or {}
    blocks: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        while table and not any(text.strip() for text in table[-1]):
            table.pop()
        if table:
            blocks.append(_table(table))
            table.clear()

    for index, row in enumerate(rows):
        if not any(text.strip() for text in row):
            if table:  # a gap inside a table keeps its place
                table.append(row)
            continue
        banner = _banner(row, index, spans)
        if banner is None:
            table.append(row)
            continue
        flush()
        blocks.append(banner_text(banner))
    flush()
    return "\n\n".join(block for block in blocks if block)
