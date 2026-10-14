"""Excel workbooks (XLSX with openpyxl, XLS with xlrd) as Markdown tables.

Cells are read as written: an empty cell is empty, the text ``NA`` stays
``NA``, and a header-less column has an empty header. Sheet layout (banner
rows, one table per block) follows :mod:`markitai.converter.sheet_tables`.
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from markitai.converter.base import ConvertResult
from markitai.converter.sheet_tables import Spans, sheet_markdown
from markitai.converter.structured_text import _normalize

_MERGE_REF = re.compile(
    rb"<(?:[A-Za-z_][\w.-]*:)?mergeCell\b[^>]*?\bref=\"([A-Z]{1,3})([0-9]+)"
    rb"(?::([A-Z]{1,3})([0-9]+))?\""
)
_CHUNK = 1 << 20


def cell_text(value: Any) -> str:
    """The text a cell shows, without pandas' column-wide inference."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return ""
        if value.is_integer() and abs(value) < 2**63:
            return str(int(value))
        # Excel keeps 15 significant digits (0.1 + 0.2 shows as 0.3) and
        # never writes an exponent into a General cell this short.
        text = f"{value:.15g}"
        return format(Decimal(text), "f") if "e" in text else text
    if isinstance(value, datetime.datetime):
        if value.time() == datetime.time(0):
            return value.date().isoformat()
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return str(value)


def _column_number(letters: bytes) -> int:
    number = 0
    for letter in letters:
        number = number * 26 + letter - 64
    return number


def _xlsx_spans(sheet: Any) -> Spans:
    """Horizontal merges, read from the sheet part (read-only mode has none).

    ``<mergeCells>`` follows the cell data, so the part is scanned in chunks
    with a small overlap rather than loaded whole.
    """
    spans: Spans = {}
    tail = b""
    source = sheet._get_source()
    try:
        while chunk := source.read(_CHUNK):
            data = tail + chunk
            for match in _MERGE_REF.finditer(data):
                first, row, last, _ = match.groups()
                start = _column_number(first)
                end = _column_number(last) if last else start
                if end > start:
                    spans[(int(row) - 1, start - 1)] = end - start + 1
            tail = data[-256:]
    finally:
        source.close()
    return spans


def _xlsx_rows(sheet: Any) -> list[list[str]]:
    sheet.reset_dimensions()
    rows: list[list[str]] = []
    for cells in sheet.rows:
        row: list[str] = []
        for cell in cells:
            value = getattr(cell, "value", None)
            # An error cell shows its code (#DIV/0!, #N/A, ...).
            row.append(str(value) if cell.data_type == "e" else cell_text(value))
        rows.append(row)
    return rows


def _section(title: str, markdown: str) -> str:
    return f"## {title}\n\n{markdown}" if markdown else f"## {title}"


def convert_xlsx(path: Path) -> ConvertResult:
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    sections = []
    try:
        for sheet in book.worksheets:
            markdown = sheet_markdown(_xlsx_rows(sheet), _xlsx_spans(sheet))
            sections.append(_section(sheet.title, markdown))
    finally:
        book.close()
    return ConvertResult(
        markdown=_normalize("\n\n".join(sections).strip()),
        metadata={"source": str(path), "format": "XLSX", "converter": "openpyxl"},
    )


def _xls_value(book: Any, cell: Any) -> str:
    import xlrd

    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            return cell_text(xlrd.xldate.xldate_as_datetime(cell.value, book.datemode))
        except (ValueError, OverflowError, xlrd.xldate.XLDateError):
            return cell_text(cell.value)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return cell_text(bool(cell.value))
    if cell.ctype == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(cell.value, "#N/A")
    if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return ""
    return cell_text(cell.value)


def convert_xls(path: Path) -> ConvertResult:
    import xlrd

    try:
        book = xlrd.open_workbook(str(path), formatting_info=True, on_demand=True)
    except NotImplementedError:
        # formatting_info is BIFF8-only; older files convert without merges.
        book = xlrd.open_workbook(str(path), on_demand=True)
    sections = []
    try:
        for index in range(book.nsheets):
            sheet = book.sheet_by_index(index)
            rows = [
                [_xls_value(book, sheet.cell(r, c)) for c in range(sheet.ncols)]
                for r in range(sheet.nrows)
            ]
            spans: Spans = {
                (r0, c0): c1 - c0
                for r0, _r1, c0, c1 in getattr(sheet, "merged_cells", [])
                if c1 - c0 > 1
            }
            sections.append(_section(sheet.name, sheet_markdown(rows, spans)))
            book.unload_sheet(index)
    finally:
        book.release_resources()
    return ConvertResult(
        markdown=_normalize("\n\n".join(sections).strip()),
        metadata={"source": str(path), "format": "XLS", "converter": "xlrd"},
    )
