"""Known Excel inputs, retaining pandas cell semantics without a table DOM."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from markitai.converter.base import ConvertResult
from markitai.converter.structured_text import _normalize

_NEWLINE_WHITESPACE = re.compile(r"[\t \r\n]*[\r\n][\t \r\n]*")
_HORIZONTAL_WHITESPACE = re.compile(r"[\t ]+")


class _DataframeTable(HTMLParser):
    """Render the escaped, single-header table emitted by DataFrame.to_html.

    This is intentionally not a general HTML renderer. Excel's default reader
    supplies a flat header, and pandas escapes cell HTML before it reaches us.
    Preserve Markdownify's default whitespace and escaping rules for those
    plain-text cells. No document nodes or recursive traversal are necessary.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.renderer: Any = None
        # Default plain cells only escape emphasis markers. Avoid loading an
        # HTML parser for this; retain loaded adapters and modified defaults.
        if "markdownify" in sys.modules:
            from markitai.webextract.markdownify_compat import (
                CompatibleMarkdownConverter,
            )

            self.renderer = CompatibleMarkdownConverter()
        self.rows: list[str] = []
        self.cells: list[str] = []
        self.data: list[str] | None = None
        self.header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "thead":
            self.header = True
        elif tag == "tr":
            self.cells = []
        elif tag in {"th", "td"}:
            self.data = []

    def handle_data(self, data: str) -> None:
        if self.data is not None:
            self.data.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.data is not None:
            text = _NEWLINE_WHITESPACE.sub("\n", "".join(self.data))
            text = _HORIZONTAL_WHITESPACE.sub(" ", text)
            if self.renderer is None:
                text = text.replace("*", r"\*").replace("_", r"\_")
            else:
                text = self.renderer.escape(text, {tag})
            self.cells.append(" " + text.strip().replace("\n", " ") + " |")
            self.data = None
        elif tag == "tr":
            self.rows.append("|" + "".join(self.cells))
            if self.header:
                self.rows.append("| " + " | ".join(["---"] * len(self.cells)) + " |")
        elif tag == "thead":
            self.header = False


def dataframe_table_markdown(html: str) -> str:
    """Convert only the escaped HTML produced by pandas.to_html(index=False)."""
    parser = _DataframeTable()
    parser.feed(html)
    parser.close()
    return "\n".join(parser.rows).strip()


def dataframe_markdown(frame: Any) -> str:
    """Keep pandas formatting while avoiding HTML serialization and parsing.

    Excel's default reader produces flat columns. Let pandas select and format
    the cells exactly as its HTML formatter does, then pass their text straight
    to the existing Markdown cell renderer. Multi-level headers retain the HTML
    path, since their spans require the original table handling.
    """
    from pandas.io.formats.format import DataFrameFormatter
    from pandas.io.formats.html import HTMLFormatter
    from pandas.io.formats.printing import pprint_thing

    if frame.columns.nlevels != 1:
        return dataframe_table_markdown(frame.to_html(index=False))

    parser = _DataframeTable()
    formatter = DataFrameFormatter(frame, index=False)
    # pandas 2.2 writes literal paired spaces; newer formatters encode NBSP.
    # Query one constant cell so this follows the installed implementation
    # rather than assuming all supported releases have the same semantics.
    probe = HTMLFormatter(formatter)
    probe.write_td("x  y")
    preserve_spaces = "&nbsp;" in "".join(probe.elements)

    class MarkdownRows(HTMLFormatter):
        def write_tr(
            self,
            line: Any,
            indent: int = 0,
            indent_delta: int = 0,
            header: bool = False,
            align: str | None = None,
            tags: Any = None,
            nindex_levels: int = 0,
        ) -> None:
            parser.header = header
            parser.handle_starttag("tr", [])
            for value in line:
                kind = "th" if header else "td"
                parser.handle_starttag(kind, [])
                # HTMLFormatter strips padding and encodes paired spaces as
                # NBSP entities. Supply the text an HTML parser would deliver,
                # without escaping/unescaping literal user HTML or entities.
                text = pprint_thing(value, escape_chars={}).strip()
                if preserve_spaces:
                    text = text.replace("  ", "\u00a0\u00a0")
                parser.handle_data(text)
                parser.handle_endtag(kind)
            parser.handle_endtag("tr")

    MarkdownRows(formatter).render()
    return "\n".join(parser.rows).strip()


_PLAIN_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_AMBIGUOUS_TEXT = frozenset(
    {
        "-1.#ind",
        "-1.#qnan",
        "1.#ind",
        "1.#qnan",
        "-nan",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "<na>",
        "true",
        "false",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
)


def _plain_text(value: str) -> bool:
    text = value.strip()
    return bool(text) and not (
        "  " in value
        or any(c in value for c in "\r\n\t")
        or text.casefold() in _AMBIGUOUS_TEXT
        or text.startswith("#")
        or text.isnumeric()
        or _PLAIN_NUMBER.fullmatch(text)
    )


def plain_table_markdown(data: list[list[Any]]) -> str | None:
    """Render only homogeneous, complete scalar columns with plain formatting.

    Returning None retains pandas inference and formatting. Callers also retain
    pandas whenever it is already loaded, so runtime display options keep their
    usual effect. Ambiguous whitespace stays with the installed pandas version.
    """
    if data:
        header = data[0]
        if not header or any(type(v) not in (str, int) or v == "" for v in header):
            return None
        if len(set(header)) != len(header):
            return None
        if any(
            isinstance(v, str) and ("  " in v or any(c in v for c in "\r\n\t"))
            for v in header
        ):
            return None
        kinds = None
        for row in data[1:]:
            if len(row) != len(header):
                return None
            types = [type(v) for v in row]
            if kinds is not None and types != kinds:
                return None
            kinds = types
            for value in row:
                if type(value) is int:
                    if not -(2**63) <= value < 2**63:
                        return None
                elif type(value) is str:
                    if not _plain_text(value):
                        return None
                elif type(value) is not bool:
                    return None
    parser = _DataframeTable()
    for index, row in enumerate(data or [[]]):
        parser.header = index == 0
        parser.handle_starttag("tr", [])
        for value in row:
            kind = "th" if parser.header else "td"
            parser.handle_starttag(kind, [])
            parser.handle_data(str(value).strip())
            parser.handle_endtag(kind)
        parser.handle_endtag("tr")
    return "\n".join(parser.rows).strip()


def _sheet_rows(sheet: Any) -> list[list[Any]]:
    """Read once using pandas' default openpyxl scalar and padding semantics."""
    sheet.reset_dimensions()
    data: list[list[Any]] = []
    last_nonempty = 0
    for cells in sheet.rows:
        row = []
        for cell in cells:
            value = cell.value
            if value is None:
                value = ""
            elif cell.data_type == "e":
                value = float("nan")
            elif cell.data_type == "n":
                integer = int(value)
                value = integer if integer == value else float(value)
            row.append(value)
        while row and row[-1] == "":
            row.pop()
        data.append(row)
        if row:
            last_nonempty = len(data)
    del data[last_nonempty:]
    width = max(map(len, data), default=0)
    for row in data:
        row.extend([""] * (width - len(row)))
    return data


def _pandas_sheet(data: list[list[Any]], name: str) -> Any:
    import pandas as pd
    from pandas.io.parsers import TextParser

    if not data:
        return pd.DataFrame()
    try:
        return TextParser(data, header=0, skip_blank_lines=False).read()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as error:
        error.args = (f"{error.args[0]} (sheet: {name})", *error.args[1:])
        raise


def convert_xlsx(path: Path) -> ConvertResult:
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    sections = []
    engine = "openpyxl"
    try:
        for sheet in book.worksheets:
            data = _sheet_rows(sheet)
            markdown = (
                plain_table_markdown(data) if "pandas" not in sys.modules else None
            )
            if markdown is None:
                engine = "pandas"
                markdown = dataframe_markdown(_pandas_sheet(data, sheet.title))
            sections.append(f"## {sheet.title}\n{markdown}")
    finally:
        book.close()
    return ConvertResult(
        markdown=_normalize("\n\n".join(sections).strip()),
        metadata={"source": str(path), "format": "XLSX", "converter": engine},
    )
