"""Simple HTML tables (one header row, escaped cells) as Markdown rows.

PPTX tables are rendered through this, with Markdownify's default whitespace
and escaping rules for plain-text cells (it was the spreadsheet renderer's
table writer before spreadsheets got their own, :mod:`sheet_tables`).
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from typing import Any

_NEWLINE_WHITESPACE = re.compile(r"[\t \r\n]*[\r\n][\t \r\n]*")
_HORIZONTAL_WHITESPACE = re.compile(r"[\t ]+")


class _HtmlTable(HTMLParser):
    """Render an escaped, single-header HTML table.

    This is intentionally not a general HTML renderer: the caller builds the
    table from plain cell texts it escaped itself (``html.escape``). Preserve
    Markdownify's default whitespace and escaping rules for those plain-text
    cells. No document nodes or recursive traversal are necessary.
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


def html_table_markdown(html: str) -> str:
    """Convert a table of one ``<thead>`` row and escaped plain-text cells."""
    parser = _HtmlTable()
    parser.feed(html)
    parser.close()
    return "\n".join(parser.rows).strip()
