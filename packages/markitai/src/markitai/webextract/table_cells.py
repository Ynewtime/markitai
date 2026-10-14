"""One Markdown table cell, written on one line.

A Markdown table row is one line, so a line break inside a cell (a newline
in a spreadsheet cell or a CSV field, a ``<br>`` or a second paragraph in an
HTML or DOCX cell) is written as ``<br>``. Runs of breaks and blank lines
become one ``<br>``, breaks at the start or end of the cell are dropped, a
``|`` is escaped, and a ``<br>`` that is part of the cell's text stays text
(``&lt;br&gt;``) instead of turning into a line break.

Dependency-free: the plain-text writers (spreadsheets, CSV, OpenDocument,
RTF) and the Markdownify renderers both use it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

#: Stands for a line break found inside a table cell (``<br>``, a paragraph)
#: while Markdownify renders the cell; :func:`markdown_cell` writes it as
#: ``<br>``. A Unicode noncharacter, reserved for exactly this internal use,
#: so it never comes from a document.
CELL_BREAK = "\ufdd0"

# A <br> tag written as the cell's text (not one Markdownify escaped already)
_LITERAL_BR = re.compile(r"(?<!\\)<(\s*/?\s*br\b[^<>]*)>", re.IGNORECASE)
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
# Line breaks Markdownify leaves in a rendered cell: it folds only tabs,
# spaces, CR and LF in text, so a vertical tab, NEL or U+2028/U+2029 reaches
# the row as is. Its own LF (between list items, nested rows) is a space.
_RENDERED_BREAK = re.compile("[\ufdd0\r\v\f\x1c-\x1e\x85\u2028\u2029]")


def _literal_br(text: str) -> str:
    return _LITERAL_BR.sub(r"&lt;\1&gt;", text)


def _join(lines: Iterable[str], escape: Callable[[str], str]) -> str:
    return "<br>".join(escape(line) for line in (s.strip() for s in lines) if line)


def text_cell(value: str) -> str:
    """A plain-text cell (spreadsheet value, CSV field) as one Markdown line.

    Every line break :meth:`str.splitlines` knows counts (``\\n``, ``\\r\\n``,
    ``\\r``, a vertical tab, U+2028, ...).
    """
    return _join(
        value.replace(CELL_BREAK, "\n").splitlines(),
        lambda line: _literal_br(line.replace("|", "\\|")),
    )


def markdown_cell(text: str) -> str:
    """A cell Markdownify rendered, its breaks marked with :data:`CELL_BREAK`.

    A vertical tab, NEL, CR or U+2028/U+2029 in the text is a break too. The
    newlines Markdownify writes itself (between a cell's list items or nested
    rows) stay spaces, as it writes them; a ``|`` it did not escape is
    escaped.
    """
    return _join(
        (part.replace("\n", " ") for part in _RENDERED_BREAK.split(text)),
        lambda line: _literal_br(_UNESCAPED_PIPE.sub(r"\\|", line)),
    )


# Line breaks a parser may delete: libxml2 before 2.14 (the one lxml bundles
# in its Windows wheels) drops C0 controls from HTML text, so "bag\vtwine"
# became "bagtwine" there while 2.14 keeps the tab. U+2028 survives every
# parser and is the same kind of break to every rule after parsing.
_PARSER_DROPPED_BREAK = re.compile("[\v\f\x1c-\x1e]")


def parser_safe_breaks(html: str) -> str:
    """*html* with each vertical tab, form feed and file, group or record
    separator written as U+2028, so that a cell's line break (or the
    whitespace between two words) reads the same whatever parser and
    libxml2 version reads it."""
    if not _PARSER_DROPPED_BREAK.search(html):
        return html
    return _PARSER_DROPPED_BREAK.sub("\u2028", html)
