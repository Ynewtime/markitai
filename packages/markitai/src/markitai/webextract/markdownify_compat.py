"""Narrow Markdownify rules ported from MarkItDown 0.1.7.

Keep the HTML renderer independent of MarkItDown's package initializer, which
loads format detectors, ONNX, and every Office converter. Deliberately preserve
these rules so dependency isolation does not silently change Markdown output.
Upstream: microsoft/markitdown, converters/_markdownify.py (MIT).
Copyright (c) Microsoft Corporation. See NOTICE for the full license.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, unquote, urlparse, urlunparse

import markdownify
from bs4 import BeautifulSoup, Tag

from markitai.webextract.table_cells import (
    CELL_BREAK,
    markdown_cell,
    parser_safe_breaks,
)


class _MarkdownTag(Tag):
    """Avoid allocating a general query for Markdownify's per-tag pre check.

    Follow current parent pointers so detached or moved nodes remain correct.
    Only the exact positional call used by the renderer takes this shortcut;
    filters, callbacks, and all other queries retain BeautifulSoup semantics.
    """

    def find_parent(self, *args: Any, **kwargs: Any) -> Tag | None:
        if len(args) == 1 and type(args[0]) is str and args[0] == "pre" and not kwargs:
            parent = self.parent
            while parent is not None:
                if parent.name == "pre":
                    return parent
                parent = parent.parent
            return None
        return super().find_parent(*args, **kwargs)


def parse_markdown_html(
    markup: Any, features: str = "html.parser", **kwargs: Any
) -> BeautifulSoup:
    """Use specialized tags only in the temporary DOM built for rendering.

    Text markup has its line-break controls made parser-safe first
    (:func:`~markitai.webextract.table_cells.parser_safe_breaks`).
    """
    if isinstance(markup, str):
        markup = parser_safe_breaks(markup)
    kwargs.setdefault("element_classes", {Tag: _MarkdownTag})
    return BeautifulSoup(markup, features, **kwargs)


# Markdownify's own rules, called by name from TableCellLines (untyped there)
_MARKDOWNIFY: Any = markdownify.MarkdownConverter


def _in_cell(parent_tags: set[str]) -> bool:
    return "td" in parent_tags or "th" in parent_tags


class TableCellLines:
    """A table cell's lines joined with ``<br>``, the row kept on one line.

    Markdownify writes a ``<br>``, ``<p>`` or ``<div>`` inside a cell as a
    space and leaves a U+2028 or vertical tab in its text as is (which ends
    the table in a GFM renderer). These rules mark the breaks with
    :data:`CELL_BREAK` and write each cell through :func:`markdown_cell`.
    They call ``MarkdownConverter``'s own rules by name, so the same
    functions also serve MarkItDown's converter class
    (:func:`markitai.converter._patches.apply_markitdown_cell_patches`).
    """

    def convert_br(self: Any, el: Any, text: str, parent_tags: set[str]) -> str:
        if _in_cell(parent_tags):
            return CELL_BREAK + text
        return _MARKDOWNIFY.convert_br(self, el, text, parent_tags)

    def convert_p(self: Any, el: Any, text: str, parent_tags: set[str]) -> str:
        if _in_cell(parent_tags):
            return CELL_BREAK + text.strip() + CELL_BREAK
        return _MARKDOWNIFY.convert_p(self, el, text, parent_tags)

    def convert_div(self: Any, el: Any, text: str, parent_tags: set[str]) -> str:
        if _in_cell(parent_tags):
            return CELL_BREAK + text.strip() + CELL_BREAK
        return _MARKDOWNIFY.convert_div(self, el, text, parent_tags)

    def convert_td(self: Any, el: Any, text: str, parent_tags: set[str]) -> str:
        return _MARKDOWNIFY.convert_td(self, el, markdown_cell(text), parent_tags)

    def convert_th(self: Any, el: Any, text: str, parent_tags: set[str]) -> str:
        return _MARKDOWNIFY.convert_th(self, el, markdown_cell(text), parent_tags)


class CompatibleMarkdownConverter(TableCellLines, markdownify.MarkdownConverter):
    """A custom version of markdownify's MarkdownConverter. Changes include:

    - Altering the default heading style to use '#', '##', etc.
    - Removing javascript hyperlinks.
    - Truncating images with large data:uri sources.
    - Ensuring URIs are properly escaped, and do not conflict with Markdown syntax
    - Line breaks inside table cells written as ``<br>`` (:class:`TableCellLines`)
    """

    options: dict[str, Any]

    def __init__(self, **options: Any):
        options["heading_style"] = options.get("heading_style", markdownify.ATX)
        options["keep_data_uris"] = options.get("keep_data_uris", False)
        # Explicitly cast options to the expected type if necessary
        super().__init__(**options)

    def convert_hn(
        self,
        n: int,
        el: Any,
        text: str,
        convert_as_inline: bool | None = False,
        **kwargs,
    ) -> str:
        """Same as usual, but be sure to start with a new line"""
        if not convert_as_inline:
            if not re.search(r"^\n", text):
                return "\n" + super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

        return super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

    def convert_a(
        self,
        el: Any,
        text: str,
        convert_as_inline: bool | None = False,
        **kwargs,
    ):
        """Same as usual converter, but removes JavaScript links and escapes URIs."""
        prefix, suffix, text = markdownify.chomp(text)  # type: ignore
        if not text:
            return ""

        if el.find_parent("pre") is not None:
            return text

        href = el.get("href")
        title = el.get("title")

        # Escape URIs and skip non-http or file schemes
        if href:
            try:
                parsed_url = urlparse(href)  # type: ignore
                if parsed_url.scheme and parsed_url.scheme.lower() not in [
                    "http",
                    "https",
                    "file",
                ]:  # type: ignore
                    return f"{prefix}{text}{suffix}"
                href = urlunparse(
                    parsed_url._replace(path=quote(unquote(parsed_url.path)))
                )  # type: ignore
            except ValueError:  # It's not clear if this ever gets thrown
                return f"{prefix}{text}{suffix}"

        # For the replacement see #29: text nodes underscores are escaped
        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            # Shortcut syntax
            return f"<{href}>"
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "{}"'.format(title.replace('"', r"\"")) if title else ""
        return f"{prefix}[{text}]({href}{title_part}){suffix}" if href else text

    def convert_img(
        self,
        el: Any,
        text: str,
        convert_as_inline: bool | None = False,
        **kwargs,
    ) -> str:
        """Same as usual converter, but removes data URIs"""

        alt = el.attrs.get("alt", None) or ""
        src = el.attrs.get("src", None) or el.attrs.get("data-src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "{}"'.format(title.replace('"', r"\"")) if title else ""
        # Remove all line breaks from alt
        alt = alt.replace("\n", " ")
        if (
            convert_as_inline
            and el.parent.name not in self.options["keep_inline_images_in"]
        ):
            return alt

        # Remove dataURIs
        if src.startswith("data:") and not self.options["keep_data_uris"]:
            src = src.split(",")[0] + "..."

        return f"![{alt}]({src}{title_part})"

    def convert_input(
        self,
        el: Any,
        text: str,
        convert_as_inline: bool | None = False,
        **kwargs,
    ) -> str:
        """Convert checkboxes to Markdown [x]/[ ] syntax."""

        if el.get("type") == "checkbox":
            return "[x] " if el.has_attr("checked") else "[ ] "
        return ""

    def convert_soup(self, soup: Any) -> str:
        return super().convert_soup(soup)  # type: ignore
