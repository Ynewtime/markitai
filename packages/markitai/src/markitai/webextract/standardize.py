from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment, Tag
from bs4.element import NavigableString

from markitai.webextract.constants import ALLOWED_EMPTY_ELEMENTS
from markitai.webextract.elements.callouts import normalize_callouts
from markitai.webextract.elements.code import normalize_code_blocks
from markitai.webextract.elements.headings import normalize_headings
from markitai.webextract.elements.images import normalize_images
from markitai.webextract.elements.math import normalize_math

_PRESERVE_ELEMENTS = frozenset(
    {
        "pre",
        "code",
        "table",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "figure",
        "figcaption",
        "picture",
        "details",
        "summary",
        "blockquote",
        "form",
        "fieldset",
    }
)

_BLOCK_LEVEL = frozenset(
    {
        "div",
        "section",
        "article",
        "main",
        "aside",
        "header",
        "footer",
        "nav",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "pre",
        "blockquote",
        "figure",
        "figcaption",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "details",
        "summary",
        "address",
        "hr",
    }
)


def standardize_content(root: Tag, title: str | None, base_url: str) -> None:
    """Normalize extracted content in place.

    Args:
        root: Content root.
        title: Extracted title.
        base_url: Base URL for relative link resolution.
    """

    _remove_comments(root)
    _convert_h1_to_h2(root)
    _dedupe_title_headings(root, title)
    _resolve_relative_urls(root, base_url)
    _remove_javascript_links(root)
    _unwrap_special_links(root)
    normalize_math(root)
    normalize_code_blocks(root)
    normalize_images(root, base_url)
    normalize_headings(root)
    normalize_callouts(root)
    _unwrap_layout_tables(root)
    _flatten_wrapper_divs(root)
    _unwrap_bare_spans(root)
    _remove_empty_elements(root)
    _remove_trailing_content(root)


def _remove_comments(root: Tag) -> None:
    for comment in root.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _dedupe_title_headings(root: Tag, title: str | None) -> None:
    if not title:
        return

    seen = False
    normalized_title = " ".join(title.split())
    for heading in root.find_all(["h1", "h2", "h3"]):
        text = " ".join(heading.get_text(" ", strip=True).split())
        if text != normalized_title:
            continue
        if seen:
            heading.decompose()
        else:
            seen = True


def _resolve_relative_urls(root: Tag, base_url: str) -> None:
    for tag in root.find_all(["a", "img"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if (
            not isinstance(value, str)
            or not value
            or value.startswith(("#", "http", "data:"))
        ):
            continue
        tag[attr] = urljoin(base_url, value)


def _remove_javascript_links(root: Tag) -> None:
    for link in root.find_all("a", href=True):
        href = str(link["href"]).strip().lower()
        if href.startswith("javascript:"):
            del link["href"]


_HEADING_NAMES = ("h1", "h2", "h3", "h4", "h5", "h6")


def _unwrap_special_links(root: Tag) -> None:
    """Unwrap links that Markdown cannot represent well.

    Ported from defuddle's ``unwrapSpecialLinks`` step (standardize.ts):

    - Links inside inline code — Markdown can't render links in backticks.
    - Card links wrapping block content with a direct-child heading:
      ``<a href="/x"><h2>T</h2><p>d</p></a>`` becomes
      ``<h2><a href="/x">T</a></h2><p>d</p>``.
    - Same-page anchor links (``href="#..."``) wrapping a heading (e.g.
      clickable section headers) — unwrapped entirely.
    """
    for link in list(root.find_all("a")):
        if link.find_parent("code") is not None:
            link.unwrap()

    for link in list(root.find_all("a")):
        if link.parent is None:
            continue
        href = str(link.get("href") or "")
        if not href or href.startswith("#"):
            continue
        heading = next(
            (
                c
                for c in link.children
                if isinstance(c, Tag) and c.name in _HEADING_NAMES
            ),
            None,
        )
        if heading is None:
            continue
        # Move the href into the heading by wrapping its children
        inner = BeautifulSoup("", "html.parser").new_tag("a", href=href)
        for child in list(heading.children):
            inner.append(child.extract())
        heading.append(inner)
        link.unwrap()

    for link in list(root.find_all("a", href=True)):
        if link.parent is None:
            continue
        if str(link["href"]).startswith("#") and link.find(_HEADING_NAMES):
            link.unwrap()


def _convert_h1_to_h2(root: Tag) -> None:
    """Convert H1 tags to H2 when there are multiple H1s."""
    h1s = root.find_all("h1")
    if len(h1s) <= 1:
        return
    for h1 in h1s:
        h1.name = "h2"


def _unwrap_bare_spans(root: Tag) -> None:
    """Remove <span> elements with no attributes, keeping their content."""
    for span in root.find_all("span"):
        if not span.attrs:
            span.unwrap()


def _remove_empty_elements(root: Tag) -> None:
    """Remove elements with no text content or children.

    Preserves void elements (img, br, hr, etc.), whitespace-significant
    content inside ``<pre>``/``<code>`` (syntax highlighters emit
    whitespace-only token spans), and the root itself.
    """
    for el in root.find_all(True):
        if el is root:
            continue
        if el.name in ALLOWED_EMPTY_ELEMENTS:
            continue
        if el.find_parent(("pre", "code")) is not None:
            continue
        if not el.get_text(strip=True) and not el.find(list(ALLOWED_EMPTY_ELEMENTS)):
            el.decompose()


def _remove_trailing_content(root: Tag) -> None:
    """Remove trailing <hr> elements and leading <hr> elements."""
    # Remove trailing hr
    children = [c for c in root.children if isinstance(c, Tag)]
    while children and children[-1].name == "hr":
        children[-1].decompose()
        children = [c for c in root.children if isinstance(c, Tag)]
    # Remove leading hr
    children = [c for c in root.children if isinstance(c, Tag)]
    while children and children[0].name == "hr":
        children[0].decompose()
        children = [c for c in root.children if isinstance(c, Tag)]


def _flatten_wrapper_divs(root: Tag) -> None:
    """Unwrap wrapper divs that add no semantic value.

    Targets divs that are purely structural wrappers:
    - Empty divs (no text, no children)
    - Divs with a single block-level child
    """
    changed = True
    while changed:
        changed = False
        for div in root.find_all("div"):
            if div is root:
                continue
            if div.name in _PRESERVE_ELEMENTS:
                continue

            # Skip divs with semantic roles or classes
            if not div.attrs:
                continue
            if div.get("role"):
                continue
            # Preserve the standardized footnotes container
            if div.get("id") == "footnotes":
                continue
            classes = div.get("class")
            if isinstance(classes, list) and any(
                c.lower() in ("article", "main", "content", "footnote", "reference")
                for c in classes
            ):
                continue

            # Empty div (only whitespace text)
            if not div.get_text(strip=True) and not div.find(
                list(ALLOWED_EMPTY_ELEMENTS)
            ):
                div.decompose()
                changed = True
                continue

            # Single block-level child — unwrap the wrapper
            tag_children = [c for c in div.children if isinstance(c, Tag)]
            text_children = [
                c
                for c in div.children
                if isinstance(c, NavigableString)
                and not isinstance(c, Comment)
                and c.strip()
            ]
            if (
                len(tag_children) == 1
                and not text_children
                and tag_children[0].name in _BLOCK_LEVEL
            ):
                div.unwrap()
                changed = True


def _unwrap_layout_tables(root: Tag) -> None:
    """Unwrap layout tables, preserving data tables.

    Layout tables are detected by:
    - Single-column tables (every row has ≤1 cell) without <th> headers
    - Tables containing nested tables

    Data tables (with <th> or multi-column structure) are preserved.
    Ported from defuddle's standardize.ts table unwrapping logic.
    """
    for table in list(root.find_all("table")):
        if table.parent is None:
            continue  # already detached

        # Get direct cells and rows (not from nested tables)
        direct_cells = [
            cell
            for cell in table.find_all(["td", "th"])
            if _is_direct_table_child(cell, table)
        ]
        direct_rows = [
            row for row in table.find_all("tr") if _is_direct_table_child(row, table)
        ]

        # Has nested tables → layout table, unwrap
        if table.find("table"):
            _unwrap_table_cells(table, direct_cells)
            continue

        # Skip data tables with header cells
        if any(cell.name == "th" for cell in direct_cells):
            continue

        # Skip if no rows
        if not direct_rows:
            continue

        # Check single-column: every row has at most 1 direct cell
        is_single_column = all(
            sum(1 for cell in direct_cells if cell.parent is row) <= 1
            for row in direct_rows
        )
        if is_single_column:
            _unwrap_table_cells(table, direct_cells)


def _is_direct_table_child(el: Tag, table: Tag) -> bool:
    """Check if element belongs directly to this table (not a nested one)."""
    parent = el.parent
    while parent is not None and parent is not table:
        if isinstance(parent, Tag) and parent.name == "table":
            return False  # belongs to a nested table
        parent = parent.parent
    return parent is table


def _unwrap_table_cells(table: Tag, cells: list[Tag]) -> None:
    """Replace a layout table with its cell contents."""
    # Collect all cell content
    fragments: list[Tag | NavigableString] = []
    for cell in cells:
        for child in list(cell.children):
            extracted = child.extract()
            if isinstance(extracted, (Tag, NavigableString)):
                fragments.append(extracted)

    # Replace table with cell contents
    for fragment in reversed(fragments):
        table.insert_after(fragment)
    table.decompose()
