from __future__ import annotations

from urllib.parse import urljoin

from bs4 import Comment, NavigableString, Tag

from markitai.webextract.constants import ALLOWED_EMPTY_ELEMENTS
from markitai.webextract.elements.callouts import normalize_callouts
from markitai.webextract.elements.code import normalize_code_blocks
from markitai.webextract.elements.footnotes import normalize_footnotes
from markitai.webextract.elements.headings import normalize_headings
from markitai.webextract.elements.images import normalize_images

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
    normalize_code_blocks(root)
    normalize_footnotes(root)
    normalize_images(root, base_url)
    normalize_headings(root)
    normalize_callouts(root)
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

    Preserves void elements (img, br, hr, etc.) and the root itself.
    """
    for el in root.find_all(True):
        if el is root:
            continue
        if el.name in ALLOWED_EMPTY_ELEMENTS:
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
            if div.get("role"):
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
