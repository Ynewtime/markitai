from __future__ import annotations

from bs4 import BeautifulSoup, Tag

UNSAFE_URL_PREFIXES = ("javascript:", "data:text/html", "data:image/svg+xml")
REMOVE_TAGS = {
    "script",
    "style",
    "object",
    "embed",
    "iframe",
    "noscript",
    "form",
    "button",
    "input",
    "textarea",
    "select",
}


def sanitize_html_fragment(html: str) -> str:
    """Remove unsafe attributes, links, and obvious noise tags.

    Args:
        html: HTML fragment to sanitize.

    Returns:
        Sanitized HTML string.
    """

    soup = BeautifulSoup(html, "html.parser")
    for tag in list(soup.find_all(True)):
        _sanitize_tag(tag)
    return str(soup)


def sanitize_tag_tree(root: Tag) -> None:
    """Sanitize a parsed tag tree in place.

    Args:
        root: Root tag to sanitize.
    """

    for tag in list(root.find_all(True)):
        _sanitize_tag(tag)


def _sanitize_tag(tag: Tag) -> None:
    if tag.name in REMOVE_TAGS:
        tag.decompose()
        return

    for attr in list(tag.attrs):
        if attr.startswith("on"):
            del tag.attrs[attr]

    for attr in ("href", "src"):
        value = tag.get(attr)
        if isinstance(value, str) and value.strip().lower().startswith(
            UNSAFE_URL_PREFIXES
        ):
            del tag.attrs[attr]
