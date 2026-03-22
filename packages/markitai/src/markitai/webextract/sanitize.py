from __future__ import annotations

from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag

UNSAFE_URL_PREFIXES = (
    "javascript:",
    "data:text/html",
    "data:image/svg+xml",
    "data:text/javascript",
    "vbscript:",
)
_URL_ATTRS = ("href", "src", "action", "formaction")
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
        # Preserve checkbox inputs for task list support
        if tag.name == "input" and tag.get("type") == "checkbox":
            return
        tag.decompose()
        return

    if not tag.attrs:
        return

    for attr in list(tag.attrs):
        if attr.startswith("on"):
            del tag.attrs[attr]

    for attr in _URL_ATTRS:
        value = tag.get(attr)
        if isinstance(value, str):
            decoded = unquote(value).strip().lower()
            if decoded.startswith(UNSAFE_URL_PREFIXES):
                del tag.attrs[attr]
