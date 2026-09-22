from __future__ import annotations

from urllib.parse import unquote

from bs4 import Tag

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
    # Inert template fragments evade descendant traversal and can cause
    # mutation XSS once re-parsed. Declarative shadow roots were hoisted
    # out of their templates before parsing, so what remains is inert.
    "template",
    "form",
    "button",
    "input",
    "textarea",
    "select",
    # SVG SMIL elements can mutate sanitized URL attributes.
    "animate",
    "set",
    "animatemotion",
    "animatetransform",
    "animatecolor",
    "discard",
}


def sanitize_tag_tree(root: Tag) -> None:
    """Sanitize a parsed tag tree in place.

    ``find_all`` omits the root: a root that is itself an unsafe element is
    emptied instead of removed, so the caller serializes nothing.

    Args:
        root: Root tag to sanitize.
    """

    if _is_unsafe(root):
        root.attrs.clear()
        root.clear()
        return
    for tag in list(root.find_all(True)):
        _sanitize_tag(tag)


def _is_unsafe(tag: Tag) -> bool:
    # Checkbox inputs are kept for task-list support.
    if tag.name == "input" and tag.get("type") == "checkbox":
        return False
    return tag.name in REMOVE_TAGS


def _sanitize_tag(tag: Tag) -> None:
    if _is_unsafe(tag):
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
