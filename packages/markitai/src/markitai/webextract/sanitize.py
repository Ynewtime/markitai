from __future__ import annotations

import re
from urllib.parse import unquote

from bs4 import Tag

UNSAFE_URL_PREFIXES = (
    "javascript:",
    "data:text/html",
    "data:image/svg+xml",
    "data:text/javascript",
    "vbscript:",
)
_URL_ATTRS = (
    "href",
    "xlink:href",
    "src",
    "srcset",
    "poster",
    "data",
    "action",
    "formaction",
)
# Browsers drop ASCII tab/newline anywhere in a URL and C0 controls or spaces
# at its ends, so `java\tscript:` still runs; ignore them for the scheme test.
_URL_IGNORED_CHARS = re.compile(r"[\x00-\x20]")
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
            # srcset is a comma-separated list of candidate URLs.
            urls = value.split(",") if attr == "srcset" else [value]
            if any(_is_unsafe_url(url) for url in urls):
                del tag.attrs[attr]


def _is_unsafe_url(value: str) -> bool:
    decoded = _URL_IGNORED_CHARS.sub("", unquote(value)).lower()
    return decoded.startswith(UNSAFE_URL_PREFIXES)
