"""Remove small images and tracking pixels."""

from __future__ import annotations

import re

from bs4 import Tag

_DIMENSION_STYLE_RE = re.compile(r"(?:^|;\s*)([\w-]+)\s*:\s*(\d+)")
_MIN_SIZE = 33


def remove_small_images(root: Tag, min_size: int = _MIN_SIZE) -> int:
    """Remove images and SVGs smaller than *min_size* pixels.

    Only removes when BOTH dimensions are known and below the threshold.
    Images without dimension info are kept (conservative).

    Args:
        root: Content root element.
        min_size: Minimum dimension in pixels.

    Returns:
        Number of elements removed.
    """
    removed = 0
    for el in root.find_all(["img", "svg"]):
        w, h = _get_dimensions(el)
        if w is not None and h is not None and w < min_size and h < min_size:
            el.decompose()
            removed += 1
    return removed


def _get_dimensions(el: Tag) -> tuple[int | None, int | None]:
    """Extract width and height from attributes or inline style."""
    w = _parse_int(el.get("width"))
    h = _parse_int(el.get("height"))
    if w is not None and h is not None:
        return w, h

    style = el.get("style", "")
    if style:
        for prop, val in _DIMENSION_STYLE_RE.findall(str(style)):
            if prop == "width" and w is None:
                w = int(val)
            elif prop == "height" and h is None:
                h = int(val)
    return w, h


def _parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip().rstrip("px"))
    except (ValueError, TypeError):
        return None
