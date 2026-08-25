"""Remove small images and tracking pixels."""

from __future__ import annotations

import re

from bs4 import Tag

_STYLE_WIDTH_RE = re.compile(r"width\s*:\s*(\d+)")
_STYLE_HEIGHT_RE = re.compile(r"height\s*:\s*(\d+)")
_URL_WIDTH_RE = re.compile(r"(?:width[=:/]|[/,?&]w[_:=])(\d+)")
_SRCSET_1X_RE = re.compile(r"(\S+)\s+1x")
_LEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")
# LaTeX command detector for math-image protection (mirrors defuddle).
_LOOKS_LIKE_LATEX_RE = re.compile(r"\\[a-zA-Z]{2,}")
_MIN_SIZE = 33


def remove_small_images(root: Tag, min_size: int = _MIN_SIZE) -> int:
    """Remove images and SVGs smaller than *min_size* pixels.

    An element is small when the minimum of its known dimensions (width /
    height attributes, inline style, SVG viewBox, srcset ``1x`` URL width
    hint) falls below the threshold on either axis. Elements without any
    dimension info are kept, as are math/equation images. Mirrors
    defuddle ``removals/small-images.ts``.

    Args:
        root: Content root element.
        min_size: Minimum dimension in pixels.

    Returns:
        Number of elements removed.
    """
    removed = 0
    for el in list(root.find_all(["img", "svg"])):
        widths, heights = _collect_dimensions(el)
        if not widths and not heights:
            continue
        effective_w = min(widths) if widths else float("inf")
        effective_h = min(heights) if heights else float("inf")
        if effective_w >= min_size and effective_h >= min_size:
            continue
        if el.name == "img" and _is_math_image(el):
            continue
        el.decompose()
        removed += 1
    return removed


def _collect_dimensions(el: Tag) -> tuple[list[float], list[float]]:
    """Gather all known width/height hints for an element."""
    widths: list[float] = []
    heights: list[float] = []

    for attr, bucket in (("width", widths), ("height", heights)):
        parsed = _parse_number(el.get(attr))
        if parsed and parsed > 0:
            bucket.append(parsed)

    style = str(el.get("style") or "")
    if style:
        for pattern, bucket in (
            (_STYLE_WIDTH_RE, widths),
            (_STYLE_HEIGHT_RE, heights),
        ):
            match = pattern.search(style)
            if match and float(match.group(1)) > 0:
                bucket.append(float(match.group(1)))

    # For SVGs, viewBox dimensions count as a size hint (icon sprites
    # often declare only a viewBox).
    if el.name == "svg":
        view_box = str(el.get("viewbox") or el.get("viewBox") or "")
        parts = re.split(r"[\s,]+", view_box.strip())
        if len(parts) == 4:
            vw, vh = _parse_number(parts[2]), _parse_number(parts[3])
            if vw and vw > 0:
                widths.append(vw)
            if vh and vh > 0:
                heights.append(vh)

    # Fallback: images served at tiny widths via CDN params (e.g. ?w=32 1x)
    if not widths and not heights and el.name == "img":
        one_x = _SRCSET_1X_RE.search(str(el.get("srcset") or ""))
        if one_x:
            url_match = _URL_WIDTH_RE.search(one_x.group(1))
            if url_match and float(url_match.group(1)) > 0:
                widths.append(float(url_match.group(1)))

    return widths, heights


def _is_math_image(el: Tag) -> bool:
    """Check if an image renders an equation (must not be removed)."""
    if _LOOKS_LIKE_LATEX_RE.search(str(el.get("alt") or "")):
        return True
    classes = el.get("class")
    if isinstance(classes, list) and ({"latex", "tex"} & set(classes)):
        return True
    return bool(el.get("data-latex") or el.get("data-math"))


def _parse_number(value: object) -> float | None:
    """Parse the leading number of a dimension (parseFloat semantics).

    Mirrors defuddle: ``"1em"`` parses to 1, so em-sized icon SVGs count
    as small; ``"100%"`` parses to 100.
    """
    if value is None:
        return None
    match = _LEADING_NUMBER_RE.match(str(value))
    return float(match.group(1)) if match else None
