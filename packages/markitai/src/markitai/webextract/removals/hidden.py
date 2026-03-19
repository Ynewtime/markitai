"""Remove hidden elements from the DOM."""

from __future__ import annotations

import re

from bs4 import Tag

_HIDDEN_STYLE_RE = re.compile(
    r"(?:^|;\s*)(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0)"
    r"(?:\s*;|\s*$)",
    re.IGNORECASE,
)

_HIDDEN_CLASSES = {"hidden", "invisible"}

# Math elements should never be removed (may use aria-hidden for a11y)
_MATH_SELECTORS = (
    "math",
    "[data-mathml]",
    ".katex",
    ".katex-mathml",
    ".katex-display",
    ".MathJax",
    ".MathJax_Display",
    ".MathJax_SVG",
    "mjx-container",
)


def remove_hidden_elements(root: Tag) -> int:
    """Remove elements hidden via inline styles or CSS framework classes.

    Preserves math elements (often use aria-hidden for accessibility).

    Args:
        root: Content root element.

    Returns:
        Number of elements removed.
    """
    removed = 0
    for el in root.find_all(True):
        if _is_math_context(el):
            continue
        if _is_hidden(el):
            el.decompose()
            removed += 1
    return removed


def _is_hidden(el: Tag) -> bool:
    """Check if element is hidden via inline style or class."""
    # Inline style check
    style = el.get("style")
    if style and _HIDDEN_STYLE_RE.search(str(style)):
        return True

    # CSS framework class check
    classes = el.get("class")
    if isinstance(classes, list):
        for cls in classes:
            bare = cls.split(":")[-1]  # handle "md:hidden" → "hidden"
            if bare in _HIDDEN_CLASSES:
                return True

    # hidden attribute
    if el.has_attr("hidden"):
        return True

    return False


def _is_math_context(el: Tag) -> bool:
    """Check if element is or contains math content."""
    # Check self
    if el.name == "math":
        return True
    raw_classes = el.get("class")
    classes_str = " ".join(raw_classes) if isinstance(raw_classes, list) else ""
    if any(s.lstrip(".") in classes_str for s in _MATH_SELECTORS if s.startswith(".")):
        return True
    if el.get("data-mathml"):
        return True

    # Check ancestors
    for parent in el.parents:
        if not isinstance(parent, Tag):
            break
        if parent.name == "math":
            return True
        parent_raw = parent.get("class")
        parent_classes = " ".join(parent_raw) if isinstance(parent_raw, list) else ""
        if "katex" in parent_classes or "MathJax" in parent_classes:
            return True
    return False
