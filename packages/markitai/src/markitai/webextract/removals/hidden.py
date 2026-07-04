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

# CSS Modules / CSS-in-JS frameworks produce hashed class names like
# "isHidden-vzcyV0", "is-hidden-abc123".  Match the semantic prefix.
_HIDDEN_CSS_MODULE_RE = re.compile(r"^is[-_]?[Hh]idden[-_]")

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
    for el in list(root.find_all(True)):
        if not isinstance(el, Tag):
            continue
        # Guard against decomposed elements (parent set to None during iteration)
        if el.parent is None:
            continue
        if not _is_hidden(el):
            continue
        if _is_math_context(el) or _contains_math(el):
            continue
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
            if _HIDDEN_CSS_MODULE_RE.match(bare):
                return True

    # hidden attribute
    if el.has_attr("hidden"):
        return True

    return False


def _contains_math(el: Tag) -> bool:
    """Check if element contains math descendants.

    Sites like Wikipedia wrap MathML in ``display: none`` spans for
    accessibility (the visible version is an image/SVG fallback); these
    must survive hidden-element removal so math extraction can use them.
    Mirrors defuddle's ``removeHiddenElements`` descendant check.
    """
    if el.find("math") is not None:
        return True
    if el.find(True, attrs={"data-mathml": True}) is not None:
        return True
    return el.find(class_="katex-mathml") is not None


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
