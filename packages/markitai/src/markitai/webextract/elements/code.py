from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

# Language extraction patterns (ported from defuddle elements/code.ts)
_LANG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"language-(\w+)", re.IGNORECASE),
    re.compile(r"lang-(\w+)", re.IGNORECASE),
    re.compile(r"highlight-(\w+)", re.IGNORECASE),
    re.compile(r"syntax-(\w+)", re.IGNORECASE),
    re.compile(r"(\w+)-code\b", re.IGNORECASE),
    re.compile(r"code-(\w+)", re.IGNORECASE),
    re.compile(r"code-snippet__(\w+)", re.IGNORECASE),
    re.compile(r"(\w+)-snippet\b", re.IGNORECASE),
]


def normalize_code_blocks(root: Tag) -> None:
    """Normalize code blocks: wrap in pre, detect and normalize language class.

    Args:
        root: Content root.
    """

    for code in list(root.find_all("code")):
        raw_classes = code.get("class")
        classes: list[str] = list(raw_classes) if isinstance(raw_classes, list) else []
        style = str(code.get("style", ""))

        # Wrap in <pre> if inline code with pre-style
        if "white-space: pre" in style and code.parent and code.parent.name != "pre":
            pre = BeautifulSoup("", "html.parser").new_tag("pre")
            code.wrap(pre)

        # Detect language from code or parent element
        lang = _detect_language(code) or _detect_language_from_parent(code)
        if lang and not any(c.startswith("language-") for c in classes):
            classes = [f"language-{lang}"] + [
                c for c in classes if not _is_lang_class(c)
            ]

        if classes:
            code["class"] = classes  # type: ignore[assignment]


def _detect_language(el: Tag) -> str | None:
    """Extract language from element's class or data attributes."""
    # Check data attributes first
    for attr in ("data-lang", "data-language", "language"):
        val = el.get(attr)
        if val and isinstance(val, str):
            return val.strip().lower()

    # Check class patterns
    raw_classes = el.get("class")
    if isinstance(raw_classes, list):
        for cls in raw_classes:
            for pattern in _LANG_PATTERNS:
                match = pattern.search(cls)
                if match:
                    return match.group(1).lower()
    return None


def _detect_language_from_parent(code: Tag) -> str | None:
    """Check parent <pre> or wrapper for language hints."""
    parent = code.parent
    if parent is None:
        return None
    # Check parent <pre> or <div> with highlight/syntax classes
    for ancestor in [parent, parent.parent]:
        if ancestor is None or not isinstance(ancestor, Tag):
            continue
        lang = _detect_language(ancestor)
        if lang:
            return lang
    return None


def _is_lang_class(cls: str) -> bool:
    """Check if a class name is a language indicator."""
    return any(p.search(cls) for p in _LANG_PATTERNS)
