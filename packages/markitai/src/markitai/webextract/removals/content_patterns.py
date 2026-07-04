"""Remove content patterns: read-time, bylines, boilerplate."""

from __future__ import annotations

import re

from bs4 import Tag

_READ_TIME_RE = re.compile(r"\d+\s*min(?:ute)?s?\s+read\b", re.IGNORECASE)

_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^This (?:article|story|piece) (?:appeared|was published|originally appeared) in\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^A version of this (?:article|story) (?:appeared|was published) in\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^Originally (?:published|appeared) (?:in|on|at)\b",
        re.IGNORECASE,
    ),
]

_BYLINE_DATE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}",
    re.IGNORECASE,
)

_BYLINE_BY_RE = re.compile(r"\bBy\s+[A-Z]")

_MAX_BYLINE_WORDS = 15


def _remove_hero_headers(root: Tag) -> int:
    """Remove hero headers (title + date blocks) at the start of content.

    Targets blocks within the first few children that contain a heading
    (h1/h2) and a time/date element, with fewer than 30 words total.
    These are typically page banners, not article content.
    """
    removed = 0
    for child in list(root.children)[:5]:
        if not isinstance(child, Tag):
            continue
        if not child.find(["h1", "h2"]):
            continue
        text = child.get_text(strip=True)
        if len(text.split()) > 30:
            continue
        has_time = child.find("time") is not None
        has_date_class = (
            child.find(class_=lambda c: bool(c and "date" in str(c).lower()))
            is not None
        )
        if has_time or has_date_class:
            child.decompose()
            removed += 1
            break
    return removed


_STRUCTURED_CONTENT_TAGS = frozenset(
    {"table", "math", "pre", "code", "svg", "img", "picture", "video"}
)


def _remove_trailing_thin_sections(root: Tag) -> int:
    """Remove thin trailing sections (CTAs, newsletter prompts, etc.).

    Scans backward from the end of content. Removes blocks that have
    a heading but fewer than 25 words (typical of subscription CTAs,
    "Related articles", etc.). Stops at the first substantial block.

    Blocks containing structured content (tables, math, code, media)
    are always considered substantial regardless of word count.
    """
    removed = 0
    children = [c for c in root.children if isinstance(c, Tag)]
    for child in reversed(children):
        # Skip the standardized footnotes container (appended at the end
        # by standardize_footnotes)
        if child.get("id") == "footnotes":
            continue
        text = child.get_text(strip=True)
        word_count = len(text.split())
        if word_count > 25:
            break
        # Protect blocks with structured content (math, tables, code, etc.)
        # Check both the element itself and its descendants
        if child.name in _STRUCTURED_CONTENT_TAGS or child.find(
            _STRUCTURED_CONTENT_TAGS
        ):
            break
        if child.find(["h2", "h3", "h4", "h5", "h6"]):
            child.decompose()
            removed += 1
        else:
            break
    return removed


def remove_content_patterns(root: Tag) -> int:
    """Remove metadata patterns from content.

    Targets:
    - Read-time metadata ("3 min read")
    - Boilerplate sentences ("This article appeared in...")
    - Short author bylines with dates ("By Author · March 15, 2026")

    Args:
        root: Content root element.

    Returns:
        Number of elements removed.
    """
    removed = 0

    for el in list(root.find_all(True)):
        text = el.get_text(strip=True)
        if not text:
            continue

        # Read-time: short element containing "N min read"
        words = text.split()
        if len(words) <= 10 and _READ_TIME_RE.search(text):
            el.decompose()
            removed += 1
            continue

        # Boilerplate sentences
        for pattern in _BOILERPLATE_PATTERNS:
            if pattern.search(text):
                el.decompose()
                removed += 1
                break
        else:
            # Author byline: short text with "By" + date
            if (
                len(words) <= _MAX_BYLINE_WORDS
                and _BYLINE_BY_RE.search(text)
                and _BYLINE_DATE_RE.search(text)
            ):
                el.decompose()
                removed += 1

    removed += _remove_hero_headers(root)
    removed += _remove_trailing_thin_sections(root)

    return removed
