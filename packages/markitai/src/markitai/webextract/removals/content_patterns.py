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

_BYLINE_BY_RE = re.compile(r"\bBy\s+[A-Z]", re.IGNORECASE)

_MAX_BYLINE_WORDS = 15


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

    return removed
