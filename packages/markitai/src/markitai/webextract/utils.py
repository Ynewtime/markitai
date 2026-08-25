"""Utility functions for webextract."""

from __future__ import annotations

import re

# CJK character ranges (each character counts as one word)
_CJK_RE = re.compile(
    "["
    "\u3040-\u309f"  # Hiragana
    "\u30a0-\u30ff"  # Katakana
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\u3400-\u4dbf"  # CJK Extension A
    "\uac00-\ud7af"  # Hangul Syllables
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\U00020000-\U0002a6df"  # CJK Extension B
    "]"
)


_NORMALIZE_MAP = str.maketrans(
    {
        " ": " ",
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "…": "...",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonicalize text for title/heading comparison.

    Normalizes smart quotes, dashes, ellipses, and whitespace; lowercases.
    Two strings a human would read as "the same" compare equal after this
    pass. Ported from defuddle ``utils.ts`` ``normalizeText``.
    """
    return _WHITESPACE_RE.sub(" ", text.translate(_NORMALIZE_MAP)).strip().lower()


def count_words(text: str) -> int:
    """Count words with CJK awareness.

    CJK characters are counted individually (each character = 1 word).
    Latin/other text is counted by whitespace separation.

    Args:
        text: Input text.

    Returns:
        Word count.
    """
    if not text or not text.strip():
        return 0

    # Count CJK characters
    cjk_chars = _CJK_RE.findall(text)
    cjk_count = len(cjk_chars)

    # Remove CJK characters and non-word chars, count remaining by whitespace
    remaining = _CJK_RE.sub(" ", text)
    # Strip CJK/fullwidth punctuation and other non-alphanumeric residue
    remaining = re.sub(r"[^\w\s]", " ", remaining, flags=re.UNICODE).strip()
    latin_count = len(remaining.split()) if remaining else 0

    return cjk_count + latin_count


# Responsive Tailwind "show" utilities (e.g. "sm:block", "lg:flex") — an
# element carrying one is visible at some breakpoint even when it also has
# a "hidden" class. Ported from defuddle ``utils/dom.ts``
# ``hasResponsiveShowClass``.
_RESPONSIVE_SHOW_RE = re.compile(
    r"^(?:sm|md|lg|xl|2xl|min-\[|max-\[):(?:block|flex|grid|inline|table|contents)"
)


def has_responsive_show_class(class_name: str) -> bool:
    """Check whether a class string re-shows the element at some breakpoint.

    Args:
        class_name: Space-separated class attribute value.

    Returns:
        True if any token is a responsive show utility.
    """
    return any(_RESPONSIVE_SHOW_RE.match(t) for t in class_name.split())
