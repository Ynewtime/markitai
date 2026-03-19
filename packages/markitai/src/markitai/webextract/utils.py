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
