"""Score and remove non-content blocks from the DOM."""

from __future__ import annotations

import re

from bs4 import Tag

from markitai.webextract.constants import (
    CONTENT_INDICATOR_TOKENS,
    CONTENT_PROTECTION_SELECTORS,
    NAVIGATION_INDICATORS,
    NON_CONTENT_CLASS_PATTERNS,
)


def score_and_remove(root: Tag) -> int:
    """Score direct child blocks and remove those scoring below zero.

    Walks the direct children of *root*, scoring each block element.
    Blocks with negative scores are considered non-content and removed.

    Args:
        root: Content root element.

    Returns:
        Number of elements removed.
    """
    to_remove: list[Tag] = []
    for child in list(root.children):
        if not isinstance(child, Tag):
            continue
        if child.name not in _BLOCK_TAGS:
            continue
        if _is_likely_content(child):
            continue
        score = _score_non_content_block(child)
        if score < 0:
            to_remove.append(child)

    for el in to_remove:
        el.decompose()
    return len(to_remove)


_BLOCK_TAGS = frozenset(
    {
        "div",
        "section",
        "aside",
        "header",
        "footer",
        "nav",
        "ul",
        "ol",
        "dl",
        "figure",
        "details",
    }
)


def _is_likely_content(el: Tag) -> bool:
    """Whitelist check: return True if element likely contains real content."""
    # Check ARIA roles
    role = str(el.get("role") or "").lower()
    if role in ("article", "main", "contentinfo"):
        return True

    # Check class/id for content indicators (whole-token match)
    class_tokens: set[str] = set()
    classes_raw = el.get("class")
    for cls in classes_raw if isinstance(classes_raw, list) else []:
        # Split "related-posts" into {"related", "posts"}
        for part in re.split(r"[-_]", str(cls).lower()):
            class_tokens.add(part)
    el_id = str(el.get("id") or "")
    for part in re.split(r"[-_]", el_id.lower()):
        class_tokens.add(part)
    if class_tokens & CONTENT_INDICATOR_TOKENS:
        return True

    # Contains protected elements (code, table, math, etc.)
    for selector in CONTENT_PROTECTION_SELECTORS:
        try:
            if el.select_one(selector):
                return True
        except Exception:  # noqa: BLE001
            continue

    text = el.get_text(" ", strip=True)
    words = len(text.split())

    # Count block elements
    paragraphs = len(el.find_all("p"))
    list_items = len(el.find_all("li"))
    block_elements = paragraphs + list_items

    # Check link density before applying word-count thresholds
    link_text_len = sum(len(a.get_text(strip=True)) for a in el.find_all("a"))
    total_text_len = len(text)
    link_density = link_text_len / max(total_text_len, 1)

    # Substantial content thresholds (only if not link-heavy)
    if link_density < 0.5:
        if words >= 50 and block_elements >= 2:
            return True
        if words >= 100:
            return True
        if words >= 30 and block_elements >= 1:
            return True

    # Short text with sentence structure and low link density
    if words >= 10 and _has_sentence_ending(text):
        if total_text_len > 0 and (link_text_len / total_text_len) < 0.1:
            return True

    return False


def _score_non_content_block(el: Tag) -> float:
    """Score a block element. Negative = likely non-content."""
    text = el.get_text(" ", strip=True)
    words = text.split()
    word_count = len(words)

    if word_count < 3:
        return -1.0

    score = 0.0

    # Positive signals
    score += text.count(",")  # commas indicate prose

    # Negative signals: navigation indicators
    text_lower = text.lower()
    for indicator in NAVIGATION_INDICATORS:
        if indicator in text_lower:
            score -= 10

    # Link density
    link_text_len = sum(len(a.get_text(strip=True)) for a in el.find_all("a"))
    total_text_len = len(text)
    if total_text_len > 0:
        link_density = link_text_len / total_text_len
        if link_density > 0.5:
            score -= 15

    # High link ratio in short text
    link_count = len(el.find_all("a"))
    if word_count < 80 and link_count > 1 and total_text_len > 0:
        if link_text_len / total_text_len > 0.8:
            score -= 15

    # Non-content class patterns (substring match is intentional here —
    # patterns like "ad-" should match "ad-container")
    raw_classes = el.get("class")
    classes_str = " ".join(raw_classes).lower() if isinstance(raw_classes, list) else ""
    el_id_str = str(el.get("id") or "").lower()
    combined = f"{classes_str} {el_id_str}"
    for pattern in NON_CONTENT_CLASS_PATTERNS:
        if pattern in combined:
            score -= 8

    # Card grid detection: 3+ headings + 2+ images + sparse prose
    headings = len(el.find_all(["h2", "h3", "h4"]))
    images = len(el.find_all("img"))
    if headings >= 3 and images >= 2 and word_count < 500:
        prose_per_heading = word_count / max(headings, 1)
        if prose_per_heading < 20:
            score -= 15

    return score


def _has_sentence_ending(text: str) -> bool:
    """Check if text contains sentence-ending punctuation."""
    return any(c in text for c in ".!?。！？")
