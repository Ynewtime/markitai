from __future__ import annotations

"""Typed quality profiles for native web extraction assessment.

Each profile encodes domain-specific heuristics for deciding whether the
native markdown produced by the extraction pipeline is acceptably clean.
"""

import re

from markitai.webextract.types import QualityAssessment

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Social / X (Twitter) noise patterns
_QUOTE_CARD_PATTERN = re.compile(r"(?m)^Quote\s*$")
_DISCOVER_MORE_PATTERN = re.compile(r"Discover more")
_TRENDS_PATTERN = re.compile(r"Trends for you")

# GitHub Issues / discussion sidebar patterns
_ASSIGNEES_PATTERN = re.compile(r"(?m)^Assignees\s*$")
_LABELS_PATTERN = re.compile(r"(?m)^Labels\s*$")


def _strip_markdown_punctuation(text: str) -> str:
    """Remove markdown-specific punctuation and return plain text.

    Args:
        text: Markdown string.

    Returns:
        Plain text with markdown symbols stripped.
    """
    text = re.sub(r"[#*_\[\]()>`\-|!]", "", text)
    return " ".join(text.split())


def _word_count(text: str) -> int:
    """Count whitespace-delimited words in *text*."""
    return len(text.split()) if text.strip() else 0


# ---------------------------------------------------------------------------
# Profile implementations
# ---------------------------------------------------------------------------


def _assess_social_post(markdown: str) -> QualityAssessment:
    """Quality assessment for social posts (X/Twitter, Mastodon, etc.).

    Social posts can be very short (min 5 words). The key failures are
    structural noise leaking through from site chrome.

    Args:
        markdown: Extracted markdown string.

    Returns:
        QualityAssessment with accept/reject decision and reasons.
    """
    reasons: list[str] = []
    score = 1.0

    if not markdown or not markdown.strip():
        return QualityAssessment(accepted=False, score=0.0, reasons=["empty_content"])

    if _QUOTE_CARD_PATTERN.search(markdown):
        reasons.append("quote_card_leakage")
        score -= 0.4

    if _DISCOVER_MORE_PATTERN.search(markdown):
        reasons.append("recommendation_noise")
        score -= 0.4

    if _TRENDS_PATTERN.search(markdown):
        reasons.append("sidebar_leakage")
        score -= 0.4

    plain = _strip_markdown_punctuation(markdown)
    if not plain:
        # No readable text at all (only markdown punctuation)
        reasons.append("too_short")
        score -= 0.5

    score = max(0.0, min(1.0, score))
    accepted = len(reasons) == 0
    return QualityAssessment(accepted=accepted, score=score, reasons=reasons)


def _assess_conversation_thread(markdown: str) -> QualityAssessment:
    """Quality assessment for conversation threads (X threads, forum threads).

    Args:
        markdown: Extracted markdown string.

    Returns:
        QualityAssessment with accept/reject decision and reasons.
    """
    reasons: list[str] = []
    score = 1.0

    if not markdown or not markdown.strip():
        return QualityAssessment(accepted=False, score=0.0, reasons=["empty_content"])

    if _DISCOVER_MORE_PATTERN.search(markdown):
        reasons.append("recommendation_noise")
        score -= 0.4

    if _TRENDS_PATTERN.search(markdown):
        reasons.append("sidebar_leakage")
        score -= 0.4

    plain = _strip_markdown_punctuation(markdown)
    if _word_count(plain) < 10:
        reasons.append("too_short")
        score -= 0.5

    score = max(0.0, min(1.0, score))
    accepted = len(reasons) == 0
    return QualityAssessment(accepted=accepted, score=score, reasons=reasons)


def _assess_discussion_issue(markdown: str) -> QualityAssessment:
    """Quality assessment for GitHub Issues and similar discussion pages.

    Checks for sidebar element leakage (Assignees, Labels sections) that
    indicates the extraction grabbed chrome rather than content.

    Args:
        markdown: Extracted markdown string.

    Returns:
        QualityAssessment with accept/reject decision and reasons.
    """
    reasons: list[str] = []
    score = 1.0

    if not markdown or not markdown.strip():
        return QualityAssessment(accepted=False, score=0.0, reasons=["empty_content"])

    if _ASSIGNEES_PATTERN.search(markdown) or _LABELS_PATTERN.search(markdown):
        reasons.append("sidebar_leakage")
        score -= 0.5

    plain = _strip_markdown_punctuation(markdown)
    if _word_count(plain) < 10:
        reasons.append("too_short")
        score -= 0.5

    score = max(0.0, min(1.0, score))
    accepted = len(reasons) == 0
    return QualityAssessment(accepted=accepted, score=score, reasons=reasons)


def _assess_generic_article(markdown: str) -> QualityAssessment:
    """Quality assessment for generic web articles.

    Preserves the original is_native_markdown_acceptable() behaviour:
    min 10 chars of plain text. Additionally rejects content that is fewer
    than 3 words (pure markdown punctuation / empty headings).

    Args:
        markdown: Extracted markdown string.

    Returns:
        QualityAssessment with accept/reject decision and reasons.
    """
    reasons: list[str] = []
    score = 1.0

    if not markdown or not markdown.strip():
        return QualityAssessment(accepted=False, score=0.0, reasons=["empty_content"])

    plain = _strip_markdown_punctuation(markdown)

    if len(plain) < 10:
        reasons.append("too_short")
        score -= 0.6

    if _word_count(plain) < 3:
        if "too_short" not in reasons:
            reasons.append("too_short")
        score -= 0.4

    score = max(0.0, min(1.0, score))
    accepted = len(reasons) == 0
    return QualityAssessment(accepted=accepted, score=score, reasons=reasons)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROFILE_MAP: dict[str, object] = {
    "generic_article": _assess_generic_article,
    "social_post": _assess_social_post,
    "conversation_thread": _assess_conversation_thread,
    "discussion_issue": _assess_discussion_issue,
}


def assess_native_markdown(
    markdown: str,
    *,
    profile: str = "generic_article",
) -> QualityAssessment:
    """Assess whether native extraction markdown meets quality criteria.

    Each profile encodes domain-specific heuristics. Unknown profiles fall
    back to the ``generic_article`` profile.

    Args:
        markdown: Markdown text produced by native extraction.
        profile: Quality profile name. One of ``"generic_article"``,
            ``"social_post"``, ``"conversation_thread"``,
            ``"discussion_issue"``. Unknown values fall back to
            ``"generic_article"``.

    Returns:
        A :class:`~markitai.webextract.types.QualityAssessment` describing
        whether the extraction was accepted and why.
    """
    assessor = _PROFILE_MAP.get(profile, _assess_generic_article)
    return assessor(markdown)  # type: ignore[operator]
