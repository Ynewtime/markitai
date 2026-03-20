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
_SOCIAL_TITLE_LINE_PATTERN = re.compile(r"(?i)^post by\s+(.+)$")
_SOCIAL_HANDLE_LINE_PATTERN = re.compile(r"^@[\w.]+$")
_SOCIAL_TIMESTAMP_LINE_PATTERN = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[tT ][0-9:.\-+Z]+|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b.+|"
    r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm).*)$"
)

# GitHub Issues / discussion sidebar patterns
_ASSIGNEES_PATTERN = re.compile(r"(?m)^Assignees\s*$")
_LABELS_PATTERN = re.compile(r"(?m)^Labels\s*$")
_CJK_CHAR_PATTERN = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)


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


def _contains_cjk(text: str) -> bool:
    """Return whether text contains CJK characters."""
    return bool(_CJK_CHAR_PATTERN.search(text))


def _normalized_markdown_lines(text: str) -> list[str]:
    """Return non-empty markdown lines with punctuation stripped."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        normalized = re.sub(r"[#*_\[\]()>`!]", "", raw_line)
        normalized = " ".join(normalized.split())
        if normalized:
            lines.append(normalized)
    return lines


def _extract_social_body_text(markdown: str) -> str:
    """Return social-post text after stripping synthesized metadata lines."""
    lines = _normalized_markdown_lines(markdown)
    body_lines: list[str] = []
    title_author: str | None = None

    for line in lines:
        title_match = _SOCIAL_TITLE_LINE_PATTERN.match(line)
        if title_match is not None:
            title_author = title_match.group(1).strip() or None
            continue

        if title_author is not None and line == title_author:
            continue

        if _SOCIAL_HANDLE_LINE_PATTERN.match(line):
            continue

        if _SOCIAL_TIMESTAMP_LINE_PATTERN.match(line):
            continue

        body_lines.append(line)

    return " ".join(body_lines)


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
    elif not _extract_social_body_text(markdown):
        reasons.append("missing_body")
        score -= 0.6

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
    dense_plain = re.sub(r"\s+", "", plain)

    if len(plain) < 10:
        reasons.append("too_short")
        score -= 0.6

    has_dense_cjk_text = _contains_cjk(plain) and len(dense_plain) >= 10

    if _word_count(plain) < 3 and not has_dense_cjk_text:
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
