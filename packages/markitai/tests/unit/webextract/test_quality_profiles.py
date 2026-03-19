from __future__ import annotations

"""Tests for typed native extraction quality profiles."""


from markitai.webextract.quality import assess_native_markdown

# --- social_post profile ---

_BAD_X_MARKDOWN_QUOTE_CARD = """\
Alice @alice

This is a great post about something interesting.

Quote
Bob @bob · Apr 1
Some quoted content here

Discover more
"""

_BAD_X_MARKDOWN_TRENDS = """\
Alice @alice

Some tweet content here that is otherwise fine.

Trends for you
"""

_CLEAN_SOCIAL_POST = """\
Alice @alice

Short but valid social post content.
"""

_MINIMAL_SOCIAL = "yes"  # very short social posts are OK


def test_social_post_with_quote_card_leakage_fails_quality() -> None:
    assessment = assess_native_markdown(
        _BAD_X_MARKDOWN_QUOTE_CARD, profile="social_post"
    )
    assert assessment.accepted is False
    assert "quote_card_leakage" in assessment.reasons


def test_social_post_with_trends_sidebar_fails_quality() -> None:
    assessment = assess_native_markdown(_BAD_X_MARKDOWN_TRENDS, profile="social_post")
    assert assessment.accepted is False
    assert "sidebar_leakage" in assessment.reasons


def test_social_post_with_discover_more_fails_quality() -> None:
    bad = "Some tweet content\n\nDiscover more\n"
    assessment = assess_native_markdown(bad, profile="social_post")
    assert assessment.accepted is False
    assert "recommendation_noise" in assessment.reasons


def test_clean_social_post_passes_quality() -> None:
    assessment = assess_native_markdown(_CLEAN_SOCIAL_POST, profile="social_post")
    assert assessment.accepted is True


def test_minimal_social_post_passes_quality() -> None:
    """Social posts can be very short — any non-empty text without noise passes."""
    assessment = assess_native_markdown(_MINIMAL_SOCIAL, profile="social_post")
    assert assessment.accepted is True


def test_empty_social_post_fails_quality() -> None:
    assessment = assess_native_markdown("", profile="social_post")
    assert assessment.accepted is False


# --- conversation_thread profile ---

_CLEAN_THREAD_MARKDOWN = """\
# Discussion

**Alice** said:

This is the first meaningful comment in the thread.

**Bob** replied:

I agree with Alice's point about the topic at hand.
"""

_BAD_THREAD_WITH_DISCOVER = """\
Some thread content here.

Discover more
"""

_BAD_THREAD_WITH_TRENDS = """\
Some thread content here.

Trends for you
"""


def test_clean_thread_markdown_passes_quality() -> None:
    assessment = assess_native_markdown(
        _CLEAN_THREAD_MARKDOWN, profile="conversation_thread"
    )
    assert assessment.accepted is True


def test_thread_with_discover_more_fails_quality() -> None:
    assessment = assess_native_markdown(
        _BAD_THREAD_WITH_DISCOVER, profile="conversation_thread"
    )
    assert assessment.accepted is False
    assert "recommendation_noise" in assessment.reasons


def test_thread_with_trends_fails_quality() -> None:
    assessment = assess_native_markdown(
        _BAD_THREAD_WITH_TRENDS, profile="conversation_thread"
    )
    assert assessment.accepted is False
    assert "sidebar_leakage" in assessment.reasons


def test_thread_too_short_fails_quality() -> None:
    assessment = assess_native_markdown("hi", profile="conversation_thread")
    assert assessment.accepted is False


# --- discussion_issue profile ---

_BAD_ISSUE_WITH_ASSIGNEES = """\
## Bug: Something is broken

Steps to reproduce the bug here.

Assignees
alice
bob

Labels
bug
help wanted
"""

_BAD_ISSUE_WITH_LABELS = """\
## Feature request

Please add this feature.

Labels
enhancement
"""

_CLEAN_ISSUE_MARKDOWN = """\
## Bug: Something is broken

Steps to reproduce:

1. Do first thing
2. Do second thing
3. See error

Expected behavior: no error.
"""


def test_issue_with_assignees_sidebar_fails_quality() -> None:
    assessment = assess_native_markdown(
        _BAD_ISSUE_WITH_ASSIGNEES, profile="discussion_issue"
    )
    assert assessment.accepted is False
    assert "sidebar_leakage" in assessment.reasons


def test_issue_with_labels_sidebar_fails_quality() -> None:
    assessment = assess_native_markdown(
        _BAD_ISSUE_WITH_LABELS, profile="discussion_issue"
    )
    assert assessment.accepted is False
    assert "sidebar_leakage" in assessment.reasons


def test_clean_issue_markdown_passes_quality() -> None:
    assessment = assess_native_markdown(
        _CLEAN_ISSUE_MARKDOWN, profile="discussion_issue"
    )
    assert assessment.accepted is True


# --- generic_article profile ---

_CLEAN_ARTICLE = """\
# Introduction to Python

Python is a high-level, general-purpose programming language. Its design philosophy
emphasises code readability. Python is dynamically typed and garbage-collected.
"""

_TOO_SHORT_ARTICLE = "hi"

_BARE_MARKDOWN = "# Title"


def test_clean_article_passes_quality() -> None:
    assessment = assess_native_markdown(_CLEAN_ARTICLE, profile="generic_article")
    assert assessment.accepted is True


def test_article_too_short_fails_quality() -> None:
    assessment = assess_native_markdown(_TOO_SHORT_ARTICLE, profile="generic_article")
    assert assessment.accepted is False


def test_article_bare_markdown_fails_quality() -> None:
    """Markdown that is only punctuation/symbols with no real words fails."""
    assessment = assess_native_markdown(_BARE_MARKDOWN, profile="generic_article")
    assert assessment.accepted is False


def test_empty_string_fails_all_profiles() -> None:
    for profile in (
        "generic_article",
        "social_post",
        "conversation_thread",
        "discussion_issue",
    ):
        assessment = assess_native_markdown("", profile=profile)
        assert assessment.accepted is False, f"Expected failure for profile={profile}"


def test_unknown_profile_uses_generic_article_behaviour() -> None:
    """Unknown profiles fall back to generic_article heuristics."""
    assessment = assess_native_markdown(
        _CLEAN_ARTICLE, profile="totally_unknown_profile"
    )
    assert assessment.accepted is True


# --- assess_native_markdown return type ---


def test_assessment_has_score_field() -> None:
    assessment = assess_native_markdown(_CLEAN_ARTICLE)
    assert isinstance(assessment.score, float)
    assert 0.0 <= assessment.score <= 1.0


def test_assessment_has_reasons_list() -> None:
    assessment = assess_native_markdown(
        _BAD_X_MARKDOWN_QUOTE_CARD, profile="social_post"
    )
    assert isinstance(assessment.reasons, list)
    assert len(assessment.reasons) > 0


def test_default_profile_is_generic_article() -> None:
    """Calling without profile= uses generic_article semantics."""
    assessment = assess_native_markdown(_TOO_SHORT_ARTICLE)
    assert assessment.accepted is False
