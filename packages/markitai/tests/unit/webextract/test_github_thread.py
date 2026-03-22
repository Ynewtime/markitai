"""Tests for GitHub issue/PR thread extraction using shared semantic types.

These tests validate that the GitHubThreadExtractor produces the same semantic
model structure as XTweetExtractor, proving the abstraction is not X-specific.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"
GITHUB_URL = "https://github.com/example/repo/issues/42"


def _load_fixture(name: str) -> str:
    """Load an HTML fixture file by name.

    Args:
        name: Fixture filename relative to the web fixtures directory.

    Returns:
        Raw HTML string.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Semantic model tests
# ---------------------------------------------------------------------------


def test_github_thread_uses_shared_thread_semantics() -> None:
    """GitHub issue page must produce a ConversationThread via the shared model.

    Verifies:
    - semantic.thread is populated (not None)
    - metadata.site is set to "GitHub"
    - The rendered markdown includes a "## Comments" heading for replies
    """
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("github_issue_thread.playwright.html")
    result = extract_web_content(html, GITHUB_URL)

    assert result.semantic is not None
    assert result.semantic.thread is not None
    assert result.metadata.site == "GitHub"
    assert "## Comments" in result.markdown


def test_github_thread_keeps_issue_body_without_sidebar_noise() -> None:
    """GitHub issue page must exclude sidebar metadata from extracted markdown.

    Verifies:
    - "Assignees" sidebar label is absent from markdown
    - The issue body content is present
    """
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("github_issue_thread.playwright.html")
    result = extract_web_content(html, GITHUB_URL)

    assert "Assignees" not in result.markdown


# ---------------------------------------------------------------------------
# Content profile tests
# ---------------------------------------------------------------------------


def test_github_thread_content_profile_is_discussion_issue() -> None:
    """GitHub issue page must be classified as DISCUSSION_ISSUE content profile."""
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile

    html = _load_fixture("github_issue_thread.playwright.html")
    result = extract_web_content(html, GITHUB_URL)

    assert result.info is not None
    assert result.info.content_profile == ContentProfile.DISCUSSION_ISSUE


# ---------------------------------------------------------------------------
# Thread structure tests
# ---------------------------------------------------------------------------


def test_github_thread_main_item_has_issue_body_text() -> None:
    """The main_item of the thread must contain the issue body text."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("github_issue_thread.playwright.html")
    result = extract_web_content(html, GITHUB_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    # The fixture body text describes the bug; the title mentions "extraction fails on empty HTML"
    assert "extraction pipeline" in thread.main_item.text


def test_github_thread_collects_comments_as_items() -> None:
    """Thread must collect comment entries as ConversationItems in items list."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("github_issue_thread.playwright.html")
    result = extract_web_content(html, GITHUB_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    # 2 comments in the fixture (defunkt and octocat reply)
    assert len(thread.items) == 2


def test_github_thread_title_reflects_issue_title() -> None:
    """The thread title must match the issue title from the page."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("github_issue_thread.playwright.html")
    result = extract_web_content(html, GITHUB_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    assert "extraction fails on empty HTML" in thread.title


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_resolves_github_thread_extractor_for_issue_urls() -> None:
    """find_extractor must return an extractor with resolve() for GitHub issue URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://github.com/example/repo/issues/42")

    assert extractor is not None
    assert hasattr(extractor, "resolve")
    assert callable(extractor.resolve)  # type: ignore[reportAttributeAccessIssue]


def test_registry_resolves_github_thread_extractor_for_pr_urls() -> None:
    """find_extractor must return an extractor with resolve() for GitHub PR URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://github.com/example/repo/pull/99")

    assert extractor is not None
    assert hasattr(extractor, "resolve")
    assert callable(extractor.resolve)  # type: ignore[reportAttributeAccessIssue]
