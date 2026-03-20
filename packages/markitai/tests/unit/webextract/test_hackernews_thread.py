"""Tests for Hacker News thread extraction using shared semantic types.

These tests validate that HackerNewsThreadExtractor produces the same semantic
model structure as XTweetExtractor and GitHubThreadExtractor, proving the
shared ConversationThread abstraction spans multiple threaded platforms.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"
HN_URL = "https://news.ycombinator.com/item?id=99887766"


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


def test_hackernews_thread_reuses_thread_semantics_for_comments() -> None:
    """HN thread page must produce a ConversationThread via the shared model.

    Verifies:
    - semantic.thread is populated (not None)
    - metadata.site is set to "Hacker News"
    - The rendered markdown includes a "## Comments" heading for replies
    """
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("hackernews_thread.playwright.html")
    result = extract_web_content(html, HN_URL)

    assert result.semantic is not None
    assert result.semantic.thread is not None
    assert result.metadata.site == "Hacker News"
    assert "## Comments" in result.markdown


def test_hackernews_thread_title_reflects_story_title() -> None:
    """The thread title must match the story title from the page."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("hackernews_thread.playwright.html")
    result = extract_web_content(html, HN_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    assert "productive" in thread.title or "working from home" in thread.title


def test_hackernews_thread_collects_comments_as_items() -> None:
    """Thread must collect comment entries as ConversationItems in items list."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("hackernews_thread.playwright.html")
    result = extract_web_content(html, HN_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    assert len(thread.items) >= 2


def test_hackernews_thread_comment_authors_are_extracted() -> None:
    """Comment authors must be extracted as author_name on ConversationItems."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("hackernews_thread.playwright.html")
    result = extract_web_content(html, HN_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    # At least one comment should have an author
    authors = [item.author_name for item in thread.items if item.author_name]
    assert len(authors) >= 1


def test_hackernews_thread_content_profile_is_discussion_thread() -> None:
    """HN thread page must be classified as DISCUSSION_THREAD content profile."""
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile

    html = _load_fixture("hackernews_thread.playwright.html")
    result = extract_web_content(html, HN_URL)

    assert result.info is not None
    assert result.info.content_profile == ContentProfile.DISCUSSION_THREAD


def test_hackernews_thread_main_item_contains_story_text() -> None:
    """The main_item of the thread must contain the original story/question text."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("hackernews_thread.playwright.html")
    result = extract_web_content(html, HN_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    # Story text or title must be present in main_item
    combined = thread.main_item.text + " " + thread.title
    assert "productive" in combined or "working from home" in combined


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_resolves_hn_extractor_for_item_urls() -> None:
    """find_extractor must return an extractor with resolve() for HN item URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor(HN_URL)

    assert extractor is not None
    assert hasattr(extractor, "resolve")
    assert callable(extractor.resolve)


def test_registry_does_not_match_hn_for_non_item_urls() -> None:
    """find_extractor must NOT return the HN extractor for non-item HN URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    # news.ycombinator.com front page — not a thread
    extractor = find_extractor("https://news.ycombinator.com/")

    # Should not match HN thread extractor (may return None or a different extractor)
    if extractor is not None:
        assert extractor.name != "hackernews_thread"
