"""Tests for Reddit post thread extraction using shared semantic types.

These tests validate that RedditPostExtractor produces the same semantic
model structure as XTweetExtractor and GitHubThreadExtractor, proving the
shared ConversationThread abstraction spans multiple threaded platforms.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"
REDDIT_URL = (
    "https://old.reddit.com/r/rust/comments/abc123/whats_the_best_way_to_learn_rust/"
)


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


def test_reddit_post_reuses_thread_semantics_for_post_and_comments() -> None:
    """Reddit post page must produce a ConversationThread via the shared model.

    Verifies:
    - semantic.thread is populated (not None)
    - metadata.site is set to "Reddit"
    - The rendered markdown includes a "## Comments" heading for replies
    """
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("reddit_post.playwright.html")
    result = extract_web_content(html, REDDIT_URL)

    assert result.semantic is not None
    assert result.semantic.thread is not None
    assert result.metadata.site == "Reddit"
    assert "## Comments" in result.markdown


def test_reddit_post_main_item_contains_post_body() -> None:
    """The main_item of the thread must contain the original post text."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("reddit_post.playwright.html")
    result = extract_web_content(html, REDDIT_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    assert (
        "borrow checker" in thread.main_item.text or "Python" in thread.main_item.text
    )


def test_reddit_post_thread_title_reflects_post_title() -> None:
    """The thread title must match the post title from the page."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("reddit_post.playwright.html")
    result = extract_web_content(html, REDDIT_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    assert "Rust" in thread.title


def test_reddit_post_collects_comments_as_items() -> None:
    """Thread must collect comment entries as ConversationItems in items list."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("reddit_post.playwright.html")
    result = extract_web_content(html, REDDIT_URL)

    assert result.semantic is not None
    thread = result.semantic.thread
    assert thread is not None
    assert len(thread.items) >= 2


def test_reddit_post_excludes_sidebar_content() -> None:
    """Reddit post extraction must exclude sidebar content from markdown."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("reddit_post.playwright.html")
    result = extract_web_content(html, REDDIT_URL)

    # The sidebar has "Submit a new link" which must not appear in the thread
    assert "Submit a new link" not in result.markdown


def test_reddit_post_content_profile_is_discussion_thread() -> None:
    """Reddit post page must be classified as DISCUSSION_THREAD content profile."""
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile

    html = _load_fixture("reddit_post.playwright.html")
    result = extract_web_content(html, REDDIT_URL)

    assert result.info is not None
    assert result.info.content_profile == ContentProfile.DISCUSSION_THREAD


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_resolves_reddit_extractor_for_post_urls() -> None:
    """find_extractor must return an extractor with resolve() for Reddit post URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor(REDDIT_URL)

    assert extractor is not None
    assert hasattr(extractor, "resolve")
    assert callable(extractor.resolve)


def test_registry_resolves_reddit_extractor_for_comments_urls() -> None:
    """find_extractor must match reddit.com URLs containing /comments/."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://www.reddit.com/r/Python/comments/xyz/topic/")

    assert extractor is not None
    assert hasattr(extractor, "resolve")
