"""Semantic parity tests: freeze expected extraction behaviour.

These tests validate that the pipeline correctly classifies pages, populates
semantic models, and excludes noise sections from markdown output.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"

X_URL = "https://x.com/ixiaowenz/status/2030105637204676808"
ARTICLE_URL = "https://example.com/blog/async-python"


def _load_fixture(name: str) -> str:
    """Load an HTML fixture file by name.

    Args:
        name: Fixture filename relative to the web fixtures directory.

    Returns:
        Raw HTML string.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# X/Twitter status page
# ---------------------------------------------------------------------------


def test_x_status_semantics_match_expected_fixture() -> None:
    """X status page must be classified as SOCIAL_POST with a populated thread.

    Verifies:
    - metadata.title reflects the post author
    - info.content_profile is ContentProfile.SOCIAL_POST
    - semantic.thread is not None and carries the correct author handle
    - Noise sections ("Discover more", quote cards) are absent from markdown
    """
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile

    html = _load_fixture("x_status_2030105637204676808.playwright.html")
    result = extract_web_content(html, X_URL)

    assert result.metadata.title == "Post by @ixiaowenz"

    assert result.info is not None
    assert result.info.content_profile == ContentProfile.SOCIAL_POST
    assert result.semantic is not None
    assert result.semantic.thread is not None
    assert result.semantic.thread.main_item.author_handle == "@ixiaowenz"

    assert "Discover more" not in result.markdown
    assert "Quote" not in result.markdown


def test_x_resolver_returns_thread_semantic_model() -> None:
    """X resolver must populate a ConversationThread with correct author info."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("x_status_2030105637204676808.playwright.html")
    result = extract_web_content(html, X_URL)

    assert result.semantic is not None
    assert result.semantic.thread is not None
    assert result.semantic.thread.main_item.author_name == "\u5c0f\u6587"
    assert result.semantic.thread.main_item.author_handle == "@ixiaowenz"
    assert "AI agents" in result.semantic.thread.main_item.text


def test_x_output_excludes_recommendation_sections_and_quote_card_leakage() -> None:
    """X output must not contain recommendation or quote card noise."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("x_status_2030105637204676808.playwright.html")
    result = extract_web_content(html, X_URL)

    assert "Discover more" not in result.markdown
    assert "\nQuote\n" not in result.markdown
    assert "Trends for you" not in result.markdown


def test_x_status_preserves_spaces_between_inline_text_fragments() -> None:
    """Inline spans and links in tweet text must preserve readable spacing."""
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html><body>
    <div data-testid="primaryColumn">
      <article data-testid="tweet">
        <div data-testid="User-Name">
          <a href="/alice"><span>Alice</span><span>@alice</span></a>
        </div>
        <div data-testid="tweetText">
          <span>Hello </span><span>world</span><span> and </span>
          <a href="https://example.com"><span>friends</span></a>
        </div>
        <time datetime="2026-03-19T00:00:00Z"></time>
      </article>
    </div>
    </body></html>
    """

    result = extract_web_content(html, "https://x.com/alice/status/123")

    assert "Hello world and friends" in result.markdown


# ---------------------------------------------------------------------------
# Generic article page
# ---------------------------------------------------------------------------


def test_generic_article_baseline_stays_non_threaded_and_clean() -> None:
    """Generic article pages must not gain a semantic thread object.

    Verifies:
    - info.content_profile is ContentProfile.GENERIC_ARTICLE
    - semantic is None (no structured thread model)
    - Sidebar noise ("Related posts") is absent from markdown
    """
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile

    html = _load_fixture("generic_article.playwright.html")
    result = extract_web_content(html, ARTICLE_URL)

    assert result.info is not None
    assert result.info.content_profile == ContentProfile.GENERIC_ARTICLE
    assert result.semantic is None

    assert "Related posts" not in result.markdown
