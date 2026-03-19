"""Tests for YouTube page extraction using the non-thread resolver pattern.

YouTube pages are rich media pages, not conversation threads. These tests
verify that YouTubePageExtractor uses the RICH_MEDIA_PAGE content profile
and does NOT produce a thread semantic model.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"
YOUTUBE_URL = "https://www.youtube.com/watch?v=KVKufdTphKs"


def _load_fixture(name: str) -> str:
    """Load an HTML fixture file by name.

    Args:
        name: Fixture filename relative to the web fixtures directory.

    Returns:
        Raw HTML string.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Core behaviour tests
# ---------------------------------------------------------------------------


def test_youtube_page_sets_video_metadata_even_without_transcript() -> None:
    """YouTube page must produce metadata with site="YouTube" and rich_media_page profile.

    Verifies:
    - metadata.site is "YouTube"
    - info.content_profile is ContentProfile.RICH_MEDIA_PAGE
    """
    from markitai.webextract.pipeline import extract_web_content
    from markitai.webextract.types import ContentProfile

    html = _load_fixture("youtube_page.playwright.html")
    result = extract_web_content(html, YOUTUBE_URL)

    assert result.metadata.site == "YouTube"
    assert result.info is not None
    assert result.info.content_profile == ContentProfile.RICH_MEDIA_PAGE


def test_youtube_page_uses_non_thread_semantic_path() -> None:
    """YouTube page must NOT produce a ConversationThread semantic model.

    Video pages are rich media pages, not threaded discussions. The semantic
    field should be None or have no thread attached.
    """
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("youtube_page.playwright.html")
    result = extract_web_content(html, YOUTUBE_URL)

    assert result.semantic is None or result.semantic.thread is None


# ---------------------------------------------------------------------------
# Metadata extraction tests
# ---------------------------------------------------------------------------


def test_youtube_page_extracts_video_title() -> None:
    """YouTube page must extract the video title from the page."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("youtube_page.playwright.html")
    result = extract_web_content(html, YOUTUBE_URL)

    assert result.metadata.title is not None
    assert "Python" in result.metadata.title or "GIL" in result.metadata.title


def test_youtube_page_extracts_channel_as_author() -> None:
    """YouTube page must set the channel name as the author metadata field."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("youtube_page.playwright.html")
    result = extract_web_content(html, YOUTUBE_URL)

    assert result.metadata.author is not None
    assert "ArjanCodes" in result.metadata.author


def test_youtube_page_markdown_contains_video_link() -> None:
    """YouTube page markdown must contain a link back to the video."""
    from markitai.webextract.pipeline import extract_web_content

    html = _load_fixture("youtube_page.playwright.html")
    result = extract_web_content(html, YOUTUBE_URL)

    assert "youtube.com" in result.markdown or "KVKufdTphKs" in result.markdown


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_resolves_youtube_extractor_for_watch_urls() -> None:
    """find_extractor must return an extractor with resolve() for YouTube watch URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor(YOUTUBE_URL)

    assert extractor is not None
    assert hasattr(extractor, "resolve")
    assert callable(extractor.resolve)


def test_registry_does_not_match_non_youtube_urls() -> None:
    """find_extractor must not return a YouTubePageExtractor for non-YouTube URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://vimeo.com/123456789")

    # Either no extractor or a different extractor (not youtube_page)
    if extractor is not None:
        assert getattr(extractor, "name", "") != "youtube_page"


def test_registry_matches_youtube_short_url() -> None:
    """find_extractor must match youtu.be short URLs."""
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor("https://youtu.be/KVKufdTphKs")

    assert extractor is not None
    assert getattr(extractor, "name", "") == "youtube_page"
