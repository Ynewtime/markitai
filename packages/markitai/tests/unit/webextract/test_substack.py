"""Substack article extraction, safe preload parsing, and fallback regressions."""

from __future__ import annotations

import json

import pytest

from markitai.webextract import extract_web_content
from markitai.webextract.dom import parse_html
from markitai.webextract.extractors.registry import find_extractor
from markitai.webextract.extractors.substack_note import SubstackNoteExtractor
from markitai.webextract.types import ContentProfile

_URL = "https://writer.substack.com/p/article"
_BODY = "<p>Article body with <strong>important details</strong> and Unicode 中文.</p>"


def _preload(post: object, *, feed: bool = True, encoded: bool = True) -> str:
    data = {"feedData": {"initialPost": {"post": post}}} if feed else {"post": post}
    payload = json.dumps(data).replace("<", r"\u003c")
    expression = f"JSON.parse({json.dumps(payload)})" if encoded else payload
    return f"<script>window._preloads = {expression}; window.after = true;</script>"


@pytest.mark.parametrize("feed", [True, False])
@pytest.mark.parametrize("encoded", [True, False])
@pytest.mark.parametrize("url", [_URL, "https://newsletter.example/p/article"])
def test_preload_article_body_and_metadata(feed: bool, encoded: bool, url: str) -> None:
    html = (
        _preload(
            {
                "body_html": _BODY,
                "title": "Preloaded article",
                "subtitle": "Article summary",
                "post_date": "2026-02-24T12:34:56Z",
                "publishedBylines": [{"name": "Test Author"}],
            },
            feed=feed,
            encoded=encoded,
        )
        + '<div class="byline-wrapper">Other Author Feb 25, 2026</div><p>Shell noise</p>'
    )
    result = extract_web_content(html, url)
    assert "**important details**" in result.markdown
    assert "中文" in result.markdown
    assert "Shell noise" not in result.markdown
    assert result.metadata.title == "Preloaded article"
    assert result.metadata.description == "Article summary"
    assert result.metadata.author == "Test Author"
    assert result.metadata.published == "2026-02-24T12:34:56Z"
    assert result.metadata.site == "Substack"
    assert result.info is not None
    assert result.info.content_profile == ContentProfile.GENERIC_ARTICLE


def test_rendered_body_wins_and_preload_supplies_metadata() -> None:
    html = _preload({"body_html": "<p>Stale body</p>", "title": "Preload title"})
    html += f'<div class="body markup">{_BODY}</div>'
    result = extract_web_content(html, _URL)
    assert "important details" in result.markdown
    assert "Stale body" not in result.markdown
    assert result.metadata.title == "Preload title"


@pytest.mark.parametrize(
    ("byline", "published"),
    [
        ("Test Author Feb 24, 2026", "2026-02-24T00:00:00+00:00"),
        ("Test AuthorFeb 4 2026", "2026-02-04T00:00:00+00:00"),
        ("<span>Author</span><span>Dec 1, 2025</span>", "2025-12-01T00:00:00+00:00"),
        ("Test Author Feb 29, 2024", "2024-02-29T00:00:00+00:00"),
        ("Test Author Feb 29, 2025", None),
        ("Test Author Feb 31, 2026", None),
        ("Test Author Yesterday", None),
    ],
)
def test_rendered_article_byline_date(byline: str, published: str | None) -> None:
    html = (
        '<meta property="og:title" content="Rendered title">'
        '<meta property="og:description" content="Rendered summary">'
        '<a href="https://substack.com/@writer">Test Author</a>'
        f'<div class="post-byline-wrapper">{byline}</div>'
        f'<div class="body markup">{_BODY}</div>'
    )
    result = extract_web_content(html, _URL)
    assert result.metadata.published == published
    assert result.metadata.author == "Test Author"
    assert result.metadata.title == "Rendered title"
    assert result.metadata.description == "Rendered summary"


@pytest.mark.parametrize(
    "payload",
    [
        'window._preloads = JSON.parse("broken");',
        'window._preloads = JSON.parse("\\q");',
        'window._preloads = JSON.parse("null");',
        'window._preloads = {"feedData": []};',
        'window._preloads = {"feedData": {"initialPost": null}};',
        'window._preloads = {"post": {"body_html": 123, "title": []}};',
        'window._preloads = {"post": {"body_html": "   "}};',
        "window._preloads = runArbitraryCode();",
    ],
)
def test_bad_preloads_preserve_generic_fallback(payload: str) -> None:
    html = (
        '<meta name="author" content="Generic Author">'
        '<meta property="article:published_time" content="2025-01-01">'
        f"<script>{payload}</script><article>{_BODY}</article>"
    )
    result = extract_web_content(html, _URL)
    assert "important details" in result.markdown
    assert result.metadata.author == "Generic Author"
    assert result.metadata.published == "2025-01-01"


def test_bad_script_does_not_hide_later_preload_and_escapes_survive() -> None:
    body = '<p>A quoted "word", a backslash \\ and an &amp; entity.</p>'
    html = '<script>window._preloads = JSON.parse("broken");</script>'
    html += _preload({"body_html": body})
    resolved = SubstackNoteExtractor().resolve(parse_html(html), _URL)
    assert resolved.content_html == body


@pytest.mark.parametrize(
    "first_post", [{}, {"title": "Shell title"}, {"body_html": 123}]
)
def test_incomplete_preload_does_not_hide_later_article(first_post: object) -> None:
    html = _preload(first_post) + _preload({"body_html": _BODY, "title": "Article"})
    result = extract_web_content(html, _URL)
    assert "important details" in result.markdown
    assert result.metadata.title == "Article"


def test_incomplete_feed_envelope_falls_back_to_publication_post() -> None:
    data = {
        "feedData": {"initialPost": {}},
        "post": {"body_html": _BODY, "title": "Publication article"},
    }
    html = f"<script>window._preloads = {json.dumps(data)};</script>"
    result = extract_web_content(html, _URL)
    assert "important details" in result.markdown
    assert result.metadata.title == "Publication article"


def test_preload_body_uses_shared_sanitization_and_link_resolution() -> None:
    html = _preload(
        {
            "body_html": '<p>Article <a href="/p/related">related</a></p>'
            '<script>alert("unsafe")</script>'
            '<p onclick="alert(1)">Safe paragraph</p>'
        }
    )
    result = extract_web_content(html, _URL)
    assert "[related](https://writer.substack.com/p/related)" in result.markdown
    assert "Safe paragraph" in result.markdown
    assert "<script" not in result.clean_html
    assert "onclick" not in result.clean_html
    assert "unsafe" not in result.markdown


def test_rendered_article_survives_malformed_preload() -> None:
    html = '<script>window._preloads = JSON.parse("broken");</script>'
    html += '<div class="byline-wrapper">Author Mar 8, 2026</div>'
    html += f'<div class="body markup">{_BODY}</div>'
    result = extract_web_content(html, _URL)
    assert "important details" in result.markdown
    assert result.metadata.published == "2026-03-08T00:00:00+00:00"


def test_unrelated_json_parse_is_not_used() -> None:
    unrelated = json.dumps(json.dumps({"post": {"body_html": "<p>Wrong</p>"}}))
    html = f"<script>const other = JSON.parse({unrelated});</script>"
    html += _preload({"body_html": _BODY})
    assert SubstackNoteExtractor().resolve(parse_html(html), _URL).content_html == _BODY


def test_missing_post_metadata_preserves_generic_values() -> None:
    html = (
        '<meta name="author" content="Generic Author">'
        '<meta property="article:published_time" content="2025-01-01">'
        + _preload({"body_html": _BODY, "publishedBylines": [None], "title": 123})
    )
    result = extract_web_content(html, _URL)
    assert result.metadata.author == "Generic Author"
    assert result.metadata.published == "2025-01-01"


def test_custom_domain_rendered_article_requires_platform_signal() -> None:
    html = f'<div class="body markup">{_BODY}</div>'
    url = "https://newsletter.example/p/article"
    assert find_extractor(url, parse_html(html)) is None
    html += '<link rel="stylesheet" href="https://substackcdn.com/bundle/style.css">'
    assert isinstance(find_extractor(url, parse_html(html)), SubstackNoteExtractor)
    result = extract_web_content(html, url)
    assert result.info is not None
    assert result.info.extractor_name == "substack_note"
    assert "important details" in result.markdown


@pytest.mark.parametrize(
    "url",
    [
        "https://substack.com.evil.example/p/article",
        "https://notsubstack.com/p/article",
        "https://example.com/?url=https://substack.com/@user/note/1",
        "https://substack.com@evil.example/p/article",
        "ftp://writer.substack.com/p/article",
    ],
)
def test_url_matching_rejects_lookalikes(url: str) -> None:
    assert not SubstackNoteExtractor().matches_url(url)


def test_note_permalink_keeps_main_note_image_author_and_profile() -> None:
    html = (
        '<meta property="og:title" content="Test User (@testuser)">'
        '<meta property="og:image" content="https://example.com/photo.jpg">'
        '<div class="ProseMirror FeedProseMirror"><p>Recommendation</p></div>'
        '<div class="feedPermalinkUnit-main">'
        '<div class="ProseMirror FeedProseMirror"><p>The main note text.</p></div>'
        '</div><script>window._preloads = JSON.parse("broken");</script>'
    )
    result = extract_web_content(html, "https://substack.com/@testuser/note/c-123")
    assert "The main note text." in result.markdown
    assert "Recommendation" not in result.markdown
    assert "https://example.com/photo.jpg" in result.markdown
    assert result.metadata.author == "Test User"
    assert result.info is not None
    assert result.info.content_profile == ContentProfile.SOCIAL_POST
