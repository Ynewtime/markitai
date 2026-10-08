"""Regression tests: site extractors are chosen by hostname, not substring.

``"x.com/" in url`` matched ``dropbox.com/status/123``, the unanchored
article regex matched ``linux.com/news/article/12345``, and
``"github.com" in url`` matched ``docs.github.com/.../issues/...``. The wrong
extractor then picked the content root and the quality thresholds, so
legitimate articles lost content.
"""

from __future__ import annotations

import pytest

from markitai.webextract.extractors.registry import find_extractor


@pytest.mark.parametrize(
    "url",
    [
        "https://www.dropbox.com/status/123",
        "https://example.com/x.com/user/status/123",
        "https://notx.com/user/status/123",
        "https://linux.com/news/article/12345",
        "https://www.linux.com/news/article/12345",
        "https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues",
        "https://example.com/github.com/owner/repo/issues/1",
        "https://example.com/news/app/1/view/2",
        "https://example.com/bilibili.com/opus/123",
        "https://notyoutube.com/watch?v=abc",
        "https://example.com/?next=youtu.be/abc",
        "https://example.com/news.ycombinator.com/item?id=1",
    ],
)
def test_lookalike_urls_get_no_site_extractor(url: str) -> None:
    assert find_extractor(url) is None


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("https://x.com/user/status/123", "x_tweet"),
        ("https://mobile.twitter.com/user/status/123?s=20", "x_tweet"),
        ("https://x.com/i/web/status/123", "x_tweet"),
        ("https://x.com/user/article/123", "x_article"),
        ("https://x.com/i/articles/123", "x_article"),
        ("https://github.com/owner/repo/issues/56", "github_thread"),
        ("https://github.com/owner/repo/pull/7/files", "github_thread"),
        ("https://store.steampowered.com/news/app/1/view/2", "steam_news"),
        ("https://www.bilibili.com/opus/1234567890", "bilibili_opus"),
        ("https://www.youtube.com/watch?v=KVKufdTphKs", "youtube_page"),
        ("https://m.youtube.com/watch?v=KVKufdTphKs&t=10", "youtube_page"),
        ("https://youtu.be/KVKufdTphKs", "youtube_page"),
        ("https://news.ycombinator.com/item?id=12345", "hackernews_thread"),
    ],
)
def test_site_urls_still_match(url: str, name: str) -> None:
    extractor = find_extractor(url)

    assert extractor is not None
    assert extractor.name == name


def test_steam_event_page_on_another_host_is_recognized_by_its_data() -> None:
    """The Steam extractor no longer claims any URL with a Steam-like path;
    a Steam event page served elsewhere is recognized by its event data."""
    from bs4 import BeautifulSoup

    from markitai.webextract.extractors.registry import document_sniff_may_apply

    html = (
        '<html><body><div id="application_config" '
        "data-partnereventstore='[]'></div></body></html>"
    )
    url = "https://store.example.com/news/app/1/view/2"

    assert find_extractor(url) is None
    assert document_sniff_may_apply(html)
    extractor = find_extractor(url, BeautifulSoup(html, "html.parser"))
    assert extractor is not None
    assert extractor.name == "steam_news"
