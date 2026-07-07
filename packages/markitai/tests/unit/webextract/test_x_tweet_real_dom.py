"""Parity tests for the 2026 X.com DOM (real Playwright capture).

The fixture ``x_status_2073286406558949828.playwright.html`` is a faithful
capture of the rendered DOM that ``fetch_with_playwright`` hands to
``extract_web_content`` (captured live on 2026-07-04). The 2026 redesign
dropped every ``data-testid`` attribute, so these tests freeze the new
selector contract:

- tweet article:    ``article[data-tweet-id]``
- author block:     ``div[data-slot="hover-card-trigger"]``
- tweet text:       first ``div[dir="auto"]`` (whitespace-pre-wrap newlines)
- timestamp:        status-permalink text (no ``<time>`` element)
- quoted tweet:     ``div[role="link"][data-href="/user/status/id"]``
- video:            ``<video poster="https://…" src="blob:…">``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from markitai.webextract.pipeline import extract_web_content
from markitai.webextract.types import ContentProfile, ExtractedWebContent

FIXTURES = Path(__file__).parents[2] / "fixtures" / "web"

X_URL = "https://x.com/dotey/status/2073286406558949828"
QUOTE_URL = "https://x.com/dotey/status/2067876611873964284"
POSTER_URL = (
    "https://pbs.twimg.com/amplify_video_thumb/2073283886402981888"
    "/img/iLBiXgibPtjFs1tt.jpg"
)


@pytest.fixture(scope="module")
def result() -> ExtractedWebContent:
    html = (FIXTURES / "x_status_2073286406558949828.playwright.html").read_text(
        encoding="utf-8"
    )
    return extract_web_content(html, X_URL)


class TestClassification:
    def test_resolver_fires_on_new_dom(self, result: ExtractedWebContent) -> None:
        assert result.diagnostics["extractor"] == "resolver"
        resolver_diag = result.diagnostics["resolver_diagnostics"]
        assert resolver_diag["x_resolve"] == "success"

    def test_profile_and_metadata(self, result: ExtractedWebContent) -> None:
        assert result.info is not None
        assert result.info.content_profile == ContentProfile.SOCIAL_POST
        assert result.metadata.title == "Post by @dotey on X"

    def test_semantic_thread_populated(self, result: ExtractedWebContent) -> None:
        assert result.semantic is not None
        assert result.semantic.thread is not None
        main = result.semantic.thread.main_item
        assert main.author_name == "宝玉"
        assert main.author_handle == "@dotey"
        assert main.id == "2073286406558949828"
        assert main.timestamp == "2026-07-04T06:02:00"


class TestMainTweetBody:
    def test_single_bold_author_line_with_date(
        self, result: ExtractedWebContent
    ) -> None:
        # Author meta line is NOT rendered for tweets (in frontmatter only)
        assert "**宝玉 @dotey** · 2026-07-04" not in result.markdown
        # But quoted tweet still has author meta
        assert result.markdown.count("**宝玉 @dotey**") == 1  # only in quote
        assert "[宝玉](https://x.com/dotey)" not in result.markdown
        assert "[@dotey](https://x.com/dotey)" not in result.markdown

    def test_paragraph_breaks_preserved(self, result: ExtractedWebContent) -> None:
        assert "baoyu-design skill 更新： 支持 PPT 动画了" in result.markdown
        assert "推荐去试试看：" in result.markdown

    def test_truncated_link_expanded_to_full_url(
        self, result: ExtractedWebContent
    ) -> None:
        assert "https://github.com/jimliu/baoyu-design" in result.markdown
        assert "github.com/jimliu/baoyu-d…" not in result.markdown


class TestMedia:
    def test_video_renders_poster_image(self, result: ExtractedWebContent) -> None:
        assert f"![]({POSTER_URL})" in result.markdown

    def test_no_blob_urls_or_video_link(self, result: ExtractedWebContent) -> None:
        assert "blob:" not in result.markdown
        # No usable https video URL exists in the DOM -> no [Video] link.
        assert "[Video]" not in result.markdown

    def test_no_duration_artifact(self, result: ExtractedWebContent) -> None:
        assert "00:00" not in result.markdown

    def test_poster_not_duplicated_as_image(self, result: ExtractedWebContent) -> None:
        assert result.markdown.count(POSTER_URL) == 1


class TestQuotedTweet:
    def test_quote_rendered_as_blockquote(self, result: ExtractedWebContent) -> None:
        assert "> **宝玉 @dotey** · Jun 19" in result.markdown
        assert "> baoyu-design skill 更新：可以在制作 PPT" in result.markdown

    def test_quote_not_expanded_as_duplicate_block(
        self, result: ExtractedWebContent
    ) -> None:
        # The quoted tweet text must only appear inside the blockquote.
        for line in result.markdown.splitlines():
            if "可以在制作 PPT、动画视频或者网站时调用" in line:
                assert line.startswith(">")

    def test_quote_carries_permalink_and_upgraded_image(
        self, result: ExtractedWebContent
    ) -> None:
        assert QUOTE_URL in result.markdown
        assert (
            "https://pbs.twimg.com/media/HLKRt-AXgAAFSGw.jpg?name=orig"
            in result.markdown
        )
        assert "name=medium" not in result.markdown

    def test_quote_semantic_model(self, result: ExtractedWebContent) -> None:
        assert result.semantic is not None
        assert result.semantic.thread is not None
        quoted = result.semantic.thread.main_item.quoted_item
        assert quoted is not None
        assert quoted.author_handle == "@dotey"
        assert quoted.url == QUOTE_URL
        assert quoted.timestamp == "Jun 19"
        assert "Show more" not in quoted.text


class TestNoiseRemoval:
    def test_no_avatars(self, result: ExtractedWebContent) -> None:
        assert "profile_images" not in result.markdown
        assert "user avatar" not in result.markdown

    def test_no_cookie_banner(self, result: ExtractedWebContent) -> None:
        assert "cookies" not in result.markdown
        assert "Did someone say" not in result.markdown

    def test_no_stats_or_views_line(self, result: ExtractedWebContent) -> None:
        assert "Views" not in result.markdown
        assert "6:02 AM" not in result.markdown

    def test_no_engagement_ui(self, result: ExtractedWebContent) -> None:
        assert "Read 28 replies" not in result.markdown
        assert "Log in" not in result.markdown
        assert "Sign up" not in result.markdown
