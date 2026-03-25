"""Unit tests for the Steam news extractor and BBCode-to-HTML converter."""

from __future__ import annotations

import json

import pytest

from markitai.webextract.extractors.steam_news import (
    SteamNewsExtractor,
    _bbcode_to_html,
)


class TestSteamNewsURLMatching:
    """Tests for URL pattern matching."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://store.steampowered.com/news/app/12345/view/67890",
            "https://store.steampowered.com/news/app/1/view/2",
            "https://store.steampowered.com/news/collection/all",
        ],
    )
    def test_matches_steam_news_urls(self, url: str) -> None:
        assert SteamNewsExtractor().matches_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://store.steampowered.com/app/12345",
            "https://example.com/news/app/12345",
            "https://github.com/user/repo",
        ],
    )
    def test_rejects_non_steam_news_urls(self, url: str) -> None:
        assert not SteamNewsExtractor().matches_url(url)


class TestBBCodeToHTML:
    """Tests for BBCode-to-HTML conversion."""

    # --- Basic inline formatting ---

    def test_bold(self) -> None:
        assert "<strong>hello</strong>" in _bbcode_to_html("[b]hello[/b]")

    def test_italic(self) -> None:
        assert "<em>world</em>" in _bbcode_to_html("[i]world[/i]")

    def test_underline(self) -> None:
        assert "<u>text</u>" in _bbcode_to_html("[u]text[/u]")

    def test_strikethrough(self) -> None:
        assert "<del>gone</del>" in _bbcode_to_html("[s]gone[/s]")

    def test_nested_formatting(self) -> None:
        result = _bbcode_to_html("[b][i]nested[/i][/b]")
        assert "<strong><em>nested</em></strong>" in result

    # --- Headings ---

    def test_h1(self) -> None:
        assert "<h1>Title</h1>" in _bbcode_to_html("[h1]Title[/h1]")

    def test_h2(self) -> None:
        assert "<h2>Sub</h2>" in _bbcode_to_html("[h2]Sub[/h2]")

    def test_h3(self) -> None:
        assert "<h3>Section</h3>" in _bbcode_to_html("[h3]Section[/h3]")

    # --- Links ---

    def test_url_with_href(self) -> None:
        result = _bbcode_to_html("[url=https://example.com]click[/url]")
        assert '<a href="https://example.com">click</a>' in result

    def test_url_bare(self) -> None:
        result = _bbcode_to_html("[url]https://example.com[/url]")
        assert '<a href="https://example.com">https://example.com</a>' in result

    def test_url_special_chars_escaped(self) -> None:
        result = _bbcode_to_html("[url=https://example.com?a=1&b=2]link[/url]")
        assert "&amp;" in result

    # --- Images ---

    def test_img(self) -> None:
        result = _bbcode_to_html("[img]https://example.com/pic.jpg[/img]")
        assert '<img src="https://example.com/pic.jpg">' in result

    def test_youtube_preview(self) -> None:
        result = _bbcode_to_html("[previewyoutube=dQw4w9WgXcQ;full][/previewyoutube]")
        assert "youtube.com/watch?v=dQw4w9WgXcQ" in result

    # --- Lists ---

    def test_list(self) -> None:
        result = _bbcode_to_html("[list][*]one[*]two[/list]")
        assert "<ul>" in result
        assert "<li>" in result
        assert "</ul>" in result

    # --- Code ---

    def test_code(self) -> None:
        result = _bbcode_to_html("[code]print('hi')[/code]")
        assert "<code>" in result
        assert "print" in result

    # --- Paragraphs ---

    def test_paragraphs(self) -> None:
        result = _bbcode_to_html("[p]First[/p][p]Second[/p]")
        assert "<p>First</p>" in result
        assert "<p>Second</p>" in result

    # --- HTML injection prevention ---

    def test_html_in_heading_is_escaped(self) -> None:
        result = _bbcode_to_html("[h1]<script>alert(1)</script>[/h1]")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_html_in_bold_is_escaped(self) -> None:
        result = _bbcode_to_html("[b]<img onerror=alert(1)>[/b]")
        assert "onerror" not in result or "&lt;img" in result

    def test_html_in_code_is_escaped(self) -> None:
        result = _bbcode_to_html("[code]<div>html</div>[/code]")
        assert "&lt;div&gt;" in result

    # --- JSON unescape ---

    def test_json_escaped_slashes(self) -> None:
        result = _bbcode_to_html("[url]https:\\/\\/example.com[/url]")
        assert "https://example.com" in result

    # --- Unknown tags stripped ---

    def test_unknown_tags_removed(self) -> None:
        result = _bbcode_to_html("[spoiler]secret[/spoiler]")
        assert "[spoiler]" not in result
        assert "secret" in result

    # --- Newline handling ---

    def test_double_newline_becomes_paragraph(self) -> None:
        result = _bbcode_to_html("first\n\nsecond")
        assert "</p><p>" in result

    def test_single_newline_becomes_br(self) -> None:
        result = _bbcode_to_html("line1\nline2")
        assert "<br>" in result

    # --- Case insensitivity ---

    def test_case_insensitive_tags(self) -> None:
        assert "<strong>hi</strong>" in _bbcode_to_html("[B]hi[/B]")
        assert "<em>hi</em>" in _bbcode_to_html("[I]hi[/I]")


class TestSteamNewsResolve:
    """Tests for the full resolve pipeline."""

    def _make_html(
        self,
        *,
        headline: str = "Test Title",
        body: str = "Hello world",
        posttime: int = 1700000000,
        group_name: str = "Test Game",
    ) -> str:
        event_data = [
            {
                "announcement_body": {
                    "headline": headline,
                    "body": body,
                    "posttime": posttime,
                }
            }
        ]
        group_data = [{"group_name": group_name}]
        return (
            f'<html><body><div id="application_config" '
            f"data-partnereventstore='{json.dumps(event_data)}' "
            f"data-groupvanityinfo='{json.dumps(group_data)}'>"
            f"</div></body></html>"
        )

    def test_resolve_extracts_content(self) -> None:
        from bs4 import BeautifulSoup

        html = self._make_html(body="[b]Important[/b] news")
        soup = BeautifulSoup(html, "html.parser")
        extractor = SteamNewsExtractor()
        result = extractor.resolve(
            soup, "https://store.steampowered.com/news/app/1/view/2"
        )

        assert result.content_html
        assert "<strong>Important</strong>" in result.content_html

    def test_resolve_extracts_metadata(self) -> None:
        from bs4 import BeautifulSoup

        html = self._make_html(headline="Big Update", group_name="My Game")
        soup = BeautifulSoup(html, "html.parser")
        extractor = SteamNewsExtractor()
        result = extractor.resolve(
            soup, "https://store.steampowered.com/news/app/1/view/2"
        )

        assert result.metadata_overrides.get("title") == "Big Update"
        assert result.metadata_overrides.get("author") == "My Game"
        assert "published" in result.metadata_overrides

    def test_resolve_no_config_returns_empty(self) -> None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        extractor = SteamNewsExtractor()
        result = extractor.resolve(
            soup, "https://store.steampowered.com/news/app/1/view/2"
        )

        assert result.content_html == ""
        assert result.diagnostics.get("steam_resolve") == "no_announcement_data"
