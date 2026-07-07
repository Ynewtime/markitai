"""Unit tests for the X/Twitter FxTwitter + oEmbed enricher.

should_run() policy gating already has coverage in test_async_resolver.py;
this file covers the HTTP-calling and rendering logic that was previously
untested: FxTwitter fetch/retry, the oEmbed fallback, thread/media field
mapping, and X Article (Draft.js) rendering.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from markitai.webextract.enrichers.x_oembed import XOEmbedEnricher

TWEET_URL = "https://x.com/testuser/status/123456789"


def _http_response(json_data: dict) -> MagicMock:
    """A sync-flavored httpx.Response mock (raise_for_status/json are sync)."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=json_data)
    return resp


def _mock_client(get_side_effect) -> MagicMock:
    """An httpx.AsyncClient mock whose async context manager yields itself."""
    client = AsyncMock()
    client.get = AsyncMock(side_effect=get_side_effect)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


MOCK_TWEET = {
    "tweet": {
        "text": "Hello from FxTwitter",
        "author": {"name": "Test User", "screen_name": "testuser"},
        "created_at": "2026-03-23T10:00:00.000Z",
        "media": {
            "all": [
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/test.jpg",
                    "altText": "a photo",
                }
            ]
        },
    }
}


class TestShouldRun:
    """XOEmbedEnricher.should_run — article URL matching.

    Generic status-URL / network / async gating is already covered in
    test_async_resolver.py; this covers the article-specific path form.
    """

    def _policy(self):
        from markitai.webextract.enrichers.base import EnrichmentPolicy

        return EnrichmentPolicy(allow_network=True, allow_async=True)

    def test_matches_singular_article_path(self) -> None:
        enricher = XOEmbedEnricher()
        assert (
            enricher.should_run("https://x.com/user/article/123", self._policy())
            is True
        )

    def test_does_not_match_plural_articles_path(self) -> None:
        """Regression guard: a plural /articles/<id> (with a username) form
        was speculatively added and then reverted — verified against
        defuddle's reference XArticleExtractor/XOembedExtractor
        (github.com/kepano/defuddle), which only matches singular
        /(status|article)/ with no plural. Don't reintroduce it without
        a confirmed real-world X.com URL using this shape."""
        enricher = XOEmbedEnricher()
        assert (
            enricher.should_run("https://x.com/user/articles/123", self._policy())
            is False
        )


class TestTryFxtwitter:
    """XOEmbedEnricher._try_fxtwitter."""

    async def test_non_matching_url_returns_none(self) -> None:
        enricher = XOEmbedEnricher()
        result = await enricher._try_fxtwitter("https://example.com/not-a-tweet")
        assert result is None

    async def test_matches_singular_article_url(self) -> None:
        """The /article/<id> path (singular) must resolve to the same
        FxTwitter status API call as /status/<id>."""
        enricher = XOEmbedEnricher()
        client = _mock_client([_http_response(MOCK_TWEET)])
        with patch("httpx.AsyncClient", return_value=client):
            result = await enricher._try_fxtwitter(
                "https://x.com/testuser/article/123456789"
            )

        assert result is not None
        client.get.assert_awaited_once()
        called_url = client.get.await_args.args[0]
        assert called_url == "https://api.fxtwitter.com/testuser/status/123456789"

    async def test_does_not_match_plural_articles_url(self) -> None:
        """Regression guard: see TestShouldRun.test_does_not_match_plural_articles_path
        — plural (with a username) has no confirmed real-world basis."""
        enricher = XOEmbedEnricher()
        result = await enricher._try_fxtwitter(
            "https://x.com/testuser/articles/123456789"
        )
        assert result is None

    async def test_success_builds_resolved_page(self) -> None:
        enricher = XOEmbedEnricher()
        client = _mock_client([_http_response(MOCK_TWEET)])
        with patch("httpx.AsyncClient", return_value=client):
            result = await enricher._try_fxtwitter(TWEET_URL)

        assert result is not None
        assert result.diagnostics == {
            "enricher_name": "x_oembed",
            "source": "fxtwitter",
        }
        assert result.metadata_overrides["author"] == "@testuser"
        assert result.metadata_overrides["title"] == "Post by @testuser"
        assert "Hello from FxTwitter" in (result.content_html or "")

    async def test_missing_tweet_key_returns_none(self) -> None:
        enricher = XOEmbedEnricher()
        client = _mock_client([_http_response({"code": 404})])
        with patch("httpx.AsyncClient", return_value=client):
            result = await enricher._try_fxtwitter(TWEET_URL)

        assert result is None

    async def test_retries_on_transient_failure_then_succeeds(self) -> None:
        enricher = XOEmbedEnricher()
        client = _mock_client([ConnectionError("boom"), _http_response(MOCK_TWEET)])
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await enricher._try_fxtwitter(TWEET_URL)

        assert result is not None
        assert result.metadata_overrides["author"] == "@testuser"
        mock_sleep.assert_awaited_once()

    async def test_returns_none_after_exhausting_retries(self) -> None:
        enricher = XOEmbedEnricher()
        client = _mock_client(
            [ConnectionError("1"), ConnectionError("2"), ConnectionError("3")]
        )
        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await enricher._try_fxtwitter(TWEET_URL)

        assert result is None
        assert client.get.await_count == 3

    async def test_article_dispatches_to_build_article_result(self) -> None:
        enricher = XOEmbedEnricher()
        tweet_with_article = {
            "tweet": {
                "author": {"name": "Author", "screen_name": "author1"},
                "article": {
                    "title": "My Article",
                    "content": {"blocks": [], "entityMap": []},
                },
            }
        }
        client = _mock_client([_http_response(tweet_with_article)])
        with patch("httpx.AsyncClient", return_value=client):
            result = await enricher._try_fxtwitter(TWEET_URL)

        assert result is not None
        assert result.diagnostics["source"] == "fxtwitter_article"
        assert result.metadata_overrides["title"] == "My Article"


class TestTryOembed:
    """XOEmbedEnricher._try_oembed."""

    async def test_success_builds_resolved_page(self) -> None:
        enricher = XOEmbedEnricher()
        oembed_data = {
            "html": '<blockquote class="twitter-tweet">Hi</blockquote>',
            "author_name": "Test User",
            "author_url": "https://twitter.com/testuser",
        }
        client = _mock_client([_http_response(oembed_data)])
        with patch("httpx.AsyncClient", return_value=client):
            result = await enricher._try_oembed(TWEET_URL)

        assert result is not None
        assert result.diagnostics == {"enricher_name": "x_oembed", "source": "oembed"}
        assert result.metadata_overrides["author"] == "Test User"
        assert result.metadata_overrides["title"] == "Post by @testuser"

    async def test_empty_html_returns_none(self) -> None:
        enricher = XOEmbedEnricher()
        client = _mock_client([_http_response({"html": ""})])
        with patch("httpx.AsyncClient", return_value=client):
            result = await enricher._try_oembed(TWEET_URL)

        assert result is None


class TestEnrichFallsBackFromFxtwitterToOembed:
    """XOEmbedEnricher.enrich strategy ordering."""

    async def test_falls_back_to_oembed_when_fxtwitter_fails(self) -> None:
        enricher = XOEmbedEnricher()
        with (
            patch.object(
                XOEmbedEnricher,
                "_try_fxtwitter",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                XOEmbedEnricher,
                "_try_oembed",
                new_callable=AsyncMock,
            ) as mock_oembed,
        ):
            await enricher.enrich(TWEET_URL, None)

        mock_oembed.assert_awaited_once_with(TWEET_URL)

    async def test_skips_oembed_when_fxtwitter_succeeds(self) -> None:
        enricher = XOEmbedEnricher()
        sentinel = object()
        with (
            patch.object(
                XOEmbedEnricher,
                "_try_fxtwitter",
                new_callable=AsyncMock,
                return_value=sentinel,
            ),
            patch.object(
                XOEmbedEnricher, "_try_oembed", new_callable=AsyncMock
            ) as mock_oembed,
        ):
            result = await enricher.enrich(TWEET_URL, None)

        assert result is sentinel
        mock_oembed.assert_not_awaited()

    async def test_never_raises_when_both_strategies_error(self) -> None:
        enricher = XOEmbedEnricher()
        with (
            patch.object(
                XOEmbedEnricher,
                "_try_fxtwitter",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fx down"),
            ),
            patch.object(
                XOEmbedEnricher,
                "_try_oembed",
                new_callable=AsyncMock,
                side_effect=RuntimeError("oembed down"),
            ),
        ):
            result = await enricher.enrich(TWEET_URL, None)

        assert result is None


class TestBuildThread:
    """XOEmbedEnricher._build_thread."""

    def test_maps_author_text_and_media(self) -> None:
        enricher = XOEmbedEnricher()
        thread = enricher._build_thread(MOCK_TWEET["tweet"], "123", TWEET_URL)

        assert thread.main_item.author_name == "Test User"
        assert thread.main_item.author_handle == "@testuser"
        assert thread.main_item.text == "Hello from FxTwitter"
        assert len(thread.main_item.media) == 1
        assert thread.main_item.media[0].media_type == "image"

    def test_hides_title_and_author_meta_in_body(self) -> None:
        """Author/title are frontmatter-only; the body must not repeat them."""
        enricher = XOEmbedEnricher()
        thread = enricher._build_thread(MOCK_TWEET["tweet"], "123", TWEET_URL)

        assert thread.show_title_in_body is False
        assert thread.show_author_meta is False

    def test_falls_back_to_raw_text_when_text_missing(self) -> None:
        enricher = XOEmbedEnricher()
        data = {**MOCK_TWEET["tweet"], "text": "", "raw_text": {"text": "raw fallback"}}
        thread = enricher._build_thread(data, "123", TWEET_URL)

        assert thread.main_item.text == "raw fallback"

    def test_builds_quoted_item(self) -> None:
        enricher = XOEmbedEnricher()
        data = {
            **MOCK_TWEET["tweet"],
            "quote": {
                "text": "Quoted text",
                "author": {"name": "Quoter", "screen_name": "quoter1"},
                "url": "https://x.com/quoter1/status/999",
                "created_at": "2026-01-01T00:00:00.000Z",
            },
        }
        thread = enricher._build_thread(data, "123", TWEET_URL)

        assert thread.main_item.quoted_item is not None
        assert thread.main_item.quoted_item.author_handle == "@quoter1"
        assert thread.main_item.quoted_item.text == "Quoted text"


class TestBuildMedia:
    """XOEmbedEnricher._build_media."""

    def test_photo_normalizes_to_image(self) -> None:
        enricher = XOEmbedEnricher()
        media = enricher._build_media(
            {"media": {"all": [{"type": "photo", "url": "https://x/1.jpg"}]}}
        )
        assert media[0].media_type == "image"

    def test_video_keeps_thumbnail_as_poster(self) -> None:
        enricher = XOEmbedEnricher()
        media = enricher._build_media(
            {
                "media": {
                    "all": [
                        {
                            "type": "video",
                            "url": "https://x/1.mp4",
                            "thumbnail_url": "https://x/thumb.jpg",
                        }
                    ]
                }
            }
        )
        assert media[0].media_type == "video"
        assert media[0].poster == "https://x/thumb.jpg"

    def test_no_media_key_returns_empty_list(self) -> None:
        enricher = XOEmbedEnricher()
        assert enricher._build_media({}) == []

    def test_non_dict_media_value_returns_empty_list(self) -> None:
        enricher = XOEmbedEnricher()
        assert enricher._build_media({"media": "unexpected-string"}) == []


class TestToDateString:
    """XOEmbedEnricher._to_date_string."""

    def test_converts_iso_z_suffix_to_date(self) -> None:
        assert (
            XOEmbedEnricher._to_date_string("2026-05-01T12:00:00.000Z") == "2026-05-01"
        )

    def test_none_input_returns_none(self) -> None:
        assert XOEmbedEnricher._to_date_string(None) is None

    def test_malformed_date_returns_none(self) -> None:
        assert XOEmbedEnricher._to_date_string("not-a-date") is None


class TestNormalizeEntityMap:
    """XOEmbedEnricher._normalize_entity_map.

    FxTwitter serializes entityMap as a list of {"key", "value"} pairs
    (confirmed against defuddle's reference XOembedExtractor) — this just
    converts that list to a dict for O(1) lookup, without raising on
    malformed input.
    """

    def test_list_of_key_value_pairs_is_converted_to_dict(self) -> None:
        raw = [{"key": 0, "value": {"type": "LINK", "data": {"url": "https://x"}}}]
        assert XOEmbedEnricher._normalize_entity_map(raw) == {
            "0": {"type": "LINK", "data": {"url": "https://x"}}
        }

    def test_malformed_list_items_are_skipped(self) -> None:
        raw = [{"key": 0}, "not-a-dict", {"value": {"type": "LINK"}}]
        assert XOEmbedEnricher._normalize_entity_map(raw) == {}

    def test_unexpected_type_returns_empty_dict(self) -> None:
        assert XOEmbedEnricher._normalize_entity_map(None) == {}
        assert XOEmbedEnricher._normalize_entity_map("garbage") == {}
        assert XOEmbedEnricher._normalize_entity_map({"0": {"type": "LINK"}}) == {}


class TestBuildArticleResult:
    """XOEmbedEnricher._build_article_result (Draft.js -> HTML)."""

    def _article_data(self, blocks: list[dict], **extra: object) -> dict:
        return {
            "title": "Article Title",
            "created_at": "2026-05-01T00:00:00.000Z",
            "content": {"blocks": blocks, "entityMap": extra.pop("entityMap", [])},
            **extra,
        }

    def test_renders_unstyled_paragraph(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data([{"type": "unstyled", "text": "Hello world"}])
        result = enricher._build_article_result(
            {"author": {"screen_name": "author1"}}, article
        )

        assert result is not None
        assert result.content_html is not None
        assert "<p>Hello world</p>" in result.content_html

    def test_renders_headers(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [
                {"type": "header-two", "text": "Section"},
                {"type": "header-three", "text": "Subsection"},
            ]
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert "<h2>Section</h2>" in result.content_html
        assert "<h3>Subsection</h3>" in result.content_html

    def test_skips_empty_unstyled_blocks(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data([{"type": "unstyled", "text": "   "}])
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert "<p>" not in result.content_html

    def test_unknown_block_type_falls_back_to_paragraph(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data([{"type": "blockquote", "text": "quoted"}])
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert "<p>quoted</p>" in result.content_html

    def test_atomic_media_item_with_no_matching_entity_is_skipped(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [{"type": "atomic", "entityRanges": [{"key": 0}]}],
            entityMap=[
                {
                    "key": 0,
                    "value": {
                        "type": "MEDIA",
                        "data": {"mediaItems": [{"mediaId": "missing"}], "caption": ""},
                    },
                }
            ],
            media_entities=[],
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert "<img" not in result.content_html
        assert "<figure>" not in result.content_html

    def test_renders_consecutive_list_items_as_single_ul(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [
                {"type": "unordered-list-item", "text": "one"},
                {"type": "unordered-list-item", "text": "two"},
                {"type": "unstyled", "text": "after"},
            ]
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert "<ul><li>one</li><li>two</li></ul>" in result.content_html
        assert "<p>after</p>" in result.content_html

    def test_renders_cover_image(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [],
            cover_media={"media_info": {"original_img_url": "https://x/cover.jpg"}},
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert (
            '<img src="https://x/cover.jpg" alt="Cover image">' in result.content_html
        )

    def test_renders_atomic_image_media(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [
                {
                    "type": "atomic",
                    "entityRanges": [{"key": 0}],
                }
            ],
            entityMap=[
                {
                    "key": 0,
                    "value": {
                        "type": "MEDIA",
                        "data": {"mediaItems": [{"mediaId": "m1"}], "caption": "cap"},
                    },
                }
            ],
            media_entities=[
                {
                    "media_id": "m1",
                    "media_info": {
                        "__typename": "ApiImage",
                        "original_img_url": "https://x/pic.jpg",
                    },
                }
            ],
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert '<img src="https://x/pic.jpg" alt="cap">' in result.content_html
        assert "<figcaption>cap</figcaption>" in result.content_html

    def test_renders_atomic_video_prefers_highest_bitrate_variant(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [{"type": "atomic", "entityRanges": [{"key": 0}]}],
            entityMap=[
                {
                    "key": 0,
                    "value": {
                        "type": "MEDIA",
                        "data": {"mediaItems": [{"mediaId": "m1"}], "caption": ""},
                    },
                }
            ],
            media_entities=[
                {
                    "media_id": "m1",
                    "media_info": {
                        "__typename": "ApiVideo",
                        "preview_image": {"original_img_url": "https://x/poster.jpg"},
                        "variants": [
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 500,
                                "url": "https://x/low.mp4",
                            },
                            {
                                "content_type": "video/mp4",
                                "bit_rate": 2000,
                                "url": "https://x/high.mp4",
                            },
                        ],
                    },
                }
            ],
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert "https://x/high.mp4" in result.content_html
        assert "https://x/low.mp4" not in result.content_html

    def test_renders_markdown_code_block(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data(
            [{"type": "atomic", "entityRanges": [{"key": 0}]}],
            entityMap=[
                {
                    "key": 0,
                    "value": {
                        "type": "MARKDOWN",
                        "data": {"markdown": "```python\nprint(1)\n```"},
                    },
                }
            ],
        )
        result = enricher._build_article_result({"author": {}}, article)

        assert result is not None
        assert result.content_html is not None
        assert (
            '<pre><code class="language-python" data-lang="python">print(1)</code></pre>'
            in (result.content_html)
        )

    def test_metadata_overrides_include_title_author_published(self) -> None:
        enricher = XOEmbedEnricher()
        article = self._article_data([])
        result = enricher._build_article_result(
            {"author": {"screen_name": "author1"}}, article
        )

        assert result is not None
        assert result.metadata_overrides["title"] == "Article Title"
        assert result.metadata_overrides["author"] == "@author1"
        assert result.metadata_overrides["published"] == "2026-05-01"
        assert result.diagnostics == {
            "enricher_name": "x_oembed",
            "source": "fxtwitter_article",
        }


class TestRenderInline:
    """XOEmbedEnricher._render_inline marker rendering."""

    def test_plain_text_is_escaped(self) -> None:
        enricher = XOEmbedEnricher()
        html = enricher._render_inline({"text": "<b>raw</b>"}, {})
        assert html == "&lt;b&gt;raw&lt;/b&gt;"

    def test_bold_range_wraps_strong(self) -> None:
        enricher = XOEmbedEnricher()
        block = {
            "text": "hello world",
            "inlineStyleRanges": [{"style": "Bold", "offset": 0, "length": 5}],
        }
        assert enricher._render_inline(block, {}) == "<strong>hello</strong> world"

    def test_link_entity_wraps_anchor(self) -> None:
        enricher = XOEmbedEnricher()
        block = {
            "text": "see docs",
            "entityRanges": [{"key": 0, "offset": 4, "length": 4}],
        }
        entity_map = {"0": {"type": "LINK", "data": {"url": "https://example.com"}}}
        assert (
            enricher._render_inline(block, entity_map)
            == 'see <a href="https://example.com">docs</a>'
        )

    def test_mention_wraps_anchor_to_profile(self) -> None:
        enricher = XOEmbedEnricher()
        block = {
            "text": "cc @bob",
            "data": {"mentions": [{"text": "bob", "fromIndex": 3, "toIndex": 7}]},
        }
        assert (
            enricher._render_inline(block, {})
            == 'cc <a href="https://x.com/bob">@bob</a>'
        )

    def test_adjacent_markers_close_before_open(self) -> None:
        """Two back-to-back bold ranges at the same boundary offset must not
        nest — the closing tag for the first must precede the opening tag
        for the second."""
        enricher = XOEmbedEnricher()
        block = {
            "text": "ab",
            "inlineStyleRanges": [
                {"style": "Bold", "offset": 0, "length": 1},
                {"style": "Bold", "offset": 1, "length": 1},
            ],
        }
        assert (
            enricher._render_inline(block, {}) == "<strong>a</strong><strong>b</strong>"
        )


class TestBuildResolvedFromHtml:
    """XOEmbedEnricher._build_resolved_from_html."""

    def test_empty_html_returns_none(self) -> None:
        enricher = XOEmbedEnricher()
        assert enricher._build_resolved_from_html("", "title", "", {}) is None

    def test_prefers_explicit_author_name_over_tweet_data(self) -> None:
        enricher = XOEmbedEnricher()
        result = enricher._build_resolved_from_html(
            "<p>hi</p>",
            "title",
            "",
            {"author": {"screen_name": "fromtweet"}},
            author_name="Explicit Name",
        )
        assert result is not None
        assert result.metadata_overrides["author"] == "Explicit Name"
