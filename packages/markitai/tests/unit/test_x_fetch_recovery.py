"""Regressions from X returning an empty HTTP 403 before DOM rendering."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import DomainProfileConfig, FetchConfig, ScreenshotConfig
from markitai.fetch import _fetch_with_fallback
from markitai.fetch_playwright import PlaywrightRenderer
from markitai.fetch_types import FetchError, FetchResult

URL = "https://x.com/user/status/123"


@pytest.fixture
def browser_page():
    page = AsyncMock()
    page.url = URL
    page.goto.return_value = MagicMock(status=403)
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()
    browser.new_context.return_value = context
    renderer = PlaywrightRenderer()
    # The new-headless build is unavailable here: x.com answers the default
    # browser (the tests below set the status) and the embed fallback runs
    with (
        patch.object(renderer, "_ensure_browser", AsyncMock(return_value=browser)),
        patch.object(renderer, "_ensure_new_headless", AsyncMock(return_value=None)),
    ):
        yield renderer, page, context


@pytest.mark.parametrize("status", [403, 404, 429, 503])
async def test_http_failure_skips_dom_wait_and_closes_context(browser_page, status):
    renderer, page, context = browser_page
    page.goto.return_value.status = status
    with (
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("", None, "")),
        ),
        patch("markitai.fetch_playwright.asyncio.sleep", AsyncMock()) as sleep,
        pytest.raises(FetchError, match=f"HTTP {status}"),
    ):
        await renderer.fetch(
            URL, wait_for_selector='[data-testid="tweet"]', extra_wait_ms=500
        )
    page.wait_for_selector.assert_not_awaited()
    page.content.assert_not_awaited()
    sleep.assert_not_awaited()
    context.close.assert_awaited_once()


async def test_http_failure_can_recover_with_enrichment_and_screenshot(
    browser_page, tmp_path
):
    renderer, page, context = browser_page
    screenshot = tmp_path / "page.png"
    with (
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(
                return_value=(
                    "Full tweet content",
                    {"title": "Tweet", "author": "@user"},
                    "fxtwitter",
                )
            ),
        ),
        patch(
            "markitai.fetch_playwright._capture_screenshot",
            AsyncMock(return_value=(screenshot, [screenshot])),
        ) as capture,
    ):
        result = await renderer.fetch(
            URL,
            wait_for_selector='[data-testid="tweet"]',
            screenshot_config=ScreenshotConfig(enabled=True),
            output_dir=tmp_path,
        )
    assert result.content == "Full tweet content"
    assert result.metadata["http_status"] == 403
    assert result.metadata["_enricher_source"] == "fxtwitter"
    assert result.metadata["source_frontmatter"]["word_count"] == 3
    assert result.screenshot_path == screenshot
    capture.assert_awaited_once()
    # x.com refused the browser: X's embedded-post page stands in for it,
    # and the tweet selector of the refused page is never waited for
    assert page.goto.await_args_list[-1].args[0] == (
        "https://platform.twitter.com/embed/Tweet.html?id=123&dnt=true"
    )
    page.wait_for_selector.assert_awaited_once()
    assert page.wait_for_selector.await_args.args[0] == "article"
    assert result.metadata["page_source"] == "x-embed"
    context.close.assert_awaited_once()


async def test_refused_x_post_keeps_the_embedded_post_as_its_page(
    browser_page, tmp_path
):
    renderer, page, _ = browser_page
    drawn = "<!DOCTYPE html><html><body><article><p>Post text that X drew</p>"
    drawn += "<img src='https://pbs.twimg.com/media/a.jpg'></article></body></html>"
    page.evaluate = AsyncMock(
        return_value={
            "html": drawn,
            "text": "Post text that X drew",
            "media": 1,
            "shadow": 0,
        }
    )
    with patch.object(
        renderer,
        "_try_enricher_fallback_async",
        AsyncMock(return_value=("Full tweet content", {"title": "T"}, "fxtwitter")),
    ):
        result = await renderer.fetch(URL)
    assert result.html == drawn
    assert result.metadata["page_source"] == "x-embed"
    assert "page_problem" not in result.metadata


async def test_refused_page_that_stays_empty_keeps_no_page_or_screenshot(
    browser_page, tmp_path
):
    renderer, page, _ = browser_page
    page.wait_for_selector.side_effect = TimeoutError("no article")
    page.evaluate = AsyncMock(
        return_value={
            "html": "<html><body></body></html>",
            "text": "",
            "media": 0,
            "shadow": 0,
        }
    )
    with (
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("Full tweet content", {"title": "T"}, "fxtwitter")),
        ),
        patch("markitai.fetch_playwright._capture_screenshot", AsyncMock()) as capture,
    ):
        result = await renderer.fetch(
            URL, screenshot_config=ScreenshotConfig(enabled=True), output_dir=tmp_path
        )
    assert result.content == "Full tweet content"
    assert result.html is None
    assert result.screenshot_path is None
    assert result.metadata["page_problem"] == "blank"
    assert "page_source" not in result.metadata
    capture.assert_not_awaited()


@pytest.mark.parametrize("selector_found", [False, True])
async def test_only_successful_selector_wait_gets_stabilization(
    browser_page, selector_found
):
    renderer, page, _ = browser_page
    page.goto.return_value.status = 200
    page.title.return_value = "Content"
    page.content.return_value = "<p>Real content from a slow page</p>"
    if not selector_found:
        page.wait_for_selector.side_effect = TimeoutError("selector missing")
    with (
        patch(
            "markitai.fetch_playwright.extract_web_content",
            side_effect=RuntimeError("no native extraction"),
        ),
        patch(
            "markitai.fetch_playwright._html_to_markdown",
            return_value="Real content " * 20,
        ),
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("", None, "")),
        ),
        patch(
            "markitai.fetch_playwright._wait_until_settled",
            AsyncMock(return_value=True),
        ) as settle,
    ):
        result = await renderer.fetch(
            URL, wait_for_selector="article", extra_wait_ms=500, skip_auto_scroll=True
        )
    assert "Real content" in result.content
    page.wait_for_selector.assert_awaited_once_with("article", timeout=10000)
    if selector_found:
        # Waits for the page to settle, at most extra_wait_ms
        settle.assert_awaited_once()
        assert settle.await_args_list[0].args[2] == 0.5
    else:
        settle.assert_not_awaited()


@pytest.mark.parametrize(
    "mode, public_ips, expected",
    [
        ("always", ("93.184.216.34",), True),
        ("never", ("93.184.216.34",), False),
        ("always", ("127.0.0.1",), False),
    ],
)
async def test_x_enrichment_fake_ip_verification(mode, public_ips, expected):
    with (
        patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            AsyncMock(return_value=("198.18.0.158",)),
        ),
        patch(
            "markitai.fetch_policy.resolve_public_hostname_addresses",
            AsyncMock(return_value=public_ips),
        ) as public,
        patch(
            "markitai.webextract.enrichers.x_oembed.XOEmbedEnricher.enrich",
            AsyncMock(return_value=None),
        ) as enrich,
    ):
        await PlaywrightRenderer()._try_enricher_fallback_async(URL, mode)
    assert enrich.await_count == int(expected)
    assert public.await_count == int(mode != "never")


async def test_fake_ip_consent_decline_prevents_both_dns_and_enrichment():
    from markitai.ports import get_interaction

    with (
        patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            AsyncMock(return_value=("198.18.0.158",)),
        ),
        patch(
            "markitai.fetch_policy.resolve_public_hostname_addresses", AsyncMock()
        ) as public,
        patch.object(get_interaction(), "can_prompt", return_value=True),
        patch.object(get_interaction(), "confirm", return_value=False),
        patch(
            "markitai.webextract.enrichers.x_oembed.XOEmbedEnricher.enrich", AsyncMock()
        ) as enrich,
    ):
        await PlaywrightRenderer()._try_enricher_fallback_async(URL, "ask")
    public.assert_not_awaited()
    enrich.assert_not_awaited()


@pytest.mark.parametrize(
    "url, overrides, screenshot, expected",
    [
        (URL, {}, False, "playwright"),
        ("https://twitter.com/user/status/123", {}, False, "playwright"),
        ("https://x.com/user", {}, False, "playwright"),
        (URL, {"remote_consent": "never"}, False, "playwright"),
        (
            URL,
            {"playwright": {"cookies": [{"name": "session", "value": "test"}]}},
            False,
            "playwright",
        ),
        (
            URL,
            {"playwright": {"session_mode": "domain_persistent"}},
            False,
            "playwright",
        ),
        (URL, {}, True, "playwright"),
        (URL, {"policy": {"strategy_priority": ["jina", "playwright"]}}, False, "jina"),
        (
            URL,
            {"domain_profiles": {"x.com": {"prefer_strategy": "static"}}},
            False,
            "static",
        ),
        (URL, {"policy": {"local_only_patterns": ["x.com"]}}, False, "playwright"),
        (URL + "?token=secret", {}, False, "playwright"),
        (URL, {"policy": {"enabled": False}}, False, "static"),
    ],
)
async def test_x_priority_preserves_explicit_preferences_and_local_context(
    url, overrides, screenshot, expected, monkeypatch
):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    def runner_for(strategy):
        runner = MagicMock()
        runner.unavailable_reason.return_value = None
        runner.requires_remote_consent = strategy in {"defuddle", "jina", "cloudflare"}
        runner.fetch = AsyncMock(
            return_value=FetchResult(
                content=(
                    "A complete article describes how browsers retrieve pages over "
                    "secure connections. Responses include status codes and headers. "
                    "Successful documents contain readable text, images, links and "
                    "other useful information. Extraction preserves the main story "
                    "while removing navigation menus and unrelated sidebar elements. "
                ),
                strategy_used=strategy,
            )
        )
        return runner

    with patch("markitai.fetch.get_runner", side_effect=runner_for):
        result = await _fetch_with_fallback(
            url,
            FetchConfig(**overrides),
            screenshot_config=ScreenshotConfig(enabled=screenshot),
        )
    assert result.strategy_used == expected


def test_profile_strategy_override_preserves_browser_tuning_and_false_values():
    from markitai.domain_profiles import BUILTIN_DOMAIN_PROFILES, resolve_domain_profile
    from markitai.fetch_support import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(
            prefer_strategy="defuddle",
            skip_auto_scroll=False,
            reject_resource_patterns=[],
        )
    }
    profile = resolve_domain_profile("x.com", profiles)
    assert profile is not None
    assert (
        profile.wait_for_selector == BUILTIN_DOMAIN_PROFILES["x.com"].wait_for_selector
    )
    overrides = _resolve_playwright_profile_overrides(URL, profiles)
    assert overrides["skip_auto_scroll"] is False
    assert overrides["reject_resource_patterns"] == []
    assert BUILTIN_DOMAIN_PROFILES["x.com"].skip_auto_scroll is True


async def test_authenticated_article_uses_existing_browser_context(browser_page):
    renderer, page, _ = browser_page
    page.goto.side_effect = RuntimeError("browser navigation attempted")
    with (
        patch.object(renderer, "_try_enricher_fallback_async", AsyncMock()) as enrich,
        pytest.raises(RuntimeError, match="browser navigation attempted"),
    ):
        await renderer.fetch(
            "https://x.com/user/article/123",
            cookies=[{"name": "session", "value": "test", "domain": ".x.com"}],
        )
    enrich.assert_not_awaited()


async def test_custom_remote_first_order_retains_browser_profile(
    monkeypatch,
):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    from markitai.fetch_support import _get_playwright_fetch_kwargs

    config = FetchConfig(
        domain_profiles={
            "x.com": DomainProfileConfig(
                extra_wait_ms=250, strategy_priority=["defuddle", "playwright"]
            )
        }
    )
    calls = []

    async def fetch_remote(*_args):
        calls.append("defuddle")
        raise FetchError("reader temporarily unavailable")

    async def fetch_browser(url, ctx):
        calls.append("playwright")
        with patch("markitai.fetch_support._detect_proxy", return_value=""):
            kwargs = _get_playwright_fetch_kwargs(url, ctx.config)
        assert (
            kwargs["wait_for_selector"]
            == 'article[data-tweet-id], [data-testid="tweet"], main article'
        )
        assert kwargs["skip_auto_scroll"] is True
        assert kwargs["extra_wait_ms"] == 250
        return FetchResult(
            content="Recovered tweet containing complete usable text.",
            strategy_used="playwright(fxtwitter)",
        )

    def runner_for(strategy):
        runner = MagicMock()
        runner.unavailable_reason.return_value = None
        runner.requires_remote_consent = strategy == "defuddle"
        runner.fetch = fetch_remote if strategy == "defuddle" else fetch_browser
        return runner

    with patch("markitai.fetch.get_runner", side_effect=runner_for):
        result = await _fetch_with_fallback(URL, config)
    assert calls == ["defuddle", "playwright"]
    assert result.strategy_used == "playwright(fxtwitter)"


@pytest.mark.parametrize("source", ["config", "no_proxy"])
def test_local_only_policy_disables_browser_remote_enrichment(source, monkeypatch):
    from markitai.fetch_support import _get_playwright_fetch_kwargs

    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    config = FetchConfig()
    if source == "config":
        config.policy.local_only_patterns = ["x.com"]
    else:
        monkeypatch.setenv("NO_PROXY", "x.com")
    with patch("markitai.fetch_support._detect_proxy", return_value=""):
        kwargs = _get_playwright_fetch_kwargs(URL, config)
    assert kwargs["remote_consent"] == "never"


@pytest.mark.parametrize(
    "url",
    [URL, "https://twitter.com/user/status/123", "https://x.com/user/article/123"],
)
async def test_anonymous_x_fast_path_never_starts_browser(url):
    renderer = PlaywrightRenderer()
    with (
        patch.object(renderer, "_ensure_browser", AsyncMock()) as browser,
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(
                return_value=(
                    "Complete tweet",
                    {"published": "2026-09-14"},
                    "fxtwitter",
                )
            ),
        ) as enrich,
    ):
        result = await renderer.fetch(url, remote_consent="always")
    browser.assert_not_awaited()
    enrich.assert_awaited_once()
    assert result.metadata["_enricher_source"] == "fxtwitter"
    assert result.metadata["source_frontmatter"]["published"] == "2026-09-14"


async def test_failed_fast_path_does_not_retry_same_remote_calls_after_403(
    browser_page,
):
    renderer, page, context = browser_page
    with (
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("", None, "")),
        ) as enrich,
        pytest.raises(FetchError, match="HTTP 403"),
    ):
        await renderer.fetch(URL, remote_consent="always")
    enrich.assert_awaited_once()
    page.goto.assert_awaited_once()
    page.wait_for_selector.assert_not_awaited()
    context.close.assert_awaited_once()


async def test_auto_accepts_short_verified_tweet_without_falling_back():
    from markitai.fetch_strategies.playwright import PlaywrightRunner

    with (
        patch.object(PlaywrightRunner, "unavailable_reason", return_value=None),
        patch.object(
            PlaywrightRunner,
            "fetch",
            AsyncMock(
                return_value=FetchResult(
                    content="你好",
                    strategy_used="playwright(fxtwitter)",
                    metadata={"_enricher_source": "fxtwitter"},
                )
            ),
        ),
    ):
        result = await _fetch_with_fallback(URL, FetchConfig())
    assert result.content == "你好"
    assert result.strategy_used == "playwright(fxtwitter)"
    assert result.metadata["policy_order"][0] == "playwright"


async def test_a_refused_page_that_never_answers_is_not_waited_for(
    browser_page, tmp_path
):
    import asyncio

    renderer, page, _ = browser_page

    async def hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    page.evaluate = AsyncMock(side_effect=hang)
    screenshot = tmp_path / "page.png"
    with (
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("Full tweet content", {"title": "T"}, "fxtwitter")),
        ),
        patch(
            "markitai.fetch_playwright._capture_screenshot",
            AsyncMock(return_value=(screenshot, [screenshot])),
        ),
        patch("markitai.fetch_playwright._wait_until_settled", AsyncMock()),
    ):
        result = await asyncio.wait_for(
            renderer.fetch(
                URL,
                timeout=200,
                screenshot_config=ScreenshotConfig(enabled=True),
                output_dir=tmp_path,
            ),
            5,
        )
    assert result.content == "Full tweet content"
    assert result.html is None


def _browser_with(page):
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()
    browser.new_context.return_value = context
    return browser, context


def test_direct_capture_is_the_default():
    import markitai.fetch_playwright as fp

    # conftest pins X_CAPTURE to "embed" for tests that mock one browser
    assert fp.X_CAPTURE_DEFAULT == "direct"


async def test_an_x_post_loads_in_the_new_headless_mode_without_headless_ua(
    monkeypatch,
):
    """X_CAPTURE "direct": x.com itself, in Chromium's new headless mode."""
    import markitai.fetch_playwright as fp

    monkeypatch.setattr(fp, "X_CAPTURE", "direct")
    page = AsyncMock()
    page.url = URL
    page.goto.return_value = MagicMock(status=200)
    page.title.return_value = "Post"
    page.content.return_value = "<main><article><p>Post text</p></article></main>"
    direct_browser, direct_context = _browser_with(page)
    default_browser, _ = _browser_with(page)
    agent = "Mozilla/5.0 (X11; Linux aarch64) Chrome/141.0.0.0 Safari/537.36"
    renderer = PlaywrightRenderer()
    with (
        patch.object(
            renderer, "_ensure_browser", AsyncMock(return_value=default_browser)
        ),
        patch.object(
            renderer,
            "_ensure_new_headless",
            AsyncMock(return_value=(direct_browser, agent)),
        ),
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("Full tweet content", {"title": "T"}, "fxtwitter")),
        ),
        patch("markitai.fetch_playwright._wait_until_settled", AsyncMock()),
    ):
        result = await renderer.fetch(URL, extra_wait_ms=0)
    direct_browser.new_context.assert_awaited_once()
    assert direct_browser.new_context.await_args.kwargs["user_agent"] == agent
    default_browser.new_context.assert_not_awaited()
    assert "page_source" not in result.metadata  # the page is x.com's own
    direct_context.close.assert_awaited_once()


@pytest.mark.parametrize(
    ("capture", "url"),
    [("embed", URL), ("direct", "https://example.com/post/123")],
)
async def test_other_pages_and_the_embed_setting_use_the_default_browser(
    monkeypatch, capture, url
):
    import markitai.fetch_playwright as fp

    monkeypatch.setattr(fp, "X_CAPTURE", capture)
    page = AsyncMock()
    page.url = url
    page.goto.return_value = MagicMock(status=200)
    page.title.return_value = "Page"
    page.content.return_value = "<article><p>" + "Plain words. " * 40 + "</p></article>"
    default_browser, _ = _browser_with(page)
    renderer = PlaywrightRenderer()
    new_headless = AsyncMock()
    with (
        patch.object(
            renderer, "_ensure_browser", AsyncMock(return_value=default_browser)
        ),
        patch.object(renderer, "_ensure_new_headless", new_headless),
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("Full tweet content", {"title": "T"}, "fxtwitter")),
        ),
    ):
        await renderer.fetch(url, extra_wait_ms=0)
    new_headless.assert_not_awaited()
    default_browser.new_context.assert_awaited_once()


async def test_a_build_without_the_new_headless_mode_is_tried_once():
    renderer = PlaywrightRenderer()
    playwright = MagicMock()
    playwright.chromium.launch = AsyncMock(side_effect=RuntimeError("no chromium"))
    renderer._playwright = playwright
    assert await renderer._ensure_new_headless() is None
    assert await renderer._ensure_new_headless() is None
    playwright.chromium.launch.assert_awaited_once()
    launch = playwright.chromium.launch.await_args
    assert launch is not None and launch.kwargs["channel"] == "chromium"


async def test_the_new_headless_agent_drops_the_headless_word():
    renderer = PlaywrightRenderer()
    page = AsyncMock()
    page.evaluate.return_value = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "HeadlessChrome/141.0.0.0 Safari/537.36"
    )
    browser = AsyncMock()
    browser.new_page.return_value = page
    playwright = MagicMock()
    playwright.chromium.launch = AsyncMock(return_value=browser)
    playwright.stop = AsyncMock()
    renderer._playwright = playwright
    direct = await renderer._ensure_new_headless()
    assert direct is not None
    assert direct[0] is browser
    assert "HeadlessChrome" not in direct[1] and "Chrome/141.0.0.0" in direct[1]
    await renderer.close()
    browser.close.assert_awaited_once()
