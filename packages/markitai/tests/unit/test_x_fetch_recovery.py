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
    with patch.object(renderer, "_ensure_browser", AsyncMock(return_value=browser)):
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
    page.wait_for_selector.assert_not_awaited()
    context.close.assert_awaited_once()


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
        patch("markitai.fetch_playwright.extract_web_content", None),
        patch(
            "markitai.fetch_playwright._html_to_markdown",
            return_value="Real content " * 20,
        ),
        patch.object(
            renderer,
            "_try_enricher_fallback_async",
            AsyncMock(return_value=("", None, "")),
        ),
        patch("markitai.fetch_playwright.asyncio.sleep", AsyncMock()) as sleep,
    ):
        result = await renderer.fetch(
            URL, wait_for_selector="article", extra_wait_ms=500, skip_auto_scroll=True
        )
    assert "Real content" in result.content
    page.wait_for_selector.assert_awaited_once_with("article", timeout=10000)
    if selector_found:
        sleep.assert_awaited_once_with(0.5)
    else:
        sleep.assert_not_awaited()


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
            == 'article[data-tweet-id], [data-testid="tweet"]'
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
