"""Tests for graceful fallbacks and actionable errors in the fetch layer.

Covers:
- service-side refusal classification for explicitly-selected remote
  strategies (jina/defuddle/cloudflare)
- fallback to the auto strategy chain (non-interactive auto-fallback,
  interactive prompt-once semantics, decline path)
- Jina error-body message extraction (no raw JSON dumps)
- actionable playwright missing-package / missing-browser errors
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai import fetch
from markitai.config import FetchConfig
from markitai.fetch import (
    FetchError,
    FetchResult,
    FetchStrategy,
    JinaAPIError,
    JinaRateLimitError,
    _classify_service_refusal,
    _extract_jina_error_message,
    fetch_url,
)

JINA_451_BODY = (
    '{"code":451,"name":"UnavailableForLegalReasonsError","status":45102,'
    '"message":"Anonymous access to domain github.com blocked until 2026",'
    '"readableMessage":"Anonymous access to domain github.com blocked until 2026"}'
)


def _make_response(status_code: int, text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


class TestExtractJinaErrorMessage:
    def test_extracts_readable_message_from_json_body(self) -> None:
        msg = _extract_jina_error_message(_make_response(451, JINA_451_BODY))
        assert msg == "Anonymous access to domain github.com blocked until 2026"
        assert "{" not in msg  # no raw JSON

    def test_falls_back_to_raw_text_for_non_json(self) -> None:
        msg = _extract_jina_error_message(_make_response(403, "Forbidden"))
        assert msg == "Forbidden"

    def test_truncates_long_raw_text(self) -> None:
        msg = _extract_jina_error_message(_make_response(400, "x" * 500))
        assert len(msg) == 200


class TestClassifyServiceRefusal:
    def test_jina_4xx_is_refusal(self) -> None:
        error = JinaAPIError(451, "Anonymous access to domain github.com blocked")
        reason = _classify_service_refusal(FetchStrategy.JINA, error)
        assert reason is not None
        assert "451" in reason
        assert "github.com blocked" in reason

    def test_jina_5xx_is_not_refusal(self) -> None:
        error = JinaAPIError(502, "Bad gateway")
        assert _classify_service_refusal(FetchStrategy.JINA, error) is None

    def test_jina_rate_limit_is_refusal(self) -> None:
        reason = _classify_service_refusal(FetchStrategy.JINA, JinaRateLimitError())
        assert reason is not None
        assert "rate limited" in reason

    def test_defuddle_rate_limit_is_refusal(self) -> None:
        error = FetchError("Defuddle rate limit exceeded. Try again later")
        reason = _classify_service_refusal(FetchStrategy.DEFUDDLE, error)
        assert reason == "rate limited"

    def test_defuddle_4xx_is_refusal(self) -> None:
        error = FetchError("Defuddle API returned HTTP 403: blocked")
        assert _classify_service_refusal(FetchStrategy.DEFUDDLE, error) is not None

    def test_cloudflare_credentials_error_is_not_refusal(self) -> None:
        """Missing local credentials must surface the actionable error as-is."""
        from markitai.utils.guidance import cloudflare_credentials_error

        error = FetchError(cloudflare_credentials_error())
        assert _classify_service_refusal(FetchStrategy.CLOUDFLARE, error) is None

    def test_cloudflare_rate_limit_is_refusal(self) -> None:
        error = FetchError("CF BR rate limit exceeded after 3 retries: u")
        reason = _classify_service_refusal(FetchStrategy.CLOUDFLARE, error)
        assert reason == "rate limited"

    def test_network_error_is_not_refusal(self) -> None:
        error = FetchError("Jina Reader request timed out after 30s: u")
        assert _classify_service_refusal(FetchStrategy.JINA, error) is None

    def test_non_fetch_error_is_not_refusal(self) -> None:
        assert _classify_service_refusal(FetchStrategy.JINA, ValueError("x")) is None


class TestExplicitStrategyFallback:
    """fetch_url falls back to the auto chain after a service refusal."""

    @pytest.mark.asyncio
    async def test_non_interactive_auto_fallback(self, monkeypatch) -> None:
        """Non-interactive: fall back automatically to the auto chain."""
        config = FetchConfig()
        auto_result = FetchResult(content="# ok", strategy_used="static")

        monkeypatch.setattr(
            "markitai.fetch_strategies.jina.fetch_with_jina",
            AsyncMock(side_effect=JinaAPIError(451, "Anonymous access blocked")),
        )
        fallback_mock = AsyncMock(return_value=auto_result)
        monkeypatch.setattr(fetch, "_fetch_with_fallback", fallback_mock)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            result = await fetch_url(
                "https://github.com/foo/bar",
                FetchStrategy.JINA,
                config,
                explicit_strategy=True,
            )

        assert result.strategy_used == "static"
        fallback_mock.assert_awaited_once()
        # Decision is remembered for the rest of the run
        assert fetch.get_default_session().consent.explicit_fallback_decision is True

    @pytest.mark.asyncio
    async def test_non_refusal_error_propagates_unchanged(self, monkeypatch) -> None:
        """5xx / network errors from explicit strategies are not intercepted."""
        config = FetchConfig()
        monkeypatch.setattr(
            "markitai.fetch_strategies.jina.fetch_with_jina",
            AsyncMock(side_effect=JinaAPIError(502, "Bad gateway")),
        )
        fallback_mock = AsyncMock()
        monkeypatch.setattr(fetch, "_fetch_with_fallback", fallback_mock)

        with pytest.raises(JinaAPIError):
            await fetch_url(
                "https://example.com",
                FetchStrategy.JINA,
                config,
                explicit_strategy=True,
            )
        fallback_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interactive_prompt_once_per_run(self, monkeypatch) -> None:
        """Interactive TTY prompts at most once; the answer is cached."""
        config = FetchConfig()
        auto_result = FetchResult(content="# ok", strategy_used="static")

        monkeypatch.setattr(
            "markitai.fetch_strategies.jina.fetch_with_jina",
            AsyncMock(side_effect=JinaAPIError(451, "Anonymous access blocked")),
        )
        monkeypatch.setattr(
            fetch, "_fetch_with_fallback", AsyncMock(return_value=auto_result)
        )

        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm", return_value=True) as mock_confirm,
        ):
            mock_stdin.isatty.return_value = True
            for _ in range(2):  # two URLs in one run
                await fetch_url(
                    "https://github.com/foo/bar",
                    FetchStrategy.JINA,
                    config,
                    explicit_strategy=True,
                )

        assert mock_confirm.call_count == 1
        prompt = mock_confirm.call_args[0][0]
        assert "jina cannot fetch this URL" in prompt
        assert "Anonymous access blocked" in prompt
        assert "auto strategy chain" in prompt

    @pytest.mark.asyncio
    async def test_interactive_decline_raises_actionable_error(
        self, monkeypatch
    ) -> None:
        """Declining the fallback yields an actionable error, not raw JSON."""
        config = FetchConfig()
        monkeypatch.setattr(
            "markitai.fetch_strategies.jina.fetch_with_jina",
            AsyncMock(
                side_effect=JinaAPIError(
                    451, "Anonymous access to domain github.com blocked"
                )
            ),
        )
        fallback_mock = AsyncMock()
        monkeypatch.setattr(fetch, "_fetch_with_fallback", fallback_mock)

        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm", return_value=False),
            pytest.raises(FetchError) as exc_info,
        ):
            mock_stdin.isatty.return_value = True
            await fetch_url(
                "https://github.com/foo/bar",
                FetchStrategy.JINA,
                config,
                explicit_strategy=True,
            )

        message = str(exc_info.value)
        assert "jina cannot fetch this URL" in message
        assert "github.com blocked" in message
        # Guidance: an API key lifts the anonymous block
        assert "markitai config set fetch.jina.api_key <key>" in message
        assert "JINA_API_KEY" in message
        assert "Config file:" in message
        fallback_mock.assert_not_awaited()
        # Decline is remembered for the rest of the run
        assert fetch.get_default_session().consent.explicit_fallback_decision is False

    @pytest.mark.asyncio
    async def test_fallback_failure_reports_both_errors(self, monkeypatch) -> None:
        """If the auto fallback also fails, both failures show compactly."""
        config = FetchConfig()
        monkeypatch.setattr(
            "markitai.fetch_strategies.jina.fetch_with_jina",
            AsyncMock(
                side_effect=JinaAPIError(
                    451, "Anonymous access to domain github.com blocked"
                )
            ),
        )
        monkeypatch.setattr(
            fetch,
            "_fetch_with_fallback",
            AsyncMock(side_effect=FetchError("All fetch strategies failed")),
        )

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            with pytest.raises(FetchError) as exc_info:
                await fetch_url(
                    "https://github.com/foo/bar",
                    FetchStrategy.JINA,
                    config,
                    explicit_strategy=True,
                )

        message = str(exc_info.value)
        assert "jina: HTTP 451" in message
        assert "auto fallback: All fetch strategies failed" in message
        assert "JINA_API_KEY" in message

    @pytest.mark.asyncio
    async def test_auto_strategy_never_triggers_fallback_wrapper(
        self, monkeypatch
    ) -> None:
        """AUTO-chain failures keep the existing combined error format."""
        config = FetchConfig()
        monkeypatch.setattr(
            fetch,
            "_fetch_with_fallback",
            AsyncMock(side_effect=FetchError("All fetch strategies failed for u")),
        )
        with pytest.raises(FetchError, match="All fetch strategies failed"):
            await fetch_url("https://example.com", FetchStrategy.AUTO, config)


class TestPlaywrightActionableErrors:
    """Explicit -s playwright with missing package/browser is actionable."""

    @pytest.mark.asyncio
    async def test_missing_browser_raises_actionable_error(self, monkeypatch) -> None:
        import markitai.fetch_playwright as fp

        config = FetchConfig()
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: False
        )

        with pytest.raises(FetchError) as exc_info:
            await fetch_url(
                "https://example.com",
                FetchStrategy.PLAYWRIGHT,
                config,
                explicit_strategy=True,
                renderer=object(),  # skip global renderer initialization
            )

        message = str(exc_info.value)
        assert "Chromium browser is missing" in message
        assert "markitai doctor --fix" in message
        assert "playwright install chromium" in message
        assert "-m playwright install chromium" in message

    @pytest.mark.asyncio
    async def test_missing_package_raises_actionable_error(self, monkeypatch) -> None:
        import markitai.fetch_playwright as fp

        config = FetchConfig()
        monkeypatch.setattr(fp, "is_playwright_available", lambda: False)

        with pytest.raises(FetchError) as exc_info:
            await fetch_url(
                "https://example.com",
                FetchStrategy.PLAYWRIGHT,
                config,
                explicit_strategy=True,
                renderer=object(),
            )

        message = str(exc_info.value)
        assert "'playwright' package" in message
        assert "pip install 'markitai[browser]'" in message
        assert "uv tool install --force 'markitai[all]'" in message

    @pytest.mark.asyncio
    async def test_auto_chain_skips_playwright_when_browser_missing(
        self, monkeypatch
    ) -> None:
        """AUTO keeps silently skipping playwright (debug log only)."""
        import markitai.fetch_playwright as fp
        from markitai.fetch import _fetch_with_fallback

        config = FetchConfig(remote_consent="never")
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: False
        )
        pw_mock = AsyncMock()
        monkeypatch.setattr(fp, "fetch_with_playwright", pw_mock)
        monkeypatch.setattr(
            "markitai.fetch_strategies.static.fetch_with_static",
            AsyncMock(side_effect=FetchError("HTTP 500 fetching URL")),
        )

        with pytest.raises(FetchError, match="All fetch strategies failed"):
            await _fetch_with_fallback("https://example.com", config)

        pw_mock.assert_not_awaited()


class TestExplicitStrategyCaptchaDetection:
    """Explicitly-chosen strategies (unlike AUTO) never validated content
    before returning it, so an anti-bot/CAPTCHA challenge page came back
    looking like a successful fetch. Regression: reported after a bilibili
    opus fetch returned a Geetest challenge page (title "验证码_...") as if
    it were the article — `fetch_strategy: playwright` with no indication
    anything was wrong."""

    @pytest.mark.asyncio
    async def test_explicit_playwright_raises_on_captcha_content(
        self, monkeypatch
    ) -> None:
        import markitai.fetch_playwright as fp

        config = FetchConfig()
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        pw_result = MagicMock()
        pw_result.content = "智能验证检测中\n\n由极验提供技术支持\n\n请在下图依次点击："
        pw_result.title = "验证码_哔哩哔哩"
        pw_result.final_url = "https://example.com"
        pw_result.metadata = {}
        pw_result.screenshot_path = None
        monkeypatch.setattr(
            fp, "fetch_with_playwright", AsyncMock(return_value=pw_result)
        )

        with pytest.raises(FetchError) as exc_info:
            await fetch_url(
                "https://example.com",
                FetchStrategy.PLAYWRIGHT,
                config,
                explicit_strategy=True,
                renderer=object(),
            )

        message = str(exc_info.value)
        assert "captcha_geetest" in message
        assert "anti-bot" in message.lower() or "captcha" in message.lower()

    @pytest.mark.asyncio
    async def test_explicit_playwright_returns_normally_for_real_content(
        self, monkeypatch
    ) -> None:
        """Regression guard: the new check must not false-positive on
        ordinary content."""
        import markitai.fetch_playwright as fp

        config = FetchConfig()
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        pw_result = MagicMock()
        pw_result.content = (
            "# Real Article\n\nThis is a perfectly normal article with "
            "plenty of real content to read and enjoy."
        )
        pw_result.title = "Real Article"
        pw_result.final_url = "https://example.com"
        pw_result.metadata = {}
        pw_result.screenshot_path = None
        monkeypatch.setattr(
            fp, "fetch_with_playwright", AsyncMock(return_value=pw_result)
        )

        result = await fetch_url(
            "https://example.com",
            FetchStrategy.PLAYWRIGHT,
            config,
            explicit_strategy=True,
            renderer=object(),
        )

        assert "Real Article" in result.content

    @pytest.mark.asyncio
    async def test_explicit_playwright_does_not_raise_for_non_captcha_reasons(
        self, monkeypatch
    ) -> None:
        """Non-CAPTCHA _is_invalid_content reasons (too_short, login_required,
        ...) stay non-fatal for explicit strategies — this check is scoped
        to anti-bot detection specifically, not a general content-quality
        gate for explicitly-chosen strategies (that would be a larger,
        separate behavior change with its own regression risk for
        legitimately short/edge-case pages)."""
        import markitai.fetch_playwright as fp

        config = FetchConfig()
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        pw_result = MagicMock()
        pw_result.content = "You must be logged in to view this page."
        pw_result.title = "Login required"
        pw_result.final_url = "https://example.com"
        pw_result.metadata = {}
        pw_result.screenshot_path = None
        monkeypatch.setattr(
            fp, "fetch_with_playwright", AsyncMock(return_value=pw_result)
        )

        result = await fetch_url(
            "https://example.com",
            FetchStrategy.PLAYWRIGHT,
            config,
            explicit_strategy=True,
            renderer=object(),
        )

        assert "logged in" in result.content


class TestAutoChainFxTwitterIntercept:
    """AUTO chain uses playwright with oEmbed enricher for tweet URLs.

    The enricher attempts FxTwitter API first, falling back to X oEmbed.
    When remote_consent is 'never', the enricher is skipped and DOM parsing
    is used directly.
    """

    @pytest.mark.asyncio
    async def test_tweet_url_uses_enricher_when_consent_allowed(
        self, monkeypatch
    ) -> None:
        """When consent is allowed, playwright uses oEmbed enricher."""
        import markitai.fetch_playwright as fp
        from markitai.fetch import _fetch_with_fallback

        url = "https://x.com/dotey/status/2073286406558949828"
        config = FetchConfig(remote_consent="always")

        # Static fails so the chain advances to playwright.
        monkeypatch.setattr(
            "markitai.fetch_strategies.static.fetch_with_static",
            AsyncMock(side_effect=FetchError("HTTP 403 fetching URL")),
        )

        # Mock playwright to succeed with enricher
        pw_result = MagicMock()
        pw_result.content = "Enriched content from oEmbed enricher. This is a tweet body text with sufficient length to pass content quality validation checks in the extraction pipeline."
        pw_result.title = "Post by @dotey"
        pw_result.final_url = url
        pw_result.metadata = {"source_frontmatter": {"title": "Post by @dotey"}}
        pw_result.screenshot_path = None

        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        pw_mock = AsyncMock(return_value=pw_result)
        monkeypatch.setattr(fp, "fetch_with_playwright", pw_mock)

        result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "playwright"
        pw_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tweet_url_falls_back_to_dom_when_consent_denied(
        self, monkeypatch
    ) -> None:
        """When consent is 'never', playwright DOM parsing is used directly."""
        import markitai.fetch_playwright as fp
        from markitai.fetch import _fetch_with_fallback

        url = "https://x.com/dotey/status/2073286406558949828"
        config = FetchConfig(remote_consent="never")

        # Static fails so the chain advances to playwright.
        monkeypatch.setattr(
            "markitai.fetch_strategies.static.fetch_with_static",
            AsyncMock(side_effect=FetchError("HTTP 403 fetching URL")),
        )

        pw_result = MagicMock()
        pw_result.content = "This is a DOM-extracted tweet body text with enough content to pass the quality validation threshold. The tweet discusses interesting topics."
        pw_result.title = "Post by @dotey"
        pw_result.final_url = url
        pw_result.metadata = {}
        pw_result.screenshot_path = None
        pw_mock = AsyncMock(return_value=pw_result)
        monkeypatch.setattr(fp, "fetch_with_playwright", pw_mock)
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )

        result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "playwright"
        pw_mock.assert_awaited_once()


class TestCloudflareActionableErrors:
    @pytest.mark.asyncio
    async def test_explicit_cloudflare_without_credentials(self) -> None:
        """-s cloudflare without credentials shows the actionable block."""
        config = FetchConfig()  # no cloudflare credentials

        with (
            patch(
                "markitai.config.CloudflareConfig.get_resolved_api_token",
                return_value=None,
            ),
            patch(
                "markitai.config.CloudflareConfig.get_resolved_account_id",
                return_value=None,
            ),
            pytest.raises(FetchError) as exc_info,
        ):
            await fetch_url(
                "https://example.com",
                FetchStrategy.CLOUDFLARE,
                config,
                explicit_strategy=True,
            )

        message = str(exc_info.value)
        assert "Cloudflare API token and account ID required" in message
        assert "https://dash.cloudflare.com/profile/api-tokens" in message
        assert "markitai config set fetch.cloudflare.api_token <token>" in message
        assert "CLOUDFLARE_API_TOKEN" in message
        assert "Config file:" in message

    @pytest.mark.asyncio
    async def test_converter_without_credentials(self) -> None:
        """File-convert entry point shows the same actionable block."""
        from pathlib import Path

        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token=None, account_id=None)
        with pytest.raises(RuntimeError) as exc_info:
            await converter.convert_async(Path("report.pdf"))

        message = str(exc_info.value)
        assert "Cloudflare API token and account ID required" in message
        assert "markitai config set fetch.cloudflare.account_id <account-id>" in message
        assert "Config file:" in message
