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
            fetch,
            "fetch_with_jina",
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
        assert fetch._explicit_fallback_decision is True

    @pytest.mark.asyncio
    async def test_non_refusal_error_propagates_unchanged(self, monkeypatch) -> None:
        """5xx / network errors from explicit strategies are not intercepted."""
        config = FetchConfig()
        monkeypatch.setattr(
            fetch,
            "fetch_with_jina",
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
            fetch,
            "fetch_with_jina",
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
            fetch,
            "fetch_with_jina",
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
        assert fetch._explicit_fallback_decision is False

    @pytest.mark.asyncio
    async def test_fallback_failure_reports_both_errors(self, monkeypatch) -> None:
        """If the auto fallback also fails, both failures show compactly."""
        config = FetchConfig()
        monkeypatch.setattr(
            fetch,
            "fetch_with_jina",
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
            fetch,
            "fetch_with_static",
            AsyncMock(side_effect=FetchError("HTTP 500 fetching URL")),
        )

        with pytest.raises(FetchError, match="All fetch strategies failed"):
            await _fetch_with_fallback("https://example.com", config)

        pw_mock.assert_not_awaited()


class TestAutoChainFxTwitterIntercept:
    """AUTO chain must serve tweet URLs via FxTwitter before any browser."""

    @pytest.mark.asyncio
    async def test_tweet_url_served_by_fxtwitter_without_browser_launch(
        self, monkeypatch
    ) -> None:
        import markitai.fetch_fxtwitter as fx
        import markitai.fetch_playwright as fp
        from markitai.fetch import _fetch_with_fallback

        url = "https://x.com/dotey/status/2073286406558949828"
        config = FetchConfig(remote_consent="never")

        fx_result = FetchResult(
            content="**tweet body**",
            strategy_used="fxtwitter",
            title="Post by @dotey",
            url=url,
            final_url=url,
        )
        fx_mock = AsyncMock(return_value=fx_result)
        monkeypatch.setattr(fx, "fetch_with_fxtwitter", fx_mock)

        # Playwright is fully available but must never be touched.
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        pw_mock = AsyncMock()
        monkeypatch.setattr(fp, "fetch_with_playwright", pw_mock)

        # Static fails so the chain advances to the playwright branch.
        monkeypatch.setattr(
            fetch,
            "fetch_with_static",
            AsyncMock(side_effect=FetchError("HTTP 403 fetching URL")),
        )

        result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "fxtwitter"
        assert result.content == "**tweet body**"
        fx_mock.assert_awaited_once_with(url)
        pw_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fxtwitter_miss_falls_back_to_playwright_dom_path(
        self, monkeypatch
    ) -> None:
        import markitai.fetch_fxtwitter as fx
        import markitai.fetch_playwright as fp
        from markitai.fetch import _fetch_with_fallback

        url = "https://x.com/dotey/status/2073286406558949828"
        config = FetchConfig(remote_consent="never")

        # FxTwitter down/rate-limited -> returns None (never raises).
        monkeypatch.setattr(fx, "fetch_with_fxtwitter", AsyncMock(return_value=None))
        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        pw_result = MagicMock()
        pw_result.content = "# Post by @dotey\n\nDOM-extracted tweet body text."
        pw_result.title = "Post by @dotey"
        pw_result.final_url = url
        pw_result.metadata = {}
        pw_result.screenshot_path = None
        pw_mock = AsyncMock(return_value=pw_result)
        monkeypatch.setattr(fp, "fetch_with_playwright", pw_mock)
        monkeypatch.setattr(
            fetch,
            "fetch_with_static",
            AsyncMock(side_effect=FetchError("HTTP 403 fetching URL")),
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
