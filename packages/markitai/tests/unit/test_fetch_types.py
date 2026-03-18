"""Tests for fetch_types module — shared types extracted from fetch.py."""

from __future__ import annotations


class TestFetchTypesImport:
    """Verify all shared types are importable from fetch_types."""

    def test_fetch_strategy_enum_members(self):
        from markitai.fetch_types import FetchStrategy

        assert FetchStrategy.AUTO.value == "auto"
        assert FetchStrategy.STATIC.value == "static"
        assert FetchStrategy.DEFUDDLE.value == "defuddle"
        assert FetchStrategy.PLAYWRIGHT.value == "playwright"
        assert FetchStrategy.CLOUDFLARE.value == "cloudflare"
        assert FetchStrategy.JINA.value == "jina"

    def test_fetch_error_hierarchy(self):
        from markitai.fetch_types import (
            FetchError,
            JinaAPIError,
            JinaRateLimitError,
        )

        assert issubclass(JinaRateLimitError, FetchError)
        assert issubclass(JinaAPIError, FetchError)
        assert issubclass(FetchError, Exception)

    def test_jina_rate_limit_error_message(self):
        from markitai.fetch_types import JinaRateLimitError

        err = JinaRateLimitError()
        assert "rate limit" in str(err).lower()

    def test_jina_api_error_preserves_status_code(self):
        from markitai.fetch_types import JinaAPIError

        err = JinaAPIError(429, "too many requests")
        assert err.status_code == 429
        assert "429" in str(err)

    def test_fetch_result_dataclass(self):
        from markitai.fetch_types import FetchResult

        result = FetchResult(content="# Hello", strategy_used="static")
        assert result.content == "# Hello"
        assert result.strategy_used == "static"
        assert result.title is None
        assert result.cache_hit is False
        assert result.screenshot_path is None
        assert result.static_content is None
        assert result.browser_content is None

    def test_conditional_fetch_result_dataclass(self):
        from markitai.fetch_types import ConditionalFetchResult, FetchResult

        result = FetchResult(content="test", strategy_used="static")
        cfr = ConditionalFetchResult(result=result, not_modified=False, etag="abc")
        assert cfr.result is result
        assert cfr.not_modified is False
        assert cfr.etag == "abc"
        assert cfr.last_modified is None

    def test_critical_invalid_reasons_constant(self):
        from markitai.fetch_types import CRITICAL_INVALID_REASONS

        assert isinstance(CRITICAL_INVALID_REASONS, set)
        assert "javascript_required" in CRITICAL_INVALID_REASONS
        assert "login_required" in CRITICAL_INVALID_REASONS


class TestBackwardCompatibility:
    """Verify types are still importable from markitai.fetch."""

    def test_fetch_strategy_from_fetch(self):
        from markitai.fetch import FetchStrategy

        assert FetchStrategy.AUTO.value == "auto"

    def test_fetch_result_from_fetch(self):
        from markitai.fetch import FetchResult

        r = FetchResult(content="x", strategy_used="s")
        assert r.content == "x"

    def test_fetch_error_from_fetch(self):
        from markitai.fetch import FetchError, JinaAPIError, JinaRateLimitError

        assert issubclass(JinaRateLimitError, FetchError)
        assert issubclass(JinaAPIError, FetchError)

    def test_conditional_fetch_result_from_fetch(self):
        from markitai.fetch import ConditionalFetchResult

        cfr = ConditionalFetchResult(result=None, not_modified=True)
        assert cfr.not_modified is True

    def test_critical_invalid_reasons_from_fetch(self):
        from markitai.fetch import CRITICAL_INVALID_REASONS

        assert "login_required" in CRITICAL_INVALID_REASONS
