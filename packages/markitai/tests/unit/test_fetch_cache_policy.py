"""URL fetch-cache policy: TTL, --no-cache-for, revalidation checks, CLI.

Regression tests for E2E findings: pages without validators were reused
forever, ``--no-cache-for`` never matched URLs, a conditional 200 that
turned into a Cloudflare challenge or JS shell bypassed every AUTO check
and overwrote the good cache entry, and ``markitai cache stats/clear``
ignored fetch_cache.db.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.config import FetchConfig
from markitai.fetch import fetch_url
from markitai.fetch_cache import FetchCache, url_matches_cache_patterns
from markitai.fetch_types import (
    ConditionalFetchResult,
    FetchError,
    FetchResult,
    FetchStrategy,
)

GOOD = "# Stable article\n\n" + ("A real paragraph with plenty of words. " * 20)
CHALLENGE = (
    "# Just a moment...\n\nChecking your browser before accessing example.com.\n\n"
    "Ray ID: 8a1b2c3d4e5f"
)


def _age_entry(cache: FetchCache, seconds: int) -> None:
    """Pretend every cache row was written ``seconds`` ago."""
    conn = cache._get_connection()
    conn.execute("UPDATE fetch_cache SET created_at = ?", (int(time.time()) - seconds,))
    conn.commit()


class TestFetchCacheTtl:
    def test_unvalidated_entry_expires(self, tmp_path: Path) -> None:
        cache = FetchCache(tmp_path / "fetch_cache.db")
        url = "https://example.com/a"
        cache.set(url, FetchResult(content=GOOD, strategy_used="static", url=url))
        assert cache.get(url, max_age_seconds=3600) is not None
        _age_entry(cache, 7200)
        assert cache.get(url, max_age_seconds=3600) is None
        assert cache.get(url) is not None  # no TTL requested: still there
        assert cache.get_with_validators(url, max_age_seconds=3600) == (
            None,
            None,
            None,
        )

    def test_validated_entry_is_kept_for_revalidation(self, tmp_path: Path) -> None:
        cache = FetchCache(tmp_path / "fetch_cache.db")
        url = "https://example.com/a"
        cache.set_with_validators(
            url, FetchResult(content=GOOD, strategy_used="static", url=url), '"v1"'
        )
        _age_entry(cache, 10**6)
        result, etag, _ = cache.get_with_validators(url, max_age_seconds=60)
        assert result is not None
        assert etag == '"v1"'

    def test_zero_ttl_never_reuses_unvalidated_entries(self, tmp_path: Path) -> None:
        cache = FetchCache(tmp_path / "fetch_cache.db")
        url = "https://example.com/a"
        cache.set(url, FetchResult(content=GOOD, strategy_used="static", url=url))
        assert cache.get(url, max_age_seconds=0) is None


class TestNoCachePatterns:
    @pytest.mark.parametrize(
        "pattern",
        [
            "https://example.com/*",
            "example.com/docs/*",
            "example.com",
            "*.example.com",
            "page.html",
            "*.html",
            "**/docs/*",
        ],
    )
    def test_patterns_match_urls(self, pattern: str) -> None:
        url = (
            "https://www.example.com/docs/page.html"
            if pattern == "*.example.com"
            else "https://example.com/docs/page.html"
        )
        assert url_matches_cache_patterns(url, [pattern])

    @pytest.mark.parametrize(
        ("url", "pattern"),
        [
            ("https://Example.COM/docs/page", "example.com/docs/*"),
            ("https://example.com/docs/page", "https://Example.com/*"),
            ("HTTPS://example.com/a", "https://example.com/*"),
            ("https://example.com/a", "Example.com"),
            ("https://WWW.Example.com/a", "*.example.com"),
        ],
    )
    def test_scheme_and_host_match_case_insensitively(
        self, url: str, pattern: str
    ) -> None:
        assert url_matches_cache_patterns(url, [pattern])

    def test_path_still_matches_case_sensitively(self) -> None:
        assert not url_matches_cache_patterns(
            "https://example.com/Docs/page", ["example.com/docs/*"]
        )
        assert not url_matches_cache_patterns(
            "https://example.com/report.PDF", ["*.pdf"]
        )

    def test_non_matching_pattern(self) -> None:
        assert not url_matches_cache_patterns(
            "https://example.com/docs/page.html", ["*.pdf", "other.org"]
        )
        assert not url_matches_cache_patterns("https://example.com/", [])


def _cached(url: str, content: str = GOOD) -> FetchResult:
    return FetchResult(content=content, strategy_used="static", url=url)


class TestFetchUrlCachePolicy:
    @pytest.mark.asyncio
    async def test_expired_entry_is_refetched(self, tmp_path: Path) -> None:
        url = "https://example.com/a"
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set(url, _cached(url, "# Old\n\n" + "stale words here " * 20))
        _age_entry(cache, 3 * 24 * 3600)
        fresh = _cached(url)
        with patch(
            "markitai.fetch._dispatch_strategy",
            AsyncMock(return_value=(fresh, None)),
        ) as dispatch:
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                cache=cache,
                cache_ttl_seconds=24 * 3600,
            )
        dispatch.assert_awaited_once()
        assert result.content == GOOD

    @pytest.mark.asyncio
    async def test_fresh_entry_is_reused(self, tmp_path: Path) -> None:
        url = "https://example.com/a"
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set(url, _cached(url))
        with patch("markitai.fetch._dispatch_strategy", AsyncMock()) as dispatch:
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                cache=cache,
                cache_ttl_seconds=3600,
            )
        dispatch.assert_not_awaited()
        assert result.cache_hit

    @pytest.mark.asyncio
    async def test_session_policy_applies_when_caller_passes_none(
        self, tmp_path: Path
    ) -> None:
        from markitai.fetch_session import get_default_session

        url = "https://example.com/a"
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set(url, _cached(url))
        session = get_default_session()
        saved = (session.fetch_cache_ttl_seconds, session.fetch_cache_skip_patterns)
        session.configure_fetch_cache(no_cache_patterns=["example.com/*"])
        try:
            with patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(return_value=(_cached(url), None)),
            ) as dispatch:
                await fetch_url(url, FetchStrategy.AUTO, FetchConfig(), cache=cache)
        finally:
            session.fetch_cache_ttl_seconds, session.fetch_cache_skip_patterns = saved
        dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_cache_patterns_skip_reading(self, tmp_path: Path) -> None:
        url = "https://example.com/a"
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set(url, _cached(url, "# cached\n\n" + "old words " * 30))
        with patch(
            "markitai.fetch._dispatch_strategy",
            AsyncMock(return_value=(_cached(url), None)),
        ) as dispatch:
            result = await fetch_url(
                url,
                FetchStrategy.AUTO,
                FetchConfig(),
                cache=cache,
                no_cache_patterns=["https://example.com/*"],
            )
        dispatch.assert_awaited_once()
        assert result.content == GOOD
        # Still written, like --no-cache
        assert cache.get(url) is not None
        assert cache.get(url).content == GOOD  # type: ignore[union-attr]


class TestConditionalRevalidation:
    def _cache_with_validated_entry(self, tmp_path: Path, url: str) -> FetchCache:
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set_with_validators(url, _cached(url), '"v1"')
        return cache

    @pytest.mark.asyncio
    async def test_challenge_after_revalidation_runs_full_chain(
        self, tmp_path: Path
    ) -> None:
        url = "https://example.com/a"
        cache = self._cache_with_validated_entry(tmp_path, url)
        challenge = ConditionalFetchResult(
            result=FetchResult(
                content=CHALLENGE,
                strategy_used="static",
                url=url,
                metadata={"content_kind": "html"},
            ),
            not_modified=False,
            etag='"v2"',
        )
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(return_value=challenge),
            ),
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(side_effect=FetchError("All fetch strategies failed")),
            ) as dispatch,
        ):
            result = await fetch_url(
                url, FetchStrategy.AUTO, FetchConfig(), cache=cache
            )
        dispatch.assert_awaited_once()
        # The full chain failed too: the last good copy is served, labelled
        assert result.content == GOOD
        assert result.cache_hit
        assert result.metadata["stale"] is True
        assert "All fetch strategies failed" in result.metadata["fetch_warning"]
        # The good entry survives: the challenge never overwrote it
        cached, etag, _ = cache.get_with_validators(url)
        assert cached is not None and cached.content == GOOD
        assert "stale" not in cached.metadata
        assert etag == '"v1"'

    @pytest.mark.asyncio
    async def test_policy_refusal_after_rejected_revalidation_is_not_masked(
        self, tmp_path: Path
    ) -> None:
        url = "https://example.com/a"
        cache = self._cache_with_validated_entry(tmp_path, url)
        challenge = ConditionalFetchResult(
            result=FetchResult(
                content=CHALLENGE,
                strategy_used="static",
                url=url,
                metadata={"content_kind": "html"},
            ),
            not_modified=False,
            etag='"v2"',
        )
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(return_value=challenge),
            ),
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(
                    side_effect=PermissionError(
                        "Fetch target resolved to a non-public address"
                    )
                ),
            ),
            pytest.raises(PermissionError),
        ):
            await fetch_url(url, FetchStrategy.AUTO, FetchConfig(), cache=cache)

    @pytest.mark.asyncio
    async def test_failed_fetch_without_revalidation_still_raises(
        self, tmp_path: Path
    ) -> None:
        """Only a rejected revalidation earns the stale fallback."""
        url = "https://example.com/a"
        cache = self._cache_with_validated_entry(tmp_path, url)
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(side_effect=FetchError("HTTP 404")),
            ),
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(side_effect=FetchError("HTTP 404")),
            ),
            pytest.raises(FetchError),
        ):
            await fetch_url(url, FetchStrategy.AUTO, FetchConfig(), cache=cache)

    @pytest.mark.asyncio
    async def test_js_shell_after_revalidation_falls_back_to_browser(
        self, tmp_path: Path
    ) -> None:
        url = "https://example.com/a"
        cache = self._cache_with_validated_entry(tmp_path, url)
        shell = ConditionalFetchResult(
            result=FetchResult(
                content="Loading...",
                strategy_used="static",
                url=url,
                metadata={"content_kind": "html"},
            ),
            not_modified=False,
            etag='"v2"',
        )
        rendered = FetchResult(
            content=GOOD, strategy_used="playwright", url=url, metadata={}
        )
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(return_value=shell),
            ),
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(return_value=(rendered, None)),
            ),
        ):
            result = await fetch_url(
                url, FetchStrategy.AUTO, FetchConfig(), cache=cache
            )
        assert result.strategy_used == "playwright"
        assert result.content == GOOD

    @pytest.mark.asyncio
    async def test_valid_revalidation_is_used_directly(self, tmp_path: Path) -> None:
        url = "https://example.com/a"
        cache = self._cache_with_validated_entry(tmp_path, url)
        updated = GOOD + "\n\nAn update."
        fresh = ConditionalFetchResult(
            result=FetchResult(
                content=updated,
                strategy_used="static",
                url=url,
                metadata={"content_kind": "html", "converter": "native-html"},
            ),
            not_modified=False,
            etag='"v2"',
        )
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(return_value=fresh),
            ),
            patch("markitai.fetch._dispatch_strategy", AsyncMock()) as dispatch,
        ):
            result = await fetch_url(
                url, FetchStrategy.AUTO, FetchConfig(), cache=cache
            )
        dispatch.assert_not_awaited()
        assert result.content == updated
        assert cache.get_with_validators(url)[1] == '"v2"'

    @pytest.mark.asyncio
    async def test_explicit_static_rejects_revalidated_challenge(
        self, tmp_path: Path
    ) -> None:
        url = "https://example.com/a"
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set_with_validators(url, _cached(url), '"v1"', strategy="static")
        challenge = ConditionalFetchResult(
            result=FetchResult(
                content=CHALLENGE, strategy_used="static", url=url, metadata={}
            ),
            not_modified=False,
            etag='"v2"',
        )
        with (
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(return_value=challenge),
            ),
            patch(
                "markitai.fetch._dispatch_strategy",
                AsyncMock(side_effect=FetchError("challenge")),
            ) as dispatch,
        ):
            result = await fetch_url(
                url,
                FetchStrategy.STATIC,
                FetchConfig(),
                explicit_strategy=True,
                cache=cache,
            )
        dispatch.assert_awaited_once()
        assert result.content == GOOD
        assert result.metadata["stale"] is True
        cached = cache.get(url, strategy="static")
        assert cached is not None and cached.content == GOOD

    @pytest.mark.asyncio
    async def test_explicit_static_accepts_revalidated_document_quoting_captcha(
        self, tmp_path: Path
    ) -> None:
        """A text file about CAPTCHAs is a document, not a challenge page."""
        url = "https://example.com/notes.txt"
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set_with_validators(url, _cached(url), '"v1"', strategy="static")
        notes = "Embed the widget with <div class='g-recaptcha'> and load hcaptcha.com."
        fresh = ConditionalFetchResult(
            result=FetchResult(
                content=notes,
                strategy_used="static",
                url=url,
                metadata={"content_kind": "text"},
            ),
            not_modified=False,
            etag='"v2"',
        )
        with (
            patch(
                "markitai.fetch.fetch_with_static_conditional",
                AsyncMock(return_value=fresh),
            ),
            patch("markitai.fetch._dispatch_strategy", AsyncMock()) as dispatch,
        ):
            result = await fetch_url(
                url,
                FetchStrategy.STATIC,
                FetchConfig(),
                explicit_strategy=True,
                cache=cache,
            )
        dispatch.assert_not_awaited()
        assert result.content == notes


class TestCacheCli:
    def _cfg(self, tmp_path: Path) -> MagicMock:
        cfg = MagicMock()
        cfg.cache.enabled = True
        cfg.cache.global_dir = str(tmp_path)
        cfg.cache.max_size_bytes = 100 * 1024 * 1024
        cfg.cache.fetch_ttl_seconds = 3600
        return cfg

    def _seed(self, tmp_path: Path) -> None:
        cache = FetchCache(tmp_path / "fetch_cache.db")
        cache.set("https://example.com/a", _cached("https://example.com/a"))
        cache.close()

    def test_stats_reports_fetch_cache(self, tmp_path: Path) -> None:
        import json

        from markitai.cli.commands.cache import cache_stats

        self._seed(tmp_path)
        with patch("markitai.cli.commands.cache.ConfigManager") as manager:
            manager.return_value.load.return_value = self._cfg(tmp_path)
            as_json = CliRunner().invoke(cache_stats, ["--json"])
            human = CliRunner().invoke(cache_stats)
        assert as_json.exit_code == 0
        assert json.loads(as_json.output)["fetch_cache"]["count"] == 1
        assert "URL fetches: 1" in human.output

    def test_clear_empties_fetch_cache(self, tmp_path: Path) -> None:
        from markitai.cli.commands.cache import cache_clear

        self._seed(tmp_path)
        with patch("markitai.cli.commands.cache.ConfigManager") as manager:
            manager.return_value.load.return_value = self._cfg(tmp_path)
            result = CliRunner().invoke(cache_clear, ["--yes"])
        assert result.exit_code == 0
        assert "No cache entries" not in result.output
        assert "Cleared 1" in result.output
        assert FetchCache(tmp_path / "fetch_cache.db").stats()["count"] == 0
