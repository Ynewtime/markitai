"""Unit tests for the fetch module."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.constants import JS_REQUIRED_PATTERNS
from markitai.fetch import (
    FetchStrategy,
    SPADomainCache,
    _url_to_screenshot_filename,
    _url_to_session_id,
    detect_js_required,
    should_use_browser_for_domain,
)


class TestFetchStrategy:
    """Tests for FetchStrategy enum."""

    def test_strategy_values(self) -> None:
        """Test that all strategy values are correct."""
        assert FetchStrategy.AUTO.value == "auto"
        assert FetchStrategy.STATIC.value == "static"
        assert FetchStrategy.PLAYWRIGHT.value == "playwright"
        assert FetchStrategy.JINA.value == "jina"

    def test_strategy_from_string(self) -> None:
        """Test creating strategy from string value."""
        assert FetchStrategy("auto") == FetchStrategy.AUTO
        assert FetchStrategy("static") == FetchStrategy.STATIC
        assert FetchStrategy("playwright") == FetchStrategy.PLAYWRIGHT
        assert FetchStrategy("jina") == FetchStrategy.JINA


class TestDetectJsRequired:
    """Tests for detect_js_required function."""

    def test_empty_content_requires_js(self) -> None:
        """Empty content should indicate JS is required."""
        assert detect_js_required("") is True
        assert detect_js_required("   ") is True

    def test_short_content_requires_js(self) -> None:
        """Very short content should indicate JS is required."""
        assert detect_js_required("Hello") is True
        assert detect_js_required("# Title\n\nShort") is True

    def test_normal_content_does_not_require_js(self) -> None:
        """Normal length content should not indicate JS is required."""
        content = """# Welcome to Example

This is a sample page with enough content to be considered valid.
It contains multiple paragraphs and meaningful text that would
typically be found on a real web page.

## Section 1

Here is some more content in the first section.

## Section 2

And even more content in the second section.
"""
        assert detect_js_required(content) is False

    def test_js_disabled_pattern(self) -> None:
        """Content with JS disabled message should require JS."""
        content = """# Page Title

JavaScript is disabled in your browser. Please enable JavaScript
to view this page correctly.

Some other content here to make it long enough.
More content to pass the length check.
"""
        assert detect_js_required(content) is True

    def test_please_enable_javascript_pattern(self) -> None:
        """Content asking to enable JS should require JS."""
        content = """Please enable JavaScript to continue.

This website requires JavaScript to function properly.
Please update your browser settings and reload the page.
Additional content to make this long enough for testing purposes.
"""
        assert detect_js_required(content) is True

    def test_noscript_pattern(self) -> None:
        """Content with noscript tag should require JS."""
        content = """# Page

<noscript>This page requires JavaScript</noscript>

Some more content here to make the page appear long enough
for our length-based detection to pass.
Additional paragraphs of content.
"""
        assert detect_js_required(content) is True

    def test_all_js_required_patterns_detected(self) -> None:
        """All patterns in JS_REQUIRED_PATTERNS should be detected."""
        base_content = """
This is a test page with enough content to pass length check.
We need several lines of text to ensure the length threshold is met.
Here is another line of content.
And another one to be safe.
"""
        for pattern in JS_REQUIRED_PATTERNS:
            content = f"{base_content}\n{pattern}\n"
            assert detect_js_required(content) is True, (
                f"Pattern not detected: {pattern}"
            )


class TestShouldUseBrowserForDomain:
    """Tests for should_use_browser_for_domain function."""

    def test_twitter_domain(self) -> None:
        """Twitter.com should use browser."""
        patterns = ["twitter.com", "x.com"]
        assert (
            should_use_browser_for_domain("https://twitter.com/user", patterns) is True
        )
        assert (
            should_use_browser_for_domain("https://www.twitter.com/user", patterns)
            is True
        )
        assert (
            should_use_browser_for_domain("https://mobile.twitter.com/user", patterns)
            is True
        )

    def test_x_domain(self) -> None:
        """X.com should use browser."""
        patterns = ["twitter.com", "x.com"]
        assert should_use_browser_for_domain("https://x.com/user", patterns) is True
        assert should_use_browser_for_domain("https://www.x.com/user", patterns) is True

    def test_instagram_domain(self) -> None:
        """Instagram should use browser."""
        patterns = ["instagram.com"]
        assert (
            should_use_browser_for_domain("https://instagram.com/user", patterns)
            is True
        )
        assert (
            should_use_browser_for_domain("https://www.instagram.com/user", patterns)
            is True
        )

    def test_non_matching_domain(self) -> None:
        """Non-matching domains should not use browser."""
        patterns = ["twitter.com", "x.com", "instagram.com"]
        assert should_use_browser_for_domain("https://example.com", patterns) is False
        assert should_use_browser_for_domain("https://github.com", patterns) is False
        assert should_use_browser_for_domain("https://google.com", patterns) is False

    def test_empty_patterns(self) -> None:
        """Empty patterns should always return False."""
        assert should_use_browser_for_domain("https://twitter.com", []) is False
        assert should_use_browser_for_domain("https://x.com", []) is False

    def test_case_insensitive(self) -> None:
        """Domain matching should be case-insensitive."""
        patterns = ["Twitter.com"]
        assert (
            should_use_browser_for_domain("https://TWITTER.COM/user", patterns) is True
        )
        assert (
            should_use_browser_for_domain("https://Twitter.Com/user", patterns) is True
        )

    def test_subdomain_matching(self) -> None:
        """Subdomains should match the base pattern."""
        patterns = ["twitter.com"]
        assert (
            should_use_browser_for_domain("https://api.twitter.com/v1", patterns)
            is True
        )
        assert (
            should_use_browser_for_domain("https://mobile.twitter.com/user", patterns)
            is True
        )

    def test_similar_domain_not_matching(self) -> None:
        """Similar but different domains should not match."""
        patterns = ["twitter.com"]
        assert (
            should_use_browser_for_domain("https://nottwitter.com", patterns) is False
        )
        assert (
            should_use_browser_for_domain("https://twitter.com.fake.com", patterns)
            is False
        )


class TestFetchResult:
    """Tests for FetchResult dataclass."""

    def test_fetch_result_creation(self) -> None:
        """Test creating a FetchResult."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="# Test\n\nContent",
            strategy_used="static",
            title="Test Page",
            url="https://example.com",
        )

        assert result.content == "# Test\n\nContent"
        assert result.strategy_used == "static"
        assert result.title == "Test Page"
        assert result.url == "https://example.com"
        assert result.final_url is None
        assert result.metadata == {}

    def test_fetch_result_with_metadata(self) -> None:
        """Test creating FetchResult with metadata."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="content",
            strategy_used="playwright",
            url="https://example.com",
            metadata={"renderer": "playwright"},
        )

        assert result.metadata == {"renderer": "playwright"}


class TestFetchErrors:
    """Tests for fetch error classes."""

    def test_fetch_error(self) -> None:
        """Test FetchError base class."""
        from markitai.fetch import FetchError

        error = FetchError("Test error")
        assert str(error) == "Test error"

    def test_jina_rate_limit_error(self) -> None:
        """Test JinaRateLimitError."""
        from markitai.fetch import JinaRateLimitError

        error = JinaRateLimitError()
        assert "rate limit exceeded" in str(error).lower()
        assert "20 RPM" in str(error)

    def test_jina_api_error(self) -> None:
        """Test JinaAPIError."""
        from markitai.fetch import JinaAPIError

        error = JinaAPIError(500, "Internal Server Error")
        assert "500" in str(error)
        assert "Internal Server Error" in str(error)


class TestUrlToScreenshotFilename:
    """Tests for _url_to_screenshot_filename function."""

    def test_simple_domain(self) -> None:
        """Test simple domain URL."""
        filename = _url_to_screenshot_filename("https://example.com")
        assert filename == "example.com.full.jpg"

    def test_domain_with_path(self) -> None:
        """Test URL with path."""
        filename = _url_to_screenshot_filename("https://example.com/page")
        assert filename == "example.com_page.full.jpg"

    def test_domain_with_deep_path(self) -> None:
        """Test URL with deep path."""
        filename = _url_to_screenshot_filename("https://example.com/a/b/c")
        assert filename == "example.com_a_b_c.full.jpg"

    def test_twitter_url(self) -> None:
        """Test Twitter/X URL."""
        filename = _url_to_screenshot_filename("https://x.com/user/status/123456")
        assert filename == "x.com_user_status_123456.full.jpg"

    def test_special_characters_removed(self) -> None:
        """Test that special characters are removed."""
        filename = _url_to_screenshot_filename(
            "https://example.com/page?query=1&foo=bar"
        )
        # Query string is not included in the path parts
        assert filename == "example.com_page.full.jpg"

    def test_root_path(self) -> None:
        """Test URL with root path only."""
        filename = _url_to_screenshot_filename("https://example.com/")
        assert filename == "example.com.full.jpg"

    def test_long_url_truncated(self) -> None:
        """Test that very long URLs are truncated."""
        long_path = "/".join(["x" * 20 for _ in range(20)])
        filename = _url_to_screenshot_filename(f"https://example.com{long_path}")
        # Should be truncated and end with .full.jpg
        assert filename.endswith(".full.jpg")
        assert len(filename) <= 210  # 200 base + .full.jpg extension

    def test_invalid_url_fallback(self) -> None:
        """Test fallback for invalid URL."""
        # Should fall back to hash-based filename
        filename = _url_to_screenshot_filename("")
        assert filename.startswith("screenshot_")
        assert filename.endswith(".full.jpg")


class TestFetchResultWithScreenshot:
    """Tests for FetchResult with screenshot_path field."""

    def test_fetch_result_without_screenshot(self) -> None:
        """Test FetchResult creation without screenshot."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="# Test",
            strategy_used="static",
            url="https://example.com",
        )
        assert result.screenshot_path is None

    def test_fetch_result_with_screenshot(self) -> None:
        """Test FetchResult with screenshot path."""
        from markitai.fetch import FetchResult

        screenshot = Path("/tmp/screenshots/example.com.full.jpg")
        result = FetchResult(
            content="# Test",
            strategy_used="browser",
            url="https://example.com",
            screenshot_path=screenshot,
        )
        assert result.screenshot_path == screenshot
        assert result.screenshot_path is not None
        assert result.screenshot_path.name == "example.com.full.jpg"

    def test_fetch_result_cache_hit_preserves_screenshot(self) -> None:
        """Test that cache_hit can coexist with screenshot_path."""
        from markitai.fetch import FetchResult

        screenshot = Path("/tmp/screenshots/test.full.jpg")
        result = FetchResult(
            content="# Cached Content",
            strategy_used="browser",
            url="https://example.com",
            cache_hit=True,
            screenshot_path=screenshot,
        )
        assert result.cache_hit is True
        assert result.screenshot_path == screenshot


class TestUrlToSessionId:
    """Tests for _url_to_session_id function."""

    def test_generates_stable_id(self) -> None:
        """Test that same URL always generates same session ID."""
        url = "https://example.com/page"
        id1 = _url_to_session_id(url)
        id2 = _url_to_session_id(url)
        assert id1 == id2

    def test_different_urls_different_ids(self) -> None:
        """Test that different URLs generate different session IDs."""
        id1 = _url_to_session_id("https://example.com/page1")
        id2 = _url_to_session_id("https://example.com/page2")
        assert id1 != id2

    def test_id_format(self) -> None:
        """Test that session ID has correct format."""
        session_id = _url_to_session_id("https://example.com")
        assert session_id.startswith("markitai-")
        assert len(session_id) == len("markitai-") + 8  # 8 hex chars

    def test_handles_special_characters(self) -> None:
        """Test that URLs with special characters work."""
        url = "https://example.com/page?query=1&foo=bar#section"
        session_id = _url_to_session_id(url)
        assert session_id.startswith("markitai-")
        assert len(session_id) == len("markitai-") + 8


class TestProxyDetection:
    """Tests for proxy auto-detection functions."""

    def setup_method(self) -> None:
        """Reset proxy cache before each test."""
        from markitai import fetch

        fetch._detected_proxy = None

    def teardown_method(self) -> None:
        """Reset proxy cache after each test."""
        from markitai import fetch

        fetch._detected_proxy = None

    def test_detects_https_proxy_env(self) -> None:
        """Test detection from HTTPS_PROXY environment variable."""
        from markitai.fetch import _detect_proxy

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:7890"}):
            result = _detect_proxy(force_recheck=True)
            assert result == "http://127.0.0.1:7890"

    def test_detects_http_proxy_env(self) -> None:
        """Test detection from HTTP_PROXY environment variable."""
        from markitai.fetch import _detect_proxy

        # Clear HTTPS_PROXY to test HTTP_PROXY fallback
        with (
            patch.dict(
                "os.environ", {"HTTP_PROXY": "http://localhost:8080"}, clear=False
            ),
            patch.dict("os.environ", {"HTTPS_PROXY": ""}, clear=False),
        ):
            result = _detect_proxy(force_recheck=True)
            assert result == "http://localhost:8080"

    def test_caches_result(self) -> None:
        """Test that proxy detection result is cached."""
        from markitai.fetch import _detect_proxy

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://cached:1234"}):
            result1 = _detect_proxy(force_recheck=True)

        # Change env, but cached result should be returned
        with patch.dict("os.environ", {"HTTPS_PROXY": "http://different:5678"}):
            result2 = _detect_proxy()  # No force_recheck
            assert result2 == result1

    def test_force_recheck_bypasses_cache(self) -> None:
        """Test that force_recheck bypasses cache."""
        from markitai.fetch import _detect_proxy

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://first:1111"}):
            _detect_proxy(force_recheck=True)

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://second:2222"}):
            result = _detect_proxy(force_recheck=True)
            assert result == "http://second:2222"

    def test_get_proxy_for_url_returns_proxy_for_blocked_sites(self) -> None:
        """Test that proxy is returned for commonly blocked sites."""
        from markitai.fetch import get_proxy_for_url

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy:7890"}):
            # Reset cache
            from markitai import fetch

            fetch._detected_proxy = None

            assert get_proxy_for_url("https://x.com/user") == "http://proxy:7890"
            fetch._detected_proxy = None
            assert get_proxy_for_url("https://twitter.com/user") == "http://proxy:7890"
            fetch._detected_proxy = None
            assert (
                get_proxy_for_url("https://www.youtube.com/watch")
                == "http://proxy:7890"
            )

    def test_get_proxy_for_url_returns_empty_for_normal_sites(self) -> None:
        """Test that no proxy is returned for normal sites."""
        from markitai.fetch import get_proxy_for_url

        # Normal sites should not trigger proxy
        assert get_proxy_for_url("https://example.com") == ""
        assert get_proxy_for_url("https://baidu.com") == ""


class TestSPADomainCache:
    """Tests for SPADomainCache class."""

    def test_initialization_creates_cache_file_directory(self) -> None:
        """Test that initialization creates parent directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "subdir" / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            assert cache_path.parent.exists()
            # Cache instance should be created even if file doesn't exist yet
            assert cache is not None

    def test_record_spa_domain_creates_entry(self) -> None:
        """Test recording a new SPA domain."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://example.com/page")

            assert cache.is_known_spa("https://example.com/other")
            assert cache.is_known_spa("https://example.com")

    def test_is_known_spa_returns_false_for_unknown(self) -> None:
        """Test that unknown domains return False."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            assert cache.is_known_spa("https://unknown.com") is False

    def test_record_spa_domain_increments_hits(self) -> None:
        """Test that recording same domain increments hits."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://example.com/page1")
            cache.record_spa_domain("https://example.com/page2")
            cache.record_spa_domain("https://example.com/page3")

            domains = cache.list_domains()
            assert len(domains) == 1
            assert domains[0]["domain"] == "example.com"
            assert domains[0]["hits"] == 3

    def test_record_hit_updates_stats(self) -> None:
        """Test that record_hit updates hits and last_hit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://example.com")
            initial_domains = cache.list_domains()
            initial_hits = initial_domains[0]["hits"]

            cache.record_hit("https://example.com/page")

            domains = cache.list_domains()
            assert domains[0]["hits"] == initial_hits + 1

    def test_record_hit_ignores_unknown_domain(self) -> None:
        """Test that record_hit does nothing for unknown domains."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            # Should not raise any error
            cache.record_hit("https://unknown.com")

            assert cache.list_domains() == []

    def test_clear_removes_all_domains(self) -> None:
        """Test that clear removes all learned domains."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://example1.com")
            cache.record_spa_domain("https://example2.com")
            cache.record_spa_domain("https://example3.com")

            count = cache.clear()

            assert count == 3
            assert cache.list_domains() == []
            assert cache.is_known_spa("https://example1.com") is False

    def test_list_domains_sorted_by_hits(self) -> None:
        """Test that list_domains returns sorted by hits descending."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://low.com")
            cache.record_spa_domain("https://high.com")
            cache.record_spa_domain("https://high.com")
            cache.record_spa_domain("https://high.com")
            cache.record_spa_domain("https://medium.com")
            cache.record_spa_domain("https://medium.com")

            domains = cache.list_domains()
            assert len(domains) == 3
            assert domains[0]["domain"] == "high.com"
            assert domains[0]["hits"] == 3
            assert domains[1]["domain"] == "medium.com"
            assert domains[1]["hits"] == 2
            assert domains[2]["domain"] == "low.com"
            assert domains[2]["hits"] == 1

    def test_extract_domain_handles_subdomains(self) -> None:
        """Test domain extraction with various URL formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://www.example.com/page")

            # Should match www.example.com, not example.com
            assert cache.is_known_spa("https://www.example.com/other") is True
            # Different subdomain is a different domain
            assert cache.is_known_spa("https://api.example.com") is False
            assert cache.is_known_spa("https://example.com") is False

    def test_extract_domain_is_case_insensitive(self) -> None:
        """Test that domain matching is case-insensitive."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            cache.record_spa_domain("https://Example.COM/page")

            assert cache.is_known_spa("https://example.com/other") is True
            assert cache.is_known_spa("https://EXAMPLE.COM") is True

    def test_persistence_across_instances(self) -> None:
        """Test that cache persists across different instances."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            # First instance records domain
            cache1 = SPADomainCache(cache_path)
            cache1.record_spa_domain("https://example.com")

            # Second instance should see it
            cache2 = SPADomainCache(cache_path)
            assert cache2.is_known_spa("https://example.com") is True

    def test_expiry_removes_old_entries(self) -> None:
        """Test that expired entries are removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            # Manually create an expired entry
            old_date = (datetime.now() - timedelta(days=31)).isoformat()
            data = {
                "version": 1,
                "domains": {
                    "expired.com": {
                        "learned_at": old_date,
                        "hits": 5,
                        "last_hit": old_date,
                    }
                },
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            cache = SPADomainCache(cache_path)

            # Checking should remove expired entry
            assert cache.is_known_spa("https://expired.com") is False
            assert cache.list_domains() == []

    def test_version_mismatch_resets_cache(self) -> None:
        """Test that version mismatch resets the cache."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            # Create cache with different version
            data = {
                "version": 999,  # Future version
                "domains": {
                    "example.com": {
                        "learned_at": datetime.now().isoformat(),
                        "hits": 1,
                        "last_hit": datetime.now().isoformat(),
                    }
                },
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            cache = SPADomainCache(cache_path)

            # Should not see the old data
            assert cache.is_known_spa("https://example.com") is False

    def test_invalid_json_handles_gracefully(self) -> None:
        """Test that invalid JSON file is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            # Write invalid JSON
            with open(cache_path, "w") as f:
                f.write("not valid json {{{")

            # Should not raise, just start fresh
            cache = SPADomainCache(cache_path)
            assert cache.list_domains() == []

    def test_list_domains_shows_expired_status(self) -> None:
        """Test that list_domains includes expired status."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            # Create mix of fresh and old entries
            now = datetime.now()
            old_date = (now - timedelta(days=31)).isoformat()
            fresh_date = now.isoformat()

            data = {
                "version": 1,
                "domains": {
                    "fresh.com": {
                        "learned_at": fresh_date,
                        "hits": 1,
                        "last_hit": fresh_date,
                    },
                    "expired.com": {
                        "learned_at": old_date,
                        "hits": 1,
                        "last_hit": old_date,
                    },
                },
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            cache = SPADomainCache(cache_path)
            domains = cache.list_domains()

            # Find each domain in the list
            fresh_entry = next((d for d in domains if d["domain"] == "fresh.com"), None)
            expired_entry = next(
                (d for d in domains if d["domain"] == "expired.com"), None
            )

            assert fresh_entry is not None
            assert fresh_entry["expired"] is False

            assert expired_entry is not None
            assert expired_entry["expired"] is True


class TestFetchCacheWithValidators:
    """Tests for FetchCache HTTP conditional caching (ETag/Last-Modified)."""

    def test_db_migration_adds_columns(self, tmp_path: Path) -> None:
        """Test that database migration adds etag and last_modified columns."""
        from markitai.fetch import FetchCache

        db_path = tmp_path / "test_cache.db"

        # Create old-style cache without validators
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE fetch_cache (
                key TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                content TEXT NOT NULL,
                strategy_used TEXT NOT NULL,
                title TEXT,
                final_url TEXT,
                metadata TEXT,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        # Initialize FetchCache - should trigger migration
        cache = FetchCache(db_path)

        # Verify columns exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(fetch_cache)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "etag" in columns
        assert "last_modified" in columns
        cache.close()

    def test_get_with_validators_no_cache(self, tmp_path: Path) -> None:
        """Test get_with_validators returns None for missing URL."""
        from markitai.fetch import FetchCache

        cache = FetchCache(tmp_path / "test_cache.db")
        result, etag, last_modified = cache.get_with_validators(
            "https://example.com/missing"
        )

        assert result is None
        assert etag is None
        assert last_modified is None
        cache.close()

    def test_set_and_get_with_validators(self, tmp_path: Path) -> None:
        """Test storing and retrieving validators."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com/page"

        # Create result with validators
        result = FetchResult(
            content="# Test Content",
            strategy_used="static",
            url=url,
        )

        cache.set_with_validators(
            url,
            result,
            etag='"abc123"',
            last_modified="Mon, 27 Jan 2026 12:00:00 GMT",
        )

        # Retrieve and verify
        cached_result, etag, last_modified = cache.get_with_validators(url)

        assert cached_result is not None
        assert cached_result.content == "# Test Content"
        assert etag == '"abc123"'
        assert last_modified == "Mon, 27 Jan 2026 12:00:00 GMT"
        cache.close()

    def test_set_with_validators_no_validators(self, tmp_path: Path) -> None:
        """Test storing result without validators."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com/no-validators"

        result = FetchResult(
            content="# Content",
            strategy_used="static",
            url=url,
        )

        # Store without validators
        cache.set_with_validators(url, result, etag=None, last_modified=None)

        # Retrieve
        cached_result, etag, last_modified = cache.get_with_validators(url)

        assert cached_result is not None
        assert etag is None
        assert last_modified is None
        cache.close()

    def test_update_accessed_at(self, tmp_path: Path) -> None:
        """Test updating accessed_at timestamp."""
        import time

        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com/page"

        result = FetchResult(
            content="# Test",
            strategy_used="static",
            url=url,
        )
        cache.set_with_validators(url, result, etag='"123"', last_modified=None)

        # Get initial accessed_at
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "test_cache.db"))
        row = conn.execute(
            "SELECT accessed_at FROM fetch_cache WHERE url = ?", (url,)
        ).fetchone()
        initial_accessed = row[0]
        conn.close()

        # Wait briefly and update
        time.sleep(0.1)
        cache.update_accessed_at(url)

        # Verify accessed_at was updated
        conn = sqlite3.connect(str(tmp_path / "test_cache.db"))
        row = conn.execute(
            "SELECT accessed_at FROM fetch_cache WHERE url = ?", (url,)
        ).fetchone()
        updated_accessed = row[0]
        conn.close()

        assert updated_accessed >= initial_accessed
        cache.close()


class TestConditionalFetchResult:
    """Tests for ConditionalFetchResult dataclass."""

    def test_not_modified_result(self) -> None:
        """Test creating a 304 Not Modified result."""
        from markitai.fetch import ConditionalFetchResult

        result = ConditionalFetchResult(
            result=None,
            not_modified=True,
            etag='"abc123"',
            last_modified="Mon, 27 Jan 2026 12:00:00 GMT",
        )

        assert result.result is None
        assert result.not_modified is True
        assert result.etag == '"abc123"'
        assert result.last_modified == "Mon, 27 Jan 2026 12:00:00 GMT"

    def test_modified_result(self) -> None:
        """Test creating a 200 OK result with new content."""
        from markitai.fetch import ConditionalFetchResult, FetchResult

        fetch_result = FetchResult(
            content="# New Content",
            strategy_used="static",
            url="https://example.com",
        )

        result = ConditionalFetchResult(
            result=fetch_result,
            not_modified=False,
            etag='"def456"',
            last_modified="Tue, 28 Jan 2026 12:00:00 GMT",
        )

        assert result.result is not None
        assert result.result.content == "# New Content"
        assert result.not_modified is False
        assert result.etag == '"def456"'


class TestNormalizeBypassList:
    """Tests for _normalize_bypass_list function."""

    def test_empty_input(self) -> None:
        """Test with empty input."""
        from markitai.fetch import _normalize_bypass_list

        assert _normalize_bypass_list("") == ""

    def test_windows_local_marker_removed(self) -> None:
        """Test that <local> marker is removed."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("<local>,localhost")
        assert "<local>" not in result
        assert "localhost" in result

    def test_wildcard_domain_normalized(self) -> None:
        """Test *.domain.com -> .domain.com conversion."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("*.example.com")
        assert result == ".example.com"

    def test_wildcard_prefix_normalized(self) -> None:
        """Test *-prefix.domain.com extraction."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("*-internal.company.com")
        assert result == ".company.com"

    def test_ip_wildcard_to_cidr(self) -> None:
        """Test IP wildcard to CIDR conversion."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("127.*")
        assert result == "127.0.0.0/8"

        result = _normalize_bypass_list("10.*")
        assert result == "10.0.0.0/8"

        result = _normalize_bypass_list("192.168.*")
        assert result == "192.168.0.0/16"

    def test_172_range_to_cidr(self) -> None:
        """Test 172.16-31.* to CIDR conversion."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("172.16.*")
        assert result == "172.16.0.0/12"

        result = _normalize_bypass_list("172.31.*")
        assert result == "172.16.0.0/12"

    def test_partial_ip_wildcard(self) -> None:
        """Test partial IP wildcards like 100.64.*."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("100.64.*")
        assert result == "100.64"

    def test_multiple_entries_deduplicated(self) -> None:
        """Test that duplicate entries are removed."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("*.example.com,*.example.com,localhost")
        entries = result.split(",")
        assert len(entries) == 2
        assert ".example.com" in entries
        assert "localhost" in entries

    def test_plain_hostname_preserved(self) -> None:
        """Test that plain hostnames are preserved."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("localhost,myserver.local")
        assert "localhost" in result
        assert "myserver.local" in result

    def test_complex_bypass_list(self) -> None:
        """Test complex bypass list with multiple types."""
        from markitai.fetch import _normalize_bypass_list

        input_list = "<local>,*.internal.corp,127.*,192.168.*,myhost"
        result = _normalize_bypass_list(input_list)

        assert "<local>" not in result
        assert ".internal.corp" in result
        assert "127.0.0.0/8" in result
        assert "192.168.0.0/16" in result
        assert "myhost" in result


class TestIsInvalidContent:
    """Tests for _is_invalid_content function."""

    def test_empty_content_is_invalid(self) -> None:
        """Empty content should be invalid."""
        from markitai.fetch import _is_invalid_content

        is_invalid, reason = _is_invalid_content("")
        assert is_invalid is True
        assert reason == "empty"

        is_invalid, reason = _is_invalid_content("   \n\t  ")
        assert is_invalid is True
        assert reason == "empty"

    def test_javascript_disabled_pattern(self) -> None:
        """JavaScript disabled message should be detected."""
        from markitai.fetch import _is_invalid_content

        content = "JavaScript is not available in your browser. Please enable it."
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "javascript_disabled"

        content = "JavaScript is disabled in this browser settings."
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "javascript_disabled"

    def test_javascript_required_pattern(self) -> None:
        """JavaScript required message should be detected."""
        from markitai.fetch import _is_invalid_content

        content = "Please enable JavaScript to use this site."
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "javascript_required"

    def test_unsupported_browser_pattern(self) -> None:
        """Unsupported browser message should be detected."""
        from markitai.fetch import _is_invalid_content

        content = "Your browser is not supported. Please switch to a supported browser."
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "unsupported_browser"

    def test_error_page_pattern(self) -> None:
        """Error page pattern should be detected."""
        from markitai.fetch import _is_invalid_content

        content = "Something went wrong. Hmm, let's give it another shot and try again."
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "error_page"

    def test_login_required_pattern(self) -> None:
        """Login required message should be detected."""
        from markitai.fetch import _is_invalid_content

        # Pattern: "You must be logged in"
        content = """You must be logged in to view this content.
        Please sign in first to access this page.
        This is a members-only area that requires authentication.
        Visit the login page to continue with your account.
        """
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "login_required"

        # Pattern: "Log in.*Sign up.*to continue" (all three parts must be present)
        content = """Please Log in to access this feature.
        Don't have an account? Sign up for free!
        Click the button below to continue to the login page.
        """
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "login_required"

    def test_too_short_content(self) -> None:
        """Very short content should be detected as too short."""
        from markitai.fetch import _is_invalid_content

        # Content with markdown that becomes too short after cleaning
        content = "# Title\n\n**Bold** text"
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "too_short"

    def test_valid_content(self) -> None:
        """Valid content should not be marked as invalid."""
        from markitai.fetch import _is_invalid_content

        content = """# Welcome to Our Website

        This is a comprehensive article about programming best practices.
        We will cover multiple topics including testing, documentation,
        and code organization. Here are some key points to remember:

        - Always write tests for your code
        - Document your functions properly
        - Keep your code modular and maintainable

        Following these practices will improve your software quality.
        """
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is False
        assert reason == ""

    def test_content_with_only_links_and_images(self) -> None:
        """Content with only links and images should be too short."""
        from markitai.fetch import _is_invalid_content

        content = "![Image](https://example.com/img.png)\n[Link](https://example.com)"
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "too_short"


class TestHtmlToText:
    """Tests for _html_to_text function."""

    def test_simple_html_extraction(self) -> None:
        """Test basic HTML to text extraction."""
        from markitai.fetch import _html_to_text

        html = """
        <html>
        <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <p>Another paragraph here.</p>
        </body>
        </html>
        """
        result = _html_to_text(html)
        assert "# Title" in result
        assert "This is a paragraph" in result
        assert "Another paragraph" in result

    def test_removes_script_and_style(self) -> None:
        """Test that script and style elements are removed."""
        from markitai.fetch import _html_to_text

        html = """
        <html>
        <head>
        <style>body { color: red; }</style>
        <script>alert('hello');</script>
        </head>
        <body>
        <h1>Content</h1>
        <p>Real text here.</p>
        </body>
        </html>
        """
        result = _html_to_text(html)
        assert "color: red" not in result
        assert "alert" not in result
        assert "# Content" in result
        assert "Real text" in result

    def test_heading_levels(self) -> None:
        """Test all heading levels are converted."""
        from markitai.fetch import _html_to_text

        html = """
        <body>
        <h1>H1</h1>
        <h2>H2</h2>
        <h3>H3</h3>
        <h4>H4</h4>
        <h5>H5</h5>
        <h6>H6</h6>
        </body>
        """
        result = _html_to_text(html)
        assert "# H1" in result
        assert "## H2" in result
        assert "### H3" in result
        assert "#### H4" in result
        assert "##### H5" in result
        assert "###### H6" in result

    def test_list_items(self) -> None:
        """Test list items are converted."""
        from markitai.fetch import _html_to_text

        html = """
        <body>
        <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        </ul>
        </body>
        """
        result = _html_to_text(html)
        assert "- Item 1" in result
        assert "- Item 2" in result

    def test_blockquote(self) -> None:
        """Test blockquote is converted."""
        from markitai.fetch import _html_to_text

        html = """
        <body>
        <blockquote>This is a quote.</blockquote>
        </body>
        """
        result = _html_to_text(html)
        assert "> This is a quote" in result

    def test_code_block(self) -> None:
        """Test code block is converted."""
        from markitai.fetch import _html_to_text

        html = """
        <body>
        <pre>def hello():
    print("world")</pre>
        </body>
        """
        result = _html_to_text(html)
        assert "```" in result
        assert "def hello" in result

    def test_main_content_area(self) -> None:
        """Test extraction from main content area."""
        from markitai.fetch import _html_to_text

        html = """
        <html>
        <body>
        <nav>Navigation</nav>
        <main>
        <h1>Main Content</h1>
        <p>Important text.</p>
        </main>
        <footer>Footer</footer>
        </body>
        </html>
        """
        result = _html_to_text(html)
        # nav and footer should be removed
        assert "Navigation" not in result
        assert "Footer" not in result
        assert "Main Content" in result

    def test_empty_html(self) -> None:
        """Test empty HTML returns empty string."""
        from markitai.fetch import _html_to_text

        result = _html_to_text("")
        assert result == ""

    def test_no_body(self) -> None:
        """Test HTML without body."""
        from markitai.fetch import _html_to_text

        html = "<html><head><title>Test</title></head></html>"
        result = _html_to_text(html)
        # Should return empty as no content found
        assert result == ""


class TestCompressScreenshot:
    """Tests for _compress_screenshot function."""

    def test_compress_screenshot_basic(self, tmp_path: Path) -> None:
        """Test basic screenshot compression."""
        from markitai.fetch import _compress_screenshot

        # Create a test image
        try:
            from PIL import Image
        except ImportError:
            import pytest

            pytest.skip("Pillow not installed")

        # Create a simple test image
        img = Image.new("RGB", (800, 600), color="red")
        screenshot_path = tmp_path / "test.jpg"
        img.save(screenshot_path, "JPEG", quality=100)
        original_size = screenshot_path.stat().st_size

        # Compress
        _compress_screenshot(screenshot_path, quality=50)

        # Check that file was compressed
        new_size = screenshot_path.stat().st_size
        assert new_size < original_size

    def test_compress_screenshot_rgba_conversion(self, tmp_path: Path) -> None:
        """Test RGBA to RGB conversion during compression."""
        from markitai.fetch import _compress_screenshot

        try:
            from PIL import Image
        except ImportError:
            import pytest

            pytest.skip("Pillow not installed")

        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        screenshot_path = tmp_path / "test_rgba.png"
        img.save(screenshot_path, "PNG")

        # Compress (should convert to RGB for JPEG)
        _compress_screenshot(screenshot_path, quality=75)

        # Verify it's now a valid file
        assert screenshot_path.exists()

    def test_compress_screenshot_resize_tall_image(self, tmp_path: Path) -> None:
        """Test resizing of very tall images."""
        from markitai.fetch import _compress_screenshot

        try:
            from PIL import Image
        except ImportError:
            import pytest

            pytest.skip("Pillow not installed")

        # Create a very tall image (20000 pixels)
        img = Image.new("RGB", (800, 20000), color="blue")
        screenshot_path = tmp_path / "tall.jpg"
        img.save(screenshot_path, "JPEG")

        # Compress with max_height limit
        _compress_screenshot(screenshot_path, quality=75, max_height=5000)

        # Check that image was resized
        with Image.open(screenshot_path) as compressed:
            assert compressed.height <= 5000

    def test_compress_screenshot_missing_pillow(self, tmp_path: Path) -> None:
        """Test handling when Pillow is not installed."""
        from markitai.fetch import _compress_screenshot

        # Create a dummy file
        screenshot_path = tmp_path / "test.jpg"
        screenshot_path.write_bytes(b"dummy data")

        # Mock PIL import to fail
        with patch.dict("sys.modules", {"PIL": None}):
            # Should not raise, just log warning
            _compress_screenshot(screenshot_path, quality=75)

        # File should still exist
        assert screenshot_path.exists()


class TestGetProxyForUrl:
    """Additional tests for get_proxy_for_url function."""

    def setup_method(self) -> None:
        """Reset proxy cache before each test."""
        from markitai import fetch

        fetch._detected_proxy = None
        fetch._detected_proxy_bypass = None

    def teardown_method(self) -> None:
        """Reset proxy cache after each test."""
        from markitai import fetch

        fetch._detected_proxy = None
        fetch._detected_proxy_bypass = None

    def test_subdomain_matching(self) -> None:
        """Test that subdomains of blocked sites use proxy."""
        from markitai.fetch import get_proxy_for_url

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy:8080"}):
            from markitai import fetch

            fetch._detected_proxy = None
            assert (
                get_proxy_for_url("https://api.twitter.com/v1") == "http://proxy:8080"
            )

            fetch._detected_proxy = None
            assert get_proxy_for_url("https://mobile.x.com/user") == "http://proxy:8080"

    def test_google_domain(self) -> None:
        """Test Google domain uses proxy."""
        from markitai.fetch import get_proxy_for_url

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy:8080"}):
            from markitai import fetch

            fetch._detected_proxy = None
            assert (
                get_proxy_for_url("https://www.google.com/search")
                == "http://proxy:8080"
            )

    def test_github_domain(self) -> None:
        """Test GitHub domain uses proxy."""
        from markitai.fetch import get_proxy_for_url

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy:8080"}):
            from markitai import fetch

            fetch._detected_proxy = None
            assert (
                get_proxy_for_url("https://github.com/user/repo") == "http://proxy:8080"
            )

    def test_invalid_url_returns_empty(self) -> None:
        """Test invalid URL returns empty proxy."""
        from markitai.fetch import get_proxy_for_url

        # Invalid URLs should not trigger proxy
        assert get_proxy_for_url("not a url") == ""
        assert get_proxy_for_url("") == ""


class TestDetectJsRequiredEdgeCases:
    """Additional edge case tests for detect_js_required."""

    def test_cloudflare_challenge(self) -> None:
        """Cloudflare challenge page should require JS."""
        content = """Just a moment...

        Checking if the site connection is secure.
        Verifying you are human.
        Ray ID: abc123
        """
        assert detect_js_required(content) is True

    def test_low_content_diversity(self) -> None:
        """Content with low word diversity should require JS."""
        # Repetitive content with few unique words
        content = "loading loading loading loading loading loading"
        assert detect_js_required(content) is True

    def test_spa_loading_state(self) -> None:
        """SPA loading state should require JS."""
        content = "Loading..."
        assert detect_js_required(content) is True

    def test_whitespace_only_content(self) -> None:
        """Whitespace-only content should require JS."""
        content = "   \n\n\t\t   "
        assert detect_js_required(content) is True

    def test_content_with_only_markdown_formatting(self) -> None:
        """Content with only markdown formatting should require JS."""
        content = "# \n## \n### \n---\n***"
        assert detect_js_required(content) is True

    def test_noscript_in_markdown(self) -> None:
        """Noscript tag text in markdown should require JS."""
        content = """This page works best with JavaScript.

        <noscript>Please enable JavaScript to continue.</noscript>

        Additional content here to make this longer than the minimum threshold.
        More content to ensure we pass the length check properly.
        """
        assert detect_js_required(content) is True


class TestFetchResultMultiSource:
    """Tests for FetchResult with multi-source content fields."""

    def test_fetch_result_with_static_content(self) -> None:
        """Test FetchResult with static_content field."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="Primary content",
            strategy_used="static",
            url="https://example.com",
            static_content="Static content from direct fetch",
            browser_content=None,
        )

        assert result.static_content == "Static content from direct fetch"
        assert result.browser_content is None

    def test_fetch_result_with_browser_content(self) -> None:
        """Test FetchResult with browser_content field."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="Primary content",
            strategy_used="browser",
            url="https://example.com",
            static_content=None,
            browser_content="Browser rendered content",
        )

        assert result.static_content is None
        assert result.browser_content == "Browser rendered content"

    def test_fetch_result_with_both_sources(self) -> None:
        """Test FetchResult with both static and browser content."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="Primary content",
            strategy_used="multi",
            url="https://example.com",
            static_content="Static version",
            browser_content="Browser version",
            screenshot_path=Path("/tmp/screenshot.jpg"),
        )

        assert result.static_content == "Static version"
        assert result.browser_content == "Browser version"
        assert result.screenshot_path is not None


class TestCriticalInvalidReasons:
    """Tests for CRITICAL_INVALID_REASONS constant."""

    def test_critical_reasons_defined(self) -> None:
        """Test that critical invalid reasons are properly defined."""
        from markitai.fetch import CRITICAL_INVALID_REASONS

        assert "javascript_disabled" in CRITICAL_INVALID_REASONS
        assert "javascript_required" in CRITICAL_INVALID_REASONS
        assert "login_required" in CRITICAL_INVALID_REASONS

    def test_critical_reasons_used_correctly(self) -> None:
        """Test that critical reasons match _is_invalid_content reasons."""
        from markitai.fetch import CRITICAL_INVALID_REASONS, _is_invalid_content

        # JavaScript disabled content
        _, reason = _is_invalid_content("JavaScript is disabled in your browser")
        assert reason in CRITICAL_INVALID_REASONS

        # Login required content
        _, reason = _is_invalid_content("You must be logged in to view this")
        assert reason in CRITICAL_INVALID_REASONS


class TestGetProxyBypass:
    """Tests for _get_proxy_bypass function."""

    def setup_method(self) -> None:
        """Reset proxy cache before each test."""
        from markitai import fetch

        fetch._detected_proxy = None
        fetch._detected_proxy_bypass = None

    def teardown_method(self) -> None:
        """Reset proxy cache after each test."""
        from markitai import fetch

        fetch._detected_proxy = None
        fetch._detected_proxy_bypass = None

    def test_get_proxy_bypass_from_env(self) -> None:
        """Test getting bypass list from NO_PROXY env var."""
        from markitai.fetch import _get_proxy_bypass

        with patch.dict(
            "os.environ",
            {"HTTPS_PROXY": "http://proxy:8080", "NO_PROXY": "localhost,127.0.0.1"},
        ):
            from markitai import fetch

            fetch._detected_proxy = None
            fetch._detected_proxy_bypass = None
            bypass = _get_proxy_bypass()
            assert "localhost" in bypass

    def test_get_proxy_bypass_empty(self) -> None:
        """Test empty bypass list."""
        from markitai.fetch import _get_proxy_bypass

        with patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy:8080"}, clear=True):
            from markitai import fetch

            fetch._detected_proxy = None
            fetch._detected_proxy_bypass = None
            bypass = _get_proxy_bypass()
            assert bypass == ""


class TestFetchCacheComputeHash:
    """Tests for FetchCache._compute_hash method."""

    def test_hash_consistency(self, tmp_path: Path) -> None:
        """Test that same URL always produces same hash."""
        from markitai.fetch import FetchCache

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page?query=1"

        hash1 = cache._compute_hash(url)
        hash2 = cache._compute_hash(url)

        assert hash1 == hash2
        assert len(hash1) == 32  # SHA256 truncated to 32 chars
        cache.close()

    def test_different_urls_different_hashes(self, tmp_path: Path) -> None:
        """Test that different URLs produce different hashes."""
        from markitai.fetch import FetchCache

        cache = FetchCache(tmp_path / "test.db")

        hash1 = cache._compute_hash("https://example.com/page1")
        hash2 = cache._compute_hash("https://example.com/page2")

        assert hash1 != hash2
        cache.close()


class TestSPADomainCacheEdgeCases:
    """Additional edge case tests for SPADomainCache."""

    def test_save_error_handling(self) -> None:
        """Test graceful handling of save errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            # Make the file read-only to trigger save error
            cache_path.touch()
            import os

            os.chmod(cache_path, 0o444)

            try:
                # Should not raise, just log warning
                cache.record_spa_domain("https://example.com")
            finally:
                # Restore permissions for cleanup
                os.chmod(cache_path, 0o644)

    def test_expired_entry_with_missing_dates(self) -> None:
        """Test handling of entries with missing date fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            # Create entry with missing dates
            data = {
                "version": 1,
                "domains": {
                    "nodates.com": {
                        "hits": 1,
                        # No learned_at or last_hit
                    }
                },
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            cache = SPADomainCache(cache_path)
            # Should be treated as expired due to missing dates
            assert cache.is_known_spa("https://nodates.com") is False

    def test_expired_entry_with_invalid_date(self) -> None:
        """Test handling of entries with invalid date format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"

            data = {
                "version": 1,
                "domains": {
                    "baddate.com": {
                        "learned_at": "not-a-valid-date",
                        "hits": 1,
                        "last_hit": "also-invalid",
                    }
                },
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            cache = SPADomainCache(cache_path)
            # Should be treated as expired due to invalid date
            assert cache.is_known_spa("https://baddate.com") is False


class TestGlobalCacheInstances:
    """Tests for global cache instance management."""

    def test_get_spa_domain_cache_returns_same_instance(self) -> None:
        """Test that get_spa_domain_cache returns singleton."""
        from markitai.fetch import get_spa_domain_cache

        cache1 = get_spa_domain_cache()
        cache2 = get_spa_domain_cache()

        assert cache1 is cache2

    def test_get_fetch_cache_creates_instance(self, tmp_path: Path) -> None:
        """Test that get_fetch_cache creates instance."""
        from markitai import fetch
        from markitai.fetch import get_fetch_cache

        # Reset global instance
        fetch._fetch_cache = None

        cache = get_fetch_cache(tmp_path)
        assert cache is not None
        cache.close()

        # Reset for other tests
        fetch._fetch_cache = None


class TestFetchCacheConnection:
    """Tests for FetchCache connection management."""

    def test_connection_reuse(self, tmp_path: Path) -> None:
        """Test that connection is reused across operations."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")

        # Perform multiple operations
        result = FetchResult(
            content="Test", strategy_used="static", url="https://example.com"
        )
        cache.set("https://example.com", result)
        cache.get("https://example.com")
        cache.stats()

        # Connection should be reused (not None)
        assert cache._connection is not None
        cache.close()
        assert cache._connection is None

    def test_close_idempotent(self, tmp_path: Path) -> None:
        """Test that close can be called multiple times safely."""
        from markitai.fetch import FetchCache

        cache = FetchCache(tmp_path / "test.db")
        cache.close()
        cache.close()  # Should not raise
        cache.close()  # Should not raise


class TestFetchWithStatic:
    """Tests for fetch_with_static function."""

    @pytest.mark.asyncio
    async def test_fetch_with_static_success(self) -> None:
        """Test successful static fetch."""
        from markitai.fetch import fetch_with_static

        # Mock markitdown conversion
        mock_result = type(
            "MockResult",
            (),
            {"text_content": "# Test Page\n\nSome content here.", "title": "Test Page"},
        )()

        with patch("markitai.fetch._get_markitdown") as mock_get_md:
            mock_md = type("MockMD", (), {"convert": lambda _self, _url: mock_result})()
            mock_get_md.return_value = mock_md

            result = await fetch_with_static("https://example.com")

            assert result.content == "# Test Page\n\nSome content here."
            assert result.strategy_used == "static"
            assert result.title == "Test Page"
            assert result.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_fetch_with_static_no_content(self) -> None:
        """Test static fetch with no content raises error."""
        from markitai.fetch import FetchError, fetch_with_static

        mock_result = type("MockResult", (), {"text_content": "", "title": None})()

        with patch("markitai.fetch._get_markitdown") as mock_get_md:
            mock_md = type("MockMD", (), {"convert": lambda _self, _url: mock_result})()
            mock_get_md.return_value = mock_md

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_static("https://example.com")

            assert "No content extracted" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_with_static_exception(self) -> None:
        """Test static fetch handles exceptions."""
        from markitai.fetch import FetchError, fetch_with_static

        with patch("markitai.fetch._get_markitdown") as mock_get_md:
            mock_md = type(
                "MockMD",
                (),
                {
                    "convert": lambda _self, _url: (_ for _ in ()).throw(
                        Exception("Network error")
                    )
                },
            )()
            mock_get_md.return_value = mock_md

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_static("https://example.com")

            assert "Failed to fetch URL" in str(exc_info.value)


class TestFetchWithJina:
    """Tests for fetch_with_jina function."""

    @pytest.mark.asyncio
    async def test_fetch_with_jina_success(self) -> None:
        """Test successful Jina fetch."""
        from markitai import fetch
        from markitai.fetch import fetch_with_jina

        # Reset global client
        fetch._jina_client = None

        json_data = {
            "code": 200,
            "data": {
                "title": "Test Page",
                "content": "# Jina Content\n\nExtracted text.",
                "url": "https://example.com",
            },
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await fetch_with_jina("https://example.com")

            assert result.content == "# Jina Content\n\nExtracted text."
            assert result.strategy_used == "jina"
            assert result.title == "Test Page"

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_rate_limit(self) -> None:
        """Test Jina rate limit error."""
        from markitai import fetch
        from markitai.fetch import JinaRateLimitError, fetch_with_jina

        fetch._jina_client = None

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(JinaRateLimitError):
                await fetch_with_jina("https://example.com")

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_api_error(self) -> None:
        """Test Jina API error."""
        from markitai import fetch
        from markitai.fetch import JinaAPIError, fetch_with_jina

        fetch._jina_client = None

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(JinaAPIError) as exc_info:
                await fetch_with_jina("https://example.com")

            assert exc_info.value.status_code == 500

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_empty_content(self) -> None:
        """Test Jina with empty content."""
        from markitai import fetch
        from markitai.fetch import FetchError, fetch_with_jina

        fetch._jina_client = None

        json_data = {"code": 200, "data": {"title": "Test", "content": ""}}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_jina("https://example.com")

            assert "No content returned" in str(exc_info.value)

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_invalid_json(self) -> None:
        """Test Jina with invalid JSON response."""
        from markitai import fetch
        from markitai.fetch import FetchError, fetch_with_jina

        fetch._jina_client = None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
        mock_response.text = "not json"

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_jina("https://example.com")

            assert "invalid JSON" in str(exc_info.value)

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_timeout(self) -> None:
        """Test Jina timeout error."""
        import httpx

        from markitai import fetch
        from markitai.fetch import FetchError, fetch_with_jina

        fetch._jina_client = None

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_get_client.return_value = mock_client

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_jina("https://example.com", timeout=10)

            assert "timed out" in str(exc_info.value)

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_with_api_key(self) -> None:
        """Test Jina fetch with API key."""
        from markitai import fetch
        from markitai.fetch import fetch_with_jina

        fetch._jina_client = None

        json_data = {
            "code": 200,
            "data": {"title": "Test", "content": "Content with API key"},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await fetch_with_jina(
                "https://example.com", api_key="test-api-key"
            )

            assert result.content == "Content with API key"
            # Verify get was called (API key would be in headers)
            mock_client.get.assert_called_once()

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_fetch_with_jina_error_message_in_response(self) -> None:
        """Test Jina with error message in JSON response."""
        from markitai import fetch
        from markitai.fetch import JinaAPIError, fetch_with_jina

        fetch._jina_client = None

        json_data = {
            "code": 400,
            "message": "Invalid URL provided",
            "data": None,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch._get_jina_client") as mock_get_client,
        ):
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(JinaAPIError) as exc_info:
                await fetch_with_jina("https://example.com")

            assert "Invalid URL" in str(exc_info.value)

        fetch._jina_client = None


class TestFetchWithStaticConditional:
    """Tests for fetch_with_static_conditional function."""

    @pytest.mark.asyncio
    async def test_conditional_fetch_304_not_modified(self) -> None:
        """Test conditional fetch returns 304 Not Modified."""
        from markitai.fetch import fetch_with_static_conditional

        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_response.headers = {
            "ETag": '"abc123"',
            "Last-Modified": "Mon, 01 Jan 2026",
        }

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_client
            mock_context.__aexit__.return_value = None
            mock_client_class.return_value = mock_context

            result = await fetch_with_static_conditional(
                "https://example.com",
                cached_etag='"old-etag"',
                cached_last_modified="Sun, 01 Dec 2025",
            )

            assert result.not_modified is True
            assert result.result is None

    @pytest.mark.asyncio
    async def test_conditional_fetch_200_new_content(self) -> None:
        """Test conditional fetch returns 200 with new content."""
        from markitai.fetch import fetch_with_static_conditional

        mock_headers = {
            "ETag": '"new-etag"',
            "Last-Modified": "Mon, 01 Jan 2026",
            "Content-Type": "text/html",
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = MagicMock()
        mock_response.headers.get = lambda key, default="": mock_headers.get(
            key, default
        )
        mock_response.content = b"<html><body><h1>New Content</h1></body></html>"
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# New Content"
        mock_md_result.title = "New Content"

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
            patch("markitai.fetch._get_markitdown") as mock_get_md,
        ):
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_client
            mock_context.__aexit__.return_value = None
            mock_client_class.return_value = mock_context

            mock_md = MagicMock()
            mock_md.convert.return_value = mock_md_result
            mock_get_md.return_value = mock_md

            result = await fetch_with_static_conditional("https://example.com")

            assert result.not_modified is False
            assert result.result is not None
            assert result.result.content == "# New Content"
            assert result.etag == '"new-etag"'

    @pytest.mark.asyncio
    async def test_conditional_fetch_http_error(self) -> None:
        """Test conditional fetch handles HTTP errors."""
        from markitai.fetch import FetchError, fetch_with_static_conditional

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.text = "Not Found"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_client
            mock_context.__aexit__.return_value = None
            mock_client_class.return_value = mock_context

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_static_conditional("https://example.com")

            assert "HTTP 404" in str(exc_info.value)


class TestCloseSharedClients:
    """Tests for close_shared_clients function."""

    @pytest.mark.asyncio
    async def test_close_shared_clients_with_jina_client(self) -> None:
        """Test closing shared Jina client."""
        from markitai import fetch
        from markitai.fetch import close_shared_clients

        # Create a mock client
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        fetch._jina_client = mock_client

        await close_shared_clients()

        mock_client.aclose.assert_called_once()
        assert fetch._jina_client is None

    @pytest.mark.asyncio
    async def test_close_shared_clients_with_fetch_cache(self) -> None:
        """Test closing shared fetch cache."""
        from markitai import fetch
        from markitai.fetch import close_shared_clients

        # Create a mock cache
        mock_cache = type("MockCache", (), {"close": lambda _self: None})()
        fetch._fetch_cache = mock_cache

        await close_shared_clients()

        assert fetch._fetch_cache is None

    @pytest.mark.asyncio
    async def test_close_shared_clients_with_playwright_renderer(self) -> None:
        """Test closing shared Playwright renderer."""
        from markitai import fetch
        from markitai.fetch import close_shared_clients

        # Create a mock renderer
        mock_renderer = AsyncMock()
        mock_renderer.close = AsyncMock()
        fetch._playwright_renderer = mock_renderer

        await close_shared_clients()

        mock_renderer.close.assert_called_once()
        assert fetch._playwright_renderer is None

    @pytest.mark.asyncio
    async def test_close_shared_clients_none_clients(self) -> None:
        """Test closing when no clients exist."""
        from markitai import fetch
        from markitai.fetch import close_shared_clients

        fetch._jina_client = None
        fetch._fetch_cache = None
        fetch._playwright_renderer = None

        # Should not raise any errors
        await close_shared_clients()


class TestGetMarkitdown:
    """Tests for _get_markitdown function."""

    def test_get_markitdown_creates_instance(self) -> None:
        """Test that _get_markitdown creates a new instance."""
        from markitai import fetch
        from markitai.fetch import _get_markitdown

        # Save original value
        original = fetch._markitdown_instance
        fetch._markitdown_instance = None

        try:
            with patch.dict("sys.modules", {"markitdown": MagicMock()}):
                import sys

                mock_markitdown_class = MagicMock()
                mock_instance = MagicMock()
                mock_markitdown_class.return_value = mock_instance
                sys.modules["markitdown"].MarkItDown = mock_markitdown_class

                result = _get_markitdown()

                # Should have created an instance
                assert result is not None
        finally:
            fetch._markitdown_instance = original

    def test_get_markitdown_reuses_instance(self) -> None:
        """Test that _get_markitdown reuses existing instance."""
        from markitai import fetch
        from markitai.fetch import _get_markitdown

        # Save original value
        original = fetch._markitdown_instance

        try:
            mock_instance = MagicMock()
            fetch._markitdown_instance = mock_instance

            result = _get_markitdown()

            assert result is mock_instance
        finally:
            fetch._markitdown_instance = original


class TestGetJinaClient:
    """Tests for _get_jina_client function."""

    def test_get_jina_client_creates_instance(self) -> None:
        """Test that _get_jina_client creates a new instance."""
        from markitai import fetch
        from markitai.fetch import _get_jina_client

        fetch._jina_client = None

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_instance = type("MockClient", (), {})()
            mock_client_class.return_value = mock_instance

            result = _get_jina_client(timeout=60)

            assert result is mock_instance
            mock_client_class.assert_called_once()

        fetch._jina_client = None

    def test_get_jina_client_with_proxy(self) -> None:
        """Test that _get_jina_client uses proxy."""
        from markitai import fetch
        from markitai.fetch import _get_jina_client

        fetch._jina_client = None

        with (
            patch("markitai.fetch._detect_proxy", return_value="http://proxy:8080"),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            _get_jina_client()

            call_kwargs = mock_client_class.call_args.kwargs
            assert call_kwargs.get("proxy") == "http://proxy:8080"

        fetch._jina_client = None

    def test_get_jina_client_reuses_instance(self) -> None:
        """Test that _get_jina_client reuses existing instance."""
        from markitai import fetch
        from markitai.fetch import _get_jina_client

        mock_instance = type("MockClient", (), {})()
        fetch._jina_client = mock_instance

        result = _get_jina_client()

        assert result is mock_instance

        fetch._jina_client = None


class TestFetchWithFallback:
    """Tests for _fetch_with_fallback function."""

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_static_success(self) -> None:
        """Test fallback uses static strategy successfully."""
        from markitai.fetch import FetchResult, _fetch_with_fallback

        mock_config = type(
            "MockConfig",
            (),
            {
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {"timeout": 30000, "wait_for": "load", "extra_wait_ms": 0},
                )(),
            },
        )()

        # Good content that doesn't require JS
        good_content = """# Welcome

        This is a comprehensive article with enough content to pass validation.
        It has multiple paragraphs and meaningful text that would typically
        be found on a real web page. Here is more content to ensure it passes.
        """

        mock_result = FetchResult(
            content=good_content, strategy_used="static", url="https://example.com"
        )

        with patch(
            "markitai.fetch.fetch_with_static", new_callable=AsyncMock
        ) as mock_static:
            mock_static.return_value = mock_result

            result = await _fetch_with_fallback(
                "https://example.com", mock_config, start_with_browser=False
            )

            assert result.strategy_used == "static"
            mock_static.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_all_fail(self) -> None:
        """Test fallback raises error when all strategies fail."""
        from markitai.fetch import FetchError, _fetch_with_fallback

        mock_config = type(
            "MockConfig",
            (),
            {
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {"timeout": 30000, "wait_for": "load", "extra_wait_ms": 0},
                )(),
            },
        )()

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                side_effect=FetchError("Static failed"),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=False
            ),
            patch(
                "markitai.fetch.fetch_with_jina",
                new_callable=AsyncMock,
                side_effect=FetchError("Jina failed"),
            ),
        ):
            with pytest.raises(FetchError) as exc_info:
                await _fetch_with_fallback(
                    "https://example.com",
                    mock_config,
                    start_with_browser=False,
                )

            assert "All fetch strategies failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_browser_first(self) -> None:
        """Test fallback starts with browser when requested."""
        from markitai.fetch import _fetch_with_fallback

        mock_config = type(
            "MockConfig",
            (),
            {
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {"timeout": 30000, "wait_for": "load", "extra_wait_ms": 0},
                )(),
                "auto_proxy": False,
            },
        )()

        mock_pw_result = type(
            "PlaywrightResult",
            (),
            {
                "content": "# Browser Content",
                "title": "Test",
                "final_url": "https://example.com",
                "metadata": {},
                "screenshot_path": None,
            },
        )()

        with (
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                new_callable=AsyncMock,
                return_value=mock_pw_result,
            ),
        ):
            result = await _fetch_with_fallback(
                "https://example.com",
                mock_config,
                start_with_browser=True,
            )

            assert result.strategy_used == "playwright"


class TestFetchCacheIntegration:
    """Integration tests for FetchCache class."""

    def test_cache_set_and_get(self, tmp_path: Path) -> None:
        """Test basic set and get operations."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com/page"

        # Create a result to cache
        result = FetchResult(
            content="# Test Content\n\nThis is a test.",
            strategy_used="static",
            title="Test Page",
            url=url,
            final_url=url,
            metadata={"source": "test"},
        )

        # Cache it
        cache.set(url, result)

        # Retrieve it
        cached = cache.get(url)

        assert cached is not None
        assert cached.content == result.content
        assert cached.title == result.title
        assert cached.strategy_used == result.strategy_used
        assert cached.cache_hit is True
        cache.close()

    def test_cache_get_miss(self, tmp_path: Path) -> None:
        """Test cache miss returns None."""
        from markitai.fetch import FetchCache

        cache = FetchCache(tmp_path / "test_cache.db")
        result = cache.get("https://nonexistent.com/page")

        assert result is None
        cache.close()

    def test_cache_stats(self, tmp_path: Path) -> None:
        """Test cache statistics."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")

        # Initially empty
        stats = cache.stats()
        assert stats["count"] == 0
        assert stats["size_bytes"] == 0

        # Add an entry
        result = FetchResult(
            content="# Content",
            strategy_used="static",
            url="https://example.com",
        )
        cache.set("https://example.com", result)

        # Check stats updated
        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["size_bytes"] > 0
        assert "db_path" in stats
        cache.close()

    def test_cache_clear(self, tmp_path: Path) -> None:
        """Test clearing the cache."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")

        # Add entries
        for i in range(5):
            result = FetchResult(
                content=f"Content {i}",
                strategy_used="static",
                url=f"https://example{i}.com",
            )
            cache.set(f"https://example{i}.com", result)

        # Clear and check count
        cleared = cache.clear()
        assert cleared == 5

        # Verify empty
        stats = cache.stats()
        assert stats["count"] == 0
        cache.close()

    def test_cache_lru_eviction(self, tmp_path: Path) -> None:
        """Test LRU eviction when cache is full."""
        from markitai.fetch import FetchCache, FetchResult

        # Create a very small cache (1KB max)
        cache = FetchCache(tmp_path / "test_cache.db", max_size_bytes=1024)

        # Add entries until we exceed the limit
        large_content = "x" * 500  # 500 bytes each
        for i in range(5):
            result = FetchResult(
                content=large_content,
                strategy_used="static",
                url=f"https://example{i}.com",
            )
            cache.set(f"https://example{i}.com", result)

        # Some entries should have been evicted
        stats = cache.stats()
        assert stats["count"] < 5  # Not all entries fit
        cache.close()

    def test_cache_metadata_preserved(self, tmp_path: Path) -> None:
        """Test that metadata is preserved correctly."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com"

        result = FetchResult(
            content="Content",
            strategy_used="playwright",
            url=url,
            metadata={"key1": "value1", "key2": 123, "nested": {"a": "b"}},
        )
        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.metadata == {"key1": "value1", "key2": 123, "nested": {"a": "b"}}
        cache.close()

    def test_cache_url_hash_collision_handling(self, tmp_path: Path) -> None:
        """Test that different URLs don't collide."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")

        # Store two different URLs
        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"

        result1 = FetchResult(content="Content 1", strategy_used="static", url=url1)
        result2 = FetchResult(content="Content 2", strategy_used="static", url=url2)

        cache.set(url1, result1)
        cache.set(url2, result2)

        # Both should be retrievable with correct content
        cached1 = cache.get(url1)
        cached2 = cache.get(url2)

        assert cached1 is not None
        assert cached2 is not None
        assert cached1.content == "Content 1"
        assert cached2.content == "Content 2"
        cache.close()

    def test_cache_update_existing_entry(self, tmp_path: Path) -> None:
        """Test updating an existing cache entry."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com"

        # Initial entry
        result1 = FetchResult(content="Old content", strategy_used="static", url=url)
        cache.set(url, result1)

        # Update with new content
        result2 = FetchResult(
            content="New content", strategy_used="playwright", url=url
        )
        cache.set(url, result2)

        # Should get the updated content
        cached = cache.get(url)
        assert cached is not None
        assert cached.content == "New content"
        assert cached.strategy_used == "playwright"

        # Should still only have one entry
        stats = cache.stats()
        assert stats["count"] == 1
        cache.close()


class TestIsInvalidContentAdditional:
    """Additional tests for _is_invalid_content function."""

    def test_non_dict_json_response(self) -> None:
        """Test handling of valid content with special characters."""
        from markitai.fetch import _is_invalid_content

        # Valid long content
        content = """# Article Title

        This is a long article with plenty of meaningful content.
        It discusses various topics in depth and provides valuable
        information to the reader. Here we have multiple paragraphs
        and different sections to ensure comprehensive coverage.

        ## Section 1

        More detailed content here explaining the first topic.

        ## Section 2

        Another section with equally important information.
        """
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is False

    def test_content_with_special_patterns_not_invalid(self) -> None:
        """Test content that mentions JavaScript but is valid."""
        from markitai.fetch import _is_invalid_content

        # Content that discusses JavaScript but is actually valid
        content = """# How to Use JavaScript

        JavaScript is a powerful programming language used for web development.
        In this tutorial, we'll learn how to write JavaScript code effectively.

        ## Getting Started

        First, you'll need to understand the basics of JavaScript syntax.
        Variables, functions, and control structures are fundamental concepts.

        ## Advanced Topics

        Once you master the basics, you can explore more advanced features
        like async/await, promises, and modern ES6+ syntax.
        """
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is False


class TestFetchMultiSourceAdditional:
    """Additional tests for _fetch_multi_source function."""

    @pytest.mark.asyncio
    async def test_fetch_multi_source_both_fail_raises_error(self) -> None:
        """Test multi-source fetch raises error when all strategies fail."""
        from markitai.fetch import FetchError, _fetch_multi_source

        mock_config = type(
            "MockConfig",
            (),
            {
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {"timeout": 30000, "wait_for": "load", "extra_wait_ms": 0},
                )(),
                "auto_proxy": False,
            },
        )()

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                side_effect=Exception("Static failed"),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=False
            ),
        ):
            with pytest.raises(FetchError) as exc_info:
                await _fetch_multi_source(
                    "https://example.com",
                    mock_config,
                )

            assert "All fetch strategies failed" in str(exc_info.value)


class TestDetectProxyAdditional:
    """Additional tests for proxy detection."""

    def setup_method(self) -> None:
        """Reset proxy cache before each test."""
        from markitai import fetch

        fetch._detected_proxy = None
        fetch._detected_proxy_bypass = None

    def teardown_method(self) -> None:
        """Reset proxy cache after each test."""
        from markitai import fetch

        fetch._detected_proxy = None
        fetch._detected_proxy_bypass = None

    def test_detect_proxy_lowercase_env_vars(self) -> None:
        """Test detection from lowercase environment variables."""
        from markitai.fetch import _detect_proxy

        with patch.dict(
            "os.environ",
            {"https_proxy": "http://lowercase:8080"},
            clear=True,
        ):
            result = _detect_proxy(force_recheck=True)
            assert result == "http://lowercase:8080"

    def test_detect_proxy_all_proxy_env(self) -> None:
        """Test detection from ALL_PROXY environment variable."""
        from markitai.fetch import _detect_proxy

        with patch.dict("os.environ", {"ALL_PROXY": "http://all:9999"}, clear=True):
            result = _detect_proxy(force_recheck=True)
            assert result == "http://all:9999"

    def test_detect_proxy_no_proxy_found(self) -> None:
        """Test when no proxy is configured."""
        from markitai import fetch
        from markitai.fetch import _detect_proxy

        fetch._detected_proxy = None

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("markitai.fetch._get_system_proxy", return_value=("", "")),
            patch("socket.socket") as mock_socket,
        ):
            # Make all port probes fail
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 1  # Connection refused
            mock_socket.return_value = mock_sock_instance

            result = _detect_proxy(force_recheck=True)
            assert result == ""

    def test_detect_proxy_probes_common_ports(self) -> None:
        """Test that proxy detection probes common ports."""
        from markitai import fetch
        from markitai.fetch import _detect_proxy

        fetch._detected_proxy = None

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("markitai.fetch._get_system_proxy", return_value=("", "")),
            patch("socket.socket") as mock_socket,
        ):
            # First port probe succeeds
            mock_sock_instance = MagicMock()
            mock_sock_instance.connect_ex.return_value = 0
            mock_socket.return_value = mock_sock_instance

            result = _detect_proxy(force_recheck=True)
            assert result.startswith("http://127.0.0.1:")


class TestGetSystemProxy:
    """Tests for _get_system_proxy function."""

    def test_get_system_proxy_linux(self) -> None:
        """Test system proxy detection on Linux (not implemented)."""
        from markitai.fetch import _get_system_proxy

        with patch("platform.system", return_value="Linux"):
            proxy, bypass = _get_system_proxy()
            assert proxy == ""
            assert bypass == ""

    def test_get_system_proxy_unknown_platform(self) -> None:
        """Test system proxy detection on unknown platform."""
        from markitai.fetch import _get_system_proxy

        with patch("platform.system", return_value="UnknownOS"):
            proxy, bypass = _get_system_proxy()
            assert proxy == ""
            assert bypass == ""


class TestJinaRateLimitErrorMessage:
    """Tests for JinaRateLimitError error message."""

    def test_error_message_contains_hints(self) -> None:
        """Test that error message contains helpful hints."""
        from markitai.fetch import JinaRateLimitError

        error = JinaRateLimitError()
        message = str(error)

        assert "20 RPM" in message
        assert "playwright" in message.lower() or "later" in message.lower()


class TestJinaAPIErrorDetails:
    """Tests for JinaAPIError details."""

    def test_error_preserves_status_code(self) -> None:
        """Test that status code is preserved."""
        from markitai.fetch import JinaAPIError

        error = JinaAPIError(403, "Forbidden")
        assert error.status_code == 403
        assert "403" in str(error)
        assert "Forbidden" in str(error)


class TestFetchCacheEvictionEdgeCases:
    """Edge case tests for FetchCache eviction."""

    def test_eviction_with_empty_cache(self, tmp_path: Path) -> None:
        """Test eviction behavior with empty cache."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db", max_size_bytes=100)

        # Add large content to small cache
        result = FetchResult(
            content="x" * 200,  # Larger than max
            strategy_used="static",
            url="https://example.com",
        )
        cache.set("https://example.com", result)

        # Should still store (no entries to evict initially)
        stats = cache.stats()
        assert stats["count"] == 1
        cache.close()


class TestNormalizeBypassListAdditional:
    """Additional tests for _normalize_bypass_list function."""

    def test_handles_whitespace(self) -> None:
        """Test handling of whitespace in bypass list."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("  localhost  ,  127.0.0.1  ")
        assert "localhost" in result
        assert "127.0.0.1" in result

    def test_handles_empty_items(self) -> None:
        """Test handling of empty items in bypass list."""
        from markitai.fetch import _normalize_bypass_list

        result = _normalize_bypass_list("localhost,,127.0.0.1,")
        entries = [e for e in result.split(",") if e]
        assert len(entries) == 2


class TestFetchCacheThreadSafety:
    """Tests for FetchCache thread safety mechanisms."""

    def test_lock_is_used(self, tmp_path: Path) -> None:
        """Test that lock exists and is used."""
        from markitai.fetch import FetchCache

        cache = FetchCache(tmp_path / "test.db")

        # Verify lock exists
        assert hasattr(cache, "_lock")
        assert cache._lock is not None
        cache.close()


class TestGetPlaywrightRenderer:
    """Tests for _get_playwright_renderer function."""

    @pytest.mark.asyncio
    async def test_get_playwright_renderer_creates_instance(self) -> None:
        """Test that _get_playwright_renderer creates instance."""
        from markitai import fetch
        from markitai.fetch import _get_playwright_renderer

        fetch._playwright_renderer = None

        with patch(
            "markitai.fetch_playwright.PlaywrightRenderer"
        ) as mock_renderer_class:
            mock_instance = MagicMock()
            mock_renderer_class.return_value = mock_instance

            result = await _get_playwright_renderer(proxy="http://proxy:8080")

            assert result is mock_instance
            mock_renderer_class.assert_called_once_with(proxy="http://proxy:8080")

        fetch._playwright_renderer = None

    @pytest.mark.asyncio
    async def test_get_playwright_renderer_reuses_instance(self) -> None:
        """Test that _get_playwright_renderer reuses existing instance."""
        from markitai import fetch
        from markitai.fetch import _get_playwright_renderer

        mock_instance = MagicMock()
        fetch._playwright_renderer = mock_instance

        result = await _get_playwright_renderer()

        assert result is mock_instance

        fetch._playwright_renderer = None


class TestUrlToScreenshotFilenameEdgeCases:
    """Additional edge case tests for _url_to_screenshot_filename."""

    def test_unicode_url(self) -> None:
        """Test URL with unicode characters."""
        filename = _url_to_screenshot_filename("https://example.com/页面")
        assert filename.endswith(".full.jpg")

    def test_very_short_url(self) -> None:
        """Test very short URL."""
        filename = _url_to_screenshot_filename("https://a.b")
        assert filename.endswith(".full.jpg")

    def test_url_with_port(self) -> None:
        """Test URL with port number."""
        filename = _url_to_screenshot_filename("https://localhost:8080/page")
        assert "localhost" in filename
        assert filename.endswith(".full.jpg")


class TestFetchResultFields:
    """Tests for FetchResult field defaults."""

    def test_fetch_result_default_cache_hit(self) -> None:
        """Test FetchResult default cache_hit is False."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="content", strategy_used="static", url="https://example.com"
        )
        assert result.cache_hit is False

    def test_fetch_result_default_screenshot_path(self) -> None:
        """Test FetchResult default screenshot_path is None."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="content", strategy_used="static", url="https://example.com"
        )
        assert result.screenshot_path is None

    def test_fetch_result_default_multi_source_fields(self) -> None:
        """Test FetchResult default multi-source fields."""
        from markitai.fetch import FetchResult

        result = FetchResult(
            content="content", strategy_used="static", url="https://example.com"
        )
        assert result.static_content is None
        assert result.browser_content is None


class TestFetchWithFallbackJsDetection:
    """Tests for JS detection in fallback flow."""

    @pytest.mark.asyncio
    async def test_static_success_with_js_required_falls_back(self) -> None:
        """Test that static success with JS-required content triggers fallback."""
        from markitai.fetch import FetchResult, _fetch_with_fallback

        mock_config = type(
            "MockConfig",
            (),
            {
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {"timeout": 30000, "wait_for": "load", "extra_wait_ms": 0},
                )(),
                "auto_proxy": False,
            },
        )()

        # Content that requires JS (too short)
        js_required_content = "Loading..."

        static_result = FetchResult(
            content=js_required_content,
            strategy_used="static",
            url="https://example.com",
        )

        mock_pw_result = type(
            "PlaywrightResult",
            (),
            {
                "content": "# Full Content after browser render",
                "title": "Test",
                "final_url": "https://example.com",
                "metadata": {},
                "screenshot_path": None,
            },
        )()

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=static_result,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                new_callable=AsyncMock,
                return_value=mock_pw_result,
            ),
            patch("markitai.fetch.get_spa_domain_cache") as mock_spa_cache,
        ):
            mock_cache_instance = MagicMock()
            mock_spa_cache.return_value = mock_cache_instance

            result = await _fetch_with_fallback(
                "https://example.com",
                mock_config,
                start_with_browser=False,
            )

            # Should have used playwright after static detected JS required
            assert result.strategy_used == "playwright"
            # Should have recorded the SPA domain
            mock_cache_instance.record_spa_domain.assert_called()


class TestJinaResponseParsing:
    """Tests for Jina response parsing edge cases."""

    @pytest.mark.asyncio
    async def test_jina_empty_data_structure(self) -> None:
        """Test Jina with empty data structure."""
        from markitai import fetch
        from markitai.fetch import FetchError, fetch_with_jina

        fetch._jina_client = None

        json_data = {"code": 200, "data": {}}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with patch("markitai.fetch._get_jina_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_jina("https://example.com")

            assert (
                "No content" in str(exc_info.value)
                or "empty" in str(exc_info.value).lower()
            )

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_jina_whitespace_only_content(self) -> None:
        """Test Jina with whitespace-only content."""
        from markitai import fetch
        from markitai.fetch import FetchError, fetch_with_jina

        fetch._jina_client = None

        json_data = {"code": 200, "data": {"content": "   \n\t  "}}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with patch("markitai.fetch._get_jina_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_jina("https://example.com")

            assert "No content" in str(exc_info.value)

        fetch._jina_client = None

    @pytest.mark.asyncio
    async def test_jina_whitespace_title_cleaned(self) -> None:
        """Test Jina cleans whitespace from title."""
        from markitai import fetch
        from markitai.fetch import fetch_with_jina

        fetch._jina_client = None

        json_data = {
            "code": 200,
            "data": {"title": "   ", "content": "Valid content here"},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = json_data
        mock_response.text = ""

        with patch("markitai.fetch._get_jina_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await fetch_with_jina("https://example.com")

            # Whitespace-only title should be None
            assert result.title is None

        fetch._jina_client = None
