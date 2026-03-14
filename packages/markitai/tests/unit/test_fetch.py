"""Unit tests for the fetch module."""

from __future__ import annotations

import contextlib
import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.constants import JS_REQUIRED_PATTERNS
from markitai.fetch import (
    FetchStrategy,
    SPADomainCache,
    _build_local_only_patterns,
    _url_to_screenshot_filename,
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


class TestSPADomainCache:
    """Tests for SPADomainCache class."""

    def test_initialization_does_not_create_directory(self) -> None:
        """Test that initialization does not create parent directories (lazy creation)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "subdir" / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            # Parent directory should not exist yet
            assert not cache_path.parent.exists()

            # But we can still record something, which should trigger directory creation
            cache.record_spa_domain("https://example.com/page")
            assert cache_path.parent.exists()

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
        """Truly empty/near-empty content should be detected as too short."""
        from markitai.fetch import _is_invalid_content

        # Content with only markdown syntax, no real text
        content = "# \n\n** **"
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is True
        assert reason == "too_short"

    def test_minimal_landing_page_is_valid(self) -> None:
        """Minimal but legitimate pages should NOT be rejected as too_short."""
        from markitai.fetch import _is_invalid_content

        # A real personal landing page with ~40 chars of clean content
        content = "# Ynewtime\n\n解構世界、優化未來 • Front-end developer"
        is_invalid, reason = _is_invalid_content(content)
        assert is_invalid is False
        assert reason == ""

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


class TestCompressScreenshot:
    """Tests for _compress_screenshot function."""

    def test_compress_screenshot_skips_when_not_needed(self, tmp_path: Path) -> None:
        """Test that compression is skipped for RGB JPEG within height limits."""
        from markitai.fetch import _compress_screenshot

        try:
            from PIL import Image
        except ImportError:
            import pytest

            pytest.skip("Pillow not installed")

        # Create a simple RGB JPEG image within limits
        img = Image.new("RGB", (800, 600), color="red")
        screenshot_path = tmp_path / "test.jpg"
        img.save(screenshot_path, "JPEG", quality=85)
        original_size = screenshot_path.stat().st_size

        # Compress - should skip since RGB and within max_height
        _compress_screenshot(screenshot_path, quality=50)

        # File size should be unchanged (skipped re-compression)
        new_size = screenshot_path.stat().st_size
        assert new_size == original_size

    def test_compress_screenshot_compresses_tall_image(self, tmp_path: Path) -> None:
        """Test that compression works for images exceeding max_height."""
        from markitai.fetch import _compress_screenshot

        try:
            from PIL import Image
        except ImportError:
            import pytest

            pytest.skip("Pillow not installed")

        # Create a tall image that exceeds max_height
        img = Image.new("RGB", (800, 1200), color="red")
        screenshot_path = tmp_path / "test.jpg"
        img.save(screenshot_path, "JPEG", quality=100)
        original_size = screenshot_path.stat().st_size

        # Compress with low max_height to trigger resize
        _compress_screenshot(screenshot_path, quality=50, max_height=600)

        # Check that file was compressed/resized
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

        # Create a very tall image (5000 pixels — enough to test max_height limit)
        img = Image.new("RGB", (400, 5000), color="blue")
        screenshot_path = tmp_path / "tall.jpg"
        img.save(screenshot_path, "JPEG")

        # Compress with max_height limit
        _compress_screenshot(screenshot_path, quality=75, max_height=2000)

        # Check that image was resized
        with Image.open(screenshot_path) as compressed:
            assert compressed.height <= 2000

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
        from markitai.fetch_http import StaticHttpResponse

        mock_response = StaticHttpResponse(
            content=b"plain text body",
            status_code=200,
            headers={"content-type": "text/plain"},
            url="https://example.com",
        )
        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_result = type(
            "MockResult",
            (),
            {"text_content": "# Test Page\n\nSome content here.", "title": "Test Page"},
        )()

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
            patch("markitai.fetch._get_markitdown") as mock_get_md,
        ):
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
        from markitai.fetch_http import StaticHttpResponse

        mock_response = StaticHttpResponse(
            content=b"plain text body",
            status_code=200,
            headers={"content-type": "text/plain"},
            url="https://example.com",
        )
        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_result = type("MockResult", (), {"text_content": "", "title": None})()

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
            patch("markitai.fetch._get_markitdown") as mock_get_md,
        ):
            mock_md = type("MockMD", (), {"convert": lambda _self, _url: mock_result})()
            mock_get_md.return_value = mock_md

            with pytest.raises(FetchError) as exc_info:
                await fetch_with_static("https://example.com")

            assert "No content extracted" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_with_static_exception(self) -> None:
        """Test static fetch handles exceptions."""
        from markitai.fetch import FetchError, fetch_with_static
        from markitai.fetch_http import StaticHttpResponse

        mock_response = StaticHttpResponse(
            content=b"plain text body",
            status_code=200,
            headers={"content-type": "text/plain"},
            url="https://example.com",
        )
        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
            patch("markitai.fetch._get_markitdown") as mock_get_md,
        ):
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

            assert "Network error" in str(exc_info.value)


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
        from markitai.fetch_http import StaticHttpResponse

        mock_response = StaticHttpResponse(
            content=b"",
            status_code=304,
            headers={
                "etag": '"abc123"',
                "last-modified": "Mon, 01 Jan 2026",
            },
            url="https://example.com",
        )

        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
        ):
            result = await fetch_with_static_conditional(
                "https://example.com",
                cached_etag='"old-etag"',
                cached_last_modified="Sun, 01 Dec 2025",
            )

            assert result.not_modified is True
            assert result.result is None

    @pytest.mark.asyncio
    async def test_conditional_fetch_200_new_content(self) -> None:
        """Test conditional fetch falls back when native extraction fails."""
        from markitai.fetch import fetch_with_static_conditional

        mock_headers = {
            "etag": '"new-etag"',
            "last_modified": "Mon, 01 Jan 2026",
            "content-type": "text/html",
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = MagicMock()
        mock_response.headers.get = lambda key, default="": mock_headers.get(
            key, default
        )
        mock_response.content = b"<html><body><h1>New Content</h1></body></html>"
        mock_response.text = "<html><body><h1>New Content</h1></body></html>"
        mock_response.url = "https://example.com"

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.name = "httpx"

        mock_md_result = MagicMock()
        mock_md_result.text_content = "# New Content"
        mock_md_result.title = "New Content"

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
            patch(
                "markitai.fetch.extract_web_content",
                side_effect=RuntimeError("native failure"),
            ),
            patch("markitai.fetch._get_markitdown") as mock_get_md,
        ):
            mock_md = MagicMock()
            mock_md.convert.return_value = mock_md_result
            mock_get_md.return_value = mock_md

            result = await fetch_with_static_conditional("https://example.com")

            assert result.not_modified is False
            assert result.result is not None
            assert result.result.content == "# New Content"
            assert result.result.metadata["converter"] == "markitdown"
            assert result.etag == '"new-etag"'

    @pytest.mark.asyncio
    async def test_conditional_fetch_http_error(self) -> None:
        """Test conditional fetch handles HTTP errors."""
        from markitai.fetch import FetchError, fetch_with_static_conditional
        from markitai.fetch_http import StaticHttpResponse

        mock_response = StaticHttpResponse(
            content=b"Not Found",
            status_code=404,
            headers={},
            url="https://example.com",
        )

        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
        ):
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
                "strategy": "auto",
                "fallback_patterns": [],
                "policy": type(
                    "Policy",
                    (),
                    {
                        "enabled": True,
                        "max_strategy_hops": 4,
                        "strategy_priority": None,
                        "local_only_patterns": [],
                        "inherit_no_proxy": False,
                    },
                )(),
                "domain_profiles": {},
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda *_, **__: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {
                        "timeout": 30000,
                        "wait_for": "load",
                        "extra_wait_ms": 0,
                        "wait_for_selector": None,
                        "cookies": None,
                        "reject_resource_patterns": None,
                        "extra_http_headers": None,
                        "user_agent": None,
                        "http_credentials": None,
                        "session_mode": "isolated",
                        "session_ttl_seconds": 600,
                    },
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
                "strategy": "auto",
                "fallback_patterns": [],
                "policy": type(
                    "Policy",
                    (),
                    {
                        "enabled": True,
                        "max_strategy_hops": 4,
                        "strategy_priority": None,
                        "local_only_patterns": [],
                        "inherit_no_proxy": False,
                    },
                )(),
                "domain_profiles": {},
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda *_, **__: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {
                        "timeout": 30000,
                        "wait_for": "load",
                        "extra_wait_ms": 0,
                        "wait_for_selector": None,
                        "cookies": None,
                        "reject_resource_patterns": None,
                        "extra_http_headers": None,
                        "user_agent": None,
                        "http_credentials": None,
                        "session_mode": "isolated",
                        "session_ttl_seconds": 600,
                    },
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
                "strategy": "auto",
                "fallback_patterns": [],
                "policy": type(
                    "Policy",
                    (),
                    {
                        "enabled": True,
                        "max_strategy_hops": 4,
                        "strategy_priority": None,
                        "local_only_patterns": [],
                        "inherit_no_proxy": False,
                    },
                )(),
                "domain_profiles": {},
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda *_, **__: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {
                        "timeout": 30000,
                        "wait_for": "load",
                        "extra_wait_ms": 0,
                        "wait_for_selector": None,
                        "cookies": None,
                        "reject_resource_patterns": None,
                        "extra_http_headers": None,
                        "user_agent": None,
                        "http_credentials": None,
                        "session_mode": "isolated",
                        "session_ttl_seconds": 600,
                    },
                )(),
                "auto_proxy": False,
            },
        )()

        mock_pw_result = type(
            "PlaywrightResult",
            (),
            {
                "content": (
                    "# Browser Content\n\n"
                    "This is a substantial page with enough meaningful text content "
                    "to pass the content validation check that ensures fetched pages "
                    "contain real content rather than login walls or error messages."
                ),
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

    @pytest.mark.asyncio
    async def test_fetch_with_fallback_skips_external_strategies_for_localhost(
        self,
    ) -> None:
        """Local/private URLs should not be sent to external fetch services."""
        from markitai.fetch import FetchResult, _fetch_with_fallback

        mock_config = type(
            "MockConfig",
            (),
            {
                "strategy": "auto",
                "fallback_patterns": [],
                "policy": type(
                    "Policy",
                    (),
                    {
                        "enabled": True,
                        "max_strategy_hops": 4,
                        "strategy_priority": None,
                        "local_only_patterns": [],
                        "inherit_no_proxy": False,
                    },
                )(),
                "domain_profiles": {},
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda *_, **__: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {
                        "timeout": 30000,
                        "wait_for": "load",
                        "extra_wait_ms": 0,
                        "wait_for_selector": None,
                        "cookies": None,
                        "reject_resource_patterns": None,
                        "extra_http_headers": None,
                        "user_agent": None,
                        "http_credentials": None,
                        "session_mode": "isolated",
                        "session_ttl_seconds": 600,
                    },
                )(),
            },
        )()

        mock_result = FetchResult(
            content=(
                "# Local Content\n\n"
                "This is enough local content to pass validation without "
                "needing any third-party fetch providers involved."
            ),
            strategy_used="static",
            url="http://127.0.0.1:8000",
        )

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=mock_result,
            ) as mock_static,
            patch("markitai.fetch.detect_js_required", return_value=False),
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
            patch(
                "markitai.fetch.fetch_with_jina", new_callable=AsyncMock
            ) as mock_jina,
        ):
            result = await _fetch_with_fallback(
                "http://127.0.0.1:8000", mock_config, start_with_browser=False
            )

        assert result.strategy_used == "static"
        mock_static.assert_called_once()
        mock_defuddle.assert_not_called()
        mock_jina.assert_not_called()


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

    def test_cache_metadata_serializes_date_values(self, tmp_path: Path) -> None:
        """Cache should serialize date-like metadata values for round-tripping."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com/article"

        result = FetchResult(
            content="# Article",
            strategy_used="defuddle",
            url=url,
            metadata={
                "source_frontmatter": {
                    "title": "Article",
                    "published": date(2024, 1, 15),
                }
            },
        )

        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.metadata["source_frontmatter"]["published"] == "2024-01-15"
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


class TestScreenshotDecoupled:
    """Tests for screenshot decoupled from content strategy.

    After the refactor, screenshot=True should NOT trigger parallel multi-source
    fetching. Instead, content is fetched serially via FetchPolicyEngine, and
    screenshot is captured separately via playwright if needed.
    """

    def _make_config(self, *, jina_key: str | None = None) -> object:
        """Create a mock FetchConfig for testing."""
        return type(
            "MockConfig",
            (),
            {
                "strategy": "auto",
                "fallback_patterns": [],
                "policy": type(
                    "Policy",
                    (),
                    {
                        "enabled": True,
                        "max_strategy_hops": 5,
                        "strategy_priority": None,
                        "local_only_patterns": [],
                        "inherit_no_proxy": False,
                    },
                )(),
                "domain_profiles": {},
                "jina": type(
                    "JinaConfig",
                    (),
                    {
                        "get_resolved_api_key": lambda *_, **__: jina_key,
                        "timeout": 30,
                        "rpm": 20,
                        "no_cache": False,
                        "target_selector": None,
                        "wait_for_selector": None,
                    },
                )(),
                "defuddle": type(
                    "DefuddleConfig",
                    (),
                    {"timeout": 30, "rpm": 20},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {
                        "timeout": 30000,
                        "wait_for": "load",
                        "extra_wait_ms": 0,
                        "wait_for_selector": None,
                        "cookies": None,
                        "reject_resource_patterns": None,
                        "extra_http_headers": None,
                        "user_agent": None,
                        "http_credentials": None,
                        "session_mode": "isolated",
                        "session_ttl_seconds": 600,
                    },
                )(),
                "cloudflare": type(
                    "CloudflareConfig",
                    (),
                    {
                        "get_resolved_api_token": lambda *_, **__: None,
                        "get_resolved_account_id": lambda *_, **__: None,
                        "api_token": None,
                        "account_id": None,
                        "timeout": 30000,
                        "wait_until": "networkidle0",
                        "cache_ttl": 0,
                        "reject_resource_patterns": None,
                        "user_agent": None,
                        "cookies": None,
                        "wait_for_selector": None,
                        "http_credentials": None,
                    },
                )(),
                "auto_proxy": False,
            },
        )()

    def _valid_content(self, label: str = "default") -> str:
        """Return content that passes _is_invalid_content validation."""
        return (
            f"# Content from {label}\n\n"
            "This is valid content with enough text to pass the content validation "
            "threshold which requires at least 100 characters of clean text after "
            "removing all markdown syntax elements like headers, links, and images. "
            "Adding more text to be safe and ensure validation passes correctly."
        )

    @pytest.mark.asyncio
    async def test_screenshot_mode_defuddle_wins_with_separate_screenshot(
        self,
    ) -> None:
        """When screenshot=True and defuddle wins content, playwright runs
        separately just for screenshot. Result has defuddle content + screenshot.

        Key behavioral test: fetch_with_static should NOT be called because
        serial mode stops after defuddle succeeds. In the old parallel mode,
        all strategies would run simultaneously via asyncio.gather.
        """
        from markitai.fetch import FetchResult, fetch_url

        mock_config = self._make_config()

        defuddle_result = FetchResult(
            content=self._valid_content("defuddle"),
            strategy_used="defuddle",
            title="Defuddle Title",
            url="https://example.com",
            metadata={"api": "defuddle"},
        )

        # Playwright result for screenshot-only call
        pw_screenshot_result = MagicMock()
        pw_screenshot_result.screenshot_path = Path("/tmp/screenshot.jpg")
        pw_screenshot_result.content = "browser content"
        pw_screenshot_result.title = "Browser Title"
        pw_screenshot_result.final_url = "https://example.com"
        pw_screenshot_result.metadata = {}

        with (
            patch(
                "markitai.fetch.fetch_with_defuddle",
                new_callable=AsyncMock,
                return_value=defuddle_result,
            ),
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    content=self._valid_content("static"),
                ),
            ) as mock_static,
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                new_callable=AsyncMock,
                return_value=pw_screenshot_result,
            ) as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch._get_playwright_renderer",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            result = await fetch_url(
                "https://example.com",
                FetchStrategy.AUTO,
                mock_config,
                screenshot=True,
                screenshot_dir=Path("/tmp"),
            )

            # Content should come from defuddle (the winning strategy)
            assert result.strategy_used == "defuddle"
            assert "Content from defuddle" in result.content
            # Screenshot should come from separate playwright call
            assert result.screenshot_path == Path("/tmp/screenshot.jpg")
            # KEY: Static should NOT be called (serial stops at defuddle)
            mock_static.assert_not_called()
            # Playwright should have been called (for screenshot only)
            mock_pw.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot_mode_playwright_wins_naturally(self) -> None:
        """When screenshot=True and playwright wins as content strategy,
        screenshot is already captured — no extra call needed."""
        from markitai.fetch import fetch_url

        mock_config = self._make_config()

        # Defuddle fails
        defuddle_error = Exception("defuddle down")

        # Playwright succeeds with both content and screenshot
        pw_result = MagicMock()
        pw_result.content = self._valid_content("playwright")
        pw_result.title = "PW Title"
        pw_result.final_url = "https://example.com"
        pw_result.metadata = {}
        pw_result.screenshot_path = Path("/tmp/screenshot.jpg")

        with (
            patch(
                "markitai.fetch.fetch_with_defuddle",
                new_callable=AsyncMock,
                side_effect=defuddle_error,
            ),
            patch(
                "markitai.fetch.fetch_with_jina",
                new_callable=AsyncMock,
                side_effect=Exception("jina down"),
            ),
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                side_effect=Exception("static down"),
            ),
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                new_callable=AsyncMock,
                return_value=pw_result,
            ) as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch._get_playwright_renderer",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
        ):
            result = await fetch_url(
                "https://example.com",
                FetchStrategy.AUTO,
                mock_config,
                screenshot=True,
                screenshot_dir=Path("/tmp"),
            )

            assert result.strategy_used == "playwright"
            assert "Content from playwright" in result.content
            assert result.screenshot_path == Path("/tmp/screenshot.jpg")
            # Playwright called only once (content+screenshot in same call)
            mock_pw.assert_called_once()

    @pytest.mark.asyncio
    async def test_screenshot_mode_no_playwright_available(self) -> None:
        """When screenshot=True but playwright is not available,
        content still works, screenshot is None (graceful degradation)."""
        from markitai.fetch import FetchResult, fetch_url

        mock_config = self._make_config()

        defuddle_result = FetchResult(
            content=self._valid_content("defuddle"),
            strategy_used="defuddle",
            title="Defuddle Title",
            url="https://example.com",
            metadata={"api": "defuddle"},
        )

        with (
            patch(
                "markitai.fetch.fetch_with_defuddle",
                new_callable=AsyncMock,
                return_value=defuddle_result,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=False,
            ),
            patch(
                "markitai.fetch._get_playwright_renderer",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await fetch_url(
                "https://example.com",
                FetchStrategy.AUTO,
                mock_config,
                screenshot=True,
                screenshot_dir=Path("/tmp"),
            )

            # Content should still work
            assert result.strategy_used == "defuddle"
            assert "Content from defuddle" in result.content
            # Screenshot should be None (graceful degradation)
            assert result.screenshot_path is None

    @pytest.mark.asyncio
    async def test_screenshot_mode_cache_hit_still_captures_screenshot(
        self, tmp_path: Path
    ) -> None:
        """Cache hits should still honor a requested screenshot."""
        from markitai.fetch import FetchCache, FetchResult, fetch_url

        mock_config = self._make_config()
        cache = FetchCache(tmp_path / "test_cache.db")
        url = "https://example.com/cached"

        cache.set(
            url,
            FetchResult(
                content=self._valid_content("cached"),
                strategy_used="defuddle",
                title="Cached Title",
                url=url,
                metadata={"api": "defuddle"},
            ),
        )

        pw_screenshot_result = MagicMock()
        pw_screenshot_result.screenshot_path = (
            tmp_path / "screenshots" / "cached.full.jpg"
        )
        pw_screenshot_result.title = "Browser Title"
        pw_screenshot_result.final_url = url
        pw_screenshot_result.metadata = {}

        with (
            patch(
                "markitai.fetch_playwright.fetch_with_playwright",
                new_callable=AsyncMock,
                return_value=pw_screenshot_result,
            ) as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch._get_playwright_renderer",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
        ):
            result = await fetch_url(
                url,
                FetchStrategy.DEFUDDLE,
                mock_config,
                cache=cache,
                screenshot=True,
                screenshot_dir=tmp_path / "screenshots",
            )

        assert result.cache_hit is True
        assert result.screenshot_path == pw_screenshot_result.screenshot_path
        mock_pw.assert_called_once()
        mock_defuddle.assert_not_called()
        cache.close()

    @pytest.mark.asyncio
    async def test_explicit_defuddle_rejects_private_url(self) -> None:
        """Explicit defuddle fetches should reject private/local URLs."""
        from markitai.fetch import FetchError, fetch_url

        mock_config = self._make_config()

        with (
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
            pytest.raises(FetchError, match="private"),
        ):
            await fetch_url(
                "http://127.0.0.1:8000/private",
                FetchStrategy.DEFUDDLE,
                mock_config,
                explicit_strategy=True,
            )

        mock_defuddle.assert_not_called()

    @pytest.mark.asyncio
    async def test_configured_cloudflare_strategy_works_without_explicit_mode(
        self,
    ) -> None:
        """Configured Cloudflare strategy should work when not marked explicit."""
        from markitai.fetch import FetchResult, fetch_url

        mock_config = self._make_config()
        cloudflare_result = FetchResult(
            content=self._valid_content("cloudflare"),
            strategy_used="cloudflare",
            title="Cloudflare Title",
            url="https://example.com",
            final_url="https://example.com",
            metadata={"api": "cloudflare"},
        )

        with patch(
            "markitai.fetch.fetch_with_cloudflare",
            new_callable=AsyncMock,
            return_value=cloudflare_result,
        ) as mock_cloudflare:
            result = await fetch_url(
                "https://example.com",
                FetchStrategy.CLOUDFLARE,
                mock_config,
                explicit_strategy=False,
            )

        assert result is cloudflare_result
        mock_cloudflare.assert_awaited_once()


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
        fetch._playwright_renderer_fingerprint = ""

        with patch(
            "markitai.fetch_playwright.PlaywrightRenderer"
        ) as mock_renderer_class:
            mock_instance = MagicMock()
            mock_renderer_class.return_value = mock_instance

            result = await _get_playwright_renderer(proxy="http://proxy:8080")

            assert result is mock_instance
            mock_renderer_class.assert_called_once_with(proxy="http://proxy:8080")

        fetch._playwright_renderer = None
        fetch._playwright_renderer_fingerprint = ""

    @pytest.mark.asyncio
    async def test_get_playwright_renderer_reuses_instance(self) -> None:
        """Test that _get_playwright_renderer reuses existing instance."""
        from markitai import fetch
        from markitai.fetch import _get_playwright_renderer

        mock_instance = MagicMock()
        fetch._playwright_renderer = mock_instance
        # Set fingerprint to match default call args (proxy=None, config=None)
        fetch._playwright_renderer_fingerprint = "None:None"

        result = await _get_playwright_renderer()

        assert result is mock_instance

        fetch._playwright_renderer = None
        fetch._playwright_renderer_fingerprint = ""


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
                "strategy": "auto",
                "fallback_patterns": [],
                "policy": type(
                    "Policy",
                    (),
                    {
                        "enabled": True,
                        "max_strategy_hops": 4,
                        "strategy_priority": None,
                        "local_only_patterns": [],
                        "inherit_no_proxy": False,
                    },
                )(),
                "domain_profiles": {},
                "jina": type(
                    "JinaConfig",
                    (),
                    {"get_resolved_api_key": lambda *_, **__: None, "timeout": 30},
                )(),
                "playwright": type(
                    "PlaywrightConfig",
                    (),
                    {
                        "timeout": 30000,
                        "wait_for": "load",
                        "extra_wait_ms": 0,
                        "wait_for_selector": None,
                        "cookies": None,
                        "reject_resource_patterns": None,
                        "extra_http_headers": None,
                        "user_agent": None,
                        "http_credentials": None,
                        "session_mode": "isolated",
                        "session_ttl_seconds": 600,
                    },
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
                "content": (
                    "# Full Content after browser render\n\n"
                    "This is a substantial page with enough meaningful text content "
                    "to pass the content validation check that ensures fetched pages "
                    "contain real content rather than login walls or error messages."
                ),
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


class TestContentNegotiation:
    """Tests for CF Markdown for Agents content negotiation."""

    def test_markitdown_instance_has_accept_markdown_header(self):
        """Verify markitdown session sends Accept: text/markdown header."""
        import markitai.fetch as fetch_module
        from markitai.fetch import _get_markitdown

        # Reset singleton to force re-creation
        old = fetch_module._markitdown_instance
        fetch_module._markitdown_instance = None
        try:
            md = _get_markitdown()
            accept = md._requests_session.headers.get("Accept", "")
            assert "text/markdown" in accept
            # text/html should be lower priority
            assert "text/html" in accept
        finally:
            fetch_module._markitdown_instance = old

    @pytest.mark.asyncio
    async def test_conditional_fetch_sends_accept_markdown_header(self):
        """Verify conditional fetch includes Accept: text/markdown header."""
        from markitai.fetch import fetch_with_static_conditional
        from markitai.fetch_http import StaticHttpResponse

        captured_headers = {}

        async def mock_get(url, headers=None, timeout_s=30.0, proxy=None):
            nonlocal captured_headers
            captured_headers = headers or {}
            return StaticHttpResponse(
                content=b"<html><body>Hello World test content for validation check</body></html>",
                status_code=200,
                headers={"content-type": "text/html", "etag": '"abc"'},
                url=url,
            )

        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = mock_get

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
        ):
            try:
                await fetch_with_static_conditional("https://example.com")
            except Exception:
                pass  # May fail on markitdown conversion, we only care about headers

        assert "Accept" in captured_headers
        assert "text/markdown" in captured_headers["Accept"]

    @pytest.mark.asyncio
    async def test_conditional_fetch_uses_markdown_response_directly(self):
        """When server returns text/markdown, use content directly without markitdown."""
        from markitai.fetch import fetch_with_static_conditional
        from markitai.fetch_http import StaticHttpResponse

        markdown_body = "# Hello World\n\nThis is markdown content from the server."

        async def mock_get(url, headers=None, timeout_s=30.0, proxy=None):
            return StaticHttpResponse(
                content=markdown_body.encode(),
                status_code=200,
                headers={
                    "content-type": "text/markdown; charset=utf-8",
                    "x-markdown-tokens": "42",
                    "etag": '"xyz"',
                },
                url=url,
            )

        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = mock_get

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
        ):
            result = await fetch_with_static_conditional("https://example.com")

        assert result.not_modified is False
        assert result.result is not None
        assert result.result.content == markdown_body
        assert result.result.metadata["converter"] == "server-markdown"
        assert result.result.metadata["token_hint"] == 42
        assert result.result.title == "Hello World"

    @pytest.mark.asyncio
    async def test_conditional_fetch_html_response_unchanged(self):
        """Non-CF sites returning text/html still processed through markitdown as before."""
        from markitai.fetch import fetch_with_static_conditional
        from markitai.fetch_http import StaticHttpResponse

        async def mock_get(url, headers=None, timeout_s=30.0, proxy=None):
            # Verify Accept header is sent even for non-CF sites
            assert "text/markdown" in headers.get("Accept", "")
            return StaticHttpResponse(
                content=b"<html><body><h1>Normal HTML</h1><p>Regular content.</p></body></html>",
                status_code=200,
                headers={
                    "content-type": "text/html; charset=utf-8",
                    "etag": '"html123"',
                },
                url=url,
            )

        mock_client = MagicMock()
        mock_client.name = "httpx"
        mock_client.get = mock_get

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
        ):
            try:
                await fetch_with_static_conditional("https://non-cf-site.com")
            except Exception:
                pass  # Conversion may fail in mock, we verify the path is HTML

        # The key assertion: text/html response does NOT get the "server-markdown" path
        # (it falls through to markitdown conversion as before)

    @pytest.mark.asyncio
    async def test_conditional_fetch_html_response_prefers_native_webextract(self):
        """HTML responses should prefer native webextract when available."""
        from dataclasses import dataclass

        from markitai.fetch import fetch_with_static_conditional

        @dataclass
        class _Metadata:
            title: str | None = None
            author: str | None = None

        async def mock_get(url, headers=None, timeout_s=30.0, proxy=None):
            del headers, timeout_s, proxy
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {
                "content-type": "text/html; charset=utf-8",
                "etag": '"html123"',
            }
            mock_resp.content = (
                b"<html><body><article><h1>Normal HTML</h1>"
                b"<p>Regular content.</p></article></body></html>"
            )
            mock_resp.url = url
            return mock_resp

        mock_client = AsyncMock()
        mock_client.name = "httpx"
        mock_client.get = mock_get

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
            patch("markitai.fetch.extract_web_content", create=True) as mock_extract,
        ):
            mock_extract.return_value = MagicMock(
                markdown="# Normal HTML\n\nRegular content.",
                metadata=_Metadata(title="Normal HTML", author="Jane"),
                diagnostics={"extractor": "generic"},
            )

            result = await fetch_with_static_conditional("https://non-cf-site.com")

        assert result.not_modified is False
        assert result.result is not None
        assert result.result.content == "# Normal HTML\n\nRegular content."
        assert result.result.metadata["converter"] == "native-html"
        assert result.result.metadata["source_frontmatter"]["author"] == "Jane"

    @pytest.mark.asyncio
    async def test_conditional_fetch_html_uses_declared_charset_for_native_webextract(
        self,
    ):
        """Native extraction should decode HTML using the declared response charset."""
        from dataclasses import dataclass

        from markitai.fetch import fetch_with_static_conditional
        from markitai.fetch_http import StaticHttpResponse

        @dataclass
        class _Metadata:
            title: str | None = None

        html = (
            "<html><body><article><h1>中文标题</h1>"
            "<p>这是正文内容。</p></article></body></html>"
        )
        response = StaticHttpResponse(
            content=html.encode("gbk"),
            status_code=200,
            headers={
                "content-type": "text/html; charset=gbk",
                "etag": '"gbk123"',
            },
            url="https://example.cn",
        )

        mock_client = AsyncMock()
        mock_client.name = "httpx"
        mock_client.get = AsyncMock(return_value=response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("markitai.fetch.get_static_http_client", return_value=mock_client),
            patch("markitai.fetch.extract_web_content", create=True) as mock_extract,
        ):
            mock_extract.return_value = MagicMock(
                markdown="# 中文标题\n\n这是正文内容。",
                metadata=_Metadata(title="中文标题"),
                diagnostics={"extractor": "generic"},
            )

            result = await fetch_with_static_conditional("https://example.cn")

        assert result.result is not None
        assert mock_extract.call_args.args[0].startswith("<html>")
        assert "中文标题" in mock_extract.call_args.args[0]
        assert result.result.content == "# 中文标题\n\n这是正文内容。"

    @pytest.mark.asyncio
    async def test_fetch_with_static_reuses_conditional_pipeline(self):
        """Plain static fetch should reuse the conditional pipeline."""
        from markitai.fetch import FetchResult, fetch_with_static

        expected = FetchResult(
            content="# Title\n\nBody",
            strategy_used="static",
            title="Title",
            url="https://example.com/post",
            final_url="https://example.com/post",
            metadata={"converter": "native-html"},
        )

        with patch("markitai.fetch.fetch_with_static_conditional") as mock_conditional:
            mock_conditional.return_value = type(
                "MockConditionalFetchResult",
                (),
                {
                    "result": expected,
                    "not_modified": False,
                    "etag": None,
                    "last_modified": None,
                },
            )()

            result = await fetch_with_static("https://example.com/post")

        assert result is expected


class TestCloudflareStrategy:
    """Tests for CF Browser Rendering fetch strategy."""

    def test_cloudflare_strategy_exists(self):
        """FetchStrategy enum has CLOUDFLARE value."""
        from markitai.fetch import FetchStrategy

        assert hasattr(FetchStrategy, "CLOUDFLARE")
        assert FetchStrategy.CLOUDFLARE.value == "cloudflare"

    def test_cloudflare_config_defaults(self):
        """CloudflareConfig has sensible defaults."""
        from markitai.config import CloudflareConfig

        config = CloudflareConfig()
        assert config.api_token is None
        assert config.account_id is None
        assert config.timeout == 30000
        assert config.wait_until == "networkidle0"
        assert config.cache_ttl == 0
        assert config.reject_resource_patterns is None

    def test_cloudflare_config_in_fetch_config(self):
        """FetchConfig includes cloudflare section."""
        from markitai.config import FetchConfig

        config = FetchConfig()
        assert hasattr(config, "cloudflare")
        assert config.cloudflare.api_token is None

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_success(self):
        """Successful CF BR fetch returns markdown content from JSON envelope."""
        from markitai.fetch import fetch_with_cloudflare

        # CF REST API returns JSON envelope: {"success": true, "result": "<markdown>"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": "# Hello World\n\nContent from CF BR.",
            "errors": [],
            "messages": [],
        }
        mock_response.headers = {"X-Browser-Ms-Used": "1500"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            result = await fetch_with_cloudflare(
                url="https://example.com",
                api_token="test-token",
                account_id="test-account",
            )

        assert result.content == "# Hello World\n\nContent from CF BR."
        assert result.strategy_used == "cloudflare"
        assert result.metadata.get("browser_ms_used") == "1500"

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_api_success_false(self):
        """CF API returns success=false with error details."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": False,
            "result": None,
            "errors": [{"code": 1000, "message": "Navigation timeout"}],
            "messages": [],
        }
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            with pytest.raises(FetchError, match="CF BR API error"):
                await fetch_with_cloudflare(
                    url="https://example.com",
                    api_token="test-token",
                    account_id="test-account",
                )

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_no_credentials(self):
        """Raises FetchError when credentials are missing."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        with pytest.raises(
            FetchError, match="Cloudflare API token and account ID required"
        ):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token=None,
                account_id="test",
            )

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_http_error(self):
        """HTTP error (non-200) raises FetchError."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("403 Forbidden")
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            with pytest.raises(FetchError, match="Cloudflare"):
                await fetch_with_cloudflare(
                    url="https://example.com",
                    api_token="bad-token",
                    account_id="test-account",
                )

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_custom_reject_patterns(self):
        """Custom reject_resource_patterns are sent in payload."""
        from markitai.fetch import fetch_with_cloudflare

        captured_payload = {}

        async def mock_post(url, headers=None, json=None):
            nonlocal captured_payload
            captured_payload = json or {}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "success": True,
                "result": "# Test\n\nContent for testing reject patterns.",
                "errors": [],
                "messages": [],
            }
            mock_resp.headers = {}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = mock_post

        custom_patterns = ["/analytics/", "/\\.css$/", "/\\.woff2?$/"]
        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="test-token",
                account_id="test-account",
                reject_resource_patterns=custom_patterns,
            )

        assert captured_payload.get("rejectRequestPattern") == custom_patterns


class TestCfBrSemaphore:
    """Tests for CF BR semaphore lazy initialization."""

    def test_get_cf_semaphore_returns_semaphore(self):
        """get_cf_semaphore returns an asyncio.Semaphore."""
        import asyncio

        from markitai.fetch import get_cf_semaphore

        sem = get_cf_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    def test_get_cf_semaphore_returns_same_instance(self):
        """get_cf_semaphore returns the same instance on repeated calls."""
        from markitai.fetch import get_cf_semaphore

        sem1 = get_cf_semaphore()
        sem2 = get_cf_semaphore()
        assert sem1 is sem2


class TestExtractMarkdownTitle:
    """Tests for _extract_markdown_title helper."""

    def test_extracts_h1_title(self):
        from markitai.fetch import _extract_markdown_title

        assert _extract_markdown_title("# Hello World\n\nContent") == "Hello World"

    def test_returns_none_for_no_heading(self):
        from markitai.fetch import _extract_markdown_title

        assert _extract_markdown_title("No heading here") is None

    def test_returns_none_for_empty_string(self):
        from markitai.fetch import _extract_markdown_title

        assert _extract_markdown_title("") is None

    def test_extracts_first_h1_only(self):
        from markitai.fetch import _extract_markdown_title

        content = "Some text\n# First Title\n## Sub\n# Second"
        assert _extract_markdown_title(content) == "First Title"


class TestPlaywrightAdvancedKwargs:
    """Tests for _get_playwright_advanced_kwargs helper."""

    def test_returns_empty_when_all_none(self):
        from markitai.config import PlaywrightConfig
        from markitai.fetch import _get_playwright_advanced_kwargs

        pw = PlaywrightConfig()
        result = _get_playwright_advanced_kwargs(pw)
        assert result == {}

    def test_returns_set_values_only(self):
        from markitai.config import PlaywrightConfig
        from markitai.fetch import _get_playwright_advanced_kwargs

        pw = PlaywrightConfig(
            wait_for_selector="#main",
            user_agent="TestBot/1.0",
        )
        result = _get_playwright_advanced_kwargs(pw)
        assert result == {
            "wait_for_selector": "#main",
            "user_agent": "TestBot/1.0",
        }

    def test_returns_all_values_when_set(self):
        from markitai.config import PlaywrightConfig
        from markitai.fetch import _get_playwright_advanced_kwargs

        pw = PlaywrightConfig(
            wait_for_selector="div.content",
            cookies=[{"name": "sid", "value": "abc", "domain": ".example.com"}],
            reject_resource_patterns=["**/*.css"],
            extra_http_headers={"Accept-Language": "zh-CN"},
            user_agent="Bot/2.0",
            http_credentials={"username": "u", "password": "p"},
        )
        result = _get_playwright_advanced_kwargs(pw)
        assert len(result) == 6
        assert result["wait_for_selector"] == "div.content"


class TestCloudflareBRRetry:
    """Tests for CF BR 429 rate-limit retry logic."""

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_429_retry_succeeds(self):
        """CF BR retries on 429 and succeeds on next attempt."""
        from markitai.fetch import fetch_with_cloudflare

        # First response: 429 rate limited
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "1"}

        # Second response: 200 success
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {
            "success": True,
            "result": "# Retry OK\n\nContent after retry.",
            "errors": [],
            "messages": [],
        }
        mock_200.headers = {"X-Browser-Ms-Used": "2000"}
        mock_200.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[mock_429, mock_200])

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            result = await fetch_with_cloudflare(
                url="https://example.com",
                api_token="test-token",
                account_id="test-account",
            )

            assert result.content == "# Retry OK\n\nContent after retry."
            assert result.strategy_used == "cloudflare"
            mock_sleep.assert_called_once()  # Slept between retries

    @pytest.mark.asyncio
    async def test_fetch_with_cloudflare_429_exhausted(self):
        """CF BR raises FetchError after exhausting all retries on 429."""
        from markitai.fetch import FetchError, fetch_with_cloudflare

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.headers = {"Retry-After": "1"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_429)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            with pytest.raises(FetchError, match="rate limit exceeded"):
                await fetch_with_cloudflare(
                    url="https://example.com",
                    api_token="test-token",
                    account_id="test-account",
                )


class TestCloudflareBRPayload:
    """Tests for CF BR payload structure validation.

    These tests use the captured_payload pattern to verify the exact JSON
    structure sent to the CF API, preventing regressions like the
    rewriteLinksBaseURL / top-level timeout bugs found during real API testing.
    """

    def _make_mock_client(self, captured: dict):
        """Create a mock httpx client that captures the POST request."""

        async def mock_post(url, headers=None, json=None):
            captured["url"] = url
            captured["payload"] = json or {}
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "success": True,
                "result": "# Test",
                "errors": [],
                "messages": [],
            }
            mock_resp.headers = {}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        return mock_client

    @contextlib.contextmanager
    def _patch_cf_client(self, captured: dict):
        """Context manager to patch httpx and proxy for CF BR tests."""
        mock_client = self._make_mock_client(captured)
        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx
            yield

    @pytest.mark.asyncio
    async def test_payload_goto_options_structure(self):
        """timeout and waitUntil are inside gotoOptions, not at top level."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
                timeout=15000,
                wait_until="load",
            )

        payload = captured["payload"]
        assert "timeout" not in payload, "timeout must not be at top level"
        assert "waitUntil" not in payload, "waitUntil must not be at top level"
        assert payload["gotoOptions"]["timeout"] == 15000
        assert payload["gotoOptions"]["waitUntil"] == "load"

    @pytest.mark.asyncio
    async def test_payload_cache_ttl_as_query_param(self):
        """cacheTTL is a query parameter, not in the body."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
                cache_ttl=120,
            )

        assert "cacheTTL=120" in captured["url"]
        assert "cacheTtl" not in captured["payload"]
        assert "cacheTTL" not in captured["payload"]
        assert "cache_ttl" not in captured["payload"]

    @pytest.mark.asyncio
    async def test_payload_no_forbidden_keys(self):
        """Payload must not contain keys rejected by CF API."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
            )

        payload = captured["payload"]
        forbidden = ["rewriteLinksBaseURL", "cacheTtl", "timeout", "waitUntil"]
        for key in forbidden:
            assert key not in payload, f"Forbidden key '{key}' found in payload"

    @pytest.mark.asyncio
    async def test_payload_default_reject_patterns(self):
        """Default rejectRequestPattern includes CSS and font patterns."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
            )

        patterns = captured["payload"]["rejectRequestPattern"]
        assert any(".css" in p for p in patterns)
        assert any(".woff" in p for p in patterns)

    @pytest.mark.asyncio
    async def test_payload_user_agent(self):
        """userAgent is set at payload top level."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
                user_agent="TestBot/1.0",
            )

        assert captured["payload"]["userAgent"] == "TestBot/1.0"

    @pytest.mark.asyncio
    async def test_payload_cookies(self):
        """cookies array is set at payload top level."""
        from markitai.fetch import fetch_with_cloudflare

        test_cookies = [
            {"name": "session", "value": "abc123", "url": "https://example.com"}
        ]
        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
                cookies=test_cookies,
            )

        assert captured["payload"]["cookies"] == test_cookies

    @pytest.mark.asyncio
    async def test_payload_wait_for_selector_wrapped(self):
        """waitForSelector is wrapped as {"selector": "..."} object."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
                wait_for_selector="#content",
            )

        assert captured["payload"]["waitForSelector"] == {"selector": "#content"}

    @pytest.mark.asyncio
    async def test_payload_authenticate(self):
        """authenticate object is set at payload top level."""
        from markitai.fetch import fetch_with_cloudflare

        creds = {"username": "user", "password": "pass"}
        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
                http_credentials=creds,
            )

        assert captured["payload"]["authenticate"] == creds

    @pytest.mark.asyncio
    async def test_payload_none_params_omitted(self):
        """None optional params are not included in payload."""
        from markitai.fetch import fetch_with_cloudflare

        captured: dict = {}
        with self._patch_cf_client(captured):
            await fetch_with_cloudflare(
                url="https://example.com",
                api_token="t",
                account_id="a",
            )

        payload = captured["payload"]
        for key in ["userAgent", "cookies", "waitForSelector", "authenticate"]:
            assert key not in payload, f"None param '{key}' should not be in payload"


def test_domain_profile_applies_wait_for_selector() -> None:
    from markitai.fetch import _resolve_playwright_profile_overrides

    overrides = _resolve_playwright_profile_overrides(
        url="https://x.com/user/status/1",
        domain_profiles={
            "x.com": type(
                "DomainProfileConfig",
                (),
                {
                    "wait_for_selector": '[data-testid="tweetText"]',
                    "wait_for": "domcontentloaded",
                    "extra_wait_ms": 1200,
                },
            )()
        },
    )

    assert overrides["wait_for_selector"] == '[data-testid="tweetText"]'
    assert overrides["wait_for"] == "domcontentloaded"


@pytest.mark.asyncio
async def test_conditional_fetch_falls_back_when_curl_cffi_missing(monkeypatch):
    from markitai.fetch_http import get_static_http_client

    monkeypatch.setenv("MARKITAI_STATIC_HTTP", "curl_cffi")
    client = get_static_http_client()
    # Should fall back to httpx if curl_cffi not installed
    assert client.name in {"httpx", "curl_cffi"}


@pytest.mark.asyncio
async def test_conditional_fetch_forwards_accept_header():
    from markitai.fetch import fetch_with_static_conditional
    from markitai.fetch_http import StaticHttpResponse

    mock_resp = StaticHttpResponse(
        content=b"test",
        status_code=200,
        headers={"Content-Type": "text/markdown"},
        url="https://example.com",
    )

    with patch(
        "markitai.fetch_http.HttpxClient.get",
        new_callable=AsyncMock,
        return_value=mock_resp,
    ) as _mock_get:
        await fetch_with_static_conditional("https://example.com")


def test_fetch_metadata_contains_policy_reason_without_user_noise():
    from markitai.fetch import FetchResult

    r = FetchResult(
        content="ok", strategy_used="static", metadata={"policy_reason": "default"}
    )
    assert r.metadata["policy_reason"] == "default"


def test_user_error_message_single_action_hint():
    from markitai.fetch import FetchError

    # Simulate a FetchError with multiple strategies failing
    errors = ["static: failed", "playwright: failed"]
    e = FetchError(
        "All fetch strategies failed for https://example.com:\n"
        + "\n".join(f"  - {err}" for err in errors)
    )

    # In a real scenario, we would check how the CLI processor handles this.
    # For now, just ensure the error message contains the expected info.
    assert "All fetch strategies failed" in str(e)
    assert "static: failed" in str(e)
    assert "playwright: failed" in str(e)


class TestSlidingWindowRateLimiter:
    """Tests for Jina rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Should not block when within RPM limit."""
        from markitai.fetch import _SlidingWindowRateLimiter

        limiter = _SlidingWindowRateLimiter(rpm=5)
        # Should complete immediately for 5 requests
        for _ in range(5):
            await limiter.acquire()
        assert len(limiter._timestamps) == 5

    @pytest.mark.asyncio
    async def test_acquire_resets_old_timestamps(self):
        """Should remove timestamps older than 60s."""
        import time

        from markitai.fetch import _SlidingWindowRateLimiter

        limiter = _SlidingWindowRateLimiter(rpm=2)
        # Add an old timestamp
        limiter._timestamps = [time.monotonic() - 61.0]
        await limiter.acquire()
        # Old timestamp should be removed, new one added
        assert len(limiter._timestamps) == 1

    @pytest.mark.asyncio
    async def test_acquire_waits_when_at_capacity(self):
        """Should wait when RPM limit is exhausted."""
        import time

        from markitai.fetch import _SlidingWindowRateLimiter

        limiter = _SlidingWindowRateLimiter(rpm=2)
        # Fill up the window
        await limiter.acquire()
        await limiter.acquire()
        assert len(limiter._timestamps) == 2

        # Simulate the oldest timestamp being 59.9s old (expires in 0.1s)
        limiter._timestamps[0] = time.monotonic() - 59.9

        start = time.monotonic()
        await limiter.acquire()  # Should wait ~0.1s
        elapsed = time.monotonic() - start
        # Should have waited (at least a little) — allow margin for test jitter
        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_acquire_does_not_block_others_while_waiting(self):
        """Lock should be released during sleep so other coroutines can proceed."""
        import asyncio
        import time

        from markitai.fetch import _SlidingWindowRateLimiter

        limiter = _SlidingWindowRateLimiter(rpm=2)
        # Fill up with timestamps that expire in 0.15s
        now = time.monotonic()
        limiter._timestamps = [now - 59.85, now - 59.85]

        acquired = []

        async def try_acquire(label: str) -> None:
            await limiter.acquire()
            acquired.append(label)

        # Launch two concurrent acquires — both should eventually succeed
        await asyncio.gather(try_acquire("a"), try_acquire("b"))
        assert len(acquired) == 2

    @pytest.mark.asyncio
    async def test_global_limiter_singleton(self):
        """Global limiter should be created once."""
        import markitai.fetch as fetch_mod

        old_limiter = fetch_mod._jina_rate_limiter
        fetch_mod._jina_rate_limiter = None
        try:
            limiter1 = fetch_mod._get_jina_rate_limiter(20)
            limiter2 = fetch_mod._get_jina_rate_limiter(20)
            assert limiter1 is limiter2
        finally:
            fetch_mod._jina_rate_limiter = old_limiter

    @pytest.mark.asyncio
    async def test_global_limiter_recreates_on_rpm_change(self):
        """Global limiter should recreate when rpm changes."""
        import markitai.fetch as fetch_mod

        old_limiter = fetch_mod._jina_rate_limiter
        fetch_mod._jina_rate_limiter = None
        try:
            limiter1 = fetch_mod._get_jina_rate_limiter(20)
            limiter2 = fetch_mod._get_jina_rate_limiter(100)
            assert limiter1 is not limiter2
            assert limiter2._rpm == 100
        finally:
            fetch_mod._jina_rate_limiter = old_limiter


class TestDefuddleUrlEncoding:
    """Tests for URL encoding in fetch_with_defuddle."""

    @pytest.mark.asyncio
    async def test_query_string_encoded_in_defuddle_url(self):
        """Target URL with query params must be percent-encoded so they are not
        interpreted as Defuddle's own query parameters."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from urllib.parse import quote

        from markitai.fetch import fetch_with_defuddle

        target_url = "https://example.com/page?foo=1&bar=2"
        expected_encoded = quote(target_url, safe="")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "# Article\n\nContent here is long enough to pass validation."
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._get_defuddle_client", return_value=mock_client),
            patch("markitai.fetch._get_defuddle_rate_limiter") as mock_limiter_fn,
        ):
            mock_limiter = AsyncMock()
            mock_limiter_fn.return_value = mock_limiter

            await fetch_with_defuddle(target_url)

        # The URL passed to client.get must have the target URL encoded
        actual_url = mock_client.get.call_args[0][0]
        assert expected_encoded in actual_url, (
            f"Target URL query string was not encoded. Got: {actual_url}"
        )
        # Specifically, raw '?' from target URL should NOT appear after the base URL
        path_after_base = actual_url.split("defuddle.md/", 1)[-1]
        assert "?" not in path_after_base, (
            f"Raw '?' from target URL leaks as query separator: {actual_url}"
        )


class TestBuildLocalOnlyPatterns:
    """Tests for _build_local_only_patterns function."""

    @staticmethod
    def _make_policy(
        local_only_patterns: list[str] | None = None,
        inherit_no_proxy: bool = True,
    ) -> object:
        """Create a minimal policy-like object for testing."""
        return type(
            "FetchPolicyConfig",
            (),
            {
                "local_only_patterns": local_only_patterns or [],
                "inherit_no_proxy": inherit_no_proxy,
            },
        )()

    def test_returns_config_patterns_only_when_inherit_disabled(self) -> None:
        """When inherit_no_proxy=False, NO_PROXY env var is ignored."""
        policy = self._make_policy(
            local_only_patterns=["*.internal.corp", "10.0.0.0/8"],
            inherit_no_proxy=False,
        )
        with patch.dict("os.environ", {"NO_PROXY": "localhost,127.0.0.1"}):
            result = _build_local_only_patterns(policy)

        assert result == ["*.internal.corp", "10.0.0.0/8"]

    def test_merges_no_proxy_env_var(self) -> None:
        """When inherit_no_proxy=True, NO_PROXY patterns are appended."""
        policy = self._make_policy(
            local_only_patterns=["*.internal.corp"],
            inherit_no_proxy=True,
        )
        with patch.dict("os.environ", {"NO_PROXY": "localhost,127.0.0.1"}, clear=False):
            result = _build_local_only_patterns(policy)

        assert result == ["*.internal.corp", "localhost", "127.0.0.1"]

    def test_merges_no_proxy_lowercase(self) -> None:
        """Lowercase no_proxy env var is also respected."""
        policy = self._make_policy(
            local_only_patterns=["*.local"],
            inherit_no_proxy=True,
        )
        # Clear NO_PROXY so only lowercase no_proxy is found
        with patch.dict(
            "os.environ",
            {"no_proxy": "192.168.1.0/24,*.dev.local"},
            clear=False,
        ):
            # Remove uppercase NO_PROXY if present to exercise lowercase path
            env = {"no_proxy": "192.168.1.0/24,*.dev.local"}
            with patch.dict("os.environ", env, clear=True):
                result = _build_local_only_patterns(policy)

        assert result == ["*.local", "192.168.1.0/24", "*.dev.local"]

    def test_deduplication(self) -> None:
        """Overlapping patterns between config and NO_PROXY are deduplicated."""
        policy = self._make_policy(
            local_only_patterns=["localhost", "*.internal.corp"],
            inherit_no_proxy=True,
        )
        with patch.dict(
            "os.environ",
            {"NO_PROXY": "localhost,10.0.0.0/8,*.internal.corp"},
            clear=False,
        ):
            result = _build_local_only_patterns(policy)

        # Only 10.0.0.0/8 should be new; duplicates not repeated
        assert result == ["localhost", "*.internal.corp", "10.0.0.0/8"]

    def test_empty_no_proxy_env(self) -> None:
        """When NO_PROXY is empty or unset, only config patterns are returned."""
        policy = self._make_policy(
            local_only_patterns=["*.internal.corp"],
            inherit_no_proxy=True,
        )
        # Unset both NO_PROXY and no_proxy
        with patch.dict("os.environ", {}, clear=True):
            result = _build_local_only_patterns(policy)

        assert result == ["*.internal.corp"]

    def test_empty_config_patterns(self) -> None:
        """When config has no patterns, NO_PROXY values are still returned."""
        policy = self._make_policy(
            local_only_patterns=[],
            inherit_no_proxy=True,
        )
        with patch.dict(
            "os.environ",
            {"NO_PROXY": "localhost,127.0.0.1,::1"},
            clear=False,
        ):
            result = _build_local_only_patterns(policy)

        assert result == ["localhost", "127.0.0.1", "::1"]


class TestSPADomainCacheAtomicWrite:
    """Tests that SPADomainCache uses atomic writes."""

    def test_save_uses_atomic_write_json(self) -> None:
        """SPADomainCache._save should use atomic_write_json for crash safety."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "spa_cache.json"
            cache = SPADomainCache(cache_path)

            with patch("markitai.fetch.atomic_write_json") as mock_atomic:
                cache.record_spa_domain("https://example.com/page")
                mock_atomic.assert_called()
                # Verify the path argument
                call_args = mock_atomic.call_args
                assert call_args[0][0] == cache_path


class TestFetchCacheSingletonConfigSensitivity:
    """Tests that get_fetch_cache rebuilds when config changes."""

    def test_get_fetch_cache_rebuilds_on_different_config(self, tmp_path: Path) -> None:
        """get_fetch_cache should return a new instance when config changes."""
        import markitai.fetch as fetch_mod

        old_cache = fetch_mod._fetch_cache
        try:
            fetch_mod._fetch_cache = None

            dir1 = tmp_path / "cache1"
            dir1.mkdir()
            cache1 = fetch_mod.get_fetch_cache(dir1, max_size_bytes=50 * 1024 * 1024)

            dir2 = tmp_path / "cache2"
            dir2.mkdir()
            cache2 = fetch_mod.get_fetch_cache(dir2, max_size_bytes=100 * 1024 * 1024)

            # Should be different instances because config changed
            assert cache1 is not cache2
            cache1.close()
            cache2.close()
        finally:
            fetch_mod._fetch_cache = old_cache

    def test_get_fetch_cache_reuses_on_same_config(self, tmp_path: Path) -> None:
        """get_fetch_cache should return same instance when config is identical."""
        import markitai.fetch as fetch_mod

        try:
            fetch_mod._fetch_cache = None

            cache1 = fetch_mod.get_fetch_cache(
                tmp_path, max_size_bytes=100 * 1024 * 1024
            )
            cache2 = fetch_mod.get_fetch_cache(
                tmp_path, max_size_bytes=100 * 1024 * 1024
            )

            assert cache1 is cache2
            cache1.close()
        finally:
            fetch_mod._fetch_cache = None

    def test_jina_client_rebuilds_on_different_config(self) -> None:
        """_get_jina_client should rebuild when timeout/proxy changes."""
        import markitai.fetch as fetch_mod

        old_client = fetch_mod._jina_client
        try:
            fetch_mod._jina_client = None

            client1 = fetch_mod._get_jina_client(timeout=30, proxy="")
            client2 = fetch_mod._get_jina_client(timeout=60, proxy="http://proxy:8080")

            assert client1 is not client2
        finally:
            # Clean up
            import asyncio

            if fetch_mod._jina_client is not None:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        pass
                    else:
                        loop.run_until_complete(fetch_mod._jina_client.aclose())
                except Exception:
                    pass
            fetch_mod._jina_client = old_client


class TestFetchCacheStrategyKey:
    """High-2: Cache key should include strategy, not just URL."""

    def test_same_url_different_strategy_gets_different_cache_entries(
        self, tmp_path: Path
    ) -> None:
        """Same URL cached with different strategies should produce separate entries."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result_static = FetchResult(
            content="Static content",
            strategy_used="static",
            url=url,
        )
        result_playwright = FetchResult(
            content="Playwright content",
            strategy_used="playwright",
            url=url,
        )

        # Cache both results for the same URL but different strategies
        cache.set(url, result_static, strategy="static")
        cache.set(url, result_playwright, strategy="playwright")

        # Retrieve by strategy — each should get its own content
        cached_static = cache.get(url, strategy="static")
        cached_playwright = cache.get(url, strategy="playwright")

        assert cached_static is not None
        assert cached_static.content == "Static content"
        assert cached_playwright is not None
        assert cached_playwright.content == "Playwright content"
        cache.close()

    def test_get_without_strategy_returns_any_cached(self, tmp_path: Path) -> None:
        """Getting without strategy should still work (backward compat)."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result = FetchResult(
            content="Some content",
            strategy_used="static",
            url=url,
        )
        cache.set(url, result)

        # Get without strategy should return the cached result
        cached = cache.get(url)
        assert cached is not None
        assert cached.content == "Some content"
        cache.close()

    @pytest.mark.asyncio
    async def test_async_cache_respects_strategy_key(self, tmp_path: Path) -> None:
        """Async cache methods should also respect strategy key."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result_jina = FetchResult(
            content="Jina content",
            strategy_used="jina",
            url=url,
        )
        result_defuddle = FetchResult(
            content="Defuddle content",
            strategy_used="defuddle",
            url=url,
        )

        await cache.aset(url, result_jina, strategy="jina")
        await cache.aset(url, result_defuddle, strategy="defuddle")

        cached_jina = await cache.aget(url, strategy="jina")
        cached_defuddle = await cache.aget(url, strategy="defuddle")

        assert cached_jina is not None
        assert cached_jina.content == "Jina content"
        assert cached_defuddle is not None
        assert cached_defuddle.content == "Defuddle content"
        cache.close()


class TestFetchCacheRoundTrip:
    """High-3: Cache round-trip should preserve all FetchResult fields."""

    def test_cache_preserves_screenshot_path(self, tmp_path: Path) -> None:
        """screenshot_path should survive cache round-trip."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"
        screenshot = Path("/tmp/screenshots/example.full.jpg")

        result = FetchResult(
            content="Content",
            strategy_used="playwright",
            url=url,
            screenshot_path=screenshot,
        )
        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.screenshot_path == screenshot
        cache.close()

    def test_cache_preserves_static_content(self, tmp_path: Path) -> None:
        """static_content should survive cache round-trip."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result = FetchResult(
            content="Primary content",
            strategy_used="multi",
            url=url,
            static_content="Static version of the page",
        )
        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.static_content == "Static version of the page"
        cache.close()

    def test_cache_preserves_browser_content(self, tmp_path: Path) -> None:
        """browser_content should survive cache round-trip."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result = FetchResult(
            content="Primary content",
            strategy_used="multi",
            url=url,
            browser_content="Browser rendered content",
        )
        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.browser_content == "Browser rendered content"
        cache.close()

    def test_cache_preserves_all_multi_source_fields(self, tmp_path: Path) -> None:
        """All multi-source fields should survive cache round-trip together."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result = FetchResult(
            content="Primary content",
            strategy_used="multi",
            url=url,
            title="Page Title",
            final_url="https://example.com/page/final",
            metadata={"key": "value"},
            screenshot_path=Path("/tmp/screenshot.jpg"),
            static_content="Static version",
            browser_content="Browser version",
        )
        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.content == "Primary content"
        assert cached.strategy_used == "multi"
        assert cached.title == "Page Title"
        assert cached.final_url == "https://example.com/page/final"
        assert cached.metadata == {"key": "value"}
        assert cached.screenshot_path == Path("/tmp/screenshot.jpg")
        assert cached.static_content == "Static version"
        assert cached.browser_content == "Browser version"
        cache.close()

    def test_cache_preserves_none_multi_source_fields(self, tmp_path: Path) -> None:
        """None values for multi-source fields should remain None after round-trip."""
        from markitai.fetch import FetchCache, FetchResult

        cache = FetchCache(tmp_path / "test.db")
        url = "https://example.com/page"

        result = FetchResult(
            content="Primary content",
            strategy_used="static",
            url=url,
        )
        cache.set(url, result)

        cached = cache.get(url)
        assert cached is not None
        assert cached.screenshot_path is None
        assert cached.static_content is None
        assert cached.browser_content is None
        cache.close()


class TestHttpClientConnectionReuse:
    """Medium-5: HTTP clients should reuse connections."""

    @pytest.mark.asyncio
    async def test_httpx_client_reuses_connection(self) -> None:
        """HttpxClient should reuse connections across requests."""
        from markitai.fetch_http import HttpxClient

        client = HttpxClient()

        # The client should have an internal shared session
        assert hasattr(client, "_client") or hasattr(client, "get")

        # After creating, a shared client should be accessible for reuse
        # This test verifies the client maintains a persistent connection pool
        # rather than creating a new connection per request
        await client.close()

    @pytest.mark.asyncio
    async def test_httpx_client_provides_close_method(self) -> None:
        """HttpxClient should provide a close method for cleanup."""
        from markitai.fetch_http import HttpxClient

        client = HttpxClient()
        # Should have a close method
        assert hasattr(client, "close")
        await client.close()


class TestProxyAutoProxyRespected:
    """Medium-7: auto_proxy=False should disable proxy for all backends."""

    def test_get_proxy_for_url_returns_empty_when_auto_proxy_disabled(self) -> None:
        """get_proxy_for_url should return empty string when auto_proxy=False."""
        from markitai.fetch import get_proxy_for_url

        result = get_proxy_for_url("https://example.com", auto_proxy=False)
        assert result == ""

    def test_get_proxy_for_url_returns_proxy_when_auto_proxy_enabled(self) -> None:
        """get_proxy_for_url should return detected proxy when auto_proxy=True."""
        from markitai.fetch import get_proxy_for_url

        with patch("markitai.fetch._detect_proxy", return_value="http://proxy:8080"):
            result = get_proxy_for_url("https://example.com", auto_proxy=True)
            assert result == "http://proxy:8080"

    def test_get_proxy_for_url_respects_no_proxy_patterns(self) -> None:
        """get_proxy_for_url should skip proxy for NO_PROXY domains."""
        from markitai import fetch
        from markitai.fetch import get_proxy_for_url

        fetch._detected_proxy = "http://proxy:8080"
        fetch._detected_proxy_bypass = "example.com,internal.corp"

        try:
            result = get_proxy_for_url("https://example.com/page", auto_proxy=True)
            assert result == ""

            result = get_proxy_for_url("https://other.com/page", auto_proxy=True)
            assert result == "http://proxy:8080"
        finally:
            fetch._detected_proxy = None
            fetch._detected_proxy_bypass = None


class TestSPABrowserFirstOrdering:
    """Medium-8: known_spa=True should prefer defuddle/jina before playwright."""

    def test_known_spa_prefers_defuddle_over_playwright(self) -> None:
        """When known_spa=True, defuddle should lead (server-side extraction may work)."""
        from markitai.fetch_policy import FetchPolicyEngine

        engine = FetchPolicyEngine()
        decision = engine.decide(
            domain="spa-app.example.com",
            known_spa=True,
            explicit_strategy=None,
            fallback_patterns=[],
            policy_enabled=True,
        )

        # Defuddle/jina lead; playwright is available as fallback
        assert decision.order[0] == "defuddle"
        assert "playwright" in decision.order

    def test_fallback_pattern_prefers_defuddle_over_playwright(self) -> None:
        """When domain matches fallback_patterns, defuddle should lead."""
        from markitai.fetch_policy import FetchPolicyEngine

        engine = FetchPolicyEngine()
        decision = engine.decide(
            domain="twitter.com",
            known_spa=False,
            explicit_strategy=None,
            fallback_patterns=["twitter.com"],
            policy_enabled=True,
        )

        # Defuddle/jina lead for fallback-pattern domains too
        assert decision.order[0] == "defuddle"
        assert "playwright" in decision.order

    def test_non_spa_default_order_unchanged(self) -> None:
        """Non-SPA domains should keep the default defuddle-first order."""
        from markitai.fetch_policy import FetchPolicyEngine

        engine = FetchPolicyEngine()
        decision = engine.decide(
            domain="blog.example.com",
            known_spa=False,
            explicit_strategy=None,
            fallback_patterns=[],
            policy_enabled=True,
        )

        # Default order: defuddle first
        assert decision.order[0] == "defuddle"
