"""Unit tests for the fetch module."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from markitai.constants import JS_REQUIRED_PATTERNS
from markitai.fetch import (
    FetchStrategy,
    SPADomainCache,
    _get_effective_agent_browser_args,
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
        assert FetchStrategy.BROWSER.value == "browser"
        assert FetchStrategy.JINA.value == "jina"

    def test_strategy_from_string(self) -> None:
        """Test creating strategy from string value."""
        assert FetchStrategy("auto") == FetchStrategy.AUTO
        assert FetchStrategy("static") == FetchStrategy.STATIC
        assert FetchStrategy("browser") == FetchStrategy.BROWSER
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
            strategy_used="browser",
            url="https://example.com",
            metadata={"renderer": "agent-browser"},
        )

        assert result.metadata == {"renderer": "agent-browser"}


class TestFetchErrors:
    """Tests for fetch error classes."""

    def test_fetch_error(self) -> None:
        """Test FetchError base class."""
        from markitai.fetch import FetchError

        error = FetchError("Test error")
        assert str(error) == "Test error"

    def test_agent_browser_not_found_error(self) -> None:
        """Test AgentBrowserNotFoundError."""
        from markitai.fetch import AgentBrowserNotFoundError

        error = AgentBrowserNotFoundError()
        assert "agent-browser is not installed" in str(error)
        assert "npm install -g agent-browser" in str(error)

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


class TestGetEffectiveAgentBrowserArgs:
    """Tests for _get_effective_agent_browser_args function."""

    def test_non_windows_returns_same_args(self) -> None:
        """Test that non-Windows platforms return args unchanged."""
        with patch("sys.platform", "linux"):
            args = ["agent-browser", "open", "https://example.com"]
            result = _get_effective_agent_browser_args(args)
            assert result == args

    def test_non_agent_browser_command_unchanged(self) -> None:
        """Test that non agent-browser commands are unchanged."""
        with patch("sys.platform", "win32"):
            args = ["python", "-c", "print('hello')"]
            result = _get_effective_agent_browser_args(args)
            assert result == args

    def test_windows_with_cached_exe(self) -> None:
        """Test that Windows uses cached native exe path."""
        from markitai.fetch import _agent_browser_exe_cache

        # Set up cache
        cached_path = "C:\\path\\to\\agent-browser.exe"
        _agent_browser_exe_cache["agent-browser"] = cached_path

        try:
            with patch("sys.platform", "win32"):
                args = ["agent-browser", "open", "https://example.com"]
                result = _get_effective_agent_browser_args(args)
                assert result[0] == cached_path
                assert result[1:] == args[1:]
        finally:
            # Clean up cache
            _agent_browser_exe_cache.pop("agent-browser", None)

    def test_empty_args_returns_empty(self) -> None:
        """Test that empty args returns empty list."""
        result = _get_effective_agent_browser_args([])
        assert result == []

    def test_preserves_all_arguments(self) -> None:
        """Test that all arguments are preserved."""
        with patch("sys.platform", "linux"):
            args = ["agent-browser", "--session", "test", "open", "https://x.com"]
            result = _get_effective_agent_browser_args(args)
            assert result == args


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
