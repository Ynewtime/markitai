"""Unit tests for the fetch module."""

from __future__ import annotations

from pathlib import Path

from markitai.constants import JS_REQUIRED_PATTERNS
from markitai.fetch import (
    FetchStrategy,
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
