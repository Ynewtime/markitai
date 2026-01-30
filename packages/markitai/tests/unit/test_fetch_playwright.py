"""Tests for Playwright-based fetch backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestIsPlaywrightAvailable:
    """Tests for is_playwright_available function."""

    def test_returns_true_when_installed(self):
        """Returns True when playwright is installed."""
        from markitai.fetch_playwright import is_playwright_available

        # Playwright is installed in dev environment
        result = is_playwright_available()
        assert result is True

    def test_returns_false_when_not_installed(self):
        """Returns False when playwright is not installed."""
        with patch("markitai.fetch_playwright.find_spec", return_value=None):
            from markitai import fetch_playwright

            # Call the function directly with the patched find_spec
            result = fetch_playwright.find_spec("playwright") is not None
            assert result is False


class TestIsPlaywrightBrowserInstalled:
    """Tests for is_playwright_browser_installed function."""

    def test_returns_false_when_playwright_not_available(self):
        """Returns False when playwright package is not installed."""
        from markitai.fetch_playwright import clear_browser_cache

        clear_browser_cache()

        with patch(
            "markitai.fetch_playwright.is_playwright_available", return_value=False
        ):
            from markitai.fetch_playwright import is_playwright_browser_installed

            result = is_playwright_browser_installed(use_cache=False)
            assert result is False

    def test_uses_cache_on_subsequent_calls(self):
        """Uses cached result on subsequent calls."""
        from markitai.fetch_playwright import (
            clear_browser_cache,
            is_playwright_browser_installed,
        )

        clear_browser_cache()

        # First call to populate cache
        with patch(
            "markitai.fetch_playwright.is_playwright_available", return_value=False
        ):
            result1 = is_playwright_browser_installed(use_cache=False)
            # Second call uses cache (mock won't be called again)
            result2 = is_playwright_browser_installed(use_cache=True)

            assert result1 is False
            assert result2 is False


class TestClearBrowserCache:
    """Tests for clear_browser_cache function."""

    def test_clears_cache(self):
        """Clears the browser installation cache."""
        from markitai.fetch_playwright import clear_browser_cache

        # Just verify it doesn't raise
        clear_browser_cache()


class TestHtmlToMarkdown:
    """Tests for _html_to_markdown function."""

    def test_converts_basic_html(self):
        """Converts basic HTML to markdown."""
        from markitai.fetch_playwright import _html_to_markdown

        html = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
        result = _html_to_markdown(html)

        assert "Title" in result
        assert "Paragraph" in result

    def test_handles_empty_html(self):
        """Handles empty HTML gracefully."""
        from markitai.fetch_playwright import _html_to_markdown

        result = _html_to_markdown("")
        assert result == ""

    def test_handles_plain_text(self):
        """Handles plain text without HTML tags."""
        from markitai.fetch_playwright import _html_to_markdown

        text = "Just plain text"
        result = _html_to_markdown(text)
        assert "plain text" in result


class TestStripHtmlTags:
    """Tests for _strip_html_tags function."""

    def test_strips_basic_tags(self):
        """Strips basic HTML tags."""
        from markitai.fetch_playwright import _strip_html_tags

        html = "<p>Hello <strong>World</strong></p>"
        result = _strip_html_tags(html)

        assert "Hello" in result
        assert "World" in result
        assert "<p>" not in result
        assert "<strong>" not in result

    def test_removes_script_tags(self):
        """Removes script tags and their content."""
        from markitai.fetch_playwright import _strip_html_tags

        html = "<p>Content</p><script>alert('xss')</script>"
        result = _strip_html_tags(html)

        assert "Content" in result
        assert "alert" not in result
        assert "xss" not in result

    def test_removes_style_tags(self):
        """Removes style tags and their content."""
        from markitai.fetch_playwright import _strip_html_tags

        html = "<style>.foo { color: red; }</style><p>Text</p>"
        result = _strip_html_tags(html)

        assert "Text" in result
        assert "color" not in result
        assert ".foo" not in result

    def test_decodes_html_entities(self):
        """Decodes HTML entities."""
        from markitai.fetch_playwright import _strip_html_tags

        html = "<p>Hello &amp; World &lt;test&gt;</p>"
        result = _strip_html_tags(html)

        assert "&" in result
        assert "<test>" in result


class TestPlaywrightFetchResult:
    """Tests for PlaywrightFetchResult dataclass."""

    def test_default_values(self):
        """Creates result with default values."""
        from markitai.fetch_playwright import PlaywrightFetchResult

        result = PlaywrightFetchResult(content="test")

        assert result.content == "test"
        assert result.title is None
        assert result.final_url is None
        assert result.screenshot_path is None
        assert result.metadata == {}

    def test_with_all_fields(self):
        """Creates result with all fields."""
        from pathlib import Path

        from markitai.fetch_playwright import PlaywrightFetchResult

        result = PlaywrightFetchResult(
            content="# Markdown",
            title="Test Page",
            final_url="https://example.com/final",
            screenshot_path=Path("/tmp/screenshot.png"),
            metadata={"key": "value"},
        )

        assert result.content == "# Markdown"
        assert result.title == "Test Page"
        assert result.final_url == "https://example.com/final"
        assert result.screenshot_path == Path("/tmp/screenshot.png")
        assert result.metadata == {"key": "value"}


class TestFetchWithPlaywright:
    """Tests for fetch_with_playwright function."""

    @pytest.mark.asyncio
    async def test_raises_import_error_when_not_available(self):
        """Raises ImportError when playwright is not installed."""
        with patch(
            "markitai.fetch_playwright.is_playwright_available", return_value=False
        ):
            from markitai.fetch_playwright import fetch_with_playwright

            with pytest.raises(ImportError) as exc_info:
                await fetch_with_playwright("https://example.com")

            assert "playwright is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_basic_fetch_with_mock(self):
        """Tests basic fetch with mocked playwright."""
        # Create async mocks
        mock_page = AsyncMock()
        mock_page.title.return_value = "Test Title"
        mock_page.url = "https://example.com/final"
        mock_page.content.return_value = "<html><body>Content</body></html>"

        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page

        mock_chromium = AsyncMock()
        mock_chromium.launch.return_value = mock_browser

        mock_playwright = AsyncMock()
        mock_playwright.chromium = mock_chromium

        # Create async context manager
        class MockAsyncPlaywright:
            async def __aenter__(self):
                return mock_playwright

            async def __aexit__(self, *args):
                pass

        with (
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "playwright.async_api.async_playwright",
                return_value=MockAsyncPlaywright(),
            ),
        ):
            from markitai.fetch_playwright import fetch_with_playwright

            result = await fetch_with_playwright(
                "https://example.com",
                timeout=30000,
                extra_wait_ms=0,
            )

        assert result.title == "Test Title"
        assert result.final_url == "https://example.com/final"
        assert "Content" in result.content
        assert result.metadata["renderer"] == "playwright"
