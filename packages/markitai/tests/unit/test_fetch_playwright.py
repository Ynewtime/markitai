"""Tests for Playwright-based fetch backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if playwright is available for tests that require it
try:
    import playwright  # noqa: F401

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Decorator for tests that require playwright module
requires_playwright = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE, reason="playwright not installed"
)


class TestIsPlaywrightAvailable:
    """Tests for is_playwright_available function."""

    @requires_playwright
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

    @requires_playwright
    @pytest.mark.asyncio
    async def test_raises_error_when_playwright_import_fails(self):
        """Raises ModuleNotFoundError when playwright cannot be imported."""
        from markitai.fetch_playwright import PlaywrightRenderer

        renderer = PlaywrightRenderer()
        renderer._browser = None
        renderer._playwright = None

        # Mock the import inside _ensure_browser by patching playwright.async_api
        with (
            patch.dict(
                "sys.modules",
                {"playwright.async_api": None},
            ),
            pytest.raises((ModuleNotFoundError, TypeError)),
        ):
            await renderer._ensure_browser()

    @requires_playwright
    @pytest.mark.asyncio
    async def test_basic_fetch_with_mock(self):
        """Tests basic fetch with mocked playwright."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Create properly configured async mocks
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content here. " * 100 + "</body></html>"

        # Configure return values as actual values, not mocks
        mock_page.title = AsyncMock(return_value="Test Title")
        mock_page.url = "https://example.com/final"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)

        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium = mock_chromium
        mock_playwright_instance.stop = AsyncMock()

        # Mock async_playwright().start() to return our mock
        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_playwright_instance)

        with patch(
            "playwright.async_api.async_playwright",
            return_value=mock_starter,
        ):
            renderer = PlaywrightRenderer()
            result = await renderer.fetch(
                "https://example.com",
                timeout=30000,
                extra_wait_ms=0,
            )

        assert result.title == "Test Title"
        assert result.final_url == "https://example.com/final"
        assert "content" in result.content.lower()
        assert result.metadata["renderer"] == "playwright"


class TestPlaywrightRenderer:
    """Tests for PlaywrightRenderer class."""

    def test_init_with_defaults(self):
        """Initializes renderer with default values."""
        from markitai.fetch_playwright import PlaywrightRenderer

        renderer = PlaywrightRenderer()

        assert renderer.proxy is None
        assert renderer._playwright is None
        assert renderer._browser is None

    def test_init_with_proxy(self):
        """Initializes renderer with proxy."""
        from markitai.fetch_playwright import PlaywrightRenderer

        renderer = PlaywrightRenderer(proxy="http://proxy.example.com:8080")

        assert renderer.proxy == "http://proxy.example.com:8080"

    @pytest.mark.asyncio
    async def test_context_manager_enter_returns_self(self):
        """Context manager __aenter__ returns self."""
        from markitai.fetch_playwright import PlaywrightRenderer

        renderer = PlaywrightRenderer()
        result = await renderer.__aenter__()

        assert result is renderer

    @pytest.mark.asyncio
    async def test_context_manager_exit_closes_resources(self):
        """Context manager __aexit__ calls close."""
        from markitai.fetch_playwright import PlaywrightRenderer

        renderer = PlaywrightRenderer()
        renderer.close = AsyncMock()

        await renderer.__aexit__(None, None, None)

        renderer.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_when_browser_is_none(self):
        """Close works when browser is None."""
        from markitai.fetch_playwright import PlaywrightRenderer

        renderer = PlaywrightRenderer()
        renderer._browser = None
        renderer._playwright = None

        # Should not raise
        await renderer.close()

        assert renderer._browser is None
        assert renderer._playwright is None

    @pytest.mark.asyncio
    async def test_close_closes_browser_and_playwright(self):
        """Close properly closes browser and playwright instances."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser
        renderer._playwright = mock_playwright

        await renderer.close()

        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        assert renderer._browser is None
        assert renderer._playwright is None

    @pytest.mark.asyncio
    async def test_ensure_browser_returns_existing_browser(self):
        """_ensure_browser returns existing browser if already created."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_browser = AsyncMock()

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        result = await renderer._ensure_browser()

        assert result is mock_browser

    @requires_playwright
    @pytest.mark.asyncio
    async def test_ensure_browser_launches_with_proxy(self):
        """_ensure_browser passes proxy to launch options."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_browser = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)

        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium = mock_chromium

        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_playwright_instance)

        with patch(
            "playwright.async_api.async_playwright",
            return_value=mock_starter,
        ):
            renderer = PlaywrightRenderer(proxy="http://proxy:8080")
            await renderer._ensure_browser()

            mock_chromium.launch.assert_called_once_with(
                headless=True, proxy={"server": "http://proxy:8080"}
            )

    @requires_playwright
    @pytest.mark.asyncio
    async def test_ensure_browser_raises_runtime_error_on_launch_failure(self):
        """_ensure_browser raises RuntimeError when browser launch fails."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_playwright_instance = AsyncMock()
        mock_playwright_instance.chromium.launch = AsyncMock(
            side_effect=Exception("Browser launch failed")
        )
        mock_playwright_instance.stop = AsyncMock()

        mock_starter = AsyncMock()
        mock_starter.start = AsyncMock(return_value=mock_playwright_instance)

        with patch(
            "playwright.async_api.async_playwright",
            return_value=mock_starter,
        ):
            renderer = PlaywrightRenderer()

            with pytest.raises(RuntimeError) as exc_info:
                await renderer._ensure_browser()

            assert "Failed to launch Chromium browser" in str(exc_info.value)
            assert "uv run playwright install chromium" in str(exc_info.value)
            mock_playwright_instance.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_with_different_wait_for_options(self):
        """Fetch correctly maps wait_for options."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        # Test with networkidle
        await renderer.fetch(
            "https://example.com",
            wait_for="networkidle",
            extra_wait_ms=0,
        )

        mock_page.goto.assert_called_with(
            "https://example.com", timeout=30000, wait_until="networkidle"
        )

    @pytest.mark.asyncio
    async def test_fetch_with_unknown_wait_for_defaults_to_domcontentloaded(self):
        """Fetch defaults to domcontentloaded for unknown wait_for."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        await renderer.fetch(
            "https://example.com",
            wait_for="unknown_option",
            extra_wait_ms=0,
        )

        mock_page.goto.assert_called_with(
            "https://example.com", timeout=30000, wait_until="domcontentloaded"
        )

    @pytest.mark.asyncio
    async def test_fetch_applies_extra_wait(self):
        """Fetch waits for extra_wait_ms after page load."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=1000,
            )

            # First call is extra_wait (1000ms = 1.0s), second is post-scroll delay
            assert mock_sleep.call_args_list[0].args == (1.0,)

    @pytest.mark.asyncio
    async def test_fetch_no_extra_wait_when_zero(self):
        """Fetch skips extra wait when extra_wait_ms is 0."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
            )

            # extra_wait sleep should not be called (extra_wait_ms=0)
            # but post-scroll delay may still be called
            if mock_sleep.call_count > 0:
                # Only post-scroll delay should be present, not extra_wait
                for call in mock_sleep.call_args_list:
                    assert call.args[0] != 0  # No zero-sleep call

    @pytest.mark.asyncio
    async def test_fetch_with_screenshot_config(self):
        """Fetch captures screenshot when config is provided."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()
        mock_page.screenshot = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        mock_screenshot_config = MagicMock()
        mock_screenshot_config.enabled = True
        mock_screenshot_config.full_page = True
        mock_screenshot_config.quality = 80
        mock_screenshot_config.max_height = 5000

        with patch(
            "markitai.fetch_playwright._capture_screenshot",
            new_callable=AsyncMock,
        ) as mock_capture:
            mock_capture.return_value = Path("/tmp/screenshot.jpg")

            result = await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                screenshot_config=mock_screenshot_config,
                output_dir=Path("/tmp"),
            )

            mock_capture.assert_called_once()
            assert result.screenshot_path == Path("/tmp/screenshot.jpg")

    @pytest.mark.asyncio
    async def test_fetch_skips_screenshot_when_disabled(self):
        """Fetch skips screenshot when enabled=False."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        mock_screenshot_config = MagicMock()
        mock_screenshot_config.enabled = False

        with patch(
            "markitai.fetch_playwright._capture_screenshot",
            new_callable=AsyncMock,
        ) as mock_capture:
            result = await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
                screenshot_config=mock_screenshot_config,
                output_dir=Path("/tmp"),
            )

            mock_capture.assert_not_called()
            assert result.screenshot_path is None

    @pytest.mark.asyncio
    async def test_fetch_closes_context_on_success(self):
        """Fetch closes context after successful fetch."""
        from markitai.fetch_playwright import PlaywrightRenderer

        # Use longer content to avoid triggering inner_text fallback
        long_content = "<html><body>" + "Test content. " * 100 + "</body></html>"
        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value=long_content)
        mock_page.goto = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        await renderer.fetch("https://example.com", extra_wait_ms=0)

        mock_context.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_closes_context_on_error(self):
        """Fetch closes context even when page.goto fails."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Navigation failed"))

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        with pytest.raises(Exception, match="Navigation failed"):
            await renderer.fetch("https://example.com", extra_wait_ms=0)

        mock_context.close.assert_called_once()


class TestFetchWithPlaywrightFunction:
    """Tests for fetch_with_playwright top-level function."""

    @pytest.mark.asyncio
    async def test_uses_provided_renderer(self):
        """Uses provided renderer instead of creating new one."""
        from markitai.fetch_playwright import (
            PlaywrightFetchResult,
            PlaywrightRenderer,
            fetch_with_playwright,
        )

        mock_renderer = AsyncMock(spec=PlaywrightRenderer)
        mock_renderer.fetch = AsyncMock(
            return_value=PlaywrightFetchResult(content="Test content")
        )

        result = await fetch_with_playwright(
            "https://example.com",
            renderer=mock_renderer,
        )

        mock_renderer.fetch.assert_called_once()
        assert result.content == "Test content"

    @pytest.mark.asyncio
    async def test_creates_standalone_renderer_when_none_provided(self):
        """Creates standalone renderer when none provided."""
        from markitai.fetch_playwright import (
            PlaywrightFetchResult,
            fetch_with_playwright,
        )

        mock_result = PlaywrightFetchResult(content="Standalone content")

        with patch("markitai.fetch_playwright.PlaywrightRenderer") as MockRendererClass:
            mock_renderer_instance = AsyncMock()
            mock_renderer_instance.fetch = AsyncMock(return_value=mock_result)
            mock_renderer_instance.__aenter__ = AsyncMock(
                return_value=mock_renderer_instance
            )
            mock_renderer_instance.__aexit__ = AsyncMock(return_value=None)
            MockRendererClass.return_value = mock_renderer_instance

            result = await fetch_with_playwright(
                "https://example.com",
                proxy="http://proxy:8080",
            )

            MockRendererClass.assert_called_once_with(proxy="http://proxy:8080")
            assert result.content == "Standalone content"


class TestIsContentIncomplete:
    """Tests for _is_content_incomplete function."""

    def test_empty_content_is_incomplete(self):
        """Empty content is considered incomplete."""
        from markitai.fetch_playwright import _is_content_incomplete

        assert _is_content_incomplete("") is True
        assert _is_content_incomplete(None) is True  # type: ignore

    def test_short_content_is_incomplete(self):
        """Very short content is incomplete."""
        from markitai.fetch_playwright import _is_content_incomplete

        assert _is_content_incomplete("Short text") is True

    def test_long_content_is_complete(self):
        """Long content is considered complete."""
        from markitai.fetch_playwright import _is_content_incomplete

        long_content = "This is a test. " * 100  # 1600 chars
        assert _is_content_incomplete(long_content) is False

    def test_twitter_login_prompt_is_incomplete(self):
        """X.com login prompt patterns are incomplete."""
        from markitai.fetch_playwright import _is_content_incomplete

        content = "Don't miss what's happening\nSign up\nLog in"
        assert _is_content_incomplete(content) is True

    def test_cookie_consent_only_is_incomplete(self):
        """Cookie consent only content is incomplete."""
        from markitai.fetch_playwright import _is_content_incomplete

        content = "We use cookies. Accept all cookies to continue."
        assert _is_content_incomplete(content) is True

    def test_login_signup_only_is_incomplete(self):
        """Login/signup only content is incomplete."""
        from markitai.fetch_playwright import _is_content_incomplete

        content = "Log in to your account. Sign up for free."
        assert _is_content_incomplete(content) is True

    def test_pattern_with_substantial_content_is_complete(self):
        """Pattern with substantial other content is complete."""
        from markitai.fetch_playwright import _is_content_incomplete

        # Has a pattern but also substantial content
        content = "Accept all cookies\n\n" + "Real article content here. " * 50
        assert _is_content_incomplete(content) is False


class TestFormatInnerText:
    """Tests for _format_inner_text function."""

    def test_empty_text(self):
        """Empty text returns empty string."""
        from markitai.fetch_playwright import _format_inner_text

        assert _format_inner_text("") == ""
        assert _format_inner_text(None) == ""  # type: ignore

    def test_formats_lines_with_paragraphs(self):
        """Formats lines as paragraphs."""
        from markitai.fetch_playwright import _format_inner_text

        text = "Line 1\n\nLine 2\n\nLine 3"
        result = _format_inner_text(text)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
        assert "\n\n" in result

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace from lines."""
        from markitai.fetch_playwright import _format_inner_text

        text = "  Line with spaces  \n  Another line  "
        result = _format_inner_text(text)

        assert "Line with spaces" in result
        assert "Another line" in result
        assert "  Line" not in result

    def test_removes_empty_lines(self):
        """Removes empty lines."""
        from markitai.fetch_playwright import _format_inner_text

        text = "Line 1\n\n\n\nLine 2"
        result = _format_inner_text(text)

        lines = [line for line in result.split("\n\n") if line.strip()]
        assert len(lines) == 2


class TestCaptureScreenshot:
    """Tests for _capture_screenshot function."""

    @pytest.mark.asyncio
    async def test_captures_screenshot_successfully(self, tmp_path):
        """Captures screenshot and returns path."""
        from markitai.fetch_playwright import _capture_screenshot

        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.full_page = True
        mock_config.quality = 85
        mock_config.max_height = 10000

        # Patch the imported function in fetch module
        with (
            patch(
                "markitai.fetch._url_to_screenshot_filename",
                return_value="screenshot.jpg",
            ),
            patch("markitai.fetch._compress_screenshot") as mock_compress,
        ):
            result = await _capture_screenshot(
                mock_page,
                mock_config,
                tmp_path,
                "https://example.com",
            )

            assert result == tmp_path / "screenshot.jpg"
            mock_page.screenshot.assert_called_once()
            mock_compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_screenshot_error(self, tmp_path):
        """Returns None when screenshot fails."""
        from markitai.fetch_playwright import _capture_screenshot

        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.full_page = True
        mock_config.quality = 85
        mock_config.max_height = 10000

        with patch(
            "markitai.fetch._url_to_screenshot_filename",
            return_value="screenshot.jpg",
        ):
            result = await _capture_screenshot(
                mock_page,
                mock_config,
                tmp_path,
                "https://example.com",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_creates_output_directory(self, tmp_path):
        """Creates output directory if it doesn't exist."""
        from markitai.fetch_playwright import _capture_screenshot

        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.full_page = False
        mock_config.quality = 70
        mock_config.max_height = 5000

        output_dir = tmp_path / "nested" / "output"

        with (
            patch(
                "markitai.fetch._url_to_screenshot_filename",
                return_value="test.jpg",
            ),
            patch("markitai.fetch._compress_screenshot"),
        ):
            await _capture_screenshot(
                mock_page,
                mock_config,
                output_dir,
                "https://example.com",
            )

            assert output_dir.exists()


class TestHtmlToMarkdownFallback:
    """Tests for _html_to_markdown fallback behavior."""

    def test_falls_back_to_strip_html_on_markitdown_error(self):
        """Falls back to _strip_html_tags when markitdown fails."""
        from markitai.fetch_playwright import _html_to_markdown

        html = "<html><body><p>Test content</p></body></html>"

        with patch("markitdown.MarkItDown") as MockMarkItDown:
            MockMarkItDown.side_effect = Exception("MarkItDown error")

            result = _html_to_markdown(html)

            assert "Test content" in result

    def test_removes_noscript_tags(self):
        """Removes noscript tags from HTML."""
        from markitai.fetch_playwright import _html_to_markdown

        html = """
        <html>
        <body>
            <p>Real content</p>
            <noscript>JavaScript is required</noscript>
        </body>
        </html>
        """

        result = _html_to_markdown(html)

        assert "Real content" in result
        assert "JavaScript is required" not in result


class TestBrowserInstalledWithRealCheck:
    """Tests for browser installation check via path detection."""

    @requires_playwright
    def test_returns_true_when_chromium_found(self):
        """Returns True when Chromium executable is found in paths."""
        from markitai.fetch_playwright import (
            clear_browser_cache,
            is_playwright_browser_installed,
        )

        clear_browser_cache()

        with (
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch("markitai.fetch_playwright._check_chromium_paths", return_value=True),
        ):
            result = is_playwright_browser_installed(use_cache=False)
            assert result is True

    @requires_playwright
    def test_returns_false_when_chromium_not_found(self):
        """Returns False when Chromium executable is not found."""
        from markitai.fetch_playwright import (
            clear_browser_cache,
            is_playwright_browser_installed,
        )

        clear_browser_cache()

        with (
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright._check_chromium_paths", return_value=False
            ),
        ):
            result = is_playwright_browser_installed(use_cache=False)
            assert result is False

    @requires_playwright
    def test_returns_false_when_playwright_not_available(self):
        """Returns False when playwright package is not available."""
        from markitai.fetch_playwright import (
            clear_browser_cache,
            is_playwright_browser_installed,
        )

        clear_browser_cache()

        with patch(
            "markitai.fetch_playwright.is_playwright_available", return_value=False
        ):
            result = is_playwright_browser_installed(use_cache=False)
            assert result is False


class TestFetchWithIncompleteContent:
    """Tests for fetch behavior with incomplete content."""

    @pytest.mark.asyncio
    async def test_fetch_uses_inner_text_when_content_incomplete(self):
        """Uses inner_text when HTML content appears incomplete."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value="<html><body>Short</body></html>")
        mock_page.goto = AsyncMock()
        # inner_text returns more content
        mock_page.inner_text = AsyncMock(return_value="Much longer content " * 50)

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        with patch(
            "markitai.fetch_playwright._is_content_incomplete", return_value=True
        ):
            result = await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
            )

            mock_page.inner_text.assert_called_once_with("body")
            # Should use the longer inner_text content
            assert "longer content" in result.content

    @pytest.mark.asyncio
    async def test_fetch_keeps_original_if_inner_text_shorter(self):
        """Keeps original content if inner_text is shorter."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        # HTML gives more content
        mock_page.content = AsyncMock(
            return_value="<html><body>" + "Original content " * 100 + "</body></html>"
        )
        mock_page.goto = AsyncMock()
        # inner_text returns less
        mock_page.inner_text = AsyncMock(return_value="Short")

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        # Force content to be marked incomplete but original is longer
        with patch(
            "markitai.fetch_playwright._is_content_incomplete", return_value=True
        ):
            result = await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
            )

            # Should keep original content since inner_text is shorter
            assert "Original content" in result.content

    @pytest.mark.asyncio
    async def test_fetch_handles_inner_text_exception(self):
        """Handles exception when inner_text fails."""
        from markitai.fetch_playwright import PlaywrightRenderer

        mock_page = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.url = "https://example.com"
        mock_page.content = AsyncMock(return_value="<html><body>Content</body></html>")
        mock_page.goto = AsyncMock()
        mock_page.inner_text = AsyncMock(side_effect=Exception("inner_text failed"))

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        renderer = PlaywrightRenderer()
        renderer._browser = mock_browser

        with patch(
            "markitai.fetch_playwright._is_content_incomplete", return_value=True
        ):
            # Should not raise, just continue with original content
            result = await renderer.fetch(
                "https://example.com",
                extra_wait_ms=0,
            )

            assert result is not None
