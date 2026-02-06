"""Playwright-based URL fetch backend.

This module provides browser automation using Playwright Python as an
alternative to agent-browser, eliminating the Node.js dependency.

Features:
- Pure Python implementation (no external CLI)
- Native async support
- Cross-platform (Windows/Linux/macOS)
- Automatic proxy detection
- Screenshot capture support

Usage:
    from markitai.fetch_playwright import fetch_with_playwright, is_playwright_available

    if is_playwright_available():
        result = await fetch_with_playwright(url, config)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import (
    DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    DEFAULT_PLAYWRIGHT_WAIT_FOR,
)

if TYPE_CHECKING:
    from markitai.config import ScreenshotConfig


def is_playwright_available() -> bool:
    """Check if playwright is installed.

    Returns:
        True if playwright can be imported
    """
    return find_spec("playwright") is not None


# Cache for browser installation check
_browser_installed_cache: bool | None = None


def is_playwright_browser_installed(use_cache: bool = True) -> bool:
    """Check if playwright browser (Chromium) is installed.

    This function checks for browser executable existence without launching it,
    avoiding potential hangs in environments like WSL2 or headless servers.

    Args:
        use_cache: Whether to use cached result

    Returns:
        True if Chromium browser is available
    """
    global _browser_installed_cache

    if use_cache and _browser_installed_cache is not None:
        return _browser_installed_cache

    if not is_playwright_available():
        _browser_installed_cache = False
        return False

    # Check browser executable by path - never launch the browser
    _browser_installed_cache = _check_chromium_paths()
    return _browser_installed_cache


def _check_chromium_paths() -> bool:
    """Check common Playwright Chromium installation paths.

    Returns:
        True if Chromium executable found
    """
    import os
    import sys

    # Playwright stores browsers in these locations
    if sys.platform == "win32":
        base_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright",
            Path.home() / "AppData" / "Local" / "ms-playwright",
        ]
    elif sys.platform == "darwin":
        base_paths = [
            Path.home() / "Library" / "Caches" / "ms-playwright",
        ]
    else:  # Linux
        base_paths = [
            Path.home() / ".cache" / "ms-playwright",
        ]

    for base in base_paths:
        if not base.exists():
            continue
        # Look for chromium-* directories
        chromium_dirs = list(base.glob("chromium-*"))
        if chromium_dirs:
            # Check if chrome executable exists
            for chromium_dir in chromium_dirs:
                if sys.platform == "win32":
                    # Try both old (chrome-win) and new (chrome-win64) paths
                    exe_paths = [
                        chromium_dir / "chrome-win64" / "chrome.exe",
                        chromium_dir / "chrome-win" / "chrome.exe",
                    ]
                elif sys.platform == "darwin":
                    exe_paths = [
                        chromium_dir
                        / "chrome-mac"
                        / "Chromium.app"
                        / "Contents"
                        / "MacOS"
                        / "Chromium",
                        chromium_dir
                        / "chrome-mac-arm64"
                        / "Chromium.app"
                        / "Contents"
                        / "MacOS"
                        / "Chromium",
                    ]
                else:
                    # Try both old (chrome-linux) and new (chrome-linux64) paths
                    exe_paths = [
                        chromium_dir / "chrome-linux64" / "chrome",
                        chromium_dir / "chrome-linux" / "chrome",
                    ]

                for exe in exe_paths:
                    if exe.exists():
                        logger.debug(f"Found Chromium at: {exe}")
                        return True

    return False


def clear_browser_cache() -> None:
    """Clear the browser installation cache."""
    global _browser_installed_cache
    _browser_installed_cache = None


@dataclass
class PlaywrightFetchResult:
    """Result from Playwright fetch."""

    content: str
    title: str | None = None
    final_url: str | None = None
    screenshot_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PlaywrightRenderer:
    """Reusable Playwright renderer to avoid browser cold starts."""

    def __init__(self, proxy: str | None = None) -> None:
        self.proxy = proxy
        self._playwright: Any = None
        self._browser: Any = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> PlaywrightRenderer:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _ensure_browser(self) -> Any:
        if self._browser is not None:
            return self._browser

        from playwright.async_api import async_playwright

        async with self._lock:
            if self._browser is not None:
                return self._browser

            self._playwright = await async_playwright().start()
            launch_options: dict[str, Any] = {"headless": True}
            if self.proxy:
                launch_options["proxy"] = {"server": self.proxy}

            try:
                self._browser = await self._playwright.chromium.launch(**launch_options)
            except Exception as e:
                await self._playwright.stop()
                self._playwright = None
                raise RuntimeError(
                    f"Failed to launch Chromium browser: {e}. "
                    "Install browser with: uv run playwright install chromium "
                    "(Linux: also run 'uv run playwright install-deps chromium')"
                )
            return self._browser

    async def fetch(
        self,
        url: str,
        timeout: int = 30000,
        wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
        extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
        screenshot_config: ScreenshotConfig | None = None,
        output_dir: Path | None = None,
    ) -> PlaywrightFetchResult:
        """Fetch URL using a persistent browser instance."""
        browser = await self._ensure_browser()
        context = await browser.new_context()
        try:
            page = await context.new_page()

            # Map wait_for string to Playwright's literal type
            wait_until_map = {
                "load": "load",
                "domcontentloaded": "domcontentloaded",
                "networkidle": "networkidle",
            }
            wait_until = wait_until_map.get(wait_for, "domcontentloaded")

            await page.goto(url, timeout=timeout, wait_until=wait_until)

            if extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)

            title = await page.title()
            final_url = page.url
            html_content = await page.content()
            markdown_content = _html_to_markdown(html_content)

            if _is_content_incomplete(markdown_content):
                try:
                    rendered_text = await page.inner_text("body")
                    if rendered_text and len(rendered_text.strip()) > len(
                        markdown_content.strip()
                    ):
                        markdown_content = _format_inner_text(rendered_text)
                except Exception:
                    pass

            screenshot_path = None
            if screenshot_config and output_dir:
                enabled = getattr(screenshot_config, "enabled", True)
                if enabled:
                    screenshot_path = await _capture_screenshot(
                        page, screenshot_config, output_dir, url
                    )

            return PlaywrightFetchResult(
                content=markdown_content,
                title=title,
                final_url=final_url,
                screenshot_path=screenshot_path,
                metadata={"renderer": "playwright", "wait_for": wait_for},
            )
        finally:
            await context.close()

    async def close(self) -> None:
        """Close browser and playwright instances."""
        async with self._lock:
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None


async def fetch_with_playwright(
    url: str,
    timeout: int = 30000,
    wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
    extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    proxy: str | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    output_dir: Path | None = None,
    renderer: PlaywrightRenderer | None = None,
) -> PlaywrightFetchResult:
    """Fetch URL using Playwright (reuses renderer if provided)."""
    if renderer:
        return await renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
        )

    # Legacy one-off path
    async with PlaywrightRenderer(proxy=proxy) as standalone_renderer:
        return await standalone_renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
        )


def _is_content_incomplete(content: str) -> bool:
    """Check if content appears incomplete (likely Shadow DOM or JS rendering issue).

    Args:
        content: Markdown content to check

    Returns:
        True if content appears incomplete
    """
    if not content:
        return True

    # Remove markdown syntax and whitespace
    clean = re.sub(r"[#\-*_>\[\]`|()!]", "", content)
    clean = re.sub(r"\[.*?\]\(.*?\)", "", clean)  # Remove links
    clean = " ".join(clean.split())

    # Check for common incomplete content patterns
    incomplete_patterns = [
        r"Don't miss what's happening",  # X.com login prompt
        r"Accept all cookies",  # Cookie consent
        r"Log in.*Sign up",  # Login/signup only
        r"Terms of Service.*Privacy Policy",  # Footer only
    ]

    # Content is mostly boilerplate
    for pattern in incomplete_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            # Check if there's substantial other content
            if len(clean) < 500:
                return True

    # Too short to be meaningful content
    return len(clean) < 200


def _format_inner_text(text: str) -> str:
    """Format inner_text content as basic markdown.

    Args:
        text: Raw text from page.inner_text()

    Returns:
        Formatted markdown content
    """
    if not text:
        return ""

    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Preserve the line with basic cleanup
        formatted_lines.append(stripped)

    # Join with double newlines for paragraph separation
    return "\n\n".join(formatted_lines)


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown.

    Tries markitdown first, falls back to basic HTML stripping.

    Args:
        html: HTML content

    Returns:
        Markdown content
    """
    # Remove <noscript> tags - Playwright has JS enabled, so these are irrelevant
    # and often contain fallback messages like "JavaScript is not available"
    html = re.sub(
        r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE
    )

    try:
        import io

        from markitdown import MarkItDown

        md = MarkItDown()
        # MarkItDown uses convert_stream for in-memory content
        stream = io.BytesIO(html.encode("utf-8"))
        result = md.convert_stream(stream, file_extension=".html")
        return result.text_content if result and result.text_content else ""
    except Exception as e:
        logger.debug(f"markitdown conversion failed, using fallback: {e}")
        # Fallback: basic HTML tag stripping
        return _strip_html_tags(html)


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags from content (fallback converter).

    Args:
        html: HTML content

    Returns:
        Plain text with HTML tags removed
    """
    # Remove script, style, and noscript elements
    html = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(
        r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    try:
        import html as html_module

        text = html_module.unescape(text)
    except Exception:
        pass

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    return text.strip()


async def _capture_screenshot(
    page: Any,
    config: ScreenshotConfig,
    output_dir: Path,
    url: str,
) -> Path | None:
    """Capture page screenshot.

    Args:
        page: Playwright page object
        config: Screenshot configuration
        output_dir: Output directory
        url: Original URL (for filename)

    Returns:
        Path to screenshot file, or None on failure
    """
    from markitai.fetch import _compress_screenshot, _url_to_screenshot_filename

    try:
        # Generate filename using the same logic as fetch.py
        filename = _url_to_screenshot_filename(url)

        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / filename

        # Get settings from config
        full_page = getattr(config, "full_page", True)
        quality = getattr(config, "quality", 85)
        max_height = getattr(config, "max_height", 10000)

        await page.screenshot(
            path=str(screenshot_path),
            full_page=full_page,
            type="jpeg",
            quality=quality,
        )

        # Compress and resize if needed (handles max_height limit)
        _compress_screenshot(screenshot_path, quality=quality, max_height=max_height)

        logger.debug(f"Screenshot saved: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        logger.warning(f"Screenshot capture failed: {e}")
        return None
