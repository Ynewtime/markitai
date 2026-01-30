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
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

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

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
                browser.close()
                _browser_installed_cache = True
                return True
            except Exception as e:
                logger.debug(f"Playwright browser check failed: {e}")
                _browser_installed_cache = False
                return False
    except Exception as e:
        logger.debug(f"Playwright sync_playwright failed: {e}")
        _browser_installed_cache = False
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


async def fetch_with_playwright(
    url: str,
    timeout: int = 30000,
    wait_for: str = "domcontentloaded",
    extra_wait_ms: int = 3000,
    proxy: str | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    output_dir: Path | None = None,
) -> PlaywrightFetchResult:
    """Fetch URL using Playwright headless browser.

    Args:
        url: URL to fetch
        timeout: Page load timeout in milliseconds
        wait_for: Wait condition (load, domcontentloaded, networkidle)
        extra_wait_ms: Extra wait after load state for JS rendering
        proxy: Proxy URL (e.g., http://127.0.0.1:7890)
        screenshot_config: Screenshot settings
        output_dir: Directory for screenshots

    Returns:
        PlaywrightFetchResult with markdown content

    Raises:
        ImportError: If playwright is not installed
        RuntimeError: If browser is not installed or launch fails
    """
    if not is_playwright_available():
        raise ImportError(
            "playwright is not installed. "
            "Install with: pip install playwright && playwright install chromium"
        )

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        # Configure browser launch options
        launch_options: dict[str, Any] = {
            "headless": True,
        }

        if proxy:
            launch_options["proxy"] = {"server": proxy}

        try:
            browser = await p.chromium.launch(**launch_options)
        except Exception as e:
            raise RuntimeError(
                f"Failed to launch Chromium browser: {e}. "
                "Install browser with: playwright install chromium"
            )

        try:
            page = await browser.new_page()

            # Navigate to URL
            # Map wait_for string to Playwright's literal type
            wait_until_map = {
                "load": "load",
                "domcontentloaded": "domcontentloaded",
                "networkidle": "networkidle",
            }
            wait_until = wait_until_map.get(wait_for, "domcontentloaded")

            await page.goto(url, timeout=timeout, wait_until=wait_until)  # type: ignore[arg-type]

            # Extra wait for JS rendering
            if extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)

            # Get page info
            title = await page.title()
            final_url = page.url

            # Get page content as HTML and convert to markdown
            html_content = await page.content()
            markdown_content = _html_to_markdown(html_content)

            # Capture screenshot if requested
            screenshot_path = None
            if screenshot_config and output_dir:
                # Check if screenshot is enabled (handle both attribute and dict access)
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
            await browser.close()


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown.

    Tries markitdown first, falls back to basic HTML stripping.

    Args:
        html: HTML content

    Returns:
        Markdown content
    """
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
    # Remove script and style elements
    html = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

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
    try:
        # Generate filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_{url_hash}.png"

        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / filename

        # Get full_page setting
        full_page = getattr(config, "full_page", True)

        await page.screenshot(
            path=str(screenshot_path),
            full_page=full_page,
            type="png",
        )

        logger.debug(f"Screenshot saved: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        logger.warning(f"Screenshot capture failed: {e}")
        return None
