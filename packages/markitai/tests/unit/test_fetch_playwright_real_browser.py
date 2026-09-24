"""Real-Chromium checks for the Playwright renderer (screenshot order, hangs).

These load ``data:`` URLs, so they need no network, but they DO need
Playwright and Chromium. Marked ``slow`` (deselected by default), like
test_fetch_playwright_dom_normalize.py.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import pytest

from markitai.config import ScreenshotConfig


def _skip_if_no_playwright_browser() -> None:
    # The sync-API probe other browser tests use cannot run inside these
    # async tests' event loop, so check the install on disk instead.
    from markitai.fetch_playwright import (
        is_playwright_available,
        is_playwright_browser_installed,
    )

    if not is_playwright_available():
        pytest.skip("playwright not installed")
    if not is_playwright_browser_installed(use_cache=False):
        pytest.skip("playwright browsers not installed")


def _data_url(html: str) -> str:
    return "data:text/html;charset=utf-8," + quote(html)


STYLED_PAGE = (
    "<!doctype html><html><head><title>Styled</title><style>"
    "body{margin:0;background:#00a000;color:#fff;font:20px sans-serif}"
    "nav{background:#ff0000;height:200px}</style></head><body>"
    "<nav>NAVIGATION</nav><main>"
    + "<p>Body text that is long enough to be extracted as the main content. </p>" * 40
    + "</main></body></html>"
)


@pytest.mark.slow
async def test_screenshot_shows_the_styled_page_not_the_cleaned_dom(
    tmp_path: Path,
) -> None:
    _skip_if_no_playwright_browser()
    from PIL import Image

    from markitai.fetch_playwright import PlaywrightRenderer

    async with PlaywrightRenderer() as renderer:
        result = await renderer.fetch(
            _data_url(STYLED_PAGE),
            timeout=15000,
            extra_wait_ms=0,
            skip_auto_scroll=True,
            screenshot_config=ScreenshotConfig(
                enabled=True, viewport_width=800, viewport_height=600
            ),
            output_dir=tmp_path,
        )

    assert result.screenshot_path is not None
    with Image.open(result.screenshot_path) as image:
        rgb = image.convert("RGB")
        nav = rgb.getpixel((400, 100))
        body = rgb.getpixel((400, 400))
    # The nav bar is removed by the extraction cleanup; the capture predates it
    assert isinstance(nav, tuple) and nav[0] > 200 and nav[1] < 60
    assert isinstance(body, tuple) and body[1] > 120 and body[0] < 60
    assert "Body text" in result.content


@pytest.mark.slow
async def test_script_loop_after_load_fails_within_the_timeout() -> None:
    _skip_if_no_playwright_browser()
    from markitai.fetch_playwright import (
        PlaywrightPageTimeoutError,
        PlaywrightRenderer,
    )

    page = (
        "<!doctype html><html><body><p>content</p>"
        "<script>setTimeout(() => { while (true) {} }, 100)</script></body></html>"
    )
    started = time.monotonic()
    async with PlaywrightRenderer() as renderer:
        with pytest.raises(PlaywrightPageTimeoutError):
            await renderer.fetch(
                _data_url(page),
                timeout=2000,
                extra_wait_ms=500,
                skip_auto_scroll=True,
            )
    # goto (<2s) + 0.5s wait + 2s page budget + closing, far from "forever"
    assert time.monotonic() - started < 30


@pytest.mark.slow
async def test_screenshot_blocked_by_a_busy_page_keeps_the_text(
    tmp_path: Path,
) -> None:
    """A page that is busy for a while right after load blocks the capture.

    The screenshot then fails on its own budget; once the page is idle
    again the text is still extracted instead of the whole fetch failing.
    """
    _skip_if_no_playwright_browser()
    from markitai.fetch_playwright import PlaywrightRenderer

    page = (
        "<!doctype html><html><body>"
        + "<p>Readable paragraph that survives a slow screenshot.</p>" * 20
        + "<script>setTimeout(() => { const end = Date.now() + 3000; "
        "while (Date.now() < end) {} }, 50)</script></body></html>"
    )
    started = time.monotonic()
    async with PlaywrightRenderer() as renderer:
        result = await renderer.fetch(
            _data_url(page),
            timeout=2000,
            extra_wait_ms=200,
            skip_auto_scroll=True,
            screenshot_config=ScreenshotConfig(enabled=True),
            output_dir=tmp_path,
        )
    assert "Readable paragraph" in result.content
    if result.screenshot_path is None:
        assert result.metadata.get("screenshot_error")
    assert time.monotonic() - started < 30
