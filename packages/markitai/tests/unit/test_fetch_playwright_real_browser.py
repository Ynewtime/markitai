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

    assert result.screenshot_path is not None, result.metadata.get("screenshot_error")
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


@pytest.mark.slow
async def test_content_from_a_late_request_is_still_waited_for() -> None:
    """The render wait ends when the page settles, not before its data came.

    The page starts its request after a 300 ms timer and the answer takes
    another 700 ms: both inside the quiet window and the in-flight check.
    """
    _skip_if_no_playwright_browser()
    import threading
    from http.server import BaseHTTPRequestHandler

    from markitai.fetch_playwright import PlaywrightRenderer
    from tests.local_http import LocalHTTPServer

    article = "Late paragraph delivered by a slow API call. " * 30
    page = (
        "<!doctype html><html><head><title>Late</title></head><body>"
        "<main id='m'>Loading</main><script>setTimeout(() => fetch('/data')"
        ".then(r => r.text()).then(t => { document.getElementById('m')"
        ".innerHTML = '<p>' + t + '</p>'; }), 300)</script></body></html>"
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/data":
                time.sleep(0.7)
                body = article.encode()
            else:
                body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = LocalHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    try:
        async with PlaywrightRenderer() as renderer:
            result = await renderer.fetch(
                f"http://127.0.0.1:{server.server_address[1]}/",
                timeout=15000,
                extra_wait_ms=3000,
                skip_auto_scroll=True,
            )
    finally:
        server.shutdown()
        server.server_close()
    assert "Late paragraph delivered by a slow API call." in result.content


@pytest.mark.slow
async def test_a_complete_page_does_not_sit_out_the_render_bound() -> None:
    _skip_if_no_playwright_browser()
    from markitai.fetch_playwright import PlaywrightRenderer

    page = (
        "<!doctype html><html><head><title>Done</title></head><body>"
        + "<p>Everything on this page is there when it loads.</p>" * 30
        + "</body></html>"
    )
    async with PlaywrightRenderer() as renderer:
        await renderer._ensure_browser()  # time the page, not the launch
        started = time.monotonic()
        result = await renderer.fetch(
            _data_url(page), timeout=15000, extra_wait_ms=3000
        )
        elapsed = time.monotonic() - started
    assert "Everything on this page" in result.content
    # It used to sleep 3 s, scroll for 0.6 s and pause another 0.8 s
    assert elapsed < 2.5


@pytest.mark.slow
async def test_extra_http_headers_reach_the_initial_origin_only() -> None:
    """A secret in extra_http_headers reaches the page's own origin and its
    same-origin subresources, never a cross-origin subresource."""
    _skip_if_no_playwright_browser()
    import threading
    from http.server import BaseHTTPRequestHandler

    from markitai.fetch_playwright import PlaywrightRenderer
    from tests.local_http import LocalHTTPServer

    other_seen: list[tuple[str, str]] = []
    main_seen: list[tuple[str, str]] = []

    def make_handler(seen: list[tuple[str, str]], page: str | None):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                seen.append((self.path, self.headers.get("X-Account", "")))
                if page is not None and self.path == "/":
                    body = page.encode()
                    mime = "text/html; charset=utf-8"
                else:
                    body = b"GIF87a"
                    mime = "image/gif"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        return Handler

    other = LocalHTTPServer(("127.0.0.1", 0), make_handler(other_seen, None))
    other_port = other.server_address[1]
    page = (
        "<!doctype html><html><head><title>Origins</title></head><body>"
        "<main><p>A page with a same-origin image and a cross-origin one.</p>"
        "<img src='/same.gif'>"
        f"<img src='http://127.0.0.1:{other_port}/other.gif'></main></body></html>"
    )
    main = LocalHTTPServer(("127.0.0.1", 0), make_handler(main_seen, page))
    main_port = main.server_address[1]
    for server in (main, other):
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
    try:
        async with PlaywrightRenderer() as renderer:
            await renderer.fetch(
                f"http://127.0.0.1:{main_port}/",
                timeout=15000,
                extra_wait_ms=1500,
                skip_auto_scroll=True,
                extra_http_headers={"X-Account": "private-token"},
            )
    finally:
        for server in (main, other):
            server.shutdown()
            server.server_close()

    # The document and its same-origin image carry the header.
    assert ("/", "private-token") in main_seen
    assert ("/same.gif", "private-token") in main_seen
    # The other origin was reached but never saw the header.
    assert any(path == "/other.gif" for path, _ in other_seen), other_seen
    assert all(account == "" for _, account in other_seen), other_seen
