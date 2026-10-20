"""Real-Chromium checks for the Playwright renderer (screenshot order, hangs,
where headers and redirects may go).

These load ``data:`` URLs or local servers, so they need no network, but
they DO need Playwright and Chromium. Marked ``slow`` (deselected by default), like
test_fetch_playwright_dom_normalize.py.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
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


def _redirecting_server(
    host: str, seen: list[tuple[str, str]], redirects: dict[str, str]
) -> Any:
    """A local server that logs (path, X-Account) and answers *redirects*
    (path -> Location) with 302, anything else with a small page or image."""
    from http.server import BaseHTTPRequestHandler

    from tests.local_http import LocalHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            seen.append((self.path, self.headers.get("X-Account", "")))
            location = redirects.get(self.path)
            if location is not None:
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.endswith(".gif"):
                body, mime = b"GIF87a", "image/gif"
            else:
                body = (
                    b"<!doctype html><html><head><title>Landed</title></head>"
                    b"<body><main><p>The page a redirect led to, with enough"
                    b" words to count as its content.</p><img src='pic.gif'>"
                    b"</main></body></html>"
                )
                mime = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = LocalHTTPServer((host, 0), Handler)
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    return server


@pytest.mark.slow
async def test_extra_http_headers_follow_same_origin_redirects_only() -> None:
    """The browser follows a redirect without asking the route again, so
    headers handed to it would go wherever a redirect points. A same-origin
    redirect keeps them; a redirect to another origin is reached without
    them."""
    _skip_if_no_playwright_browser()
    from markitai.fetch_playwright import PlaywrightRenderer

    main_seen: list[tuple[str, str]] = []
    other_seen: list[tuple[str, str]] = []
    other = _redirecting_server("127.0.0.1", other_seen, {})
    other_port = other.server_address[1]
    main = _redirecting_server(
        "127.0.0.1",
        main_seen,
        {
            "/away": f"http://127.0.0.1:{other_port}/landing",
            "/hop": "/doc/",
        },
    )
    main_port = main.server_address[1]
    try:
        async with PlaywrightRenderer() as renderer:
            away = await renderer.fetch(
                f"http://127.0.0.1:{main_port}/away",
                timeout=15000,
                extra_wait_ms=300,
                skip_auto_scroll=True,
                extra_http_headers={"X-Account": "private-token"},
            )
        async with PlaywrightRenderer() as renderer:
            hop = await renderer.fetch(
                f"http://127.0.0.1:{main_port}/hop",
                timeout=15000,
                extra_wait_ms=300,
                skip_auto_scroll=True,
                extra_http_headers={"X-Account": "private-token"},
            )
    finally:
        for server in (main, other):
            server.shutdown()
            server.server_close()

    assert ("/away", "private-token") in main_seen
    assert any(path == "/landing" for path, _ in other_seen), other_seen
    assert all(account == "" for _, account in other_seen), other_seen
    assert away.final_url == f"http://127.0.0.1:{other_port}/landing"
    # The same-origin hop and the page's resources still carry the header,
    # and relative links resolve against where the page came from.
    assert ("/doc/", "private-token") in main_seen
    assert ("/doc/pic.gif", "private-token") in main_seen
    assert hop.final_url == f"http://127.0.0.1:{main_port}/doc/"
    assert "The page a redirect led to" in hop.content


@pytest.mark.slow
async def test_a_public_page_cannot_redirect_the_restricted_browser_inward() -> None:
    """With public_network_only on, every redirect target is checked before
    it is asked. A "public" server (127.0.0.2, treated as a global address
    for this test) redirects to one on 127.0.0.1, which must never be
    reached; a redirect within the public site still works."""
    _skip_if_no_playwright_browser()
    import ipaddress
    from unittest.mock import patch

    import markitai.fetch_policy as fetch_policy
    from markitai.fetch_playwright import PlaywrightRenderer

    internal_seen: list[tuple[str, str]] = []
    public_seen: list[tuple[str, str]] = []
    internal = _redirecting_server("127.0.0.1", internal_seen, {})
    secret = f"http://127.0.0.1:{internal.server_address[1]}/secret"
    try:
        public = _redirecting_server(
            "127.0.0.2", public_seen, {"/start": secret, "/hop": "/doc/"}
        )
    except OSError:  # macOS has only 127.0.0.1 on lo0 by default
        internal.shutdown()
        internal.server_close()
        pytest.skip("127.0.0.2 is not available here")
    port = public.server_address[1]

    real_assess = fetch_policy.assess_url_for_remote
    real_resolve = fetch_policy.resolve_hostname_addresses
    real_is_global = ipaddress.IPv4Address.is_global

    async def assess(url: str, **kwargs: Any) -> Any:
        if "public.example" in url:
            return fetch_policy.RemoteURLAssessment(True)
        return await real_assess(url, **kwargs)

    async def resolve(host: str) -> Any:
        if host == "public.example":
            return ("127.0.0.2",)
        return await real_resolve(host)

    is_global = property(
        lambda self: str(self) == "127.0.0.2" or real_is_global.fget(self)  # type: ignore[attr-defined]
    )
    errors: list[str] = []
    token = fetch_policy.public_network_only.set(True)
    try:
        with (
            patch.object(fetch_policy, "assess_url_for_remote", assess),
            patch.object(fetch_policy, "resolve_hostname_addresses", resolve),
            patch.object(ipaddress.IPv4Address, "is_global", is_global),
        ):
            async with PlaywrightRenderer() as renderer:
                try:
                    await renderer.fetch(
                        f"http://public.example:{port}/start",
                        timeout=15000,
                        extra_wait_ms=300,
                        skip_auto_scroll=True,
                    )
                except Exception as exc:
                    errors.append(str(exc))
                hop = await renderer.fetch(
                    f"http://public.example:{port}/hop",
                    timeout=15000,
                    extra_wait_ms=300,
                    skip_auto_scroll=True,
                )
    finally:
        fetch_policy.public_network_only.reset(token)
        for server in (public, internal):
            server.shutdown()
            server.server_close()

    assert internal_seen == []
    assert errors, "the redirect inward should have failed the fetch"
    assert any(path == "/start" for path, _ in public_seen)
    assert hop.final_url == f"http://public.example:{port}/doc/"
    assert ("/doc/pic.gif", "") in public_seen
    assert "The page a redirect led to" in hop.content


@pytest.mark.slow
async def test_a_restricted_browser_names_the_refusal_of_a_private_page() -> None:
    """A user token's private URL fails as the static path's does, not with
    the browser's ``net::ERR_BLOCKED_BY_CLIENT``, and nothing is asked."""
    _skip_if_no_playwright_browser()
    import markitai.fetch_policy as fetch_policy
    from markitai.fetch_playwright import PlaywrightRenderer
    from markitai.fetch_types import FetchError

    seen: list[tuple[str, str]] = []
    server = _redirecting_server("127.0.0.1", seen, {})
    token = fetch_policy.public_network_only.set(True)
    try:
        async with PlaywrightRenderer() as renderer:
            with pytest.raises(FetchError) as raised:
                await renderer.fetch(
                    f"http://127.0.0.1:{server.server_address[1]}/",
                    timeout=15000,
                    extra_wait_ms=0,
                    skip_auto_scroll=True,
                )
    finally:
        fetch_policy.public_network_only.reset(token)
        server.shutdown()
        server.server_close()

    assert "Fetch target refused: private_or_local_host" in str(raised.value)
    assert "ERR_BLOCKED_BY_CLIENT" not in str(raised.value)
    assert seen == []
