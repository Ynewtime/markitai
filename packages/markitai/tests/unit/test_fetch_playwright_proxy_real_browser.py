"""Real-Chromium checks for proxied browsers (bypass list, coexisting renderers).

A local page server and a logging forward proxy run on loopback, so no
network is needed, but Playwright and Chromium are. Marked ``slow``
(deselected by default), like test_fetch_playwright_real_browser.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import pytest


def _skip_if_no_playwright_browser() -> None:
    from markitai.fetch_playwright import (
        is_playwright_available,
        is_playwright_browser_installed,
    )

    if not is_playwright_available():
        pytest.skip("playwright not installed")
    if not is_playwright_browser_installed(use_cache=False):
        pytest.skip("playwright browsers not installed")


@dataclass
class _Servers:
    page_port: int
    proxy_port: int
    proxied: list[str] = field(default_factory=list)

    @property
    def proxy(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}"


def _page(path: str, page_port: int) -> bytes:
    if path.endswith(".png"):
        return b"\x89PNG\r\n\x1a\n"
    return (
        "<!doctype html><html><head><title>Page</title></head><body><main>"
        f"<h1>Heading {path}</h1>"
        + "<p>Body text long enough to be extracted as the main content.</p>" * 5
        + f'<img src="http://127.0.0.1:{page_port}/loopback.png">'
        + f'<img src="http://exempt.test:{page_port}/exempt.png">'
        + "</main></body></html>"
    ).encode()


@pytest.fixture
async def servers() -> AsyncIterator[_Servers]:
    """Page server plus a forward proxy that records every request it sees."""
    state = _Servers(page_port=0, proxy_port=0)

    async def _respond(writer: asyncio.StreamWriter, body: bytes) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    async def page(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        path = head.split(" ", 2)[1]
        await asyncio.sleep(0.3)  # keep concurrent fetches overlapping
        await _respond(writer, _page(path, state.page_port))

    async def proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        method, target, _ = head.split(" ", 2)
        state.proxied.append(target)
        if method == "CONNECT":
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        # Every host the proxy is asked for is served by the page server
        await _respond(writer, _page(urlsplit(target).path, state.page_port))

    page_server = await asyncio.start_server(page, "127.0.0.1", 0)
    proxy_server = await asyncio.start_server(proxy, "127.0.0.1", 0)
    state.page_port = page_server.sockets[0].getsockname()[1]
    state.proxy_port = proxy_server.sockets[0].getsockname()[1]
    try:
        yield state
    finally:
        for server in (page_server, proxy_server):
            server.close()
            await server.wait_closed()


async def _fetch(renderer, url: str):
    return await renderer.fetch(
        url, timeout=15000, extra_wait_ms=300, skip_auto_scroll=True
    )


@pytest.mark.slow
async def test_proxied_browser_reaches_loopback_and_no_proxy_hosts_directly(
    servers: _Servers,
) -> None:
    """Playwright proxies loopback by default (<-loopback>); NO_PROXY hosts of
    subresources used to go through the proxy too."""
    _skip_if_no_playwright_browser()
    from markitai.fetch_playwright import PlaywrightRenderer

    async with PlaywrightRenderer(
        proxy=servers.proxy, proxy_bypass=["exempt.test"]
    ) as renderer:
        result = await _fetch(renderer, f"http://site.test:{servers.page_port}/p")

    assert "Heading /p" in result.content
    # Only the proxied page itself went through the proxy: neither the
    # loopback image nor the NO_PROXY host's image did
    assert servers.proxied == [f"http://site.test:{servers.page_port}/p"]


@pytest.mark.slow
async def test_renderers_of_different_proxies_serve_concurrent_fetches(
    servers: _Servers,
) -> None:
    """Regression: asking the session for an unproxied renderer closed the
    proxied one while it was mid-fetch (TargetClosedError)."""
    _skip_if_no_playwright_browser()
    from markitai.fetch_session import FetchSession

    session = FetchSession()
    session.detected_proxy = servers.proxy
    # The OS exception list the session hands to its proxied browser
    session.detected_proxy_bypass = "exempt.test"
    try:
        proxied = await session.get_playwright_renderer(proxy=servers.proxy)
        first = asyncio.create_task(
            _fetch(proxied, f"http://site.test:{servers.page_port}/a")
        )
        await asyncio.sleep(0.2)  # the proxied browser is now in use
        direct = await session.get_playwright_renderer(proxy=None)
        results = await asyncio.gather(
            first,
            _fetch(direct, f"http://127.0.0.1:{servers.page_port}/b"),
            _fetch(proxied, f"http://site.test:{servers.page_port}/c"),
        )
    finally:
        await session.close()

    assert direct is not proxied
    assert "Heading /a" in results[0].content
    assert "Heading /b" in results[1].content
    assert "Heading /c" in results[2].content
    assert sorted(servers.proxied) == [
        f"http://site.test:{servers.page_port}/a",
        f"http://site.test:{servers.page_port}/c",
    ]
