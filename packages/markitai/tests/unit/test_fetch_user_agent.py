"""Static fetches identify markitai, and a refused agent is retried as a browser.

Wikipedia (and many other sites) answer the generic ``python-httpx/x.y``
agent with 403, so AUTO launched a browser for pages a plain request gets in
half a second. The static client now sends a descriptive agent, and a 401/403
earns one retry with a browser's headers before AUTO moves on to Playwright.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai import fetch_http
from markitai.config import FetchConfig
from markitai.fetch_http import (
    BROWSER_REQUEST_HEADERS,
    HttpxClient,
    StaticHttpResponse,
    default_user_agent,
)
from markitai.fetch_strategies.static import fetch_with_static_conditional
from markitai.fetch_types import FetchError, FetchResult

ARTICLE = (
    "<!doctype html><html><head><title>Markdown</title></head><body><article>"
    "<h1>Markdown</h1>"
    + "<p>Markdown is a lightweight markup language for creating formatted "
    "text using a plain-text editor, created by John Gruber in 2004.</p>"
    * 8
    + "</article></body></html>"
).encode()

# Installed metadata, as default_user_agent reads it (an editable install's
# metadata can lag markitai.__version__ until reinstalled)
EXPECTED_AGENT = (
    f"markitai/{metadata.version('markitai')} (+https://github.com/Ynewtime/markitai)"
)


class _Site:
    """A loopback site that records each request's User-Agent."""

    def __init__(self, accept: Callable[[str], bool]) -> None:
        self.agents: list[str] = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                agent = self.headers.get("User-Agent", "")
                site.agents.append(agent)
                status, body = (200, ARTICLE) if accept(agent) else (403, b"no")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/wiki/Markdown"
        threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        ).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def site_factory() -> Iterator[Callable[[Callable[[str], bool]], _Site]]:
    sites: list[_Site] = []

    def make(accept: Callable[[str], bool]) -> _Site:
        sites.append(_Site(accept))
        return sites[-1]

    with (
        patch("markitai.fetch_strategies.static._detect_proxy", return_value=""),
        patch.object(fetch_http, "_httpx_client", None),
        patch.object(fetch_http, "_curl_cffi_client", None),
        patch.dict("os.environ", {"MARKITAI_STATIC_HTTP": "httpx"}),
    ):
        yield make
    for site in sites:
        site.close()


def _not_generic(agent: str) -> bool:
    """Wikipedia's rule: generic HTTP library agents are refused."""
    return not agent.startswith("python-")


def _browser_only(agent: str) -> bool:
    return "Chrome/" in agent


class TestDefaultUserAgent:
    def test_names_markitai_its_version_and_a_contact(self) -> None:
        assert default_user_agent() == EXPECTED_AGENT

    async def test_static_fetch_sends_it(self, site_factory) -> None:
        site = site_factory(_not_generic)

        result = (await fetch_with_static_conditional(site.url)).result

        assert result is not None and "John Gruber" in result.content
        assert site.agents == [EXPECTED_AGENT]

    async def test_a_user_agent_in_the_request_headers_wins(self, site_factory) -> None:
        site = site_factory(lambda _agent: True)

        await HttpxClient().get(
            site.url, headers={"User-Agent": "Configured/1.0"}, timeout_s=5.0
        )

        assert site.agents == ["Configured/1.0"]

    async def test_public_network_requests_send_it(self) -> None:
        client = fetch_http._public_http_client(None, 5.0)
        try:
            assert client.headers["User-Agent"] == EXPECTED_AGENT
        finally:
            await client.aclose()


class TestRefusedAgentRetry:
    async def test_a_refusal_is_retried_once_with_browser_headers(
        self, site_factory
    ) -> None:
        site = site_factory(_browser_only)

        result = (await fetch_with_static_conditional(site.url)).result

        assert result is not None and "John Gruber" in result.content
        assert site.agents == [EXPECTED_AGENT, BROWSER_REQUEST_HEADERS["User-Agent"]]

    async def test_a_second_refusal_fails_without_a_third_request(
        self, site_factory
    ) -> None:
        site = site_factory(lambda _agent: False)

        with pytest.raises(FetchError, match="HTTP 403"):
            await fetch_with_static_conditional(site.url)

        assert len(site.agents) == 2

    @pytest.mark.parametrize("status", [404, 429, 500])
    async def test_other_errors_are_not_retried(self, status: int) -> None:
        client = MagicMock(name="client", impersonates_browser=False)
        client.name = "httpx"
        client.get = AsyncMock(
            return_value=StaticHttpResponse(b"", status, {}, "https://e.test/")
        )
        with (
            patch("markitai.fetch_strategies.static._detect_proxy", return_value=""),
            patch(
                "markitai.fetch_strategies.static.get_static_http_client",
                return_value=client,
            ),
            pytest.raises(FetchError, match=f"HTTP {status}"),
        ):
            await fetch_with_static_conditional("https://e.test/")
        assert client.get.await_count == 1

    async def test_a_browser_impersonating_client_is_not_retried(self) -> None:
        """curl_cffi already sends Chrome's agent: a retry would change nothing."""
        client = MagicMock(name="client", impersonates_browser=True)
        client.name = "curl_cffi"
        client.get = AsyncMock(
            return_value=StaticHttpResponse(b"", 403, {}, "https://e.test/")
        )
        with (
            patch("markitai.fetch_strategies.static._detect_proxy", return_value=""),
            patch(
                "markitai.fetch_strategies.static.get_static_http_client",
                return_value=client,
            ),
            pytest.raises(FetchError, match="HTTP 403"),
        ):
            await fetch_with_static_conditional("https://e.test/")
        assert client.get.await_count == 1


class TestAutoChainOrder:
    """AUTO: descriptive agent, then browser headers, then the real browser."""

    @pytest.fixture
    def browser(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        import markitai.fetch_playwright as fp

        monkeypatch.setattr(fp, "is_playwright_available", lambda: True)
        monkeypatch.setattr(
            fp, "is_playwright_browser_installed", lambda *_a, **_k: True
        )
        rendered = MagicMock(
            content="Rendered by the browser. " * 20,
            title="Markdown",
            final_url=None,
            metadata={},
            screenshot_path=None,
            screenshot_tiles=[],
        )
        mock = AsyncMock(return_value=rendered)
        monkeypatch.setattr(fp, "fetch_with_playwright", mock)
        return mock

    async def test_a_page_served_to_markitai_never_reaches_the_browser(
        self, site_factory, browser: AsyncMock
    ) -> None:
        from markitai.fetch import _fetch_with_fallback

        site = site_factory(_not_generic)

        result = await _fetch_with_fallback(site.url, FetchConfig())

        assert isinstance(result, FetchResult)
        assert result.strategy_used == "static"
        assert site.agents == [EXPECTED_AGENT]
        browser.assert_not_awaited()

    async def test_a_browser_only_page_is_fetched_statically(
        self, site_factory, browser: AsyncMock
    ) -> None:
        from markitai.fetch import _fetch_with_fallback

        site = site_factory(_browser_only)

        result = await _fetch_with_fallback(site.url, FetchConfig())

        assert result.strategy_used == "static"
        assert site.agents == [EXPECTED_AGENT, BROWSER_REQUEST_HEADERS["User-Agent"]]
        browser.assert_not_awaited()

    async def test_a_page_refused_to_both_agents_goes_to_the_browser(
        self, site_factory, browser: AsyncMock
    ) -> None:
        from markitai.fetch import _fetch_with_fallback

        site = site_factory(lambda _agent: False)

        result = await _fetch_with_fallback(site.url, FetchConfig())

        assert result.strategy_used == "playwright"
        assert len(site.agents) == 2
        browser.assert_awaited_once()
