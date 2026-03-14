"""Static HTTP client adapter for URL fetching."""

from __future__ import annotations

import codecs
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger


@dataclass
class StaticHttpResponse:
    """Standardized response from static HTTP clients."""

    content: bytes
    status_code: int
    headers: dict[str, str]
    url: str

    @property
    def encoding(self) -> str | None:
        """Best-effort charset parsed from the response headers."""
        content_type = self.headers.get("content-type", "")
        match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        if not match:
            return None

        encoding = match.group(1).strip().strip("\"'")
        try:
            codecs.lookup(encoding)
        except LookupError:
            return None
        return encoding

    @property
    def text(self) -> str:
        """Get response content as string."""
        if self.encoding:
            try:
                return self.content.decode(self.encoding)
            except UnicodeDecodeError:
                pass
        return self.content.decode("utf-8", errors="replace")


@runtime_checkable
class StaticHttpClient(Protocol):
    """Protocol for static HTTP clients."""

    name: str

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        """Perform a GET request."""
        ...

    async def close(self) -> None:
        """Close the client and release resources."""
        ...


class HttpxClient:
    """Static HTTP client using httpx with connection pooling.

    Maintains a persistent AsyncClient for connection reuse across requests.
    The client is lazily initialized on first use and rebuilt when proxy config
    changes. Call close() to release resources.
    """

    name = "httpx"

    def __init__(self) -> None:
        self._client: Any = None
        self._client_proxy: str | None = None  # Track proxy for rebuild

    def _get_or_create_client(self, timeout_s: float, proxy: str | None) -> Any:
        """Get or create a shared httpx.AsyncClient.

        Rebuilds the client if proxy configuration changes.
        """
        if self._client is not None and self._client_proxy == proxy:
            return self._client

        import httpx

        # Close old client if proxy changed
        if self._client is not None:
            import asyncio

            try:
                asyncio.get_running_loop().create_task(self._client.aclose())
            except RuntimeError:
                pass

        client_kwargs: dict[str, Any] = {
            "follow_redirects": True,
            "timeout": timeout_s,
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        if proxy:
            client_kwargs["proxy"] = proxy

        self._client = httpx.AsyncClient(**client_kwargs)
        self._client_proxy = proxy
        return self._client

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        """Perform a GET request using the shared client."""
        import httpx

        client = self._get_or_create_client(timeout_s, proxy)
        resp = await client.get(url, headers=headers, timeout=httpx.Timeout(timeout_s))
        return StaticHttpResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items()},
            url=str(resp.url),
        )

    async def close(self) -> None:
        """Close the shared client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._client_proxy = None


class CurlCffiClient:
    """Static HTTP client using curl-cffi (impersonation support).

    Maintains a persistent AsyncSession for connection reuse.
    Call close() to release resources.
    """

    name = "curl_cffi"

    def __init__(self) -> None:
        self._session: Any = None
        self._session_proxy: str | None = None

    def _get_or_create_session(self, proxy: str | None) -> Any:
        """Get or create a shared curl-cffi AsyncSession."""
        if self._session is not None and self._session_proxy == proxy:
            return self._session

        from curl_cffi.requests import AsyncSession  # type: ignore[import-not-found]

        # Close old session if proxy changed
        if self._session is not None:
            import asyncio

            try:
                asyncio.get_running_loop().create_task(self._session.close())
            except RuntimeError:
                pass

        proxies = {"http": proxy, "https": proxy} if proxy else None
        self._session = AsyncSession(
            impersonate="chrome",
            proxies=proxies,  # type: ignore[arg-type]  # ProxySpec TypedDict accepts str values
        )
        self._session_proxy = proxy
        return self._session

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        """Perform a GET request using the shared session."""
        session = self._get_or_create_session(proxy)
        resp = await session.get(url, headers=headers, timeout=timeout_s)  # type: ignore[reportArgumentType]  # curl_cffi stub mismatch
        return StaticHttpResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items() if v is not None},
            url=resp.url,
        )

    async def close(self) -> None:
        """Close the shared session and release resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None
            self._session_proxy = None


# Global singleton clients (lazily initialized, reused across calls)
_httpx_client: HttpxClient | None = None
_curl_cffi_client: CurlCffiClient | None = None


def get_static_http_client() -> StaticHttpClient:
    """Get the configured static HTTP client (singleton).

    Returns a shared client instance for connection reuse across requests.
    """
    global _httpx_client, _curl_cffi_client

    mode = os.getenv("MARKITAI_STATIC_HTTP", "httpx").lower()

    if mode == "curl_cffi":
        import importlib.util

        if importlib.util.find_spec("curl_cffi") is not None:
            if _curl_cffi_client is None:
                _curl_cffi_client = CurlCffiClient()
            return _curl_cffi_client
        else:
            logger.debug("curl-cffi not installed, falling back to httpx")

    if _httpx_client is None:
        _httpx_client = HttpxClient()
    return _httpx_client


async def close_static_http_clients() -> None:
    """Close all shared static HTTP client instances."""
    global _httpx_client, _curl_cffi_client
    if _httpx_client is not None:
        await _httpx_client.close()
        _httpx_client = None
    if _curl_cffi_client is not None:
        await _curl_cffi_client.close()
        _curl_cffi_client = None
