"""Static HTTP client adapter for URL fetching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from loguru import logger


@dataclass
class StaticHttpResponse:
    """Standardized response from static HTTP clients."""

    content: bytes
    status_code: int
    headers: dict[str, str]
    url: str

    @property
    def text(self) -> str:
        """Get response content as string."""
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


class HttpxClient:
    """Static HTTP client using httpx."""

    name = "httpx"

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        import httpx

        client_kwargs = {"follow_redirects": True, "timeout": timeout_s}
        if proxy:
            client_kwargs["proxy"] = proxy

        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(url, headers=headers)
            return StaticHttpResponse(
                content=resp.content,
                status_code=resp.status_code,
                headers={k.lower(): v for k, v in resp.headers.items()},
                url=str(resp.url),
            )


class CurlCffiClient:
    """Static HTTP client using curl-cffi (impersonation support)."""

    name = "curl_cffi"

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        from curl_cffi.requests import AsyncSession  # type: ignore[import-not-found]

        async with AsyncSession(impersonate="chrome") as s:
            proxies = {"http": proxy, "https": proxy} if proxy else None
            resp = await s.get(url, headers=headers, timeout=timeout_s, proxies=proxies)
            return StaticHttpResponse(
                content=resp.content,
                status_code=resp.status_code,
                headers={k.lower(): v for k, v in resp.headers.items()},
                url=resp.url,
            )


def get_static_http_client() -> StaticHttpClient:
    """Get the configured static HTTP client."""
    mode = os.getenv("MARKITAI_STATIC_HTTP", "httpx").lower()

    if mode == "curl_cffi":
        import importlib.util

        if importlib.util.find_spec("curl_cffi") is not None:
            return CurlCffiClient()
        else:
            logger.debug("curl-cffi not installed, falling back to httpx")
            return HttpxClient()

    return HttpxClient()
