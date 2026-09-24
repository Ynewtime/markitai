"""Static HTTP client adapter for URL fetching."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger

# Strong references to pending close tasks so they are not garbage
# collected before completing (create_task only keeps a weak reference).
_pending_close_tasks: set[Any] = set()


def _public_http_client(proxy: str | None, timeout: float) -> Any:
    import httpx

    return httpx.AsyncClient(trust_env=False, proxy=proxy, timeout=timeout)


async def public_http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
    timeout: float = 30.0,
    proxy: str | None = None,
    follow_redirects: bool = True,
) -> Any:
    """Fetch public destinations only, pinning each connection to checked DNS.

    The numeric connection URL prevents a second DNS lookup from rebinding
    the destination. Host and TLS SNI retain the original hostname. Every
    redirect gets a fresh address check before a request can be sent.
    """
    import ipaddress
    from urllib.parse import urljoin

    import httpx

    from markitai.fetch_policy import (
        assess_url_for_remote,
        resolve_hostname_addresses,
    )

    current = httpx.URL(url)
    request_headers = httpx.Headers(headers)
    cookies = httpx.Cookies()
    async with _public_http_client(proxy, timeout) as client:
        for _ in range(11):
            assessment = await assess_url_for_remote(str(current))
            if not assessment.allowed:
                raise PermissionError(f"Fetch target refused: {assessment.reason}")
            addresses = await resolve_hostname_addresses(current.host)
            if not addresses or any(
                not ipaddress.ip_address(address).is_global
                or ipaddress.ip_address(address).is_multicast
                for address in addresses
            ):
                raise PermissionError("Fetch target resolved to a non-public address")
            request_headers["Host"] = current.netloc.decode("ascii")
            logical_request = httpx.Request(method, current, headers=request_headers)
            cookies.set_cookie_header(logical_request)
            client.cookies.clear()  # Never scope cookies to the pinned numeric IP.
            for index, address in enumerate(addresses):
                try:
                    response = await client.request(
                        method,
                        current.copy_with(host=address),
                        headers=logical_request.headers,
                        content=content,
                        extensions={"sni_hostname": current.host},
                        follow_redirects=False,
                    )
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    # IPv6 or one replica can be unavailable. Retry only
                    # connection failures, before the request was sent.
                    if index == len(addresses) - 1:
                        raise
            # Consumers see the logical URL, never the transport's numeric IP.
            response.request = logical_request
            cookies.extract_cookies(response)
            location = response.headers.get("location")
            if not follow_redirects or not response.is_redirect or not location:
                return response
            target = httpx.URL(urljoin(str(current), location))
            if (
                target.host != current.host
                or target.scheme != current.scheme
                or target.port != current.port
            ):
                for name in ("Authorization", "Cookie", "Proxy-Authorization"):
                    request_headers.pop(name, None)
            if (response.status_code == 303 and method != "HEAD") or (
                response.status_code in (301, 302) and method == "POST"
            ):
                method, content = "GET", None
                request_headers.pop("Content-Length", None)
                request_headers.pop("Content-Type", None)
            current = target
    raise PermissionError("Too many redirects")


def schedule_client_close(coro: Any, name: str) -> None:
    """Schedule an async client close on the running loop (best-effort).

    Keeps a strong reference to the task until it completes. If no event
    loop is running, the close is skipped (logged) and the coroutine is
    closed to avoid a "never awaited" warning.
    """
    import asyncio

    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()
        logger.debug(f"[{name}] No running event loop; old client left to GC")
        return
    _pending_close_tasks.add(task)
    task.add_done_callback(_pending_close_tasks.discard)


# Provider indirection (same shape as fetch_consent's state provider):
# fetch_session registers the session-owned bypass patterns, which merge the
# NO_PROXY env var with the OS proxy exception list. Importing fetch_session
# from here would drag this foundation module into the fetch_playwright /
# webextract layers (import-linter contract), so the dependency is inverted.
# The fallback keeps NO_PROXY working even if fetch_session was never imported.
_proxy_bypass_provider: Callable[[], list[str]] | None = None


def set_proxy_bypass_provider(provider: Callable[[], list[str]]) -> None:
    """Register the source of NO_PROXY-style proxy bypass patterns.

    Called by ``markitai.fetch_session`` at import time so the patterns come
    from the process-wide FetchSession (env var + OS exception list).
    """
    global _proxy_bypass_provider
    _proxy_bypass_provider = provider


def _proxy_bypass_patterns() -> list[str]:
    """Return the active NO_PROXY patterns (session-owned once registered)."""
    if _proxy_bypass_provider is not None:
        return _proxy_bypass_provider()

    from markitai.fetch_policy import parse_no_proxy

    return parse_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))


def is_loopback_url(url: str) -> bool:
    """Whether *url* targets this machine (``localhost``, 127.0.0.0/8, ``::1``).

    Loopback hosts are always reached without a proxy, matching the browser
    (``fetch_playwright.chromium_proxy_bypass``): a proxy would dial its own
    loopback, not ours.
    """
    import ipaddress
    from urllib.parse import urlsplit

    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback


def resolve_proxy_for_url(url: str, proxy: str | None) -> str | None:
    """Apply NO_PROXY bypass rules to a caller-supplied proxy candidate.

    Callers resolve *which* proxy exists (env/system detection); this is the
    connection-level bypass deciding whether the host being fetched may use
    it at all. It lives here so every user of the shared static clients
    honors NO_PROXY, not only the ones that remember to ask.

    Args:
        url: URL about to be fetched.
        proxy: Proxy URL the caller wants to use (may be empty/None).

    Loopback hosts are always exempt (see :func:`is_loopback_url`).

    Returns:
        The proxy to use, or None when the host is exempt or no proxy was given.
    """
    if not proxy:
        return None

    from markitai.fetch_policy import host_bypasses_proxy

    if is_loopback_url(url):
        logger.debug("[HTTP] Loopback host: fetching {} without proxy", url)
        return None
    if host_bypasses_proxy(url, _proxy_bypass_patterns()):
        logger.debug("[HTTP] NO_PROXY bypass: fetching {} without proxy", url)
        return None
    return proxy


@dataclass
class StaticHttpResponse:
    """Standardized response from static HTTP clients."""

    content: bytes
    status_code: int
    headers: dict[str, str]
    url: str

    @property
    def encoding(self) -> str | None:
        """Codec for the ``Content-Type`` charset, widened per WHATWG.

        ``gb2312`` resolves to GB18030, ``iso-8859-1`` to Windows-1252 and so
        on (see :mod:`markitai.utils.charset`); None when the header has no
        usable charset.
        """
        from markitai.utils.charset import charset_from_content_type

        return charset_from_content_type(self.headers.get("content-type", ""))

    @property
    def text(self) -> str:
        """Get response content as string (BOM > header > meta > detection)."""
        from markitai.utils.charset import decode_body

        content_type = self.headers.get("content-type", "")
        is_html = not content_type or "html" in content_type.lower()
        return decode_body(self.content, content_type, html=is_html)[0]


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


#: httpx mount keys its environment proxies live under; mapping them to None
#: routes every request through the client's own direct transport.
_DIRECT_MOUNT_KEYS = ("http://", "https://", "all://")


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
        self._client_loop: Any = None  # Event loop the client is bound to

    def _get_or_create_client(self, timeout_s: float, proxy: str | None) -> Any:
        """Get or create a shared httpx.AsyncClient.

        Rebuilds the client if the proxy configuration changes or the
        running event loop differs from the one the client was created on
        (each CLI invocation runs its own asyncio.run loop; reusing a
        client bound to a closed loop raises "Event loop is closed").
        """
        import asyncio

        try:
            loop: Any = asyncio.get_running_loop()
        except RuntimeError:  # sync/test context: skip the loop check
            loop = None
        if (
            self._client is not None
            and self._client_proxy == proxy
            and (loop is None or self._client_loop is loop)
        ):
            return self._client

        import httpx

        # Close old client if proxy or loop changed
        if self._client is not None:
            if loop is None or self._client_loop is loop:
                schedule_client_close(self._client.aclose(), self.name)
            else:
                # The old loop is gone; its connections died with it
                logger.debug("[HTTP] Dropping httpx client bound to a stale loop")

        client_kwargs: dict[str, Any] = {
            "follow_redirects": True,
            "timeout": timeout_s,
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        if proxy:
            client_kwargs["proxy"] = proxy
        else:
            # No proxy means direct: resolve_proxy_for_url already decided
            # (NO_PROXY, loopback), so httpx must not re-add HTTP(S)_PROXY
            # from the environment. Other env settings (CA bundle) still apply.
            client_kwargs["mounts"] = dict.fromkeys(_DIRECT_MOUNT_KEYS)

        self._client = httpx.AsyncClient(**client_kwargs)
        self._client_proxy = proxy
        self._client_loop = loop
        return self._client

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        """Perform a GET request using the shared client.

        The client is keyed by the *effective* proxy, so a NO_PROXY host and
        a proxied host use separate pooled clients.
        """
        import httpx

        from markitai.fetch_policy import public_network_only

        if public_network_only.get():
            resp = await public_http_request(
                url,
                headers=headers,
                timeout=timeout_s,
                proxy=resolve_proxy_for_url(url, proxy),
            )
        else:
            client = self._get_or_create_client(
                timeout_s, resolve_proxy_for_url(url, proxy)
            )
            resp = await client.get(
                url, headers=headers, timeout=httpx.Timeout(timeout_s)
            )
        return StaticHttpResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items()},
            url=str(resp.url),
        )

    async def close(self) -> None:
        """Close the shared client and release resources.

        A client bound to a different (dead) event loop cannot be aclosed
        from the current loop — its connections died with that loop, so
        just drop the reference.
        """
        import asyncio

        if self._client is not None:
            try:
                current = asyncio.get_running_loop()
            except RuntimeError:
                current = None
            if self._client_loop is None or self._client_loop is current:
                await self._client.aclose()
            else:
                logger.debug(
                    "[HTTP] Dropping httpx client bound to a stale loop (close)"
                )
            self._client = None
            self._client_proxy = None
            self._client_loop = None


class CurlCffiClient:
    """Static HTTP client using curl-cffi (impersonation support).

    Maintains a persistent AsyncSession for connection reuse.
    Call close() to release resources.
    """

    name = "curl_cffi"

    def __init__(self) -> None:
        self._session: Any = None
        self._session_proxy: str | None = None
        self._session_loop: Any = None  # Event loop the session is bound to

    def _get_or_create_session(self, proxy: str | None) -> Any:
        """Get or create a shared curl-cffi AsyncSession.

        Rebuilds when the proxy or the running event loop changes (a
        session bound to a closed loop raises "Event loop is closed").
        """
        import asyncio

        try:
            loop: Any = asyncio.get_running_loop()
        except RuntimeError:  # sync/test context: skip the loop check
            loop = None
        if (
            self._session is not None
            and self._session_proxy == proxy
            and (loop is None or self._session_loop is loop)
        ):
            return self._session

        from curl_cffi.requests import AsyncSession  # type: ignore[import-not-found]

        # Close old session if proxy or loop changed
        if self._session is not None:
            if loop is None or self._session_loop is loop:
                schedule_client_close(self._session.close(), self.name)
            else:
                logger.debug("[HTTP] Dropping curl session bound to a stale loop")

        # An empty proxy string makes libcurl connect directly even when
        # http_proxy/HTTPS_PROXY are set; None would let it re-read them for
        # a host resolve_proxy_for_url already exempted (NO_PROXY, loopback).
        proxies = {"http": proxy, "https": proxy} if proxy else {"all": ""}
        self._session = AsyncSession(
            impersonate="chrome",
            proxies=proxies,  # type: ignore[arg-type]  # ProxySpec TypedDict accepts str values
        )
        self._session_proxy = proxy
        self._session_loop = loop
        return self._session

    async def get(
        self,
        url: str,
        headers: dict[str, str],
        timeout_s: float,
        proxy: str | None = None,
    ) -> StaticHttpResponse:
        """Perform a GET request using the shared session.

        The session is keyed by the *effective* proxy, so a NO_PROXY host and
        a proxied host use separate pooled sessions.
        """
        from markitai.fetch_policy import public_network_only

        if public_network_only.get():
            return await HttpxClient().get(url, headers, timeout_s, proxy)
        session = self._get_or_create_session(resolve_proxy_for_url(url, proxy))
        resp = await session.get(url, headers=headers, timeout=timeout_s)
        return StaticHttpResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers={k.lower(): v for k, v in resp.headers.items() if v is not None},
            url=resp.url,
        )

    async def close(self) -> None:
        """Close the shared session and release resources.

        A session bound to a different (dead) event loop cannot be closed
        from the current loop — drop the reference instead.
        """
        import asyncio

        if self._session is not None:
            try:
                current = asyncio.get_running_loop()
            except RuntimeError:
                current = None
            if self._session_loop is None or self._session_loop is current:
                await self._session.close()
            else:
                logger.debug(
                    "[HTTP] Dropping curl session bound to a stale loop (close)"
                )
            self._session = None
            self._session_proxy = None
            self._session_loop = None


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
