"""Process-wide fetch session: owns every shared fetch resource.

:class:`FetchSession` collects the mutable state that ``markitai.fetch``
historically kept in module-level globals — lazily built HTTP clients,
caches, rate limiters, the shared Playwright renderer, detected proxy, and
the remote-consent state — behind one object with a single ``close()``.

``markitai.fetch`` exposes thin module-level delegates to the default
session, so the public API (and test patch points) on ``markitai.fetch``
are unchanged.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.fetch_cache import FetchCache, SPADomainCache
from markitai.fetch_consent import ConsentState, set_consent_state_provider

if TYPE_CHECKING:
    from markitai.config import FetchConfig


class _SlidingWindowRateLimiter:
    """Simple sliding-window rate limiter for API calls."""

    def __init__(self, rpm: int, name: str = "API") -> None:
        self._rpm = rpm
        self._name = name
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return  # Slot acquired

                wait_time = self._timestamps[0] - cutoff

            # Sleep OUTSIDE the lock so other coroutines can proceed
            if wait_time > 0:
                logger.debug(f"[{self._name}] Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)


# Common proxy ports used by popular proxy software.
# Only HTTP proxy ports are listed: the detected proxy is labeled http://,
# and SOCKS-only ports (1080 SOCKS5, 10808 V2Ray SOCKS, 9050 Tor) would
# produce a broken proxy URL (httpx lacks the socks extra).
_COMMON_PROXY_PORTS = [
    7897,  # Clash Verge default
    7890,  # Clash default
    7891,  # Clash mixed
    1082,  # Shadowrocket
    10809,  # V2Ray HTTP
    8080,  # HTTP proxy common
    8118,  # Privoxy
]


def _get_system_proxy() -> tuple[str, str]:
    """Get system proxy settings from OS configuration.

    Returns:
        Tuple of (proxy_url, bypass_list) where bypass_list is comma-separated hosts
        The bypass list is normalized to Linux no_proxy compatible format.
    """
    import platform
    import subprocess

    system = platform.system()

    if system == "Windows":
        try:
            import winreg  # type: ignore[import-not-found]  # Windows-only module

            with winreg.OpenKey(  # type: ignore[attr-defined]
                winreg.HKEY_CURRENT_USER,  # type: ignore[attr-defined]
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as key:
                proxy_enable, _ = winreg.QueryValueEx(key, "ProxyEnable")  # type: ignore[attr-defined]
                if proxy_enable:
                    proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")  # type: ignore[attr-defined]
                    # Handle format: "http=host:port;https=host:port" or "host:port"
                    if "=" in proxy_server:
                        # Parse protocol-specific proxies
                        for part in proxy_server.split(";"):
                            if part.startswith("https=") or part.startswith("http="):
                                proxy_addr = part.split("=", 1)[1]
                                if not proxy_addr.startswith("http"):
                                    proxy_addr = f"http://{proxy_addr}"
                                break
                        else:
                            proxy_addr = ""
                    else:
                        proxy_addr = (
                            f"http://{proxy_server}"
                            if not proxy_server.startswith("http")
                            else proxy_server
                        )

                    # Get bypass list
                    try:
                        bypass, _ = winreg.QueryValueEx(key, "ProxyOverride")  # type: ignore[attr-defined]
                        # Windows uses semicolon, convert to comma
                        bypass = bypass.replace(";", ",") if bypass else ""
                    except FileNotFoundError:
                        bypass = ""

                    if proxy_addr:
                        # Silent - system proxy detection is routine
                        return proxy_addr, bypass  # Return raw, normalize at usage
        except Exception:
            # Silent - registry read failure is not critical
            pass

    elif system == "Darwin":  # macOS
        try:
            # Get network service (usually Wi-Fi or Ethernet)
            result = subprocess.run(
                ["scutil", "--proxy"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout
                # Parse scutil output
                https_enable = "HTTPSEnable : 1" in output
                http_enable = "HTTPEnable : 1" in output

                proxy_addr = ""
                if https_enable:
                    # Extract HTTPS proxy
                    host = ""
                    for line in output.split("\n"):
                        if "HTTPSProxy :" in line:
                            host = line.split(":")[1].strip()
                        elif "HTTPSPort :" in line:
                            port = line.split(":")[1].strip()
                            if host:
                                proxy_addr = f"http://{host}:{port}"
                            break
                elif http_enable:
                    # Extract HTTP proxy
                    host = ""
                    for line in output.split("\n"):
                        if "HTTPProxy :" in line:
                            host = line.split(":")[1].strip()
                        elif "HTTPPort :" in line:
                            port = line.split(":")[1].strip()
                            if host:
                                proxy_addr = f"http://{host}:{port}"
                            break

                # Get exceptions list
                bypass = ""
                if "ExceptionsList" in output:
                    # Parse exception list from scutil output
                    in_exceptions = False
                    exceptions = []
                    for line in output.split("\n"):
                        if "ExceptionsList" in line:
                            in_exceptions = True
                        elif in_exceptions:
                            if "}" in line:
                                break
                            # Extract host from "0 : localhost" format
                            if ":" in line:
                                host = line.split(":", 1)[1].strip()
                                if host:
                                    exceptions.append(host)
                    bypass = ",".join(exceptions)

                if proxy_addr:
                    # Silent - system proxy detection is routine
                    return proxy_addr, bypass  # Return raw, normalize at usage
        except Exception:
            # Silent - scutil failure is not critical
            pass

    return "", ""


class FetchSession:
    """Process-wide fetch session: owns every resource fetch.py used to keep
    in module globals.

    All builders are lazy and rebuild on config-fingerprint changes, exactly
    as the former module-level getters did. ``close()`` releases everything
    (the former ``close_shared_clients()``).
    """

    def __init__(self) -> None:
        # Remote-fetch consent (privacy) state — see markitai.fetch_consent.
        self.consent = ConsentState()

        # Event-loop-bound concurrency primitives.
        # CF Browser Rendering: Free plan allows 2 concurrent browser
        # instances. Lazily initialized to avoid binding to a wrong event
        # loop at import time.
        self.cf_br_semaphore: asyncio.Semaphore | None = None

        # Caches (initialized lazily).
        self.spa_domain_cache: SPADomainCache | None = None
        self.fetch_cache: FetchCache | None = None
        self.fetch_cache_fingerprint: str = ""

        # Shared MarkItDown instance (reused for static fetching).
        # Note: MarkItDown's requests.Session is NOT thread-safe. However,
        # since fetch_with_static runs in the asyncio event loop (not in a
        # thread pool), only one md.convert() call executes at a time,
        # avoiding thread safety issues. If fetch_with_static is ever moved
        # to run_in_executor with threads, this should be changed to use
        # threading.local() for thread-local instances.
        self.markitdown_instance: Any = None

        # Shared HTTP clients (+ config fingerprints for rebuild-on-change).
        self.jina_client: Any = None
        self.jina_client_fingerprint: str = ""
        self.defuddle_client: Any = None
        self.defuddle_client_fingerprint: str = ""

        # Rate limiters.
        self.jina_rate_limiter: _SlidingWindowRateLimiter | None = None
        self.defuddle_rate_limiter: _SlidingWindowRateLimiter | None = None

        # Shared Playwright renderer (reused to avoid browser cold starts).
        self.playwright_renderer: Any = None
        self.playwright_renderer_fingerprint: str = ""

        # Cached proxy detection result
        # (None = not checked, "" = no proxy, "http://..." = proxy URL).
        self.detected_proxy: str | None = None
        self.detected_proxy_bypass: str | None = None

    def get_cf_semaphore(self) -> asyncio.Semaphore:
        """Get or create the CF BR rate-limiting semaphore.

        Lazily initialized to avoid binding to a wrong event loop at import time.
        CF Free plan allows 2 concurrent browser instances.
        """
        if self.cf_br_semaphore is None:
            self.cf_br_semaphore = asyncio.Semaphore(2)
        return self.cf_br_semaphore

    def get_spa_domain_cache(self) -> SPADomainCache:
        """Get or create the session's SPA domain cache instance.

        Returns:
            SPADomainCache instance
        """
        if self.spa_domain_cache is None:
            self.spa_domain_cache = SPADomainCache()
        return self.spa_domain_cache

    def get_fetch_cache(
        self, cache_dir: Path, max_size_bytes: int = 100 * 1024 * 1024
    ) -> FetchCache:
        """Get or create the session's fetch cache instance.

        Rebuilds the cache when configuration (cache_dir or max_size_bytes)
        changes, using a fingerprint to detect config drift.

        Args:
            cache_dir: Directory to store cache database
            max_size_bytes: Maximum cache size

        Returns:
            FetchCache instance
        """
        fingerprint = f"{cache_dir}:{max_size_bytes}"
        if self.fetch_cache is None or self.fetch_cache_fingerprint != fingerprint:
            if self.fetch_cache is not None:
                self.fetch_cache.close()
                logger.debug(
                    "[FetchCache] Rebuilding: config changed "
                    f"(was {self.fetch_cache_fingerprint!r}, now {fingerprint!r})"
                )
            db_path = cache_dir / "fetch_cache.db"
            self.fetch_cache = FetchCache(db_path, max_size_bytes)
            self.fetch_cache_fingerprint = fingerprint
        return self.fetch_cache

    def get_markitdown(self) -> Any:
        """Get or create the shared MarkItDown instance.

        Reusing a single instance avoids repeated initialization overhead.
        Includes Accept header for CF Markdown for Agents content negotiation.
        """
        if self.markitdown_instance is None:
            from markitdown import MarkItDown

            self.markitdown_instance = MarkItDown()
            # Enable Cloudflare Markdown for Agents content negotiation.
            # CF-enabled sites return text/markdown directly (higher quality,
            # fewer tokens). Non-CF sites return text/html as usual — zero
            # impact on existing behavior.
            #
            # Note: This patches the singleton's internal requests.Session
            # headers. Safe because get_markitdown() builds at most one
            # instance per session (guarded by `if ... is None`), and the
            # session is never internally rebuilt by MarkItDown. If markitdown
            # ever changes this, the test
            # `test_markitdown_instance_has_accept_markdown_header` will catch it.
            self.markitdown_instance._requests_session.headers.update(
                {"Accept": "text/markdown, text/html;q=0.9, */*;q=0.5"}
            )
        return self.markitdown_instance

    def get_defuddle_rate_limiter(self, rpm: int) -> _SlidingWindowRateLimiter:
        """Get or create the session's Defuddle rate limiter."""
        if self.defuddle_rate_limiter is None or self.defuddle_rate_limiter._rpm != rpm:
            self.defuddle_rate_limiter = _SlidingWindowRateLimiter(rpm, name="Defuddle")
        return self.defuddle_rate_limiter

    def get_defuddle_client(self, timeout: int = 30) -> Any:
        """Get or create the shared httpx.AsyncClient for Defuddle fetching.

        Rebuilds when ``timeout``, the detected proxy, or the running event
        loop change (config-fingerprint check, same scheme as the Jina
        client).
        """
        # Loop identity is part of the fingerprint: each CLI invocation runs
        # its own asyncio.run loop, and a client bound to a closed loop raises
        # "Event loop is closed" when reused
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:  # sync/test context
            loop_id = 0
        effective_proxy = self.detect_proxy()
        fingerprint = f"{timeout}:{effective_proxy}:{loop_id}"
        if (
            self.defuddle_client is None
            or self.defuddle_client_fingerprint != fingerprint
        ):
            import httpx

            if self.defuddle_client is not None:
                # Schedule close of old client (best-effort, non-blocking)
                logger.debug(
                    "[Defuddle] Rebuilding client: config changed "
                    f"(was {self.defuddle_client_fingerprint!r}, now {fingerprint!r})"
                )
                from markitai.fetch_http import schedule_client_close

                schedule_client_close(self.defuddle_client.aclose(), "Defuddle")

            client_kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(timeout, connect=10),
                "follow_redirects": True,
                "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
            }
            if effective_proxy:
                client_kwargs["proxy"] = effective_proxy
            self.defuddle_client = httpx.AsyncClient(**client_kwargs)
            self.defuddle_client_fingerprint = fingerprint
        return self.defuddle_client

    def get_jina_rate_limiter(self, rpm: int) -> _SlidingWindowRateLimiter:
        """Get or create the session's Jina rate limiter."""
        if self.jina_rate_limiter is None or self.jina_rate_limiter._rpm != rpm:
            self.jina_rate_limiter = _SlidingWindowRateLimiter(rpm, name="Jina")
        return self.jina_rate_limiter

    def get_jina_client(self, timeout: int = 30, proxy: str = "") -> Any:
        """Get or create the shared httpx.AsyncClient for Jina fetching.

        Reusing a single client instance avoids repeated connection setup
        overhead. The client uses connection pooling for better performance.
        Rebuilds when ``timeout`` or ``proxy`` change (config-fingerprint
        check).

        Args:
            timeout: Request timeout in seconds
            proxy: Proxy URL

        Returns:
            httpx.AsyncClient instance
        """
        # Loop identity is part of the fingerprint: each CLI invocation runs
        # its own asyncio.run loop, and a client bound to a closed loop raises
        # "Event loop is closed" when reused
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:  # sync/test context
            loop_id = 0
        fingerprint = f"{timeout}:{proxy}:{loop_id}"
        if self.jina_client is None or self.jina_client_fingerprint != fingerprint:
            import httpx

            if self.jina_client is not None:
                # Schedule close of old client (best-effort, non-blocking)
                logger.debug(
                    "[Jina] Rebuilding client: config changed "
                    f"(was {self.jina_client_fingerprint!r}, now {fingerprint!r})"
                )
                from markitai.fetch_http import schedule_client_close

                schedule_client_close(self.jina_client.aclose(), "Jina")

            # Use detected proxy if not explicitly provided
            effective_proxy = proxy or self.detect_proxy()
            client_kwargs: dict[str, Any] = {
                "timeout": timeout,
                "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
            }
            if effective_proxy:
                client_kwargs["proxy"] = effective_proxy
                logger.debug(f"[Jina] Using proxy: {effective_proxy}")

            self.jina_client = httpx.AsyncClient(**client_kwargs)
            self.jina_client_fingerprint = fingerprint
        return self.jina_client

    async def get_playwright_renderer(
        self, proxy: str | None = None, config: FetchConfig | None = None
    ) -> Any:
        """Get or create the shared PlaywrightRenderer.

        Rebuilds when ``proxy`` or session-mode configuration changes.

        Args:
            proxy: Optional proxy URL
            config: Optional fetch configuration to enable session cache

        Returns:
            PlaywrightRenderer instance
        """
        session_mode = config.playwright.session_mode if config else None
        fingerprint = f"{proxy}:{session_mode}"
        if (
            self.playwright_renderer is None
            or self.playwright_renderer_fingerprint != fingerprint
        ):
            if self.playwright_renderer is not None:
                logger.debug(
                    "[Playwright] Rebuilding renderer: config changed "
                    f"(was {self.playwright_renderer_fingerprint!r}, "
                    f"now {fingerprint!r})"
                )
                await self.playwright_renderer.close()

            from markitai.fetch_playwright import PlaywrightRenderer

            self.playwright_renderer = PlaywrightRenderer(proxy=proxy)

            # Enable domain-persistent session cache if configured
            if config and config.playwright.session_mode == "domain_persistent":
                self.playwright_renderer.enable_domain_session_cache(
                    ttl_seconds=config.playwright.session_ttl_seconds,
                    max_contexts=8,  # Default limit
                )

            self.playwright_renderer_fingerprint = fingerprint

        return self.playwright_renderer

    def detect_proxy(self, force_recheck: bool = False) -> str:
        """Detect proxy settings from environment, system config, or common local ports.

        Detection order:
        1. Environment variables: HTTPS_PROXY, HTTP_PROXY, ALL_PROXY
        2. System proxy settings (Windows registry / macOS scutil)
        3. Probe common proxy ports on localhost

        Args:
            force_recheck: Force re-detection even if cached

        Returns:
            Proxy URL string (e.g., "http://127.0.0.1:7890") or empty string
            if no proxy
        """
        if self.detected_proxy is not None and not force_recheck:
            return self.detected_proxy

        import os
        import socket

        # Check environment variables first (highest priority - user explicit config)
        for var in [
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "https_proxy",
            "http_proxy",
            "all_proxy",
        ]:
            proxy = os.environ.get(var, "").strip()
            if proxy:
                # Silent - proxy from env is routine, no need to log
                self.detected_proxy = proxy
                # Also check NO_PROXY env var
                self.detected_proxy_bypass = os.environ.get(
                    "NO_PROXY", os.environ.get("no_proxy", "")
                )
                return proxy

        # Check system proxy settings (Windows/macOS)
        system_proxy, system_bypass = _get_system_proxy()
        if system_proxy:
            self.detected_proxy = system_proxy
            self.detected_proxy_bypass = system_bypass
            return system_proxy

        # Probe common proxy ports on localhost (fallback)
        for port in _COMMON_PROXY_PORTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)  # 100ms timeout
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                if result == 0:
                    proxy_url = f"http://127.0.0.1:{port}"
                    logger.warning(f"[Proxy] Auto-detected local proxy at port {port}")
                    self.detected_proxy = proxy_url
                    self.detected_proxy_bypass = ""
                    return proxy_url
            except Exception as e:
                logger.debug("[Proxy] Auto-detection failed: {}", e)

        # Silent - no proxy is common, no need to log
        self.detected_proxy = ""
        self.detected_proxy_bypass = ""
        return ""

    async def close(self) -> None:
        """Close every shared resource owned by this session.

        Call this during cleanup to release resources (formerly
        ``markitai.fetch.close_shared_clients``).
        """
        if self.jina_client is not None:
            try:
                await self.jina_client.aclose()
            except RuntimeError:
                # Client bound to a previous (closed) event loop; its
                # connections died with that loop — just drop it
                logger.debug("[Fetch] Dropping Jina client bound to a stale loop")
            self.jina_client = None
        self.jina_client_fingerprint = ""
        if self.defuddle_client is not None:
            try:
                await self.defuddle_client.aclose()
            except RuntimeError:
                logger.debug("[Fetch] Dropping Defuddle client bound to a stale loop")
            self.defuddle_client = None
        self.defuddle_client_fingerprint = ""
        if self.fetch_cache is not None:
            self.fetch_cache.close()
            self.fetch_cache = None
        self.fetch_cache_fingerprint = ""
        if self.playwright_renderer is not None:
            await self.playwright_renderer.close()
            self.playwright_renderer = None
        self.playwright_renderer_fingerprint = ""
        self.jina_rate_limiter = None
        self.defuddle_rate_limiter = None

        # Reset state that may be bound to the current event loop
        self.cf_br_semaphore = None
        self.spa_domain_cache = None
        self.markitdown_instance = None
        self.detected_proxy = None
        self.detected_proxy_bypass = None

        # Reset heavy task semaphore (bound to event loop)
        from markitai.utils.executor import reset_heavy_task_semaphore

        reset_heavy_task_semaphore()

        # Close shared static HTTP clients
        from markitai.fetch_http import close_static_http_clients

        await close_static_http_clients()


_default_session = FetchSession()


def get_default_session() -> FetchSession:
    """Return the process-wide default FetchSession."""
    return _default_session


def reset_default_session() -> None:
    """Replace the default session with a fresh one (mainly for tests).

    Does not close the old session's resources; callers that need cleanup
    should ``await get_default_session().close()`` first.
    """
    global _default_session
    _default_session = FetchSession()


# Consent functions in markitai.fetch_consent operate on the default
# session's ConsentState (single source of truth). The lambda resolves the
# session at call time, so reset_default_session() is honored.
set_consent_state_provider(lambda: get_default_session().consent)
