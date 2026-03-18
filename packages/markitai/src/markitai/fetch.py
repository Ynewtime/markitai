"""URL fetch module for handling static and JS-rendered pages.

This module provides a unified interface for fetching web pages using different
strategies:
- defuddle: Free content extraction API (best content cleaning, no auth)
- jina: Jina Reader API (cloud-based, no local dependencies)
- static: Direct HTTP request via httpx/curl-cffi (fastest, no external deps)
- playwright: Headless browser via Playwright Python (JS-rendered pages)
- cloudflare: Cloudflare Browser Rendering API (cloud browser)
- auto: Policy engine orders strategies and falls back through them

Example usage:
    from markitai.fetch import fetch_url, FetchStrategy

    # Auto-detect strategy (defuddle → jina → static → playwright → cloudflare)
    result = await fetch_url("https://example.com", FetchStrategy.AUTO, config.fetch)

    # Force Defuddle
    result = await fetch_url("https://example.com", FetchStrategy.DEFUDDLE, config.fetch)
"""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger

from markitai.constants import (
    DEFAULT_DEFUDDLE_BASE_URL,
    DEFAULT_DEFUDDLE_RPM,
    DEFAULT_JINA_BASE_URL,
    DEFAULT_JINA_RPM,
    JS_REQUIRED_PATTERNS,
)
from markitai.fetch_http import get_static_http_client

try:
    from markitai.webextract import (
        coerce_source_frontmatter,
        extract_web_content,
        is_native_markdown_acceptable,
    )
except ImportError:  # pragma: no cover - optional during staged implementation
    extract_web_content = None  # type: ignore[assignment]
    coerce_source_frontmatter = None  # type: ignore[assignment]
    is_native_markdown_acceptable = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from markitai.config import (
        FetchConfig,
        FetchPolicyConfig,
        PlaywrightConfig,
        ScreenshotConfig,
    )


from markitai.fetch_cache import FetchCache as FetchCache  # noqa: F401
from markitai.fetch_cache import SPADomainCache as SPADomainCache  # noqa: F401
from markitai.fetch_cache import _make_json_safe as _make_json_safe  # noqa: F401
from markitai.fetch_types import (
    CRITICAL_INVALID_REASONS as CRITICAL_INVALID_REASONS,  # noqa: F401
)
from markitai.fetch_types import (
    ConditionalFetchResult as ConditionalFetchResult,  # noqa: F401
)
from markitai.fetch_types import FetchError as FetchError  # noqa: F401
from markitai.fetch_types import FetchResult as FetchResult  # noqa: F401
from markitai.fetch_types import FetchStrategy as FetchStrategy  # noqa: F401
from markitai.fetch_types import JinaAPIError as JinaAPIError  # noqa: F401
from markitai.fetch_types import JinaRateLimitError as JinaRateLimitError  # noqa: F401

# Global SPA domain cache instance (initialized lazily)
_spa_domain_cache: SPADomainCache | None = None

# CF Browser Rendering: Free plan allows 2 concurrent browser instances.
# Semaphore is lazily initialized to avoid binding to a wrong event loop at import time.
_cf_br_semaphore: asyncio.Semaphore | None = None


def get_cf_semaphore() -> asyncio.Semaphore:
    """Get or create the CF BR rate-limiting semaphore.

    Lazily initialized to avoid binding to a wrong event loop at import time.
    CF Free plan allows 2 concurrent browser instances.
    """
    global _cf_br_semaphore
    if _cf_br_semaphore is None:
        _cf_br_semaphore = asyncio.Semaphore(2)
    return _cf_br_semaphore


def _extract_markdown_title(content: str) -> str | None:
    """Extract the first H1 title from markdown content."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1) if match else None


def _get_playwright_advanced_kwargs(pw: PlaywrightConfig) -> dict[str, Any]:
    """Extract advanced Playwright kwargs from config, omitting None values."""
    return {
        k: v
        for k, v in {
            "wait_for_selector": pw.wait_for_selector,
            "cookies": pw.cookies,
            "reject_resource_patterns": pw.reject_resource_patterns,
            "extra_http_headers": pw.extra_http_headers,
            "user_agent": pw.user_agent,
            "http_credentials": pw.http_credentials,
        }.items()
        if v is not None
    }


def _get_playwright_fetch_kwargs(
    url: str,
    config: FetchConfig,
    screenshot_config: Any | None = None,
    output_dir: Any | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Resolve all Playwright fetch arguments including domain profile overrides."""
    profile_overrides = _resolve_playwright_profile_overrides(
        url, config.domain_profiles
    )

    kwargs = {
        "timeout": config.playwright.timeout,
        "wait_for": profile_overrides.get("wait_for", config.playwright.wait_for),
        "extra_wait_ms": profile_overrides.get(
            "extra_wait_ms", config.playwright.extra_wait_ms
        ),
        "proxy": _detect_proxy() if getattr(config, "auto_proxy", True) else None,
        "screenshot_config": screenshot_config,
        "output_dir": output_dir,
        "renderer": renderer,
    }

    # Session persistence
    if config.playwright.session_mode == "domain_persistent":
        kwargs["session_key"] = _url_to_session_key(url)
        kwargs["persist_context"] = True

    advanced_kwargs = _get_playwright_advanced_kwargs(config.playwright)
    kwargs.update(advanced_kwargs)
    kwargs.update(profile_overrides)

    return kwargs


def get_spa_domain_cache() -> SPADomainCache:
    """Get or create the global SPA domain cache instance.

    Returns:
        SPADomainCache instance
    """
    global _spa_domain_cache
    if _spa_domain_cache is None:
        _spa_domain_cache = SPADomainCache()
    return _spa_domain_cache


# Global fetch cache instance (initialized lazily)
_fetch_cache: FetchCache | None = None
_fetch_cache_fingerprint: str = ""


def get_fetch_cache(
    cache_dir: Path, max_size_bytes: int = 100 * 1024 * 1024
) -> FetchCache:
    """Get or create the global fetch cache instance.

    Rebuilds the cache when configuration (cache_dir or max_size_bytes)
    changes, using a fingerprint to detect config drift.

    Args:
        cache_dir: Directory to store cache database
        max_size_bytes: Maximum cache size

    Returns:
        FetchCache instance
    """
    global _fetch_cache, _fetch_cache_fingerprint
    fingerprint = f"{cache_dir}:{max_size_bytes}"
    if _fetch_cache is None or _fetch_cache_fingerprint != fingerprint:
        if _fetch_cache is not None:
            _fetch_cache.close()
            logger.debug(
                "[FetchCache] Rebuilding: config changed "
                f"(was {_fetch_cache_fingerprint!r}, now {fingerprint!r})"
            )
        db_path = cache_dir / "fetch_cache.db"
        _fetch_cache = FetchCache(db_path, max_size_bytes)
        _fetch_cache_fingerprint = fingerprint
    return _fetch_cache


# Global MarkItDown instance (reused for static fetching)
# Note: MarkItDown's requests.Session is NOT thread-safe. However, since
# fetch_with_static runs in the asyncio event loop (not in a thread pool),
# only one md.convert() call executes at a time, avoiding thread safety issues.
# If fetch_with_static is ever moved to run_in_executor with threads, this
# should be changed to use threading.local() for thread-local instances.
_markitdown_instance: Any = None

# Global httpx.AsyncClient for Jina fetching (reused to avoid connection overhead)
_jina_client: Any = None

# Global Playwright renderer (reused to avoid browser cold starts)
_playwright_renderer: Any = None

# Cached proxy URL (None = not checked, "" = no proxy, "http://..." = proxy URL)
_detected_proxy: str | None = None

# Common proxy ports used by popular proxy software
_COMMON_PROXY_PORTS = [
    7897,  # Clash Verge default
    7890,  # Clash default
    7891,  # Clash mixed
    1082,  # Shadowrocket
    10808,  # V2Ray default
    10809,  # V2Ray HTTP
    1080,  # SOCKS5 common
    8080,  # HTTP proxy common
    8118,  # Privoxy
    9050,  # Tor
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


# Cache for system proxy bypass list
_detected_proxy_bypass: str | None = None


def _detect_proxy(force_recheck: bool = False) -> str:
    """Detect proxy settings from environment, system config, or common local ports.

    Detection order:
    1. Environment variables: HTTPS_PROXY, HTTP_PROXY, ALL_PROXY
    2. System proxy settings (Windows registry / macOS scutil)
    3. Probe common proxy ports on localhost

    Args:
        force_recheck: Force re-detection even if cached

    Returns:
        Proxy URL string (e.g., "http://127.0.0.1:7890") or empty string if no proxy
    """
    global _detected_proxy, _detected_proxy_bypass

    if _detected_proxy is not None and not force_recheck:
        return _detected_proxy

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
            _detected_proxy = proxy
            # Also check NO_PROXY env var
            _detected_proxy_bypass = os.environ.get(
                "NO_PROXY", os.environ.get("no_proxy", "")
            )
            return proxy

    # Check system proxy settings (Windows/macOS)
    system_proxy, system_bypass = _get_system_proxy()
    if system_proxy:
        _detected_proxy = system_proxy
        _detected_proxy_bypass = system_bypass
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
                logger.info(f"[Proxy] Auto-detected local proxy at port {port}")
                _detected_proxy = proxy_url
                _detected_proxy_bypass = ""
                return proxy_url
        except Exception as e:
            logger.debug("[Proxy] Auto-detection failed: {}", e)

    # Silent - no proxy is common, no need to log
    _detected_proxy = ""
    _detected_proxy_bypass = ""
    return ""


def get_proxy_for_url(url: str, auto_proxy: bool = True) -> str:
    """Get proxy URL for a given URL, respecting auto_proxy setting and NO_PROXY.

    This is the unified entry point for proxy resolution. All fetch backends
    should use this instead of calling _detect_proxy() directly.

    Args:
        url: URL being fetched (checked against NO_PROXY patterns)
        auto_proxy: If False, always return empty string (proxy disabled)

    Returns:
        Proxy URL string or empty string if no proxy should be used
    """
    if not auto_proxy:
        return ""

    proxy = _detect_proxy()
    if not proxy:
        return ""

    # Check NO_PROXY bypass patterns
    bypass = _detected_proxy_bypass
    if bypass:
        from urllib.parse import urlparse

        from markitai.fetch_policy import match_local_only, parse_no_proxy

        domain = urlparse(url).netloc.lower()
        patterns = parse_no_proxy(bypass)
        if match_local_only(domain, patterns):
            return ""

    return proxy


def _get_markitdown() -> Any:
    """Get or create the shared MarkItDown instance.

    Reusing a single instance avoids repeated initialization overhead.
    Includes Accept header for CF Markdown for Agents content negotiation.
    """
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
        # Enable Cloudflare Markdown for Agents content negotiation.
        # CF-enabled sites return text/markdown directly (higher quality, fewer tokens).
        # Non-CF sites return text/html as usual — zero impact on existing behavior.
        #
        # Note: This patches the singleton's internal requests.Session headers.
        # Safe because _get_markitdown() is called once per process (guarded by
        # `if _markitdown_instance is None`), and the session is never internally
        # rebuilt by MarkItDown. If markitdown ever changes this, the test
        # `test_markitdown_instance_has_accept_markdown_header` will catch it.
        _markitdown_instance._requests_session.headers.update(
            {"Accept": "text/markdown, text/html;q=0.9, */*;q=0.5"}
        )
    return _markitdown_instance


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


# =============================================================================
# Defuddle: Free content extraction API (https://defuddle.md)
# Returns clean Markdown with YAML frontmatter from any URL.
#
# NOTE: Rate limit is undocumented — using same conservative limiter as Jina.
# NOTE: JS rendering capability is unconfirmed. SPA sites may need playwright.
# TODO: Migrate defuddle's core content extraction logic to markitai native.
#       Defuddle is open-source (https://github.com/kepano/defuddle) and its
#       HTML cleaning/article extraction could replace the external API dependency.
# =============================================================================

_defuddle_rate_limiter: _SlidingWindowRateLimiter | None = None
_defuddle_client: Any = None


def _get_defuddle_rate_limiter(rpm: int) -> _SlidingWindowRateLimiter:
    """Get or create the global Defuddle rate limiter."""
    global _defuddle_rate_limiter
    if _defuddle_rate_limiter is None or _defuddle_rate_limiter._rpm != rpm:
        _defuddle_rate_limiter = _SlidingWindowRateLimiter(rpm, name="Defuddle")
    return _defuddle_rate_limiter


def _get_defuddle_client(timeout: int = 30) -> Any:
    """Get or create the shared httpx.AsyncClient for Defuddle fetching."""
    global _defuddle_client
    if _defuddle_client is None:
        import httpx

        effective_proxy = _detect_proxy()
        client_kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(timeout, connect=10),
            "follow_redirects": True,
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        if effective_proxy:
            client_kwargs["proxy"] = effective_proxy
        _defuddle_client = httpx.AsyncClient(**client_kwargs)
    return _defuddle_client


async def fetch_with_defuddle(
    url: str,
    timeout: int = 30,
    rpm: int = DEFAULT_DEFUDDLE_RPM,
) -> FetchResult:
    """Fetch URL using Defuddle content extraction API.

    Defuddle extracts the main article content from web pages, removing clutter
    like ads, sidebars, headers, and footers. Returns clean Markdown with rich
    YAML frontmatter (title, author, published, description, word_count).

    API: GET https://defuddle.md/<url> → Markdown with YAML frontmatter

    NOTE: Rate limit is undocumented — we use a conservative default (20 RPM).
    NOTE: JS rendering capability of the API is unconfirmed. SPA-heavy sites
          may still need playwright as a fallback strategy.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        rpm: Requests per minute limit (conservative; actual limit undocumented)

    Returns:
        FetchResult with clean markdown content and extracted metadata

    Raises:
        FetchError: If fetch fails or returns empty content
    """
    limiter = _get_defuddle_rate_limiter(rpm)
    await limiter.acquire()

    import httpx

    logger.debug(f"[Defuddle] Fetching: {url}")

    from urllib.parse import quote

    defuddle_url = f"{DEFAULT_DEFUDDLE_BASE_URL}/{quote(url, safe='')}"

    try:
        client = _get_defuddle_client(timeout)
        response = await client.get(defuddle_url)

        if response.status_code == 429:
            raise FetchError(
                "Defuddle rate limit exceeded. "
                "Try again later or use --playwright for local rendering."
            )
        elif response.status_code >= 400:
            raise FetchError(
                f"Defuddle API returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        content = response.text
        if not content or not content.strip():
            raise FetchError(f"No content returned from Defuddle: {url}")

        # Parse YAML frontmatter for metadata extraction
        import yaml

        title: str | None = None
        metadata: dict[str, Any] = {"api": "defuddle"}
        source_frontmatter: dict[str, Any] = {}

        frontmatter_match = re.match(
            r"^\s*---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL
        )
        if frontmatter_match:
            try:
                fm_data = yaml.safe_load(frontmatter_match.group(1))
                if isinstance(fm_data, dict):
                    if fm_data.get("title"):
                        title = str(fm_data["title"])
                    # Preserve all source frontmatter fields for output
                    for key, value in fm_data.items():
                        if value is not None:
                            source_frontmatter[key] = value
            except yaml.YAMLError:
                logger.debug(
                    "[Defuddle] Failed to parse YAML frontmatter, using raw content"
                )
        if source_frontmatter and frontmatter_match:
            metadata["source_frontmatter"] = source_frontmatter

            # Strip frontmatter, keep only markdown body
            content = content[frontmatter_match.end() :].strip()

        if not content:
            raise FetchError(f"Defuddle returned empty content for: {url}")

        return FetchResult(
            content=content,
            strategy_used="defuddle",
            title=title,
            url=url,
            metadata=metadata,
        )

    except FetchError:
        raise
    except httpx.TimeoutException:
        raise FetchError(f"Defuddle request timed out after {timeout}s: {url}")
    except Exception as e:
        raise FetchError(f"Defuddle fetch failed: {e}")


_jina_rate_limiter: _SlidingWindowRateLimiter | None = None


def _get_jina_rate_limiter(rpm: int) -> _SlidingWindowRateLimiter:
    """Get or create the global Jina rate limiter."""
    global _jina_rate_limiter
    if _jina_rate_limiter is None or _jina_rate_limiter._rpm != rpm:
        _jina_rate_limiter = _SlidingWindowRateLimiter(rpm, name="Jina")
    return _jina_rate_limiter


_jina_client_fingerprint: str = ""


def _get_jina_client(timeout: int = 30, proxy: str = "") -> Any:
    """Get or create the shared httpx.AsyncClient for Jina fetching.

    Reusing a single client instance avoids repeated connection setup overhead.
    The client uses connection pooling for better performance.  Rebuilds when
    ``timeout`` or ``proxy`` change (config-fingerprint check).

    Args:
        timeout: Request timeout in seconds
        proxy: Proxy URL

    Returns:
        httpx.AsyncClient instance
    """
    global _jina_client, _jina_client_fingerprint
    fingerprint = f"{timeout}:{proxy}"
    if _jina_client is None or _jina_client_fingerprint != fingerprint:
        import httpx

        if _jina_client is not None:
            # Schedule close of old client (best-effort, non-blocking)
            logger.debug(
                "[Jina] Rebuilding client: config changed "
                f"(was {_jina_client_fingerprint!r}, now {fingerprint!r})"
            )
            _old = _jina_client
            try:
                asyncio.get_running_loop().create_task(_old.aclose())
            except RuntimeError:
                pass  # no event loop; old client will be GC'd

        # Use detected proxy if not explicitly provided
        effective_proxy = proxy or _detect_proxy()
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        }
        if effective_proxy:
            client_kwargs["proxy"] = effective_proxy
            logger.debug(f"[Jina] Using proxy: {effective_proxy}")

        _jina_client = httpx.AsyncClient(**client_kwargs)
        _jina_client_fingerprint = fingerprint
    return _jina_client


async def close_shared_clients() -> None:
    """Close shared client instances.

    Call this during cleanup to release resources.
    """
    global _jina_client, _fetch_cache, _playwright_renderer, _jina_rate_limiter
    global _defuddle_client, _defuddle_rate_limiter
    global _fetch_cache_fingerprint, _jina_client_fingerprint
    global _playwright_renderer_fingerprint
    global _cf_br_semaphore, _spa_domain_cache
    global _markitdown_instance, _detected_proxy, _detected_proxy_bypass
    if _jina_client is not None:
        await _jina_client.aclose()
        _jina_client = None
    _jina_client_fingerprint = ""
    if _defuddle_client is not None:
        await _defuddle_client.aclose()
        _defuddle_client = None
    if _fetch_cache is not None:
        _fetch_cache.close()
        _fetch_cache = None
    _fetch_cache_fingerprint = ""
    if _playwright_renderer is not None:
        await _playwright_renderer.close()
        _playwright_renderer = None
    _playwright_renderer_fingerprint = ""
    _jina_rate_limiter = None
    _defuddle_rate_limiter = None

    # Reset global state that may be bound to the current event loop
    _cf_br_semaphore = None
    _spa_domain_cache = None
    _markitdown_instance = None
    _detected_proxy = None
    _detected_proxy_bypass = None

    # Reset heavy task semaphore (bound to event loop)
    from markitai.utils.executor import reset_heavy_task_semaphore

    reset_heavy_task_semaphore()

    # Close shared static HTTP clients
    from markitai.fetch_http import close_static_http_clients

    await close_static_http_clients()


_playwright_renderer_fingerprint: str = ""


async def _get_playwright_renderer(
    proxy: str | None = None, config: FetchConfig | None = None
) -> Any:
    """Get or create the shared PlaywrightRenderer.

    Rebuilds when ``proxy`` or session-mode configuration changes.

    Args:
        proxy: Optional proxy URL
        config: Optional fetch configuration to enable session cache

    Returns:
        PlaywrightRenderer instance
    """
    global _playwright_renderer, _playwright_renderer_fingerprint
    session_mode = config.playwright.session_mode if config else None
    fingerprint = f"{proxy}:{session_mode}"
    if _playwright_renderer is None or _playwright_renderer_fingerprint != fingerprint:
        if _playwright_renderer is not None:
            logger.debug(
                "[Playwright] Rebuilding renderer: config changed "
                f"(was {_playwright_renderer_fingerprint!r}, now {fingerprint!r})"
            )
            await _playwright_renderer.close()

        from markitai.fetch_playwright import PlaywrightRenderer

        _playwright_renderer = PlaywrightRenderer(proxy=proxy)

        # Enable domain-persistent session cache if configured
        if config and config.playwright.session_mode == "domain_persistent":
            _playwright_renderer.enable_domain_session_cache(
                ttl_seconds=config.playwright.session_ttl_seconds,
                max_contexts=8,  # Default limit
            )

        _playwright_renderer_fingerprint = fingerprint

    return _playwright_renderer


def _url_to_session_key(url: str) -> str:
    """Extract session key (domain) from URL."""
    from urllib.parse import urlparse

    return urlparse(url).netloc.lower()


def _resolve_playwright_profile_overrides(
    url: str, domain_profiles: dict[str, Any]
) -> dict[str, Any]:
    """Resolve domain-specific Playwright overrides from config."""
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lower()
    profile = domain_profiles.get(domain)
    if not profile:
        return {}

    out: dict[str, Any] = {}
    if profile.wait_for_selector:
        out["wait_for_selector"] = profile.wait_for_selector
    if profile.wait_for:
        out["wait_for"] = profile.wait_for
    if profile.extra_wait_ms is not None:
        out["extra_wait_ms"] = profile.extra_wait_ms
    return out


def detect_js_required(content: str) -> bool:
    """Detect if content indicates JavaScript rendering is required.

    Note: This function receives MARKDOWN content (converted by markitdown),
    not raw HTML. Detection strategies must work with Markdown text.

    Uses multiple detection strategies:
    1. Simple string matching for common JS-required messages
    2. Content patterns that survive Markdown conversion
    3. Content length and quality checks

    Args:
        content: Markdown content to check (from markitdown conversion)

    Returns:
        True if content suggests JavaScript is needed
    """
    if not content:
        return True  # Empty content likely means JS-rendered

    content_lower = content.lower()

    # 1. Simple string matching for JS-required messages
    # These text patterns survive Markdown conversion
    for pattern in JS_REQUIRED_PATTERNS:
        if pattern.lower() in content_lower:
            logger.debug(f"JS required: string pattern matched '{pattern}'")
            return True

    # 2. Check for SPA/bot-protection text patterns
    # These patterns are more specific to avoid false positives
    spa_text_patterns = [
        # JS requirement messages (already covered by JS_REQUIRED_PATTERNS, but regex variants)
        r"this (?:page|site|website) requires javascript",
        r"you need (?:to enable )?javascript",
        r"enable javascript to (?:view|continue|access)",
        # Cloudflare/bot protection (only when it's the main content)
        r"^(?:\s*#?\s*)?(?:just a moment|one moment)\.{0,3}\s*$",
        r"checking (?:if the site connection is secure|your browser)",
        r"verifying (?:you are human|your browser)",
        r"ray id:",  # Cloudflare error pages include Ray ID
        # Common SPA loading states (only if very short content)
    ]
    for pattern in spa_text_patterns:
        if re.search(pattern, content_lower, re.MULTILINE):
            logger.debug(f"JS required: SPA text pattern matched '{pattern}'")
            return True

    # 3. Check for very short content (likely a JS-only page)
    # Strip markdown formatting for accurate length check
    text_only = re.sub(r"[#*_\[\]()>`\-|]", "", content)
    text_only = re.sub(r"!\[.*?\]\(.*?\)", "", text_only)  # Remove image refs
    text_only = re.sub(r"\[.*?\]\(.*?\)", "", text_only)  # Remove links
    text_only = " ".join(text_only.split()).strip()

    if len(text_only) < 100:
        logger.debug(f"JS required: content too short ({len(text_only)} chars)")
        return True

    # 4. Check for repetitive/placeholder content (SPA stub pages)
    # Some SPAs return minimal placeholder text
    unique_words = set(text_only.lower().split())
    if len(text_only) < 500 and len(unique_words) < 20:
        logger.debug(
            f"JS required: low content diversity "
            f"({len(unique_words)} unique words in {len(text_only)} chars)"
        )
        return True

    return False


def should_use_browser_for_domain(url: str, fallback_patterns: list[str]) -> bool:
    """Check if URL domain matches fallback patterns that need browser rendering.

    Args:
        url: URL to check
        fallback_patterns: List of domain patterns (e.g., ["twitter.com", "x.com"])

    Returns:
        True if domain matches any pattern
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for pattern in fallback_patterns:
            pattern_lower = pattern.lower()
            # Match exact domain or subdomain
            if domain == pattern_lower or domain.endswith("." + pattern_lower):
                logger.debug(f"Domain {domain} matches fallback pattern {pattern}")
                return True
    except Exception as e:
        logger.debug("[Fetch] Domain pattern matching failed: {}", e)

    return False


def _url_to_screenshot_filename(url: str) -> str:
    """Generate a safe filename for URL screenshot.

    Examples:
        https://example.com/path → example.com_path.full.jpg
        https://x.com/user/status/123 → x.com_user_status_123.full.jpg

    Args:
        url: URL to convert

    Returns:
        Safe filename with .full.jpg extension
    """
    try:
        parsed = urlparse(url)
        # Start with domain
        parts = [parsed.netloc] if parsed.netloc else []
        # Add path parts
        if parsed.path and parsed.path != "/":
            path_parts = parsed.path.strip("/").split("/")
            parts.extend(path_parts)

        # If no parts, fall back to hash
        if not parts or not any(parts):
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            return f"screenshot_{url_hash}.full.jpg"

        # Join with underscores
        name = "_".join(p for p in parts if p)

        # Sanitize for filesystem (remove/replace unsafe chars)
        # Windows-unsafe: < > : " / \ | ? *
        # Also remove other problematic chars
        unsafe_chars = r'<>:"/\\|?*\x00-\x1f'
        name = re.sub(f"[{unsafe_chars}]", "_", name)

        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)

        # Strip leading/trailing underscores
        name = name.strip("_")

        # Limit length (leave room for extension)
        max_length = 200
        if len(name) > max_length:
            name = name[:max_length]

        # Final check for empty name
        if not name:
            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
            return f"screenshot_{url_hash}.full.jpg"

        return f"{name}.full.jpg"
    except Exception:
        # Fallback: hash the URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"screenshot_{url_hash}.full.jpg"


def _compress_screenshot(
    screenshot_path: Path,
    quality: int = 85,
    max_height: int = 10000,
) -> None:
    """Compress a screenshot to JPEG with quality and size limits.

    Args:
        screenshot_path: Path to screenshot file (will be overwritten)
        quality: JPEG quality (1-100)
        max_height: Maximum height in pixels (will resize if exceeded)
    """
    try:
        from PIL import Image

        # Quick check: get image info without full decode
        with Image.open(screenshot_path) as img:
            width, height = img.size
            needs_resize = height > max_height
            needs_convert = img.mode in ("RGBA", "P")

        # Skip re-compression if image doesn't need resize or conversion
        # Playwright already saves JPEG with specified quality
        if not needs_resize and not needs_convert:
            logger.debug(
                f"Screenshot within limits ({width}x{height}), skipping re-compression"
            )
            return

        # Only re-process if needed
        with Image.open(screenshot_path) as img:
            if needs_convert:
                img = img.convert("RGB")

            if needs_resize:
                ratio = max_height / height
                new_width = int(width * ratio)
                img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
                logger.debug(
                    f"Resized screenshot from {width}x{height} to {new_width}x{max_height}"
                )

            img.save(screenshot_path, "JPEG", quality=quality, optimize=True)
            logger.debug(
                f"Compressed screenshot to quality={quality}: {screenshot_path}"
            )
    except ImportError:
        logger.warning("Pillow not installed, skipping screenshot compression")
    except Exception as e:
        logger.warning(f"Failed to compress screenshot: {e}")


async def fetch_with_static(url: str) -> FetchResult:
    """Fetch URL using the shared static pipeline.

    Args:
        url: URL to fetch

    Returns:
        FetchResult with markdown content

    Raises:
        FetchError: If fetch fails
    """
    logger.debug(f"Fetching URL with static strategy: {url}")
    cond_result = await fetch_with_static_conditional(url)
    if cond_result.result is None:
        raise FetchError(f"No content from conditional fetch: {url}")
    return cond_result.result


def _get_header_value(
    headers: Any, *candidates: str, default: str | None = None
) -> str | None:
    """Read a response header across case and separator variants."""
    getter = getattr(headers, "get", None)
    if callable(getter):
        for candidate in candidates:
            value = getter(candidate, None)
            if value is not None:
                return str(value)

    if isinstance(headers, dict):
        normalized = {str(key).lower(): value for key, value in headers.items()}
        for candidate in candidates:
            value = normalized.get(candidate.lower())
            if value is not None:
                return value

    return default


def _get_response_text(response: Any) -> str:
    """Decode a static HTTP response body into text."""
    content = getattr(response, "content", b"")
    if isinstance(content, bytes | bytearray):
        declared_encoding = getattr(response, "encoding", None)
        if not isinstance(declared_encoding, str) or not declared_encoding.strip():
            content_type = _get_header_value(
                getattr(response, "headers", {}),
                "content-type",
                "content_type",
            )
            if isinstance(content_type, str):
                match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
                if match:
                    declared_encoding = match.group(1).strip().strip("\"'")
                else:
                    declared_encoding = None

        if isinstance(declared_encoding, str) and declared_encoding.strip():
            try:
                codecs.lookup(declared_encoding)
                return bytes(content).decode(declared_encoding)
            except (LookupError, UnicodeDecodeError):
                logger.debug(
                    "Failed to decode response with declared charset "
                    f"{declared_encoding!r}, falling back"
                )

        for fallback_encoding in ("utf-8", "utf-8-sig"):
            try:
                return bytes(content).decode(fallback_encoding)
            except UnicodeDecodeError:
                continue

        return bytes(content).decode("utf-8", errors="replace")

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    return str(content)


def _build_native_fetch_result(
    *,
    html: str,
    url: str,
    final_url: str | None,
    strategy_used: str,
    base_metadata: dict[str, Any] | None = None,
) -> FetchResult | None:
    """Try native HTML extraction and return a FetchResult when acceptable."""
    if extract_web_content is None:
        return None
    # If extract_web_content is available, the other webextract functions are too
    assert is_native_markdown_acceptable is not None
    assert coerce_source_frontmatter is not None

    try:
        extracted = extract_web_content(html, final_url or url)
    except Exception as exc:
        logger.debug(f"Native webextract failed, falling back to markitdown: {exc}")
        return None

    markdown = getattr(extracted, "markdown", "")
    diagnostics = dict(getattr(extracted, "diagnostics", {}) or {})

    if not is_native_markdown_acceptable(markdown):
        diagnostics.setdefault("fallback_reason", "native_output_too_short")
        return None

    source_frontmatter = coerce_source_frontmatter(getattr(extracted, "metadata", None))
    merged_metadata = dict(base_metadata or {})
    merged_metadata.update(
        {
            "converter": "native-html",
            "webextract_diagnostics": diagnostics,
        }
    )
    if source_frontmatter:
        merged_metadata["source_frontmatter"] = source_frontmatter

    return FetchResult(
        content=markdown,
        strategy_used=strategy_used,
        title=source_frontmatter.get("title"),
        url=url,
        final_url=final_url,
        metadata=merged_metadata,
    )


def _merge_screenshot_result(result: FetchResult, pw_result: Any) -> FetchResult:
    """Attach a separately captured screenshot without dropping existing fields."""
    return FetchResult(
        content=result.content,
        strategy_used=result.strategy_used,
        title=result.title or getattr(pw_result, "title", None),
        url=result.url,
        final_url=result.final_url or getattr(pw_result, "final_url", None),
        metadata=result.metadata,
        cache_hit=result.cache_hit,
        screenshot_path=getattr(pw_result, "screenshot_path", None),
        static_content=result.static_content,
        browser_content=result.browser_content,
    )


async def fetch_with_static_conditional(
    url: str,
    cached_etag: str | None = None,
    cached_last_modified: str | None = None,
) -> ConditionalFetchResult:
    """Fetch URL with HTTP conditional request (single network roundtrip).

    Uses If-None-Match and If-Modified-Since headers for cache validation.
    If the server returns 304 Not Modified, the cached content should be used.

    Args:
        url: URL to fetch
        cached_etag: ETag from previous fetch (sent as If-None-Match)
        cached_last_modified: Last-Modified from previous fetch (sent as If-Modified-Since)

    Returns:
        ConditionalFetchResult with:
        - not_modified=True if 304 response (use cached content)
        - result with new content if 200 response
        - etag/last_modified for future conditional requests
    """
    logger.debug(
        f"[ConditionalFetch] URL: {url}, etag={cached_etag is not None}, "
        f"last_modified={cached_last_modified is not None}"
    )

    # Build conditional request headers
    # CF Markdown for Agents content negotiation
    headers: dict[str, str] = {
        "Accept": "text/markdown, text/html;q=0.9, */*;q=0.5",
    }
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified

    try:
        # Detect proxy
        proxy_url = _detect_proxy()
        client = get_static_http_client()
        logger.debug(f"Fetching URL with static {client.name} strategy: {url}")

        response = await client.get(
            url, headers=headers, timeout_s=30.0, proxy=proxy_url
        )

        # Extract response headers for future conditional requests
        response_etag = _get_header_value(response.headers, "etag", "ETag")
        response_last_modified = _get_header_value(
            response.headers,
            "last-modified",
            "Last-Modified",
            "last_modified",
            "Last_Modified",
        )

        # 304 Not Modified - use cached content
        if response.status_code == 304:
            return ConditionalFetchResult(
                result=None,
                not_modified=True,
                etag=response_etag or cached_etag,
                last_modified=response_last_modified or cached_last_modified,
            )

        # Non-2xx response (except 304)
        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code} fetching URL: {url}")

        # 200 OK (or other 2xx) - process new content
        logger.debug(
            f"[ConditionalFetch] {response.status_code} response, "
            f"content-length={len(response.content)}"
        )

        # Check if server returned markdown directly (CF Markdown for Agents)
        content_type_header = _get_header_value(
            response.headers,
            "content-type",
            "Content-Type",
            "content_type",
            default="",
        )
        if content_type_header and "text/markdown" in content_type_header:
            markdown_content = _get_response_text(response)
            token_hint = _get_header_value(
                response.headers,
                "x-markdown-tokens",
                "X-Markdown-Tokens",
            )
            logger.debug(
                f"[ConditionalFetch] Server returned markdown directly"
                f"{f' (~{token_hint} tokens)' if token_hint else ''}"
            )
            title = _extract_markdown_title(markdown_content)

            fetch_result = FetchResult(
                content=markdown_content,
                strategy_used="static",
                title=title,
                url=url,
                final_url=str(response.url),
                metadata={
                    "converter": "server-markdown",
                    "conditional": True,
                    "token_hint": int(token_hint) if token_hint else None,
                    "client": client.name,
                },
            )

            return ConditionalFetchResult(
                result=fetch_result,
                not_modified=False,
                etag=response_etag,
                last_modified=response_last_modified,
            )

        # Determine file extension from Content-Type or URL
        content_type = _get_header_value(
            response.headers,
            "content-type",
            "Content-Type",
            "content_type",
            default="",
        )
        response_text = _get_response_text(response)
        if content_type and "text/html" in content_type:
            native_result = _build_native_fetch_result(
                html=response_text,
                url=url,
                final_url=str(response.url),
                strategy_used="static",
                base_metadata={"conditional": True, "client": client.name},
            )
            if native_result is not None:
                return ConditionalFetchResult(
                    result=native_result,
                    not_modified=False,
                    etag=response_etag,
                    last_modified=response_last_modified,
                )

        if content_type and "text/html" in content_type:
            suffix = ".html"
        elif content_type and "application/pdf" in content_type:
            suffix = ".pdf"
        else:
            # Fallback to URL extension
            from urllib.parse import urlparse

            path = urlparse(url).path
            suffix = Path(path).suffix or ".html"

        # Save response to temp file and run sync markitdown in executor
        # to avoid blocking the event loop
        def _sync_convert(content_bytes: bytes, suffix: str) -> tuple[str, str | None]:
            """Write temp file and convert with markitdown (CPU-bound)."""
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, mode="wb"
            ) as f:
                f.write(content_bytes)
                temp_path = Path(f.name)
            try:
                md = _get_markitdown()
                md_result = md.convert(str(temp_path))
                return md_result.text_content or "", md_result.title
            finally:
                temp_path.unlink(missing_ok=True)

        loop = asyncio.get_running_loop()
        text_content, title = await loop.run_in_executor(
            None, _sync_convert, response.content, suffix
        )

        if not text_content:
            raise FetchError(f"No content extracted from URL: {url}")

        fetch_result = FetchResult(
            content=text_content,
            strategy_used="static",
            title=title,
            url=url,
            final_url=str(response.url),
            metadata={"converter": "markitdown", "conditional": True},
        )

        return ConditionalFetchResult(
            result=fetch_result,
            not_modified=False,
            etag=response_etag,
            last_modified=response_last_modified,
        )

    except Exception as e:
        if isinstance(e, FetchError):
            raise
        raise FetchError(f"Failed to fetch URL with conditional request: {e}")


async def fetch_with_cloudflare(
    url: str,
    api_token: str | None = None,
    account_id: str | None = None,
    timeout: int = 30000,
    wait_until: str = "networkidle0",
    cache_ttl: int = 0,
    reject_resource_patterns: list[str] | None = None,
    user_agent: str | None = None,
    cookies: list[dict[str, str]] | None = None,
    wait_for_selector: str | None = None,
    http_credentials: dict[str, str] | None = None,
) -> FetchResult:
    """Fetch URL using Cloudflare Browser Rendering /markdown API.

    Args:
        url: URL to fetch
        api_token: CF API token
        account_id: CF account ID
        timeout: Timeout in milliseconds
        wait_until: Wait event (load, domcontentloaded, networkidle0)
        cache_ttl: Cache TTL in seconds (0 = no cache)
        reject_resource_patterns: JS-style regex patterns to block
        user_agent: Custom User-Agent string
        cookies: Cookies to set before navigation
        wait_for_selector: CSS selector to wait for after page load
        http_credentials: HTTP Basic Auth credentials {username, password}

    Returns:
        FetchResult with markdown content

    Raises:
        FetchError: If fetch fails or credentials missing
    """
    import httpx

    if not api_token or not account_id:
        raise FetchError(
            "Cloudflare API token and account ID required. "
            "Set in config: fetch.cloudflare.api_token and fetch.cloudflare.account_id"
        )

    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/browser-rendering/markdown"
    )
    # cacheTTL is a query parameter, not a body parameter
    if cache_ttl > 0:
        endpoint += f"?cacheTTL={cache_ttl}"

    payload: dict[str, Any] = {
        "url": url,
    }
    # timeout and waitUntil go inside gotoOptions, not at top level
    goto_options: dict[str, Any] = {}
    if timeout:
        goto_options["timeout"] = timeout
    if wait_until:
        goto_options["waitUntil"] = wait_until
    if goto_options:
        payload["gotoOptions"] = goto_options

    # Resource filtering: use caller's patterns, or sensible defaults.
    # CF BR uses JS-style regex string literals with leading/trailing slashes
    # (e.g. "/\.css$/"), consistent with Puppeteer's page.route() patterns.
    if reject_resource_patterns is not None:
        payload["rejectRequestPattern"] = reject_resource_patterns
    else:
        # Default: skip fonts and stylesheets for faster rendering
        payload["rejectRequestPattern"] = [
            "/\\.css$/",
            "/\\.woff2?$/",
            "/\\.ttf$/",
            "/\\.eot$/",
            "/\\.otf$/",
        ]

    if user_agent:
        payload["userAgent"] = user_agent
    if cookies:
        payload["cookies"] = cookies
    if wait_for_selector:
        payload["waitForSelector"] = {"selector": wait_for_selector}
    if http_credentials:
        payload["authenticate"] = http_credentials

    logger.debug(f"Fetching URL with CF Browser Rendering: {url}")

    # CF Free plan: 2 concurrent browser instances.
    # Serialize requests to avoid 429 rate-limit errors.
    max_retries = 3
    retry_base_delay = 2.0  # seconds

    try:
        # Use _detect_proxy() not get_proxy_for_url(endpoint), because
        # api.cloudflare.com is not in proxy_domains but may still need
        # proxy in restricted network environments.
        proxy_url = _detect_proxy()
        proxy_config = proxy_url if proxy_url else None

        async with get_cf_semaphore():
            async with httpx.AsyncClient(
                timeout=max(timeout / 1000 + 10, 60.0),
                proxy=proxy_config,
            ) as client:
                for attempt in range(max_retries):
                    response = await client.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {api_token}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )

                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            delay = retry_base_delay * (2**attempt)
                            logger.warning(
                                f"CF BR rate limited (429), retrying in {delay}s "
                                f"(attempt {attempt + 1}/{max_retries}): {url}"
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise FetchError(
                            f"CF BR rate limit exceeded after {max_retries} retries: {url}"
                        )

                    response.raise_for_status()

                    # CF REST API returns JSON envelope
                    data = response.json()
                    if not data.get("success"):
                        errors = data.get("errors", [])
                        error_msg = "; ".join(e.get("message", str(e)) for e in errors)
                        raise FetchError(f"CF BR API error: {error_msg}")

                    markdown_content = data.get("result", "")

                    title = _extract_markdown_title(markdown_content)

                    return FetchResult(
                        content=markdown_content,
                        strategy_used="cloudflare",
                        title=title,
                        url=url,
                        final_url=url,
                        metadata={
                            "converter": "cloudflare-br",
                            "browser_ms_used": response.headers.get(
                                "X-Browser-Ms-Used"
                            ),
                        },
                    )
            # Unreachable: loop always returns or raises on 429 exhaustion
            raise FetchError(f"CF BR fetch failed after {max_retries} attempts: {url}")
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"Cloudflare BR fetch failed: {e}") from e


async def fetch_with_jina(
    url: str,
    api_key: str | None = None,
    timeout: int = 30,
    rpm: int = DEFAULT_JINA_RPM,
    *,
    no_cache: bool = False,
    target_selector: str | None = None,
    wait_for_selector: str | None = None,
) -> FetchResult:
    """Fetch URL using Jina Reader API with JSON mode.

    Uses JSON mode for reliable structured data extraction (title, content).

    Args:
        url: URL to fetch
        api_key: Optional Jina API key (for higher rate limits)
        timeout: Request timeout in seconds
        rpm: Requests per minute limit
        no_cache: Skip Jina server-side cache
        target_selector: CSS selector for content extraction
        wait_for_selector: Wait for element before extraction

    Returns:
        FetchResult with markdown content and extracted title

    Raises:
        JinaRateLimitError: If rate limit exceeded
        JinaAPIError: If API returns error
        FetchError: If fetch fails
    """
    limiter = _get_jina_rate_limiter(rpm)
    await limiter.acquire()

    import httpx

    logger.debug(f"Fetching URL with Jina Reader (JSON mode): {url}")

    jina_url = f"{DEFAULT_JINA_BASE_URL}/{url}"
    headers = {
        "Accept": "application/json",  # Use JSON mode for structured response
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if no_cache:
        headers["X-No-Cache"] = "true"
    if target_selector:
        headers["X-Target-Selector"] = target_selector
    if wait_for_selector:
        headers["X-Wait-For-Selector"] = wait_for_selector

    try:
        client = _get_jina_client(timeout)
        response = await client.get(jina_url, headers=headers)

        if response.status_code == 429:
            raise JinaRateLimitError()
        elif response.status_code >= 400:
            raise JinaAPIError(response.status_code, response.text[:200])

        # Parse JSON response
        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"Jina API returned invalid JSON: {e}")
            raise FetchError(f"Jina Reader returned invalid JSON response: {e}")

        # Extract data from JSON structure
        # Expected format: {"code": 200, "status": 20000, "data": {...}}
        if not isinstance(json_data, dict):
            raise FetchError("Jina Reader returned unexpected response format")

        data = json_data.get("data")
        if not data or not isinstance(data, dict):
            # Check if the response indicates an error
            error_msg = json_data.get("message") or json_data.get("error")
            if error_msg:
                raise JinaAPIError(
                    json_data.get("code", 500),
                    str(error_msg)[:200],
                )
            raise FetchError("Jina Reader returned empty or invalid data structure")

        # Extract title and content from data
        title = data.get("title")
        content = data.get("content", "")

        if not content or not content.strip():
            raise FetchError(f"No content returned from Jina Reader: {url}")

        # Clean up title if present
        if title:
            title = title.strip()
            if not title:
                title = None

        return FetchResult(
            content=content,
            strategy_used="jina",
            title=title,
            url=url,
            metadata={
                "api": "jina-reader",
                "mode": "json",
                "source_url": data.get("url", url),
            },
        )

    except (JinaRateLimitError, JinaAPIError):
        raise
    except httpx.TimeoutException:
        raise FetchError(f"Jina Reader request timed out after {timeout}s: {url}")
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(f"Jina Reader fetch failed: {e}")


async def fetch_url(
    url: str,
    strategy: FetchStrategy,
    config: FetchConfig,
    explicit_strategy: bool = False,
    cache: FetchCache | None = None,
    skip_read_cache: bool = False,
    *,
    screenshot: bool = False,
    screenshot_dir: Path | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    renderer: Any | None = None,
) -> FetchResult:
    """Fetch URL content using the specified strategy.

    Args:
        url: URL to fetch
        strategy: Fetch strategy to use
        config: Fetch configuration
        explicit_strategy: If True, don't fallback on error (user explicitly chose strategy)
        cache: Optional FetchCache for caching results
        skip_read_cache: If True, skip reading from cache but still write results (--no-cache)
        screenshot: If True, capture full-page screenshot (requires browser strategy)
        screenshot_dir: Directory to save screenshot
        screenshot_config: Screenshot settings (viewport, quality, etc.)
        renderer: Optional shared PlaywrightRenderer

    Returns:
        FetchResult with content and metadata

    Raises:
        FetchError: If fetch fails and no fallback available
        JinaRateLimitError: If --jina used and rate limit exceeded
    """
    from urllib.parse import urlparse

    from markitai.fetch_policy import is_private_or_local_domain

    def _ensure_external_strategy_allowed(strategy_name: str) -> None:
        if strategy_name not in {
            FetchStrategy.DEFUDDLE.value,
            FetchStrategy.JINA.value,
            FetchStrategy.CLOUDFLARE.value,
        }:
            return

        domain = urlparse(url).netloc.lower()
        if is_private_or_local_domain(domain):
            raise FetchError(
                f"{strategy_name} cannot fetch private/local URLs. "
                "Use static or playwright instead."
            )

    # Use provided renderer or get global one if needed
    _renderer = renderer
    if _renderer is None and (
        strategy == FetchStrategy.PLAYWRIGHT
        or (strategy == FetchStrategy.AUTO and screenshot)
        or screenshot
    ):
        # Only initialize global renderer if browser strategy is likely to be used
        proxy = _detect_proxy() if getattr(config, "auto_proxy", True) else None
        _renderer = await _get_playwright_renderer(proxy=proxy, config=config)

    # Screenshot kwargs for browser fetching (used by _fetch_with_fallback)
    screenshot_kwargs: dict[str, Any] = {
        "renderer": _renderer,
        "screenshot_config": screenshot_config,
        "screenshot_dir": screenshot_dir,
    }

    result: FetchResult | None = None
    cache_validators_to_write: tuple[str | None, str | None] | None = None

    # Include strategy in cache key when an explicit strategy is requested,
    # so that --playwright and --static don't return each other's cached results.
    cache_strategy: str | None = (
        strategy.value if explicit_strategy and strategy != FetchStrategy.AUTO else None
    )

    # For static strategy with cache, try HTTP conditional request for efficiency
    # This uses ETag/Last-Modified headers to avoid re-downloading unchanged content
    use_conditional_cache = (
        cache is not None
        and not skip_read_cache
        and strategy in (FetchStrategy.STATIC, FetchStrategy.AUTO)
    )

    if use_conditional_cache:
        # Type narrowing: cache is guaranteed non-None here due to condition above
        assert cache is not None
        # Get cached result with HTTP validators
        (
            cached_result,
            cached_etag,
            cached_last_modified,
        ) = await cache.aget_with_validators(url, strategy=cache_strategy)

        # If we have validators, try conditional fetch (static strategy only)
        if cached_result is not None and (cached_etag or cached_last_modified):
            # Only use conditional fetch for static strategy
            # AUTO strategy might need browser fallback, so skip conditional optimization
            if strategy == FetchStrategy.STATIC or (
                strategy == FetchStrategy.AUTO
                and not should_use_browser_for_domain(url, config.fallback_patterns)
                and not get_spa_domain_cache().is_known_spa(url)
            ):
                try:
                    cond_result = await fetch_with_static_conditional(
                        url, cached_etag, cached_last_modified
                    )
                    if cond_result.not_modified:
                        # 304 Not Modified - use cached content
                        await cache.aupdate_accessed_at(url, strategy=cache_strategy)
                        result = cached_result
                    elif cond_result.result is not None:
                        result = cond_result.result
                        cache_validators_to_write = (
                            cond_result.etag,
                            cond_result.last_modified,
                        )
                except FetchError:
                    # Conditional fetch failed, fall through to normal flow
                    logger.debug(
                        f"[ConditionalFetch] Failed, falling back to normal fetch: {url}"
                    )

        # No validators but have cached result - use it directly
        elif cached_result is not None and result is None:
            result = cached_result

    # Traditional cache check for non-conditional strategies
    elif cache is not None and not skip_read_cache:
        cached_result = await cache.aget(url, strategy=cache_strategy)
        if cached_result is not None and result is None:
            result = cached_result

    # Fetch the content
    if result is None and explicit_strategy:
        if strategy == FetchStrategy.PLAYWRIGHT:
            from markitai.fetch_playwright import (
                fetch_with_playwright,
                is_playwright_available,
            )

            if not is_playwright_available():
                raise FetchError(
                    "playwright is not installed. "
                    "Install with: uv add playwright && uv run playwright install chromium "
                    "(Linux: also run 'uv run playwright install-deps chromium')"
                )

            pw_result = await fetch_with_playwright(
                url,
                **_get_playwright_fetch_kwargs(
                    url,
                    config,
                    screenshot_config=screenshot_config,
                    output_dir=screenshot_dir,
                    renderer=_renderer,
                ),
            )

            result = FetchResult(
                content=pw_result.content,
                strategy_used="playwright",
                title=pw_result.title,
                url=url,
                final_url=pw_result.final_url,
                metadata=pw_result.metadata,
                screenshot_path=pw_result.screenshot_path,
            )
        elif strategy == FetchStrategy.CLOUDFLARE:
            _ensure_external_strategy_allowed(FetchStrategy.CLOUDFLARE.value)
            cf = config.cloudflare
            token = cf.get_resolved_api_token(strict=True)
            acct = cf.get_resolved_account_id(strict=True)
            result = await fetch_with_cloudflare(
                url=url,
                api_token=token,
                account_id=acct,
                timeout=cf.timeout,
                wait_until=cf.wait_until,
                cache_ttl=cf.cache_ttl,
                reject_resource_patterns=cf.reject_resource_patterns,
                user_agent=cf.user_agent,
                cookies=cf.cookies,
                wait_for_selector=cf.wait_for_selector,
                http_credentials=cf.http_credentials,
            )
        elif strategy == FetchStrategy.JINA:
            _ensure_external_strategy_allowed(FetchStrategy.JINA.value)
            api_key = config.jina.get_resolved_api_key()
            result = await fetch_with_jina(
                url,
                api_key,
                config.jina.timeout,
                config.jina.rpm,
                no_cache=config.jina.no_cache,
                target_selector=config.jina.target_selector,
                wait_for_selector=config.jina.wait_for_selector,
            )
        elif strategy == FetchStrategy.DEFUDDLE:
            _ensure_external_strategy_allowed(FetchStrategy.DEFUDDLE.value)
            result = await fetch_with_defuddle(
                url,
                config.defuddle.timeout,
                config.defuddle.rpm,
            )
        elif strategy == FetchStrategy.STATIC:
            # For fresh fetch, use conditional to capture validators
            cond_result = await fetch_with_static_conditional(url)
            if cond_result.result is None:
                raise FetchError(f"No content from conditional fetch: {url}")
            result = cond_result.result
            cache_validators_to_write = (
                cond_result.etag,
                cond_result.last_modified,
            )
        else:
            # AUTO with explicit=True shouldn't happen, but handle it
            strategy = FetchStrategy.AUTO
            result = await _fetch_with_fallback(
                url, config, start_with_browser=False, **screenshot_kwargs
            )
    elif result is None and strategy == FetchStrategy.AUTO:
        # Check if domain needs browser rendering
        # Priority: 1. Configured fallback_patterns, 2. Learned SPA domains
        spa_cache = get_spa_domain_cache()
        use_browser_first = False

        if should_use_browser_for_domain(url, config.fallback_patterns):
            use_browser_first = True
        elif spa_cache.is_known_spa(url):
            spa_cache.record_hit(url)
            use_browser_first = True

        result = await _fetch_with_fallback(
            url, config, start_with_browser=use_browser_first, **screenshot_kwargs
        )
    elif result is None and strategy == FetchStrategy.STATIC:
        # For fresh fetch, use conditional to capture validators
        cond_result = await fetch_with_static_conditional(url)
        if cond_result.result is None:
            raise FetchError(f"No content from conditional fetch: {url}")
        result = cond_result.result
        cache_validators_to_write = (
            cond_result.etag,
            cond_result.last_modified,
        )
    elif result is None and strategy == FetchStrategy.PLAYWRIGHT:
        from markitai.fetch_playwright import (
            fetch_with_playwright,
            is_playwright_available,
        )

        if not is_playwright_available():
            raise FetchError(
                "playwright is not installed. "
                "Install with: uv add playwright && uv run playwright install chromium "
                "(Linux: also run 'uv run playwright install-deps chromium')"
            )

        pw_result = await fetch_with_playwright(
            url,
            **_get_playwright_fetch_kwargs(
                url,
                config,
                screenshot_config=screenshot_config,
                output_dir=screenshot_dir,
                renderer=_renderer,
            ),
        )

        result = FetchResult(
            content=pw_result.content,
            strategy_used="playwright",
            title=pw_result.title,
            url=url,
            final_url=pw_result.final_url,
            metadata=pw_result.metadata,
            screenshot_path=pw_result.screenshot_path,
        )
    elif result is None and strategy == FetchStrategy.CLOUDFLARE:
        _ensure_external_strategy_allowed(FetchStrategy.CLOUDFLARE.value)
        cf = config.cloudflare
        token = cf.get_resolved_api_token(strict=True)
        acct = cf.get_resolved_account_id(strict=True)
        result = await fetch_with_cloudflare(
            url=url,
            api_token=token,
            account_id=acct,
            timeout=cf.timeout,
            wait_until=cf.wait_until,
            cache_ttl=cf.cache_ttl,
            reject_resource_patterns=cf.reject_resource_patterns,
            user_agent=cf.user_agent,
            cookies=cf.cookies,
            wait_for_selector=cf.wait_for_selector,
            http_credentials=cf.http_credentials,
        )
    elif result is None and strategy == FetchStrategy.JINA:
        _ensure_external_strategy_allowed(FetchStrategy.JINA.value)
        api_key = config.jina.get_resolved_api_key()
        result = await fetch_with_jina(
            url,
            api_key,
            config.jina.timeout,
            config.jina.rpm,
            no_cache=config.jina.no_cache,
            target_selector=config.jina.target_selector,
            wait_for_selector=config.jina.wait_for_selector,
        )
    elif result is None and strategy == FetchStrategy.DEFUDDLE:
        _ensure_external_strategy_allowed(FetchStrategy.DEFUDDLE.value)
        result = await fetch_with_defuddle(
            url,
            config.defuddle.timeout,
            config.defuddle.rpm,
        )
    elif result is None:
        raise ValueError(f"Unknown fetch strategy: {strategy}")

    assert result is not None

    # Capture screenshot separately if requested and not already captured
    if screenshot and result.screenshot_path is None:
        try:
            from markitai.fetch_playwright import (
                fetch_with_playwright,
                is_playwright_available,
            )

            if is_playwright_available():
                logger.debug("[URL] Capturing screenshot separately via playwright")
                pw_result = await fetch_with_playwright(
                    url,
                    **_get_playwright_fetch_kwargs(
                        url,
                        config,
                        screenshot_config=screenshot_config,
                        output_dir=screenshot_dir,
                        renderer=_renderer,
                    ),
                )
                result = _merge_screenshot_result(result, pw_result)
            else:
                logger.debug("[URL] Screenshot requested but playwright not available")
        except Exception as e:
            logger.warning(f"[URL] Screenshot capture failed: {e}")

    # Cache the result (for non-static strategies that don't use conditional caching)
    if cache is not None and not result.cache_hit:
        if cache_validators_to_write is not None:
            await cache.aset_with_validators(
                url,
                result,
                cache_validators_to_write[0],
                cache_validators_to_write[1],
                strategy=cache_strategy,
            )
        else:
            await cache.aset(url, result, strategy=cache_strategy)

    return result


def _is_invalid_content(content: str) -> tuple[bool, str]:
    """Check if fetched content is invalid (JS error page, login prompt, etc.).

    Args:
        content: Fetched content to check

    Returns:
        Tuple of (is_invalid, reason)
    """
    if not content or not content.strip():
        return True, "empty"

    # Check for common invalid content patterns
    invalid_patterns = [
        (r"JavaScript is (not available|disabled)", "javascript_disabled"),
        (r"Please enable JavaScript", "javascript_required"),
        (r"switch to a supported browser", "unsupported_browser"),
        (r"Something went wrong.*let's give it another shot", "error_page"),
        (r"Log in.*Sign up.*to continue", "login_required"),
        (r"You must be logged in", "login_required"),
    ]

    for pattern, reason in invalid_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True, reason

    # Check content length (after removing markdown links and images)
    clean_content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)  # Remove images
    clean_content = re.sub(r"\[[^\]]*\]\([^)]+\)", "", clean_content)  # Remove links
    clean_content = re.sub(
        r"[#\-*_>\[\]`|]", "", clean_content
    )  # Remove markdown syntax
    clean_content = " ".join(clean_content.split())  # Normalize whitespace

    if len(clean_content) < 30:
        return True, "too_short"

    return False, ""


def _build_local_only_patterns(policy: FetchPolicyConfig) -> list[str]:
    """Build effective local-only patterns from config + NO_PROXY env var.

    When ``inherit_no_proxy`` is True (default), patterns from the NO_PROXY
    environment variable are merged into the configured ``local_only_patterns``
    (deduplicated, config patterns take precedence).
    """
    import os

    from markitai.fetch_policy import parse_no_proxy

    patterns = list(policy.local_only_patterns)
    if policy.inherit_no_proxy:
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if no_proxy:
            inherited = parse_no_proxy(no_proxy)
            seen = set(patterns)
            for p in inherited:
                if p not in seen:
                    patterns.append(p)
                    seen.add(p)
    return patterns


async def _fetch_with_fallback(
    url: str,
    config: FetchConfig,
    start_with_browser: bool = False,
    renderer: Any | None = None,
    **screenshot_kwargs: Any,
) -> FetchResult:
    """Fetch URL with automatic fallback between strategies.

    Args:
        url: URL to fetch
        config: Fetch configuration
        start_with_browser: If True, try browser first (for known JS domains)
        renderer: Optional shared PlaywrightRenderer
        **screenshot_kwargs: Screenshot options (screenshot, screenshot_dir, screenshot_config)

    Returns:
        FetchResult from first successful strategy
    """
    from urllib.parse import urlparse

    from markitai.fetch_policy import (
        FetchPolicyEngine,
        is_private_or_local_domain,
    )

    errors = []

    domain = urlparse(url).netloc.lower()
    engine = FetchPolicyEngine()
    jina_key = config.jina.get_resolved_api_key()
    profile = config.domain_profiles.get(domain)
    domain_prefer = profile.prefer_strategy if profile else None
    domain_priority = profile.strategy_priority if profile else None

    # Build effective local_only_patterns (config + optional NO_PROXY merge)
    effective_local_only = _build_local_only_patterns(config.policy)

    decision = engine.decide(
        domain=domain,
        known_spa=start_with_browser,
        explicit_strategy=config.strategy if config.strategy != "auto" else None,
        fallback_patterns=config.fallback_patterns,
        policy_enabled=config.policy.enabled,
        has_jina_key=bool(jina_key),
        domain_prefer_strategy=domain_prefer,
        global_strategy_priority=config.policy.strategy_priority,
        domain_strategy_priority=domain_priority,
        local_only_patterns=effective_local_only,
    )
    strategies = decision.order[: config.policy.max_strategy_hops]
    if is_private_or_local_domain(domain):
        strategies = [s for s in strategies if s in {"static", "playwright"}]

    # Resolve domain profile for telemetry
    domain_profile_applied = decision.reason == "spa_or_pattern"

    for strat in strategies:
        try:
            if strat == "static":
                result = await fetch_with_static(url)
                # Check if JS is required
                if detect_js_required(result.content):
                    # Learn this domain for future requests
                    spa_cache = get_spa_domain_cache()
                    spa_cache.record_spa_domain(url)
                    continue
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(result.content)
                if is_invalid:
                    logger.debug(f"Strategy {strat} returned invalid content: {reason}")
                    errors.append(f"{strat}: invalid content ({reason})")
                    continue

                # Add telemetry
                result.metadata.update(
                    {
                        "policy_reason": decision.reason,
                        "policy_order": strategies,
                        "profile_applied": domain_profile_applied,
                    }
                )
                return result

            elif strat == "defuddle":
                result = await fetch_with_defuddle(
                    url,
                    config.defuddle.timeout,
                    config.defuddle.rpm,
                )
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(result.content)
                if is_invalid:
                    logger.debug(f"Strategy {strat} returned invalid content: {reason}")
                    errors.append(f"{strat}: invalid content ({reason})")
                    continue

                # Add telemetry
                result.metadata.update(
                    {
                        "policy_reason": decision.reason,
                        "policy_order": strategies,
                        "profile_applied": domain_profile_applied,
                    }
                )
                return result

            elif strat == "playwright":
                from markitai.fetch_playwright import (
                    fetch_with_playwright,
                    is_playwright_available,
                )

                if not is_playwright_available():
                    logger.debug("playwright not available, trying next strategy")
                    continue

                pw_result = await fetch_with_playwright(
                    url,
                    **_get_playwright_fetch_kwargs(
                        url,
                        config,
                        screenshot_config=screenshot_kwargs.get("screenshot_config"),
                        output_dir=screenshot_kwargs.get("screenshot_dir"),
                        renderer=renderer,
                    ),
                )

                result = FetchResult(
                    content=pw_result.content,
                    strategy_used="playwright",
                    title=pw_result.title,
                    url=url,
                    final_url=pw_result.final_url,
                    metadata=pw_result.metadata,
                    screenshot_path=pw_result.screenshot_path,
                )
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(result.content)
                if is_invalid:
                    logger.debug(f"Strategy {strat} returned invalid content: {reason}")
                    errors.append(f"{strat}: invalid content ({reason})")
                    continue

                # Add telemetry
                result.metadata.update(
                    {
                        "policy_reason": decision.reason,
                        "policy_order": strategies,
                        "profile_applied": domain_profile_applied,
                    }
                )
                return result

            elif strat == "cloudflare":
                cf = getattr(config, "cloudflare", None)
                if cf is None:
                    continue
                token = (
                    cf.get_resolved_api_token()
                    if hasattr(cf, "get_resolved_api_token")
                    else cf.api_token
                )
                acct = (
                    cf.get_resolved_account_id()
                    if hasattr(cf, "get_resolved_account_id")
                    else cf.account_id
                )
                if not token or not acct:
                    logger.debug("Cloudflare credentials not configured, skipping")
                    continue
                result = await fetch_with_cloudflare(
                    url=url,
                    api_token=token,
                    account_id=acct,
                    timeout=cf.timeout,
                    wait_until=cf.wait_until,
                    cache_ttl=cf.cache_ttl,
                    reject_resource_patterns=cf.reject_resource_patterns,
                    user_agent=cf.user_agent,
                    cookies=cf.cookies,
                    wait_for_selector=cf.wait_for_selector,
                    http_credentials=cf.http_credentials,
                )
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(result.content)
                if is_invalid:
                    logger.debug(f"Strategy {strat} returned invalid content: {reason}")
                    errors.append(f"{strat}: invalid content ({reason})")
                    continue

                # Add telemetry
                result.metadata.update(
                    {
                        "policy_reason": decision.reason,
                        "policy_order": strategies,
                        "profile_applied": domain_profile_applied,
                    }
                )
                return result

            elif strat == "jina":
                api_key = config.jina.get_resolved_api_key()
                result = await fetch_with_jina(
                    url,
                    api_key,
                    config.jina.timeout,
                    config.jina.rpm,
                    no_cache=config.jina.no_cache,
                    target_selector=config.jina.target_selector,
                    wait_for_selector=config.jina.wait_for_selector,
                )
                # Validate content quality before accepting
                is_invalid, reason = _is_invalid_content(result.content)
                if is_invalid:
                    logger.debug(f"Strategy {strat} returned invalid content: {reason}")
                    errors.append(f"{strat}: invalid content ({reason})")
                    continue

                # Add telemetry
                result.metadata.update(
                    {
                        "policy_reason": decision.reason,
                        "policy_order": strategies,
                        "profile_applied": domain_profile_applied,
                    }
                )
                return result

        except JinaRateLimitError as e:
            errors.append(str(e))
            logger.warning(str(e))
            continue
        except FetchError as e:
            errors.append(f"{strat}: {e}")
            logger.debug(f"Strategy {strat} failed: {e}")
            continue
        except Exception as e:
            errors.append(f"{strat}: {e}")
            logger.debug(f"Strategy {strat} failed: {e}")
            continue

    # All strategies failed
    raise FetchError(
        f"All fetch strategies failed for {url}:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )
