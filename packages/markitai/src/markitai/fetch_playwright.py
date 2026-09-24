"""Playwright-based URL fetch backend.

This module provides browser automation using Playwright Python as an
alternative to agent-browser, eliminating the Node.js dependency.

Features:
- Pure Python implementation (no external CLI)
- Native async support
- Cross-platform (Windows/Linux/macOS)
- Automatic proxy detection
- Screenshot capture support

Usage:
    from markitai.fetch_playwright import fetch_with_playwright, is_playwright_available

    if is_playwright_available():
        result = await fetch_with_playwright(url, config)
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import (
    DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS,
    DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS,
    DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS,
    DEFAULT_PLAYWRIGHT_WAIT_FOR,
    DEFAULT_SCREENSHOT_TILE_HEIGHT,
)
from markitai.fetch_types import FetchError

try:
    from markitai.webextract import (
        coerce_source_frontmatter,
        is_native_extraction_acceptable,
    )

    def extract_web_content(html: str, url: str) -> Any:
        """Load the full DOM extraction pipeline only when a page needs it."""
        from markitai.webextract import extract_web_content as extract

        return extract(html, url)

except ImportError:  # pragma: no cover - optional during staged implementation
    extract_web_content = None  # type: ignore[assignment]
    coerce_source_frontmatter = None  # type: ignore[assignment]
    is_native_extraction_acceptable = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from markitai.config import ScreenshotConfig


def is_playwright_available() -> bool:
    """Check if playwright is installed.

    Returns:
        True if playwright can be imported
    """
    return find_spec("playwright") is not None


# Cache for browser installation check
_browser_installed_cache: bool | None = None


def is_playwright_browser_installed(use_cache: bool = True) -> bool:
    """Check if playwright browser (Chromium) is installed.

    This function checks for browser executable existence without launching it,
    avoiding potential hangs in environments like WSL2 or headless servers.

    Args:
        use_cache: Whether to use cached result

    Returns:
        True if Chromium browser is available
    """
    global _browser_installed_cache

    if use_cache and _browser_installed_cache is not None:
        return _browser_installed_cache

    if not is_playwright_available():
        _browser_installed_cache = False
        return False

    # Check browser executable by path - never launch the browser
    _browser_installed_cache = _check_chromium_paths()
    return _browser_installed_cache


def _check_chromium_paths() -> bool:
    """Check common Playwright Chromium installation paths.

    Returns:
        True if Chromium executable found
    """
    import os
    import sys

    # Playwright stores browsers in these locations. PLAYWRIGHT_BROWSERS_PATH
    # is Playwright's own override and wins when set, so doctor and the
    # fetcher agree on where to look.
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if override and override != "0":
        base_paths = [Path(override).expanduser()]
    elif sys.platform == "win32":
        base_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright",
            Path.home() / "AppData" / "Local" / "ms-playwright",
        ]
    elif sys.platform == "darwin":
        base_paths = [
            Path.home() / "Library" / "Caches" / "ms-playwright",
        ]
    else:  # Linux
        base_paths = [
            Path.home() / ".cache" / "ms-playwright",
        ]

    for base in base_paths:
        if not base.exists():
            continue
        # Look for chromium-* directories
        chromium_dirs = list(base.glob("chromium-*"))
        if chromium_dirs:
            for chromium_dir in chromium_dirs:
                # Primary check: Playwright writes this marker after a
                # successful install. It is bundle-name and version agnostic
                # (newer Playwright ships "Google Chrome for Testing.app"
                # instead of "Chromium.app", which broke path-only checks)
                if (chromium_dir / "INSTALLATION_COMPLETE").exists():
                    logger.debug(f"Found Chromium install marker in: {chromium_dir}")
                    return True

                # Fallback: check known executable layouts
                if sys.platform == "win32":
                    # Try both old (chrome-win) and new (chrome-win64) paths
                    exe_paths = [
                        chromium_dir / "chrome-win64" / "chrome.exe",
                        chromium_dir / "chrome-win" / "chrome.exe",
                    ]
                elif sys.platform == "darwin":
                    exe_paths = []
                    for arch_dir in ("chrome-mac", "chrome-mac-arm64"):
                        exe_paths.extend(
                            [
                                chromium_dir
                                / arch_dir
                                / "Chromium.app"
                                / "Contents"
                                / "MacOS"
                                / "Chromium",
                                chromium_dir
                                / arch_dir
                                / "Google Chrome for Testing.app"
                                / "Contents"
                                / "MacOS"
                                / "Google Chrome for Testing",
                            ]
                        )
                else:
                    # Try both old (chrome-linux) and new (chrome-linux64) paths
                    exe_paths = [
                        chromium_dir / "chrome-linux64" / "chrome",
                        chromium_dir / "chrome-linux" / "chrome",
                    ]

                for exe in exe_paths:
                    if exe.exists():
                        logger.debug(f"Found Chromium at: {exe}")
                        return True

    return False


def clear_browser_cache() -> None:
    """Clear the browser installation cache."""
    global _browser_installed_cache
    _browser_installed_cache = None


def _build_auto_scroll_script(
    max_steps: int = DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS,
    step_delay_ms: int = DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS,
) -> str:
    """Build JavaScript for auto-scrolling to trigger lazy-loaded content.

    Borrowed from baoyu-skills url-to-markdown pattern:
    Scroll down incrementally, check if page height grows, stop when stable.

    Args:
        max_steps: Maximum number of scroll iterations
        step_delay_ms: Delay between scroll steps in milliseconds

    Returns:
        JavaScript code string for page.evaluate()
    """
    return f"""
    async () => {{
        let lastHeight = document.body.scrollHeight;
        for (let i = 0; i < {max_steps}; i++) {{
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, {step_delay_ms}));
            const newHeight = document.body.scrollHeight;
            if (newHeight === lastHeight) break;
            lastHeight = newHeight;
        }}
        window.scrollTo(0, 0);
    }}
    """


def _build_shadow_dom_normalize_script() -> str:
    """Build JavaScript for flattening live (open) shadow DOMs into light DOM.

    Walks every element in the document and, for those with an open
    ``shadowRoot``, moves all shadow children into the host element's light
    DOM.  This makes shadow-DOM content visible in ``page.content()`` output
    so that static HTML extraction can read it.

    Returns:
        JavaScript code string for ``page.evaluate()``.
    """
    return """
    () => {
        function flattenShadowRoots(root) {
            const walker = document.createTreeWalker(
                root,
                NodeFilter.SHOW_ELEMENT,
                null
            );
            const hosts = [];
            let node = walker.nextNode();
            while (node) {
                if (node.shadowRoot) {
                    hosts.push(node);
                }
                node = walker.nextNode();
            }
            for (const host of hosts) {
                const shadow = host.shadowRoot;
                // Move all shadow children into the host (light DOM)
                while (shadow.firstChild) {
                    host.appendChild(shadow.firstChild);
                }
            }
        }
        flattenShadowRoots(document.body || document);
    }
    """


def _build_dom_cleanup_script(url: str | None = None) -> str:
    """Build JavaScript for removing DOM noise before content extraction.

    Borrowed from baoyu-skills url-to-markdown pattern:
    Remove navigation, ads, popups, cookie banners, and inline event handlers.

    Args:
        url: Optional page URL used to look up site-specific noise selectors.

    Returns:
        JavaScript code string for page.evaluate()
    """
    import json
    from urllib.parse import urlparse

    from markitai.constants import (
        DOM_NOISE_ATTRIBUTES,
        DOM_NOISE_SELECTORS,
        SITE_NOISE_SELECTORS,
    )

    all_selectors = list(DOM_NOISE_SELECTORS)
    if url:
        try:
            domain = urlparse(url).hostname or ""
            # Strip leading 'www.' for lookup
            domain = domain.removeprefix("www.")
            site_selectors = SITE_NOISE_SELECTORS.get(domain, ())
            all_selectors.extend(site_selectors)
        except Exception:
            pass

    selectors_js = ", ".join(json.dumps(s) for s in all_selectors)
    attributes_js = ", ".join(json.dumps(a) for a in DOM_NOISE_ATTRIBUTES)

    return f"""
    () => {{
        // Remove noise elements
        const selectors = [{selectors_js}];
        for (const sel of selectors) {{
            try {{
                document.querySelectorAll(sel).forEach(el => el.remove());
            }} catch (e) {{}}
        }}

        // Clean inline event handlers and styles
        const attrs = [{attributes_js}];
        document.querySelectorAll('*').forEach(el => {{
            for (const attr of attrs) {{
                el.removeAttribute(attr);
            }}
        }});

        // Convert relative URLs to absolute
        const base = document.baseURI;
        document.querySelectorAll('a[href]').forEach(a => {{
            try {{
                const href = a.getAttribute('href');
                if (href && !href.startsWith('http') && !href.startsWith('//') && !href.startsWith('#')) {{
                    a.setAttribute('href', new URL(href, base).href);
                }}
            }} catch (e) {{}}
        }});
        document.querySelectorAll('img[src]').forEach(img => {{
            try {{
                const src = img.getAttribute('src');
                if (src && !src.startsWith('http') && !src.startsWith('data:') && !src.startsWith('//')) {{
                    img.setAttribute('src', new URL(src, base).href);
                }}
            }} catch (e) {{}}
        }});
    }}
    """


def _is_x_article_url(url: str) -> bool:
    """True for x.com/twitter.com ``/article/`` URLs (singular — confirmed
    against defuddle's reference implementation, see
    webextract/enrichers/x_oembed.py).

    X Articles are login-walled for anonymous visitors, so DOM extraction
    is guaranteed to produce nothing useful (see ``is_login_wall`` below).
    Callers use this to skip launching a browser entirely and go straight
    to the FxTwitter/oEmbed enricher — mirroring defuddle's
    ``canExtractAsync() && prefersAsync()`` early exit for these URLs.
    """
    is_x = "x.com/" in url or "twitter.com/" in url
    return is_x and "/article/" in url


#: Upper bound for closing a page/context whose renderer may be wedged.
_CLOSE_TIMEOUT_S = 10.0


class PlaywrightPageTimeoutError(FetchError):
    """A page operation after navigation outran fetch.playwright.timeout."""


class _PageDeadline:
    """One shared time budget for the page operations after navigation.

    Playwright only times out ``goto()`` (and the waits given an explicit
    timeout); ``evaluate()``, ``title()``, ``content()`` and friends wait
    forever on a renderer stuck in a script loop. Each such call runs under
    what is left of this budget. The screenshot is timed on its own (see
    :func:`_capture_screenshot_within`) and handed back with :meth:`extend`,
    so a slow capture cannot starve the text extraction.
    """

    def __init__(self, budget_s: float, timeout_ms: int) -> None:
        self._budget_s = budget_s
        self._timeout_ms = timeout_ms
        self._deadline = time.monotonic() + budget_s

    def remaining(self) -> float:
        """Seconds left in the budget (never negative)."""
        return max(0.0, self._deadline - time.monotonic())

    def extend(self, seconds: float) -> None:
        """Give back time spent on work that has its own timeout."""
        self._deadline += max(0.0, seconds)

    async def run(self, awaitable: Any, what: str) -> Any:
        """Await ``awaitable`` within the remaining budget.

        Raises:
            PlaywrightPageTimeoutError: The budget ran out first.
        """
        remaining = self.remaining()
        try:
            if remaining <= 0:
                raise TimeoutError
            return await asyncio.wait_for(awaitable, remaining)
        except TimeoutError as e:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            if remaining <= 0:
                stage = (
                    f"the {self._budget_s:.1f}s page budget was used up before "
                    f"{what} started"
                )
            else:
                stage = (
                    f"{what} was still running after {remaining:.1f}s, the rest "
                    f"of the {self._budget_s:.1f}s page budget"
                )
            raise PlaywrightPageTimeoutError(
                f"Playwright page did not respond: {stage} "
                f"(fetch.playwright.timeout={self._timeout_ms}ms). The page "
                "is likely busy (for example stuck in a script loop); the "
                "browser page was closed."
            ) from e


#: Share of the screenshot budget given to ``page.screenshot()`` itself, so
#: Playwright's own timeout fires (as an ordinary capture failure) before
#: the outer bound, which also covers compression and tiling.
_SCREENSHOT_INNER_TIMEOUT_SHARE = 0.8


async def _capture_screenshot_within(
    page: Any,
    config: ScreenshotConfig,
    output_dir: Path,
    url: str,
    *,
    timeout_ms: int,
) -> tuple[Path | None, list[Path], str | None]:
    """Capture a screenshot under its own ``timeout_ms`` budget.

    A screenshot is an optional extra: when it fails or runs out of time
    the fetch keeps its text and reports why in ``screenshot_error``.

    Returns:
        ``(primary path, tiles, error)``; ``error`` is None on success.
    """
    budget_s = max(timeout_ms, 1) / 1000
    capture_errors: list[str] = []
    try:
        path, tiles = await asyncio.wait_for(
            _capture_screenshot(
                page,
                config,
                output_dir,
                url,
                errors=capture_errors,
                timeout_ms=max(1, int(timeout_ms * _SCREENSHOT_INNER_TIMEOUT_SHARE)),
            ),
            budget_s,
        )
    except TimeoutError:
        error = (
            f"screenshot capture timed out after {budget_s:.1f}s "
            f"(fetch.playwright.timeout={timeout_ms}ms)"
        )
        logger.warning(f"Screenshot capture failed: {error}")
        return None, [], error
    if path is None:
        return None, [], capture_errors[0] if capture_errors else "screenshot failed"
    return path, tiles, None


@dataclass
class PlaywrightFetchResult:
    """Result from Playwright fetch."""

    content: str
    title: str | None = None
    final_url: str | None = None
    screenshot_path: Path | None = None
    #: All screenshot files: the single path when within tile_height, or the
    #: vertical tiles (primary path first) of a long page split for VLM reads.
    screenshot_tiles: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedContext:
    """Cached browser context with expiration."""

    context: Any
    created_at: float
    last_used_at: float
    session_key: str


async def _guard_public_context(context: Any, timeout: int, proxy: str | None) -> None:
    """Route anonymous browser HTTP traffic through DNS-pinned requests."""
    from markitai.fetch_http import public_http_request

    async def public_route(route: Any) -> None:
        request = route.request
        try:
            response = await public_http_request(
                request.url,
                method=request.method,
                headers=await request.all_headers(),
                content=request.post_data_buffer,
                timeout=timeout / 1000,
                proxy=proxy,
                follow_redirects=False,
            )
            headers = dict(response.headers)
            for name in (
                "content-encoding",
                "content-length",
                "transfer-encoding",
            ):
                headers.pop(name, None)
            await route.fulfill(
                status=response.status_code,
                headers=headers,
                body=response.content,
            )
        except Exception as exc:
            logger.debug("[Fetch] Blocked browser request: {}", exc)
            await route.abort("blockedbyclient")

    async def block_socket(socket: Any) -> None:
        await socket.close()

    # Context routing also covers popups and nested frames. Never
    # reuse a logged-in context for an anonymous remote caller.
    await context.route("**/*", public_route)
    await context.route_web_socket("**/*", block_socket)


#: Hosts a proxied browser reaches directly. Playwright launches Chromium
#: with ``<-loopback>`` (loopback traffic goes through the proxy, which then
#: dials its own loopback), so these have to be bypassed explicitly.
_LOOPBACK_PROXY_BYPASS = ("localhost", "*.localhost", "127.0.0.0/8", "[::1]")


def chromium_proxy_bypass(patterns: Sequence[str]) -> str:
    """Translate NO_PROXY patterns into a Playwright ``proxy.bypass`` value.

    The browser applies the list to every request of a page — subresources,
    redirects, frames — which a per-URL proxy choice cannot reach. Loopback
    hosts are always bypassed (see ``_LOOPBACK_PROXY_BYPASS``).

    Args:
        patterns: NO_PROXY-style patterns (``markitai.fetch_policy`` syntax).

    Returns:
        Comma-separated bypass rules.
    """
    import ipaddress

    rules: list[str] = list(_LOOPBACK_PROXY_BYPASS)
    for raw in patterns:
        pattern = raw.strip()
        # Playwright splits the value on commas; Chromium on whitespace too
        if not pattern or any(c in pattern for c in ", \t;"):
            continue
        try:
            # Chromium wants IPv6 literals bracketed ("[fd00::1]")
            if isinstance(ipaddress.ip_address(pattern), ipaddress.IPv6Address):
                pattern = f"[{pattern}]"
        except ValueError:
            pass
        if pattern not in rules:
            rules.append(pattern)
    return ",".join(rules)


class PlaywrightRenderer:
    """Reusable Playwright renderer to avoid browser cold starts."""

    def __init__(
        self,
        proxy: str | None = None,
        proxy_bypass: Sequence[str] | None = None,
    ) -> None:
        self.proxy = proxy
        # NO_PROXY patterns the proxied browser reaches directly
        self.proxy_bypass = list(proxy_bypass or [])
        self._playwright: Any = None
        self._browser: Any = None
        self._lock = asyncio.Lock()

        # Session cache (domain-persistent mode)
        self._context_cache: dict[str, CachedContext] = {}
        self._context_cache_lock = asyncio.Lock()
        self._session_cache_enabled = False
        self._session_ttl_seconds = 600
        self._max_contexts = 8

    async def __aenter__(self) -> PlaywrightRenderer:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def enable_domain_session_cache(self, ttl_seconds: int, max_contexts: int) -> None:
        """Enable domain-persistent session caching.

        Args:
            ttl_seconds: TTL for cached contexts in seconds
            max_contexts: Maximum number of contexts to cache
        """
        self._session_cache_enabled = True
        self._session_ttl_seconds = ttl_seconds
        self._max_contexts = max_contexts
        logger.debug(
            f"Playwright session cache enabled (TTL={ttl_seconds}s, max={max_contexts})"
        )

    async def _ensure_browser(self) -> Any:
        if self._browser is not None:
            return self._browser

        from playwright.async_api import (
            async_playwright,  # pyright: ignore[reportMissingImports]
        )

        async with self._lock:
            if self._browser is not None:
                return self._browser

            self._playwright = await async_playwright().start()
            launch_options: dict[str, Any] = {"headless": True}
            if self.proxy:
                launch_options["proxy"] = {
                    "server": self.proxy,
                    "bypass": chromium_proxy_bypass(self.proxy_bypass),
                }

            try:
                self._browser = await self._playwright.chromium.launch(**launch_options)
            except Exception as e:
                await self._playwright.stop()
                self._playwright = None
                from markitai.utils.guidance import (
                    playwright_browser_missing_error,
                )

                raise RuntimeError(
                    f"Failed to launch Chromium browser: {e}\n"
                    + playwright_browser_missing_error()
                )
            return self._browser

    async def _get_or_create_cached_context(
        self, session_key: str, ctx_options: dict[str, Any]
    ) -> Any:
        """Get an existing cached context or create a new one."""
        import time

        # Serialize lookup/create/expire so concurrent same-key fetches
        # don't create duplicate contexts or double-delete expired entries.
        async with self._context_cache_lock:
            now = time.time()

            # 1. Check for existing context
            if session_key in self._context_cache:
                cached = self._context_cache[session_key]
                # Check for expiration
                if now - cached.last_used_at < self._session_ttl_seconds:
                    cached.last_used_at = now
                    logger.debug(
                        f"Reusing cached Playwright context for: {session_key}"
                    )
                    return cached.context
                else:
                    # Expired
                    logger.debug(
                        f"Cached Playwright context expired for: {session_key}"
                    )
                    self._context_cache.pop(session_key, None)
                    await cached.context.close()

            # 2. Enforce max contexts (LRU-ish)
            if len(self._context_cache) >= self._max_contexts:
                # Remove oldest (based on last_used_at)
                oldest_key = min(
                    self._context_cache.keys(),
                    key=lambda k: self._context_cache[k].last_used_at,
                )
                logger.debug(f"Evicting Playwright context cache for: {oldest_key}")
                oldest = self._context_cache.pop(oldest_key, None)
                if oldest is not None:
                    await oldest.context.close()

            # 3. Create new context
            browser = await self._ensure_browser()
            context = await browser.new_context(**ctx_options)
            self._context_cache[session_key] = CachedContext(
                context=context,
                created_at=now,
                last_used_at=now,
                session_key=session_key,
            )
            logger.debug(f"Created new cached Playwright context for: {session_key}")
            return context

    async def fetch(
        self,
        url: str,
        timeout: int = 30000,
        wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
        extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
        screenshot_config: ScreenshotConfig | None = None,
        output_dir: Path | None = None,
        # Advanced browser control (aligned with CF Browser Rendering API)
        wait_for_selector: str | None = None,
        cookies: list[dict[str, str]] | None = None,
        reject_resource_patterns: list[str] | None = None,
        extra_http_headers: dict[str, str] | None = None,
        user_agent: str | None = None,
        http_credentials: dict[str, str] | None = None,
        skip_auto_scroll: bool = False,
        # Session persistence
        session_key: str | None = None,
        persist_context: bool = False,
        # Enrichment control
        remote_consent: str = "ask",
    ) -> PlaywrightFetchResult:
        """Fetch URL using a persistent browser instance."""
        # Like defuddle's server-side extraction, anonymous X posts can use
        # FxTwitter/oEmbed without first downloading an unusable page shell.
        # Existing sessions and screenshots still need the rendered DOM.
        needs_screenshot = bool(
            screenshot_config and getattr(screenshot_config, "enabled", True)
        )
        authenticated = bool(
            cookies or http_credentials or extra_http_headers or persist_context
        )
        from markitai.fetch_consent import peek_cached_remote_consent
        from markitai.webextract.enrichers.x_oembed import is_x_post_url

        enrichment_attempted = False
        if (
            is_x_post_url(url)
            and not needs_screenshot
            and not authenticated
            and (
                remote_consent == "always"
                or peek_cached_remote_consent() is True
                or _is_x_article_url(url)
            )
        ):
            enrichment_attempted = True
            enriched = await self._enriched_result(url, remote_consent)
            if enriched is not None:
                return enriched
            # Enricher blocked (e.g. remote_consent="never") or failed —
            # fall through to the normal browser-render path below so the
            # user still gets the (login-wall) page rather than nothing.

        from markitai.fetch_policy import public_network_only

        restricted = public_network_only.get()

        # Build context options from advanced config
        ctx_options: dict[str, Any] = {}
        if needs_screenshot and screenshot_config is not None:
            # screenshot.viewport_width/height described the render window
            # but nothing read them, so every capture used Playwright's own
            # 1280x720 default and setting them did nothing. Only the
            # screenshot path takes them: a plain HTML fetch keeps whatever
            # viewport it has always used.
            ctx_options["viewport"] = {
                "width": screenshot_config.viewport_width,
                "height": screenshot_config.viewport_height,
            }
        if extra_http_headers:
            ctx_options["extra_http_headers"] = extra_http_headers
        if user_agent:
            ctx_options["user_agent"] = user_agent
        if http_credentials:
            ctx_options["http_credentials"] = http_credentials

        if restricted:
            ctx_options["service_workers"] = "block"
        if (
            self._session_cache_enabled
            and persist_context
            and session_key
            and not restricted
        ):
            context = await self._get_or_create_cached_context(session_key, ctx_options)
            should_close_context = False
        else:
            browser = await self._ensure_browser()
            context = await browser.new_context(**ctx_options)
            should_close_context = True

        page = None
        timed_out = False
        try:
            if restricted:
                await _guard_public_context(context, timeout, self.proxy)

            # Inject cookies before navigation
            if cookies:
                await context.add_cookies(cookies)

            page = await context.new_page()

            # Set up resource filtering before navigation
            if reject_resource_patterns:

                async def _abort_route(route: Any) -> None:
                    await route.abort()

                for pattern in reject_resource_patterns:
                    await page.route(pattern, _abort_route)

            # Map wait_for string to Playwright's literal type
            wait_until_map = {
                "load": "load",
                "domcontentloaded": "domcontentloaded",
                "networkidle": "networkidle",
            }
            wait_until = wait_until_map.get(wait_for, "domcontentloaded")

            navigation_started = time.perf_counter()
            response = await page.goto(url, timeout=timeout, wait_until=wait_until)
            status = getattr(response, "status", None)
            logger.debug(
                "[Playwright] Navigation: {:.3f}s, HTTP {}",
                time.perf_counter() - navigation_started,
                status,
            )
            if isinstance(status, int) and status >= 400:
                # goto() does not raise for HTTP failures. An error document
                # cannot produce the requested tweet selector; try the same
                # remote enrichment used for missing DOM content immediately.
                enriched = (
                    None
                    if enrichment_attempted
                    else await self._enriched_result(url, remote_consent)
                )
                if enriched is None:
                    raise FetchError(f"Playwright navigation returned HTTP {status}")
                enriched.metadata["http_status"] = status
                if needs_screenshot and screenshot_config and output_dir:
                    (
                        enriched.screenshot_path,
                        enriched.screenshot_tiles,
                        shot_error,
                    ) = await _capture_screenshot_within(
                        page, screenshot_config, output_dir, url, timeout_ms=timeout
                    )
                    if shot_error is not None:
                        enriched.metadata["screenshot_error"] = shot_error
                return enriched

            # Precise element waiting (preferred) or time-based fallback
            if wait_for_selector:
                wait_started = time.perf_counter()
                try:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=min(timeout, 10000)
                    )
                except Exception as e:
                    logger.debug(
                        f"wait_for_selector '{wait_for_selector}' timed out: {e}"
                    )
                else:
                    # Stabilize only when the selector was actually found.
                    if extra_wait_ms > 0:
                        await asyncio.sleep(extra_wait_ms / 1000)
                logger.debug(
                    "[Playwright] Element wait: {:.3f}s",
                    time.perf_counter() - wait_started,
                )
            elif extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)

            # Every page operation from here on shares one budget derived
            # from fetch.playwright.timeout (plus the auto-scroll's own
            # steps). goto() is the only call Playwright times out by
            # itself; a page stuck in a script loop would otherwise hang
            # evaluate()/content() forever.
            scroll_allowance_s = (
                0.0
                if skip_auto_scroll
                else (
                    DEFAULT_PLAYWRIGHT_AUTO_SCROLL_STEPS
                    * DEFAULT_PLAYWRIGHT_AUTO_SCROLL_DELAY_MS
                    + DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS
                )
                / 1000
            )
            deadline = _PageDeadline(timeout / 1000 + scroll_allowance_s, timeout)

            # Auto-scroll to trigger lazy-loaded content
            if not skip_auto_scroll:
                try:
                    scroll_script = _build_auto_scroll_script()
                    await deadline.run(page.evaluate(scroll_script), "auto-scroll")
                    await asyncio.sleep(DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS / 1000)
                except PlaywrightPageTimeoutError:
                    raise
                except Exception as e:
                    logger.debug(f"Auto-scroll failed (non-critical): {e}")

            # Screenshot the page as rendered, BEFORE any DOM change below:
            # the shadow-DOM flattening and noise cleanup strip styles, SVG,
            # canvas, iframes and navigation for text extraction, which
            # would otherwise leave the capture a bare unstyled text page.
            screenshot_path = None
            screenshot_tiles: list[Path] = []
            screenshot_error: str | None = None
            if (
                screenshot_config
                and output_dir
                and getattr(screenshot_config, "enabled", True)
            ):
                # Timed on its own and given back to the page budget: a
                # slow capture costs the screenshot, never the text.
                capture_started = time.monotonic()
                (
                    screenshot_path,
                    screenshot_tiles,
                    screenshot_error,
                ) = await _capture_screenshot_within(
                    page, screenshot_config, output_dir, url, timeout_ms=timeout
                )
                deadline.extend(time.monotonic() - capture_started)

            # Browser DOM normalize: flatten live shadow roots before extraction
            try:
                shadow_script = _build_shadow_dom_normalize_script()
                await deadline.run(page.evaluate(shadow_script), "shadow DOM flatten")
            except PlaywrightPageTimeoutError:
                raise
            except Exception as e:
                logger.debug(f"Shadow DOM normalize failed (non-critical): {e}")

            # DOM cleanup: remove noise elements before extraction
            try:
                cleanup_script = _build_dom_cleanup_script(url=url)
                await deadline.run(page.evaluate(cleanup_script), "DOM cleanup")
            except PlaywrightPageTimeoutError:
                raise
            except Exception as e:
                logger.debug(f"DOM cleanup failed (non-critical): {e}")

            title = await deadline.run(page.title(), "page.title()")
            final_url = page.url
            html_content = await deadline.run(page.content(), "page.content()")
            metadata: dict[str, Any] = {"renderer": "playwright", "wait_for": wait_for}
            if screenshot_error is not None:
                metadata["screenshot_error"] = screenshot_error

            # Try native webextract FIRST to avoid redundant HTML→Markdown
            # conversion. Only fall back to _html_to_markdown if webextract
            # is unavailable or produces insufficient quality.
            extraction_started = time.perf_counter()
            markdown_content = ""
            used_native_webextract = False
            if extract_web_content is not None:
                # If extract_web_content is available, the other webextract functions are too
                assert is_native_extraction_acceptable is not None
                assert coerce_source_frontmatter is not None
                try:
                    # CPU-bound (BeautifulSoup parsing + deepcopies); run in a
                    # thread to avoid blocking the event loop. Relative links
                    # and images resolve against where the browser ended up
                    # after redirects, not the URL that was requested.
                    extracted = await asyncio.to_thread(
                        extract_web_content, html_content, final_url or url
                    )
                except Exception as e:
                    logger.debug(f"Native webextract failed, using fallback: {e}")
                else:
                    native_markdown = getattr(extracted, "markdown", "")
                    if is_native_extraction_acceptable(extracted):
                        markdown_content = native_markdown
                        used_native_webextract = True
                        # Prefer typed frontmatter builder when info is available
                        if (
                            hasattr(extracted, "info")
                            and getattr(extracted, "info", None) is not None
                        ):
                            from markitai.webextract.frontmatter import (
                                build_source_frontmatter,
                            )

                            source_frontmatter = build_source_frontmatter(extracted)
                        else:
                            source_frontmatter = coerce_source_frontmatter(
                                getattr(extracted, "metadata", None)
                            )
                        if source_frontmatter:
                            metadata["source_frontmatter"] = source_frontmatter
                            title = source_frontmatter.get("title") or title
                        metadata["webextract_diagnostics"] = dict(
                            getattr(extracted, "diagnostics", {}) or {}
                        )

            # Fallback chain:
            # 1. For X/Twitter URLs with very short native content (<50 words),
            #    try the oEmbed enricher (handles Articles via FxTwitter API)
            # 2. Use _html_to_markdown as last resort
            if not markdown_content and not enrichment_attempted:
                (
                    enriched_md,
                    overrides,
                    enricher_source,
                ) = await self._try_enricher_fallback_async(url, remote_consent)
                if enriched_md:
                    markdown_content = enriched_md
                    metadata["_enricher_source"] = enricher_source
                    if overrides:
                        metadata.update(overrides)
                        new_title = overrides.get("title")
                        if new_title:
                            title = str(new_title)
            elif used_native_webextract:
                # Structural completeness check (not a word-count guess).
                # The extractor knows whether it found real content:
                #   - Resolver explicitly signalled failure
                #   - Article page only showed a preview (<500 chars)
                #   - Login wall detected (for /article/ URLs where no
                #     resolver ran; XArticleExtractor has no resolve())
                diag = getattr(extracted, "diagnostics", {})
                resolver_diag = diag.get("resolver_diagnostics", {})
                x_resolve = resolver_diag.get("x_resolve", "")
                is_article = resolver_diag.get("is_article", False)
                is_x_url = "x.com/" in url or "twitter.com/" in url
                # Login wall: only needed for /article/ where no resolver runs
                is_login_wall = "Continue with" in html_content and "/article/" in url

                needs_enricher = (
                    x_resolve == "no_primary_tweet_found"
                    or (is_article and len(markdown_content) < 500)
                    or is_login_wall
                )
                if needs_enricher and is_x_url and not enrichment_attempted:
                    (
                        enriched_md,
                        overrides,
                        enricher_source,
                    ) = await self._try_enricher_fallback_async(url, remote_consent)
                    if enriched_md:
                        markdown_content = enriched_md
                        used_native_webextract = False  # mark as enricher output
                        metadata["_enricher_source"] = enricher_source
                        if overrides:
                            metadata.update(overrides)
                            new_title = overrides.get("title")
                            if new_title:
                                title = str(new_title)

            if not markdown_content:
                markdown_content = _html_to_markdown(html_content)

            if not used_native_webextract and _is_content_incomplete(markdown_content):
                try:
                    rendered_text = await deadline.run(
                        page.inner_text("body"), "page.inner_text()"
                    )
                    if rendered_text and len(rendered_text.strip()) > len(
                        markdown_content.strip()
                    ):
                        markdown_content = _format_inner_text(rendered_text)
                except PlaywrightPageTimeoutError:
                    raise
                except Exception as e:
                    logger.debug(
                        "[Playwright] Failed to extract inner_text fallback: {}", e
                    )

            logger.debug(
                "[Playwright] Extraction/enrichment: {:.3f}s",
                time.perf_counter() - extraction_started,
            )

            return PlaywrightFetchResult(
                content=markdown_content,
                title=title,
                final_url=final_url,
                screenshot_path=screenshot_path,
                screenshot_tiles=screenshot_tiles,
                metadata=metadata,
            )
        except PlaywrightPageTimeoutError:
            timed_out = True
            raise
        finally:
            await self._release_page(
                context,
                page,
                close_context=should_close_context or timed_out,
                session_key=None if should_close_context else session_key,
            )

    async def _release_page(
        self,
        context: Any,
        page: Any,
        *,
        close_context: bool,
        session_key: str | None,
    ) -> None:
        """Close the page (and context) without letting a hung page block us.

        A page whose script loop timed out cannot be trusted to close
        quickly, so every close is bounded; a cached context that held such
        a page is evicted from the session cache and closed with it.
        """
        if close_context and session_key is not None:
            async with self._context_cache_lock:
                cached = self._context_cache.get(session_key)
                if cached is not None and cached.context is context:
                    self._context_cache.pop(session_key, None)
        target = context if close_context else page
        if target is None:
            return
        try:
            await asyncio.wait_for(target.close(), _CLOSE_TIMEOUT_S)
        except Exception as e:
            logger.debug("[Playwright] Closing the page/context failed: {}", e)

    async def close(self) -> None:
        """Close browser and playwright instances."""
        async with self._lock:
            # Clean up context cache
            for cached in self._context_cache.values():
                try:
                    await cached.context.close()
                except Exception as e:
                    logger.debug("[Playwright] Context close failed: {}", e)
            self._context_cache.clear()

            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None

    async def _enriched_result(
        self, url: str, remote_consent: str
    ) -> PlaywrightFetchResult | None:
        """Build a result whose frontmatter describes the enriched content."""
        markdown, overrides, source = await self._try_enricher_fallback_async(
            url, remote_consent
        )
        if not markdown:
            return None
        from urllib.parse import urlsplit

        from markitai.webextract.utils import count_words

        frontmatter: dict[str, Any] = dict(overrides or {})
        frontmatter.setdefault("domain", urlsplit(url).hostname)
        frontmatter["word_count"] = count_words(markdown)
        frontmatter.setdefault("content_profile", "social_post")
        return PlaywrightFetchResult(
            content=markdown,
            title=frontmatter.get("title"),
            final_url=url,
            metadata={"_enricher_source": source, "source_frontmatter": frontmatter},
        )

    async def _try_enricher_fallback_async(
        self,
        url: str,
        remote_consent: str,
    ) -> tuple[str, dict[str, Any] | None, str]:
        """Try oEmbed/FxTwitter enrichment, returning (markdown, overrides, source).

        Returns ("", None, "") on failure.  ``source`` is one of
        ``"fxtwitter"``, ``"oembed"``, or ``""`` (no enrichment).

        FxTwitter/oEmbed are remote services like defuddle/jina/cloudflare, so
        they go through the *same* process-wide consent decision rather than a
        second prompt of their own: one Yes authorizes every remote service for
        the run, one No blocks them all, and a decision already cached by the
        main chain is reused as is. Under ``ask`` with nothing decided yet this
        prompts once on an interactive TTY and otherwise denies — the exact
        branches of ``resolve_remote_consent``.

        Consent is resolved lazily (after ``should_run()`` and the shared
        full-URL/DNS privacy policy have confirmed there is a public
        X/Twitter URL to send), so no question is asked about a URL that
        would never leave the machine.
        """
        from markitai.fetch_consent import (
            _env_no_remote_fetch,
            assess_remote_url,
            disclose_remote_use,
            resolve_remote_consent,
        )
        from markitai.webextract.enrichers.base import EnrichmentPolicy
        from markitai.webextract.enrichers.x_oembed import XOEmbedEnricher

        if remote_consent == "never" or _env_no_remote_fetch():
            return "", None, ""

        enricher = XOEmbedEnricher()
        policy = EnrichmentPolicy(allow_network=True, allow_async=True)
        if not enricher.should_run(url, policy):
            return "", None, ""
        assessment = await assess_remote_url(
            url, remote_consent, ["fxtwitter", "twitter-oembed"]
        )
        if not assessment.allowed:
            logger.debug("[Fetch] Skipping X enrichment: {}", assessment.reason)
            return "", None, ""
        # Anything but an explicit "always" is treated as the conservative "ask".
        # Under "ask" this reaches a blocking TTY prompt, so it runs off the
        # event loop: a live Playwright page needs its loop to keep servicing
        # the browser while the user reads the question. The consent state and
        # the interaction port are both plain process-wide objects (no
        # thread-locals), so the decision stays shared across threads.
        if not await asyncio.to_thread(
            resolve_remote_consent,
            "always" if remote_consent == "always" else "ask",
            services=["fxtwitter", "twitter-oembed"],
        ):
            return "", None, ""

        disclose_remote_use(["fxtwitter", "twitter-oembed"])
        logger.debug("[Fetch] Enriching via FxTwitter/oEmbed: {}", url)
        try:
            resolved = await enricher.enrich(url, None)
        except Exception as exc:
            logger.warning("[Playwright] enricher failed, falling back to DOM: {}", exc)
            return "", None, ""

        if resolved is None or not resolved.content_html:
            return "", None, ""

        from markitai.webextract.markdown import render_markdown

        enriched_md = render_markdown(resolved.content_html)
        source = str(resolved.diagnostics.get("source", ""))
        return enriched_md, resolved.metadata_overrides or None, source


async def fetch_with_playwright(
    url: str,
    timeout: int = 30000,
    wait_for: str = DEFAULT_PLAYWRIGHT_WAIT_FOR,
    extra_wait_ms: int = DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS,
    proxy: str | None = None,
    screenshot_config: ScreenshotConfig | None = None,
    output_dir: Path | None = None,
    renderer: PlaywrightRenderer | None = None,
    proxy_bypass: Sequence[str] | None = None,
    # Advanced browser control
    wait_for_selector: str | None = None,
    cookies: list[dict[str, str]] | None = None,
    reject_resource_patterns: list[str] | None = None,
    extra_http_headers: dict[str, str] | None = None,
    user_agent: str | None = None,
    http_credentials: dict[str, str] | None = None,
    # Auto-scroll control
    skip_auto_scroll: bool = False,
    # Session persistence
    session_key: str | None = None,
    persist_context: bool = False,
    # Enrichment control
    remote_consent: str = "ask",
) -> PlaywrightFetchResult:
    """Fetch URL using Playwright (reuses renderer if provided)."""
    # Collect advanced kwargs
    advanced_kwargs: dict[str, Any] = {}
    if skip_auto_scroll:
        advanced_kwargs["skip_auto_scroll"] = skip_auto_scroll
    if wait_for_selector is not None:
        advanced_kwargs["wait_for_selector"] = wait_for_selector
    if cookies is not None:
        advanced_kwargs["cookies"] = cookies
    if reject_resource_patterns is not None:
        advanced_kwargs["reject_resource_patterns"] = reject_resource_patterns
    if extra_http_headers is not None:
        advanced_kwargs["extra_http_headers"] = extra_http_headers
    if user_agent is not None:
        advanced_kwargs["user_agent"] = user_agent
    if http_credentials is not None:
        advanced_kwargs["http_credentials"] = http_credentials
    if session_key is not None:
        advanced_kwargs["session_key"] = session_key
    if persist_context:
        advanced_kwargs["persist_context"] = persist_context
    # Enrichment control — always pass through
    advanced_kwargs["remote_consent"] = remote_consent

    if renderer:
        return await renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
            **advanced_kwargs,
        )

    # Legacy one-off path
    async with PlaywrightRenderer(
        proxy=proxy, proxy_bypass=proxy_bypass
    ) as standalone_renderer:
        return await standalone_renderer.fetch(
            url,
            timeout=timeout,
            wait_for=wait_for,
            extra_wait_ms=extra_wait_ms,
            screenshot_config=screenshot_config,
            output_dir=output_dir,
            **advanced_kwargs,
        )


def _is_content_incomplete(content: str) -> bool:
    """Check if content appears incomplete (likely Shadow DOM or JS rendering issue).

    Args:
        content: Markdown content to check

    Returns:
        True if content appears incomplete
    """
    if not content:
        return True

    # Remove markdown syntax and whitespace
    clean = re.sub(r"[#\-*_>\[\]`|()!]", "", content)
    clean = re.sub(r"\[.*?\]\(.*?\)", "", clean)  # Remove links
    clean = " ".join(clean.split())

    # Check for common incomplete content patterns
    incomplete_patterns = [
        r"Don't miss what's happening",  # X.com login prompt
        r"Accept all cookies",  # Cookie consent
        r"Log in.*Sign up",  # Login/signup only
        r"Terms of Service.*Privacy Policy",  # Footer only
    ]

    # Content is mostly boilerplate
    for pattern in incomplete_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            # Check if there's substantial other content
            if len(clean) < 500:
                return True

    # Too short to be meaningful content
    return len(clean) < 200


def _format_inner_text(text: str) -> str:
    """Format inner_text content as basic markdown.

    Args:
        text: Raw text from page.inner_text()

    Returns:
        Formatted markdown content
    """
    if not text:
        return ""

    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Preserve the line with basic cleanup
        formatted_lines.append(stripped)

    # Join with double newlines for paragraph separation
    return "\n\n".join(formatted_lines)


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown.

    Tries markitdown first, falls back to basic HTML stripping.

    Args:
        html: HTML content

    Returns:
        Markdown content
    """
    # Remove <noscript> tags - Playwright has JS enabled, so these are irrelevant
    # and often contain fallback messages like "JavaScript is not available"
    html = re.sub(
        r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE
    )

    try:
        import io

        from markitdown import MarkItDown, StreamInfo

        md = MarkItDown()
        # MarkItDown uses convert_stream for in-memory content
        stream = io.BytesIO(html.encode("utf-8"))
        # Force UTF-8 to avoid charset auto-detection false positives on cleaned HTML.
        result = md.convert_stream(
            stream,
            file_extension=".html",
            stream_info=StreamInfo(
                mimetype="text/html",
                extension=".html",
                charset="utf-8",
            ),
        )
        return result.text_content if result and result.text_content else ""
    except Exception as e:
        logger.debug(f"markitdown conversion failed, using fallback: {e}")
        # Fallback: basic HTML tag stripping
        return _strip_html_tags(html)


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags from content (fallback converter).

    Args:
        html: HTML content

    Returns:
        Plain text with HTML tags removed
    """
    # Remove script, style, and noscript elements
    html = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(
        r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    try:
        import html as html_module

        text = html_module.unescape(text)
    except Exception as e:
        logger.debug("[Playwright] HTML unescape failed: {}", e)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    return text.strip()


async def _capture_screenshot(
    page: Any,
    config: ScreenshotConfig,
    output_dir: Path,
    url: str,
    *,
    errors: list[str] | None = None,
    timeout_ms: int | None = None,
) -> tuple[Path | None, list[Path]]:
    """Capture page screenshot, tiling long pages.

    Args:
        page: Playwright page object
        config: Screenshot configuration
        output_dir: Output directory
        url: Original URL (for filename)
        errors: When given, a failure reason is appended to it
        timeout_ms: Playwright timeout for the capture itself

    Returns:
        (primary screenshot path, all screenshot tiles). The primary path is
        the single file when the page fits within ``tile_height``, or the
        first tile of a long page. Both are None/empty on failure.
    """
    from markitai.fetch_screenshot import (
        _compress_screenshot,
        _url_to_screenshot_filename,
        remove_stale_screenshot_tiles,
    )

    try:
        # Generate filename using the same logic as fetch.py
        filename = _url_to_screenshot_filename(url)

        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / filename
        # Tiles of an earlier, longer capture under this name would
        # otherwise survive and be picked up alongside the new ones.
        remove_stale_screenshot_tiles(screenshot_path)

        # Get settings from config. Full-page capture is not configurable:
        # ScreenshotConfig has never declared `full_page`, so the old
        # getattr() default won every time — the knob only looked adjustable.
        # `tile_height` (per-tile cap) and `max_height` (legacy single-file
        # cap) bound a runaway page.
        full_page = True
        quality = getattr(config, "quality", 85)
        max_height = getattr(config, "max_height", 10000)
        tile_height = getattr(config, "tile_height", DEFAULT_SCREENSHOT_TILE_HEIGHT)

        screenshot_kwargs: dict[str, Any] = {
            "path": str(screenshot_path),
            "full_page": full_page,
            "type": "jpeg",
            "quality": quality,
        }
        if timeout_ms is not None:
            screenshot_kwargs["timeout"] = timeout_ms
        await page.screenshot(**screenshot_kwargs)

        # Compress and tile if needed (tall pages become N VLM-readable tiles)
        tiles = (
            _compress_screenshot(
                screenshot_path,
                quality=quality,
                max_height=max_height,
                tile_height=tile_height,
            )
            or []
        )
        primary = tiles[0] if tiles else screenshot_path
        if len(tiles) > 1:
            logger.debug(f"Screenshot saved: {primary} (+{len(tiles) - 1} tile(s))")
        else:
            logger.debug(f"Screenshot saved: {primary}")
        return primary, tiles
    except Exception as e:
        logger.warning(f"Screenshot capture failed: {e}")
        if errors is not None:
            errors.append(f"screenshot capture failed: {e}")
        return None, []
