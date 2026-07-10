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
)

try:
    from markitai.webextract import (
        coerce_source_frontmatter,
        extract_web_content,
        is_native_extraction_acceptable,
    )
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

    # Playwright stores browsers in these locations
    if sys.platform == "win32":
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


@dataclass
class PlaywrightFetchResult:
    """Result from Playwright fetch."""

    content: str
    title: str | None = None
    final_url: str | None = None
    screenshot_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedContext:
    """Cached browser context with expiration."""

    context: Any
    created_at: float
    last_used_at: float
    session_key: str


class PlaywrightRenderer:
    """Reusable Playwright renderer to avoid browser cold starts."""

    def __init__(self, proxy: str | None = None) -> None:
        self.proxy = proxy
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
                launch_options["proxy"] = {"server": self.proxy}

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
        # X Articles are login-walled for anonymous visitors — the browser
        # render (launch + navigate + fixed post-load wait + auto-scroll)
        # is guaranteed wasted work, so skip it and go straight to the
        # FxTwitter/oEmbed enricher (mirrors defuddle's
        # canExtractAsync()+prefersAsync() early exit). Screenshots need an
        # actual page render, so they keep the normal path.
        needs_screenshot = bool(
            screenshot_config and getattr(screenshot_config, "enabled", True)
        )
        if _is_x_article_url(url) and not needs_screenshot:
            (
                enriched_md,
                overrides,
                enricher_source,
            ) = await self._try_enricher_fallback_async(url, remote_consent)
            if enriched_md:
                # Build source_frontmatter directly from the enriched
                # content — cli/processors/url.py reads this specific key
                # for the output YAML frontmatter. Computing word_count
                # here (rather than skipping it) also avoids the stale
                # values the slow path used to produce: it ran native
                # webextract against the discarded login-wall page first,
                # so word_count/content_profile described that ~2-word
                # page rather than the actual enriched article.
                from markitai.webextract.utils import count_words

                source_frontmatter: dict[str, Any] = dict(overrides or {})
                source_frontmatter["word_count"] = count_words(enriched_md)
                # x_article gets the same profile as regular tweets in the
                # native pipeline (see webextract/pipeline.py's
                # _EXTRACTOR_CONTENT_PROFILES).
                source_frontmatter.setdefault("content_profile", "social_post")
                return PlaywrightFetchResult(
                    content=enriched_md,
                    title=source_frontmatter.get("title"),
                    final_url=url,
                    screenshot_path=None,
                    metadata={
                        "_enricher_source": enricher_source,
                        "source_frontmatter": source_frontmatter,
                    },
                )
            # Enricher blocked (e.g. remote_consent="never") or failed —
            # fall through to the normal browser-render path below so the
            # user still gets the (login-wall) page rather than nothing.

        # Build context options from advanced config
        ctx_options: dict[str, Any] = {}
        if extra_http_headers:
            ctx_options["extra_http_headers"] = extra_http_headers
        if user_agent:
            ctx_options["user_agent"] = user_agent
        if http_credentials:
            ctx_options["http_credentials"] = http_credentials

        if self._session_cache_enabled and persist_context and session_key:
            context = await self._get_or_create_cached_context(session_key, ctx_options)
            should_close_context = False
        else:
            browser = await self._ensure_browser()
            context = await browser.new_context(**ctx_options)
            should_close_context = True

        page = None
        try:
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

            await page.goto(url, timeout=timeout, wait_until=wait_until)

            # Precise element waiting (preferred) or time-based fallback
            if wait_for_selector:
                try:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=min(timeout, 10000)
                    )
                except Exception as e:
                    logger.debug(
                        f"wait_for_selector '{wait_for_selector}' timed out: {e}"
                    )
                # Short stabilization wait after selector found
                if extra_wait_ms > 0:
                    await asyncio.sleep(extra_wait_ms / 1000)
            elif extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)

            # Auto-scroll to trigger lazy-loaded content
            if not skip_auto_scroll:
                try:
                    scroll_script = _build_auto_scroll_script()
                    await page.evaluate(scroll_script)
                    await asyncio.sleep(DEFAULT_PLAYWRIGHT_POST_SCROLL_DELAY_MS / 1000)
                except Exception as e:
                    logger.debug(f"Auto-scroll failed (non-critical): {e}")

            # Browser DOM normalize: flatten live shadow roots before extraction
            try:
                shadow_script = _build_shadow_dom_normalize_script()
                await page.evaluate(shadow_script)
            except Exception as e:
                logger.debug(f"Shadow DOM normalize failed (non-critical): {e}")

            # DOM cleanup: remove noise elements before extraction
            try:
                cleanup_script = _build_dom_cleanup_script(url=url)
                await page.evaluate(cleanup_script)
            except Exception as e:
                logger.debug(f"DOM cleanup failed (non-critical): {e}")

            title = await page.title()
            final_url = page.url
            html_content = await page.content()
            metadata: dict[str, Any] = {"renderer": "playwright", "wait_for": wait_for}

            # Try native webextract FIRST to avoid redundant HTML→Markdown
            # conversion. Only fall back to _html_to_markdown if webextract
            # is unavailable or produces insufficient quality.
            markdown_content = ""
            used_native_webextract = False
            if extract_web_content is not None:
                # If extract_web_content is available, the other webextract functions are too
                assert is_native_extraction_acceptable is not None
                assert coerce_source_frontmatter is not None
                try:
                    # CPU-bound (BeautifulSoup parsing + deepcopies); run in a
                    # thread to avoid blocking the event loop
                    extracted = await asyncio.to_thread(
                        extract_web_content, html_content, url
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
            if not markdown_content:
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
                if not markdown_content:
                    markdown_content = _html_to_markdown(html_content)
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
                if needs_enricher and is_x_url and "/" in url:
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

            if not used_native_webextract and _is_content_incomplete(markdown_content):
                try:
                    rendered_text = await page.inner_text("body")
                    if rendered_text and len(rendered_text.strip()) > len(
                        markdown_content.strip()
                    ):
                        markdown_content = _format_inner_text(rendered_text)
                except Exception as e:
                    logger.debug(
                        "[Playwright] Failed to extract inner_text fallback: {}", e
                    )

            screenshot_path = None
            if screenshot_config and output_dir:
                enabled = getattr(screenshot_config, "enabled", True)
                if enabled:
                    screenshot_path = await _capture_screenshot(
                        page, screenshot_config, output_dir, url
                    )

            return PlaywrightFetchResult(
                content=markdown_content,
                title=title,
                final_url=final_url,
                screenshot_path=screenshot_path,
                metadata=metadata,
            )
        finally:
            if should_close_context:
                await context.close()
            elif page is not None:
                # In persistent mode, close the page but keep the context
                await page.close()

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

    async def _try_enricher_fallback_async(
        self,
        url: str,
        remote_consent: str,
    ) -> tuple[str, dict[str, Any] | None, str]:
        """Try oEmbed/FxTwitter enrichment, returning (markdown, overrides, source).

        Returns ("", None, "") on failure.  ``source`` is one of
        ``"fxtwitter"``, ``"oembed"``, or ``""`` (no enrichment).

        Unlike defuddle/jina/cloudflare, this never prompts for consent:
        ``should_run()`` first restricts candidates to x.com/twitter.com
        status/article URLs, then the shared full-URL and DNS privacy policy
        verifies that the URL is public. "ask" behaves like "always" here.
        Explicit opt-outs
        (``remote_consent="never"``, ``MARKITAI_NO_REMOTE_FETCH``) are
        still honored.
        """
        from markitai.fetch import (
            _env_no_remote_fetch,
            disclose_remote_use,
            peek_cached_remote_consent,
        )
        from markitai.fetch_policy import assess_url_for_remote
        from markitai.webextract.enrichers.base import EnrichmentPolicy
        from markitai.webextract.enrichers.x_oembed import XOEmbedEnricher

        if remote_consent == "never" or _env_no_remote_fetch():
            return "", None, ""
        # ``ask`` intentionally does not open a second prompt on this narrowly
        # scoped public X/Twitter path. It must still honor a process-wide No
        # decision already made by the complete-service consent prompt.
        if peek_cached_remote_consent() is False:
            return "", None, ""

        enricher = XOEmbedEnricher()
        policy = EnrichmentPolicy(allow_network=True, allow_async=True)
        if not enricher.should_run(url, policy):
            return "", None, ""
        if not (await assess_url_for_remote(url)).allowed:
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
        from markitai.webextract.pipeline import _create_markitdown

        md_instance = _create_markitdown()
        enriched_md = render_markdown(resolved.content_html, md_instance=md_instance)
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
    async with PlaywrightRenderer(proxy=proxy) as standalone_renderer:
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
) -> Path | None:
    """Capture page screenshot.

    Args:
        page: Playwright page object
        config: Screenshot configuration
        output_dir: Output directory
        url: Original URL (for filename)

    Returns:
        Path to screenshot file, or None on failure
    """
    from markitai.fetch import _compress_screenshot, _url_to_screenshot_filename

    try:
        # Generate filename using the same logic as fetch.py
        filename = _url_to_screenshot_filename(url)

        output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = output_dir / filename

        # Get settings from config
        full_page = getattr(config, "full_page", True)
        quality = getattr(config, "quality", 85)
        max_height = getattr(config, "max_height", 10000)

        await page.screenshot(
            path=str(screenshot_path),
            full_page=full_page,
            type="jpeg",
            quality=quality,
        )

        # Compress and resize if needed (handles max_height limit)
        _compress_screenshot(screenshot_path, quality=quality, max_height=max_height)

        logger.debug(f"Screenshot saved: {screenshot_path}")
        return screenshot_path
    except Exception as e:
        logger.warning(f"Screenshot capture failed: {e}")
        return None
