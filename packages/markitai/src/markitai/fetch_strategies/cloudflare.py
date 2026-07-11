"""Cloudflare Browser Rendering API fetch strategy (cloud browser)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.fetch_session import get_default_session
from markitai.fetch_strategies._shared import (
    _build_native_fetch_result,
    _markitdown_convert_bytes,
)
from markitai.fetch_support import _detect_proxy
from markitai.fetch_types import FetchError, FetchResult, FetchStrategy
from markitai.utils.text import format_error_message

if TYPE_CHECKING:
    from markitai.fetch_strategies import StrategyContext


def get_cf_semaphore() -> asyncio.Semaphore:
    """Get or create the CF BR rate-limiting semaphore.

    Lazily initialized to avoid binding to a wrong event loop at import time.
    CF Free plan allows 2 concurrent browser instances.
    """
    return get_default_session().get_cf_semaphore()


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
    """Fetch URL using Cloudflare Browser Rendering /content API.

    Fetches the rendered HTML (not CF's server-side /markdown conversion) so
    the page goes through the same native webextract pipeline as every other
    strategy — site-specific extractors included. CF's /markdown endpoint
    hands back already-converted generic markdown, which bypasses
    webextract entirely and lets site chrome through (share buttons,
    stat counters, back-to-top links).

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
        FetchResult with markdown extracted from the rendered HTML
        (native webextract preferred, markitdown fallback)

    Raises:
        FetchError: If fetch fails or credentials missing
    """
    import httpx

    if not api_token or not account_id:
        from markitai.utils.guidance import cloudflare_credentials_error

        raise FetchError(cloudflare_credentials_error())

    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/browser-rendering/content"
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

    html_content = ""
    browser_ms_used: str | None = None

    try:
        # Use _detect_proxy() not get_proxy_for_url(endpoint), because
        # api.cloudflare.com is not in proxy_domains but may still need
        # proxy in restricted network environments.
        proxy_url = _detect_proxy()
        proxy_config = proxy_url if proxy_url else None

        async with (
            get_cf_semaphore(),
            httpx.AsyncClient(
                timeout=max(timeout / 1000 + 10, 60.0),
                proxy=proxy_config,
            ) as client,
        ):
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

                html_content = data.get("result", "")
                browser_ms_used = response.headers.get("X-Browser-Ms-Used")
                # Convert outside the semaphore/client scope below, so the
                # CF concurrency slot is freed as soon as the fetch is done
                break
    except FetchError:
        raise
    except Exception as e:
        raise FetchError(
            f"Cloudflare BR fetch failed: {format_error_message(e)}"
        ) from e

    if not html_content:
        raise FetchError(f"CF BR returned no content: {url}")

    base_metadata: dict[str, Any] = {"browser_ms_used": browser_ms_used}

    # Same native-extraction path as static/playwright HTML, so site-specific
    # extractors (webextract/extractors/registry.py) apply to CF-fetched pages
    native_result = await _build_native_fetch_result(
        html=html_content,
        url=url,
        final_url=url,
        strategy_used="cloudflare",
        base_metadata=base_metadata,
    )
    if native_result is not None:
        return native_result

    try:
        loop = asyncio.get_running_loop()
        text_content, title = await loop.run_in_executor(
            None, _markitdown_convert_bytes, html_content.encode("utf-8"), ".html"
        )
    except Exception as e:
        raise FetchError(
            f"Cloudflare BR content conversion failed: {format_error_message(e)}"
        ) from e

    if not text_content:
        raise FetchError(f"No content extracted from URL: {url}")

    return FetchResult(
        content=text_content,
        strategy_used="cloudflare",
        title=title,
        url=url,
        final_url=url,
        metadata={**base_metadata, "converter": "markitdown"},
    )


class CloudflareRunner:
    """Cloudflare Browser Rendering API fetch."""

    strategy: FetchStrategy = FetchStrategy.CLOUDFLARE
    requires_remote_consent: bool = True

    def unavailable_reason(self, ctx: StrategyContext) -> str | None:
        if ctx.explicit:
            # Explicit dispatch reports missing credentials via
            # fetch_with_cloudflare's actionable error instead of skipping.
            return None
        cf = getattr(ctx.config, "cloudflare", None)
        if cf is None:
            return ""  # historical silent skip: no error recorded
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
            return "credentials not configured"
        return None

    async def fetch(self, url: str, ctx: StrategyContext) -> FetchResult:
        cf = ctx.config.cloudflare
        if ctx.explicit:
            token = cf.get_resolved_api_token(strict=True)
            acct = cf.get_resolved_account_id(strict=True)
            # Missing credentials are reported by fetch_with_cloudflare with
            # an actionable error (see markitai.utils.guidance).
        else:
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
        return await fetch_with_cloudflare(
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
