"""Jina Reader API fetch strategy (cloud-based, no local dependencies)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import DEFAULT_JINA_BASE_URL, DEFAULT_JINA_RPM
from markitai.fetch_session import _SlidingWindowRateLimiter, get_default_session
from markitai.fetch_types import (
    FetchError,
    FetchResult,
    FetchStrategy,
    JinaAPIError,
    JinaRateLimitError,
)
from markitai.utils.text import format_error_message

if TYPE_CHECKING:
    from markitai.fetch_strategies import StrategyContext


def _extract_jina_error_message(response: Any) -> str:
    """Extract a human-readable message from a Jina Reader error response.

    Jina returns JSON error envelopes (e.g. HTTP 451 with
    ``{"code": 451, "message": "Anonymous access to domain ... blocked"}``);
    surface the message instead of dumping truncated raw JSON.
    """
    text = str(getattr(response, "text", "") or "")
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text[:200]
    if isinstance(data, dict):
        for key in ("readableMessage", "message", "error"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:300]
    return text[:200]


def _get_jina_rate_limiter(rpm: int) -> _SlidingWindowRateLimiter:
    """Get or create the global Jina rate limiter."""
    return get_default_session().get_jina_rate_limiter(rpm)


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
    return get_default_session().get_jina_client(timeout, proxy)


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
            raise JinaAPIError(
                response.status_code, _extract_jina_error_message(response)
            )

        # Parse JSON response
        try:
            json_data = response.json()
        except json.JSONDecodeError as e:
            logger.warning(f"Jina API returned invalid JSON: {e}")
            raise FetchError(
                f"Jina Reader returned invalid JSON response: {format_error_message(e)}"
            )

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
        raise FetchError(f"Jina Reader fetch failed: {format_error_message(e)}")


class JinaRunner:
    """Jina Reader API fetch."""

    strategy: FetchStrategy = FetchStrategy.JINA
    requires_remote_consent: bool = True

    def unavailable_reason(self, ctx: StrategyContext) -> str | None:
        return None

    async def fetch(self, url: str, ctx: StrategyContext) -> FetchResult:
        api_key = ctx.config.jina.get_resolved_api_key()
        return await fetch_with_jina(
            url,
            api_key,
            ctx.config.jina.timeout,
            ctx.config.jina.rpm,
            no_cache=ctx.config.jina.no_cache,
            target_selector=ctx.config.jina.target_selector,
            wait_for_selector=ctx.config.jina.wait_for_selector,
        )
