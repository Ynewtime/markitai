"""Static HTTP fetch strategy (httpx/curl-cffi, no browser).

Fetches via the shared static HTTP client, prefers native webextract
HTML->markdown conversion, and supports HTTP conditional requests
(ETag/If-Modified-Since) for cache revalidation.
"""

from __future__ import annotations

import asyncio
import codecs
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.fetch_http import get_static_http_client
from markitai.fetch_strategies._shared import (
    _build_native_fetch_result,
    _markitdown_convert_bytes,
)
from markitai.fetch_support import _detect_proxy
from markitai.fetch_types import (
    ConditionalFetchResult,
    FetchError,
    FetchResult,
    FetchStrategy,
)
from markitai.utils.text import format_error_message

if TYPE_CHECKING:
    from markitai.fetch_strategies import StrategyContext


def _extract_markdown_title(content: str) -> str | None:
    """Extract the first H1 title from markdown content."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1) if match else None


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
    result = cond_result.result
    # Stash HTTP validators in metadata so AUTO dispatch can store them
    # in the cache for future conditional revalidation (popped by
    # _dispatch_strategy before caching).
    if cond_result.etag:
        result.metadata["_markitai_etag"] = cond_result.etag
    if cond_result.last_modified:
        result.metadata["_markitai_last_modified"] = cond_result.last_modified
    return result


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
            native_result = await _build_native_fetch_result(
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
        loop = asyncio.get_running_loop()
        text_content, title = await loop.run_in_executor(
            None, _markitdown_convert_bytes, response.content, suffix
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
        raise FetchError(
            f"Failed to fetch URL with conditional request: {format_error_message(e)}"
        )


class StaticRunner:
    """Static HTTP fetch (conditional variant on explicit dispatch)."""

    strategy: FetchStrategy = FetchStrategy.STATIC
    requires_remote_consent: bool = False

    def unavailable_reason(self, ctx: StrategyContext) -> str | None:
        return None

    async def fetch(self, url: str, ctx: StrategyContext) -> FetchResult:
        if not ctx.explicit:
            return await fetch_with_static(url)

        # For fresh explicit fetch, use conditional to capture validators
        cond_result = await fetch_with_static_conditional(
            url, ctx.cached_etag, ctx.cached_last_modified
        )
        if cond_result.result is None:
            raise FetchError(f"No content from conditional fetch: {url}")
        result = cond_result.result
        # Stash HTTP validators in metadata (same convention as
        # fetch_with_static) for _dispatch_strategy to pop into the
        # cache-validators channel.
        if cond_result.etag:
            result.metadata["_markitai_etag"] = cond_result.etag
        if cond_result.last_modified:
            result.metadata["_markitai_last_modified"] = cond_result.last_modified
        return result
