"""Static HTTP fetch strategy (httpx/curl-cffi, no browser).

Fetches via the shared static HTTP client, prefers native webextract
HTML->markdown conversion, and supports HTTP conditional requests
(ETag/If-Modified-Since) for cache revalidation.
"""

from __future__ import annotations

import asyncio
import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.fetch_http import get_static_http_client
from markitai.fetch_strategies._shared import (
    _build_native_fetch_result,
    _convert_document_bytes,
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


_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})

# Content-Type -> file suffix. The suffix picks the converter, the same way
# a local file's extension does.
_MIME_SUFFIXES: dict[str, str] = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "application/pdf": ".pdf",
    "application/x-pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        ".docx"
    ),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": (
        ".pptx"
    ),
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/epub+zip": ".epub",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "message/rfc822": ".eml",
    "application/vnd.ms-outlook": ".msg",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "application/json": ".json",
    "text/json": ".json",
    "application/x-ipynb+json": ".ipynb",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}

# Types that say nothing specific about the format: the file name or the
# bytes decide (text/plain is how many servers label a raw .md or .csv).
_GENERIC_MIME_TYPES = frozenset(
    {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/download",
        "application/force-download",
        "application/x-download",
        "application/unknown",
        "application/zip",
        "application/x-zip-compressed",
        "text/plain",
    }
)

# Binary documents markitai converts with its own local-file converters,
# instead of handing them to markitdown.
_DOCUMENT_SUFFIXES = frozenset(
    {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".odt",
        ".ods",
        ".epub",
        ".rtf",
        ".eml",
        ".msg",
    }
)

# Text formats: decoded with the response charset and re-encoded as UTF-8
# before conversion, so a GBK CSV or a Shift_JIS text file is not mojibake.
_TEXT_SUFFIXES = frozenset(
    {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".ipynb", ".xml"}
)


def _mime_type(content_type: str | None) -> str:
    """Return the lowercase media type of a Content-Type header value."""
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _is_html_content_type(content_type: str | None) -> bool:
    """HTML, or no declared type at all (browsers sniff those as HTML)."""
    mime = _mime_type(content_type)
    return not mime or mime in {"text/html", "application/xhtml+xml"}


def _content_disposition_filename(value: str | None) -> str | None:
    """Return the file name a Content-Disposition header suggests."""
    if not value:
        return None
    from email.message import Message

    message = Message()
    message["content-disposition"] = value
    try:
        filename = message.get_filename()
    except Exception:
        return None
    if not filename:
        return None
    return Path(str(filename).replace("\\", "/")).name or None


#: Upper bound for an OpenDocument/EPUB ``mimetype`` entry read while sniffing.
_MIMETYPE_MAX_BYTES = 256


def _sniff_document_suffix(body: bytes) -> str | None:
    """Recognise common document containers by their leading bytes."""
    if body.startswith(b"%PDF-"):
        return ".pdf"
    if body.startswith(b"{\\rtf"):
        return ".rtf"
    if body.startswith(b"PK\x03\x04"):
        import io
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                names = set(archive.namelist())
                if "word/document.xml" in names:
                    return ".docx"
                if "xl/workbook.xml" in names:
                    return ".xlsx"
                if "ppt/presentation.xml" in names:
                    return ".pptx"
                if "mimetype" in names:
                    # A real mimetype entry is a few dozen bytes. Never inflate
                    # more than that: the header's size can lie, and a zip
                    # bomb must not blow up memory while only sniffing.
                    if archive.getinfo("mimetype").file_size > _MIMETYPE_MAX_BYTES:
                        return None
                    with archive.open("mimetype") as entry:
                        raw = entry.read(_MIMETYPE_MAX_BYTES + 1)
                    if len(raw) > _MIMETYPE_MAX_BYTES:
                        return None
                    kind = raw.decode("ascii", "ignore")
                    return _MIME_SUFFIXES.get(kind.strip().lower())
        except Exception:
            return None
    return None


def _url_suffix(url: str | None) -> str:
    """Return the lowercase extension of a URL's path ('' when none)."""
    from urllib.parse import unquote, urlparse

    if not url:
        return ""
    return Path(unquote(urlparse(url).path)).suffix.lower()


def _url_filename(url: str | None) -> str | None:
    """Return the last path segment of a URL, if it has one."""
    from urllib.parse import unquote, urlparse

    if not url:
        return None
    return Path(unquote(urlparse(url).path)).name or None


def _response_suffix(
    content_type: str | None,
    content_disposition: str | None,
    urls: tuple[str | None, ...],
    body: bytes,
) -> tuple[str, str | None]:
    """Choose the converter suffix for a response body.

    Order: a specific Content-Type, the Content-Disposition file name, the
    (final, then requested) URL path, the body's magic bytes, the generic
    type's own default (``text/plain`` -> ``.txt``), then HTML.

    Returns:
        (suffix, file name suggested by Content-Disposition or None)
    """
    from markitai.converter.base import EXTENSION_MAP

    mime = _mime_type(content_type)
    filename = _content_disposition_filename(content_disposition)
    if mime not in _GENERIC_MIME_TYPES and mime in _MIME_SUFFIXES:
        return _MIME_SUFFIXES[mime], filename

    known = set(EXTENSION_MAP) | set(_MIME_SUFFIXES.values())
    candidates = [Path(filename).suffix.lower()] if filename else []
    candidates.extend(_url_suffix(url) for url in urls)
    for candidate in candidates:
        if candidate in known:
            return candidate, filename

    sniffed = _sniff_document_suffix(body)
    if sniffed:
        return sniffed, filename
    if mime in _MIME_SUFFIXES:
        return _MIME_SUFFIXES[mime], filename
    return ".html", filename


def _get_response_text(response: Any) -> str:
    """Decode a static HTTP response body into text.

    BOM, then the Content-Type charset, then (HTML only) the ``<meta>``
    prescan, then detection; every label is widened to the superset a
    browser decodes it with. See :mod:`markitai.utils.charset`.
    """
    from markitai.utils.charset import decode_body

    content = getattr(response, "content", b"")
    if isinstance(content, bytes | bytearray):
        content_type = _get_header_value(
            getattr(response, "headers", {}),
            "content-type",
            "content_type",
        )
        text, _ = decode_body(
            bytes(content),
            content_type,
            html=_is_html_content_type(content_type),
        )
        return text

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
        final_url = str(response.url) or url
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
                final_url=final_url,
                metadata={
                    "converter": "server-markdown",
                    "conditional": True,
                    "token_hint": int(token_hint) if token_hint else None,
                    "client": client.name,
                    # Markdown for Agents (token hint present) is a converted
                    # HTML page and can still be a JavaScript shell; a plain
                    # .md file served as text/markdown is a document.
                    "content_kind": "html" if token_hint else "text",
                    "content_type": "text/markdown",
                },
            )

            return ConditionalFetchResult(
                result=fetch_result,
                not_modified=False,
                etag=response_etag,
                last_modified=response_last_modified,
            )

        # The suffix picks the converter, exactly like a local file's
        # extension: Content-Type, then Content-Disposition, then the URL,
        # then the body's magic bytes.
        content_type = content_type_header or ""
        body = bytes(response.content)
        suffix, suggested_name = _response_suffix(
            content_type,
            _get_header_value(
                response.headers, "content-disposition", "Content-Disposition"
            ),
            (final_url, url),
            body,
        )
        mime = _mime_type(content_type) or None

        if suffix in _HTML_SUFFIXES:
            response_text = _get_response_text(response)
            native_result = await _build_native_fetch_result(
                html=response_text,
                url=url,
                final_url=final_url,
                strategy_used="static",
                base_metadata={
                    "conditional": True,
                    "client": client.name,
                    "content_kind": "html",
                    "content_type": mime,
                },
            )
            if native_result is not None:
                return ConditionalFetchResult(
                    result=native_result,
                    not_modified=False,
                    etag=response_etag,
                    last_modified=response_last_modified,
                )
            # markitdown sniffs the charset from the bytes and would believe
            # a <meta charset> over the header; hand it the text as decoded
            # above, BOM-marked so the BOM wins over any <meta>.
            convert_bytes = b"\xef\xbb\xbf" + response_text.encode("utf-8")
            content_kind = "html"
        elif suffix in _TEXT_SUFFIXES:
            convert_bytes = _get_response_text(response).encode("utf-8")
            content_kind = "text"
        else:
            convert_bytes = body
            content_kind = "document" if suffix in _DOCUMENT_SUFFIXES else "binary"

        loop = asyncio.get_running_loop()
        if content_kind == "document":
            # Binary documents go to markitai's own converters (the PDF
            # converter for PDFs), the same path a downloaded file takes.
            try:
                text_content, title = await loop.run_in_executor(
                    None,
                    functools.partial(
                        _convert_document_bytes,
                        convert_bytes,
                        suffix,
                        filename=suggested_name or _url_filename(final_url),
                    ),
                )
            except Exception as e:
                raise FetchError(
                    f"Failed to convert {suffix} document from {url}: "
                    f"{format_error_message(e)}"
                ) from e
            converter_name = "markitai"
            title = title or _extract_markdown_title(text_content)
            if not title and suggested_name:
                title = Path(suggested_name).stem
        else:
            # Save response to temp file and run sync markitdown in executor
            # to avoid blocking the event loop
            text_content, title = await loop.run_in_executor(
                None, _markitdown_convert_bytes, convert_bytes, suffix
            )
            converter_name = "markitdown"

        if not text_content or not text_content.strip():
            raise FetchError(f"No content extracted from URL: {url}")

        fetch_result = FetchResult(
            content=text_content,
            strategy_used="static",
            title=title,
            url=url,
            final_url=final_url,
            metadata={
                "converter": converter_name,
                "conditional": True,
                "content_kind": content_kind,
                "content_type": mime,
                "document_suffix": suffix,
            },
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
