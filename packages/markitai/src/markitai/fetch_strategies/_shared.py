"""Helpers shared by more than one fetch strategy implementation.

Native webextract HTML->markdown conversion and the markitdown temp-file
fallback are used by both the static and cloudflare strategies, so they
live here instead of in either strategy module.

Test patch anchors: strategy modules bind these names into their own
namespaces via ``from ... import``, so patches must target the consuming
module (e.g. ``markitai.fetch_strategies.static._markitdown_convert_bytes``)
— except ``extract_web_content``/``_get_markitdown``, which are looked up
in *this* module by ``_build_native_fetch_result``/``_markitdown_convert_bytes``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.fetch_session import get_default_session
from markitai.fetch_types import FetchResult

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


def _get_markitdown() -> Any:
    """Get or create the shared MarkItDown instance.

    Reusing a single instance avoids repeated initialization overhead.
    Includes Accept header for CF Markdown for Agents content negotiation.
    """
    return get_default_session().get_markitdown()


async def _build_native_fetch_result(
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
    assert is_native_extraction_acceptable is not None
    assert coerce_source_frontmatter is not None

    try:
        # CPU-bound (BeautifulSoup parsing + deepcopies); run in a thread
        # to avoid blocking the event loop
        extracted = await asyncio.to_thread(extract_web_content, html, url)
    except Exception as exc:
        logger.debug(f"Native webextract failed, falling back to markitdown: {exc}")
        return None

    markdown = getattr(extracted, "markdown", "")
    diagnostics = dict(getattr(extracted, "diagnostics", {}) or {})

    if not is_native_extraction_acceptable(extracted):
        diagnostics.setdefault("fallback_reason", "native_output_too_short")
        return None

    # Prefer build_source_frontmatter (typed result with content_profile/word_count)
    # over coerce_source_frontmatter (metadata-only legacy fallback).
    if hasattr(extracted, "info") and getattr(extracted, "info", None) is not None:
        from markitai.webextract.frontmatter import build_source_frontmatter

        source_frontmatter = build_source_frontmatter(extracted)
    else:
        source_frontmatter = coerce_source_frontmatter(
            getattr(extracted, "metadata", None)
        )
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


def _markitdown_convert_bytes(
    content_bytes: bytes, suffix: str
) -> tuple[str, str | None]:
    """Write a temp file and convert with markitdown (CPU-bound).

    Returns:
        (text_content, title) tuple.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as f:
        f.write(content_bytes)
        temp_path = Path(f.name)
    try:
        md = _get_markitdown()
        md_result = md.convert(str(temp_path))
        return md_result.text_content or "", md_result.title
    finally:
        temp_path.unlink(missing_ok=True)
