from __future__ import annotations

import io
from dataclasses import asdict

from bs4 import BeautifulSoup, Tag

from markitai.webextract.constants import ADAPTIVE_RETRY_MIN_WORDS
from markitai.webextract.dom import parse_html
from markitai.webextract.extractors.registry import find_extractor
from markitai.webextract.metadata import extract_metadata
from markitai.webextract.sanitize import sanitize_tag_tree
from markitai.webextract.schema import (
    extract_schema_text,
    find_smallest_matching_element,
    should_use_schema_fallback,
)
from markitai.webextract.scoring import select_best_candidate
from markitai.webextract.standardize import standardize_content
from markitai.webextract.types import ExtractedWebContent


def extract_web_content(html: str, url: str) -> ExtractedWebContent:
    """Extract the primary content from raw HTML.

    Args:
        html: Raw HTML content.
        url: Source URL.

    Returns:
        Extracted web content with cleaned HTML and derived Markdown.
    """

    soup = parse_html(html)
    extractor = find_extractor(url)
    root = _pick_root(soup, extractor)
    metadata = extract_metadata(soup, url)
    diagnostics: dict[str, object] = {
        "extractor": extractor.name if extractor is not None else "generic",
        "schema_fallback_used": False,
        "adaptive_retry_used": False,
        "removed_partial_selectors": False,
    }

    root = _maybe_apply_schema_fallback(soup, root, diagnostics)
    if isinstance(root, Tag):
        standardize_content(root, title=metadata.title, base_url=url)
    sanitize_tag_tree(root)
    clean_html = str(root)

    # Create a shared MarkItDown instance to avoid repeated construction
    md_instance = _create_markitdown()
    markdown = _html_fragment_to_markdown(clean_html, md_instance)

    if len(markdown.split()) <= ADAPTIVE_RETRY_MIN_WORDS and not diagnostics.get(
        "schema_fallback_used"
    ):
        # Adaptive retry: broaden extraction by falling back to <body>
        retry_root = _retry_with_broader_root(soup, root)
        if retry_root is not None and retry_root is not root:
            if isinstance(retry_root, Tag):
                standardize_content(retry_root, title=metadata.title, base_url=url)
            sanitize_tag_tree(retry_root)
            retry_html = str(retry_root)
            retry_markdown = _html_fragment_to_markdown(retry_html, md_instance)
            if len(retry_markdown.split()) > len(markdown.split()):
                clean_html = retry_html
                markdown = retry_markdown
                diagnostics["adaptive_retry_used"] = True

    return ExtractedWebContent(
        clean_html=clean_html,
        markdown=markdown,
        metadata=metadata,
        word_count=len(markdown.split()),
        diagnostics={**diagnostics, "metadata": asdict(metadata)},
    )


def _pick_root(soup: BeautifulSoup, extractor: object | None) -> Tag | BeautifulSoup:
    if extractor is not None and hasattr(extractor, "extract_root"):
        root: Tag | None = extractor.extract_root(soup)  # type: ignore[union-attr]
        if root is not None:
            return root
    return select_best_candidate(soup) or soup.find("article") or soup.body or soup


def _maybe_apply_schema_fallback(
    soup: BeautifulSoup,
    root: Tag | BeautifulSoup,
    diagnostics: dict[str, object],
) -> Tag | BeautifulSoup:
    schema_text = extract_schema_text(soup)
    if schema_text:
        candidate = find_smallest_matching_element(soup, schema_text)
        if candidate is not None:
            candidate_text = " ".join(candidate.get_text(" ", strip=True).split())
            normalized_schema = " ".join(schema_text.split())
            extracted_text = root.get_text(" ", strip=True)
            if candidate is not root or candidate_text == normalized_schema:
                diagnostics["schema_fallback_used"] = True
                return candidate
            if should_use_schema_fallback(schema_text, extracted_text):
                diagnostics["schema_fallback_used"] = True
                return candidate
        if should_use_schema_fallback(schema_text, root.get_text(" ", strip=True)):
            diagnostics["schema_fallback_used"] = True
    return root


def _retry_with_broader_root(
    soup: BeautifulSoup,
    original_root: Tag | BeautifulSoup,
) -> Tag | BeautifulSoup | None:
    """Attempt a broader extraction when initial root yielded too few words.

    Strategy: fall back to ``<body>`` (or the full soup when ``<body>`` is
    absent).  This captures content that sits outside the original scored
    candidate—e.g. paragraphs placed directly under ``<body>``.

    Returns:
        A broader root element, or *None* if no better candidate exists.
    """
    body = soup.body
    if body is None:
        return None
    # Avoid returning the same element the caller already tried.
    if body is original_root:
        return None
    return body


def _candidate_count(soup: BeautifulSoup) -> int:
    return len(soup.find_all(["article", "main", "section", "div"])) or 1


def _create_markitdown() -> object:
    """Create a MarkItDown instance for HTML-to-Markdown conversion.

    Returns:
        A MarkItDown instance that can be reused across multiple conversions.
    """
    from markitdown import MarkItDown

    return MarkItDown()


def _html_fragment_to_markdown(html: str, md: object | None = None) -> str:
    """Convert an HTML fragment to Markdown.

    Args:
        html: HTML content to convert.
        md: Optional pre-created MarkItDown instance. If None, creates a new one.

    Returns:
        Markdown text.
    """
    from markitdown import StreamInfo

    if md is None:
        md = _create_markitdown()

    stream = io.BytesIO(html.encode("utf-8"))
    result = md.convert_stream(  # type: ignore[union-attr]
        stream,
        file_extension=".html",
        stream_info=StreamInfo(
            mimetype="text/html",
            extension=".html",
            charset="utf-8",
        ),
    )
    return result.text_content if result and result.text_content else ""
