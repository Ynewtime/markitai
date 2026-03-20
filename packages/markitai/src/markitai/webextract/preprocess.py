from __future__ import annotations

"""Raw HTML preprocessing before BeautifulSoup parsing.

This module provides pure string/regex operations applied to raw HTML
before it is handed to BeautifulSoup.  Keeping these transformations at the
string level is intentional: they need to run before the parser sees the HTML
so that the parser receives clean, normalised markup.

Phases (in order):
1. Declarative Shadow DOM flattening — ``<template shadowrootmode="open">``
2. ``<wbr>`` removal — word-break hints that pollute extracted text
3. Streamed / incomplete HTML normalisation — nothing to do yet; handled by
   tolerant parsers, but the hook is here for future use
4. ``<noscript>`` content promotion — for static HTML where the main body
   contains only script tags
"""

import re

# ---------------------------------------------------------------------------
# Phase 1: Declarative Shadow DOM flattening
# ---------------------------------------------------------------------------

# Matches <template shadowrootmode="...">...</template> (case-insensitive attr value)
_SHADOW_TEMPLATE_RE = re.compile(
    r'<template\s[^>]*shadowrootmode\s*=\s*["\']open["\'][^>]*>(.*?)</template>',
    re.IGNORECASE | re.DOTALL,
)


def _flatten_declarative_shadow_dom(html: str) -> str:
    """Replace ``<template shadowrootmode="open">`` with its inner content.

    Declarative Shadow DOM is invisible to static parsers because it lives
    inside a ``<template>`` element.  By replacing the wrapper with its
    content we make the text visible to downstream extraction.

    Args:
        html: Raw HTML string.

    Returns:
        HTML with declarative shadow roots flattened.
    """
    return _SHADOW_TEMPLATE_RE.sub(r"\1", html)


# ---------------------------------------------------------------------------
# Phase 2: <wbr> removal
# ---------------------------------------------------------------------------

_WBR_RE = re.compile(r"<wbr\s*/?>", re.IGNORECASE)


def _remove_wbr_tags(html: str) -> str:
    """Remove ``<wbr>`` and ``<wbr/>`` tags.

    These are optional line-break hints that break word-boundary detection
    when HTML is converted to plain text or Markdown.

    Args:
        html: Raw HTML string.

    Returns:
        HTML with ``<wbr>`` tags removed.
    """
    return _WBR_RE.sub("", html)


# ---------------------------------------------------------------------------
# Phase 3: Streamed / incomplete HTML (no-op hook for future use)
# ---------------------------------------------------------------------------


def _normalize_streamed_html(html: str) -> str:
    """Normalise streamed or incomplete HTML patterns.

    Currently a no-op — tolerant parsers (lxml / html.parser) already handle
    unclosed tags.  This hook exists so future normalisation can be added here
    without touching call sites.

    Args:
        html: Raw HTML string.

    Returns:
        HTML string (unchanged for now).
    """
    return html


# ---------------------------------------------------------------------------
# Phase 4: <noscript> promotion
# ---------------------------------------------------------------------------

# Detects a <body> that contains only script / noscript / whitespace
_BODY_SCRIPT_ONLY_RE = re.compile(
    r"<body[^>]*>((?:\s|<script[^>]*>.*?</script>|<noscript[^>]*>.*?</noscript>)*)</body>",
    re.IGNORECASE | re.DOTALL,
)
_HAS_REAL_CONTENT_RE = re.compile(
    r"<(?!script|noscript|style|meta|link|title|head|html|!)[a-z]",
    re.IGNORECASE,
)
_NOSCRIPT_CONTENT_RE = re.compile(
    r"<noscript[^>]*>(.*?)</noscript>",
    re.IGNORECASE | re.DOTALL,
)


def _promote_noscript_content(html: str) -> str:
    """Promote ``<noscript>`` content when the body is JS-dependent.

    When the document body contains only ``<script>`` elements (and the page
    requires JS to render) the useful content often lives in ``<noscript>``
    fallback blocks.  We promote that content so static extraction can read it.

    This transform only fires when the ``<body>`` has *no* real HTML elements
    outside of ``<script>`` and ``<noscript>`` tags.

    Args:
        html: Raw HTML string.

    Returns:
        HTML with ``<noscript>`` content promoted to the body when appropriate.
    """
    match = _BODY_SCRIPT_ONLY_RE.search(html)
    if match is None:
        return html

    body_inner = match.group(1)

    # Strip out script tags to see what remains
    without_scripts = re.sub(
        r"<script[^>]*>.*?</script>", "", body_inner, flags=re.IGNORECASE | re.DOTALL
    )
    # Strip out noscript tags temporarily
    without_noscript = re.sub(
        r"<noscript[^>]*>.*?</noscript>",
        "",
        without_scripts,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # If real HTML elements remain, don't promote
    if _HAS_REAL_CONTENT_RE.search(without_noscript):
        return html

    # Collect all noscript content to verify there is something to promote
    noscript_parts = _NOSCRIPT_CONTENT_RE.findall(body_inner)
    if not noscript_parts:
        return html

    # Replace <noscript>…</noscript> wrappers with their inner content
    return _NOSCRIPT_CONTENT_RE.sub(r"\1", html)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_html(html: str) -> str:
    """Apply all raw HTML preprocessing phases before BeautifulSoup parsing.

    Phases run in order:
    1. Declarative Shadow DOM flattening
    2. ``<wbr>`` removal
    3. Streamed HTML normalisation (no-op hook)
    4. ``<noscript>`` content promotion

    Args:
        html: Raw HTML string to preprocess.

    Returns:
        Preprocessed HTML string ready for BeautifulSoup parsing.
    """
    if not html or not html.strip():
        return html

    html = _flatten_declarative_shadow_dom(html)
    html = _remove_wbr_tags(html)
    html = _normalize_streamed_html(html)
    html = _promote_noscript_content(html)
    return html
