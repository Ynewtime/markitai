from __future__ import annotations

import copy
import io
from dataclasses import asdict

from bs4 import BeautifulSoup, Tag

from markitai.webextract.dom import parse_html
from markitai.webextract.extractors.registry import find_extractor
from markitai.webextract.markdown import (
    html_to_markdown,
    postprocess_markdown,
    preserve_figure_captions,
    render_markdown,
    resolve_srcset,
)
from markitai.webextract.metadata import extract_metadata
from markitai.webextract.quality import assess_native_markdown
from markitai.webextract.removals import apply_removals
from markitai.webextract.resolver import ResolvedPage, resolve_page
from markitai.webextract.sanitize import sanitize_tag_tree
from markitai.webextract.schema import (
    extract_schema_text,
    find_smallest_matching_element,
    should_use_schema_fallback,
)
from markitai.webextract.scoring import select_best_candidate
from markitai.webextract.standardize import standardize_content
from markitai.webextract.types import (
    ContentProfile,
    ExtractedWebContent,
    ExtractionInfo,
)
from markitai.webextract.utils import count_words

_EXTRACTOR_CONTENT_PROFILES: dict[str, ContentProfile] = {
    "x_tweet": ContentProfile.SOCIAL_POST,
    "x_article": ContentProfile.SOCIAL_POST,
    "github_thread": ContentProfile.DISCUSSION_ISSUE,
    "reddit_post": ContentProfile.DISCUSSION_THREAD,
    "hackernews_thread": ContentProfile.DISCUSSION_THREAD,
    "youtube_page": ContentProfile.RICH_MEDIA_PAGE,
}


def extract_web_content(html: str, url: str) -> ExtractedWebContent:
    """Extract the primary content from raw HTML.

    Args:
        html: Raw HTML content.
        url: Source URL.

    Returns:
        Extracted web content with cleaned HTML and derived Markdown.
    """
    from loguru import logger

    # Try resolver path first (structured extraction for known sites)
    resolved = resolve_page(html, url)
    if resolved is not None and (resolved.content_html or resolved.content_root):
        return _build_from_resolved(html, url, resolved)

    if resolved is not None:
        diag = resolved.diagnostics
        logger.debug(
            "[Webextract] Resolver matched but returned no content for {}: {}",
            url,
            diag,
        )

    # Generic pipeline path
    return _extract_generic(html, url)


def _build_from_resolved(
    html: str,
    url: str,
    resolved: ResolvedPage,
) -> ExtractedWebContent:
    """Build ExtractedWebContent from a resolved page.

    Applies the same standardization and sanitization as the generic path
    to ensure ``clean_html`` is a true canonical representation regardless
    of which extraction path produced it.

    Args:
        html: Raw HTML source (for metadata extraction).
        url: Source URL.
        resolved: The resolved page from a site-specific extractor.

    Returns:
        Fully populated ExtractedWebContent.
    """
    soup = parse_html(html)
    metadata = extract_metadata(soup, url)

    # Apply metadata overrides from the resolver
    for key, value in resolved.metadata_overrides.items():
        if hasattr(metadata, key):
            setattr(metadata, key, value)

    # Obtain content HTML — from content_html directly or by rendering content_root
    if resolved.content_html:
        content_html = resolved.content_html
    elif resolved.content_root is not None:
        content_html = str(resolved.content_root)
    else:
        content_html = ""

    # Apply the same standardization and sanitization as the generic path
    # so that clean_html is truly canonical (no unsanitized tags, resolved links)
    content_soup = BeautifulSoup(content_html, "html.parser")
    standardize_content(content_soup, title=metadata.title, base_url=url)
    sanitize_tag_tree(content_soup)
    content_html = str(content_soup)

    # Convert the sanitized content_html to markdown
    md_instance = _create_markitdown()
    markdown = render_markdown(content_html, md_instance=md_instance)

    word_count = count_words(markdown)

    # Determine content profile and extractor name from resolver diagnostics
    content_profile_str = resolved.diagnostics.get(
        "content_profile", ContentProfile.SOCIAL_POST.value
    )
    try:
        content_profile = ContentProfile(content_profile_str)
    except ValueError:
        content_profile = ContentProfile.SOCIAL_POST
    extractor_name = resolved.diagnostics.get("extractor_name", "resolved")

    info = ExtractionInfo(
        content_profile=content_profile,
        extractor_name=str(extractor_name),
        word_count=word_count,
    )

    quality = assess_native_markdown(markdown, profile=content_profile.value)

    diagnostics: dict[str, object] = {
        "extractor": "resolver",
        "resolver_diagnostics": resolved.diagnostics,
        "schema_fallback_used": False,
        "adaptive_retry_used": False,
        "metadata": asdict(metadata),
    }

    return ExtractedWebContent(
        clean_html=content_html,
        markdown=markdown,
        metadata=metadata,
        word_count=word_count,
        info=info,
        quality=quality,
        semantic=resolved.semantic,
        diagnostics=diagnostics,
    )


class _ExtractionContext:
    """Cache expensive computations across retry levels.

    Note on mutation: Level 1 extraction operates directly on
    ``original_soup`` (no copy), so removals/standardization mutate it
    in place.  Subsequent retry levels call ``fresh_soup_and_root()``
    which deep-copies the (now-mutated) soup.  This is correct because
    ``_pick_root`` re-selects the content root on each copy, and the
    mutations from Level 1 (decomposed noise elements) are desirable —
    they won't reappear in copies.
    """

    def __init__(self, html: str, url: str) -> None:
        self.raw_html = html
        self.url = url
        self.original_soup = parse_html(html)
        self.metadata = extract_metadata(self.original_soup, url)
        self.md_instance = _create_markitdown()

    def fresh_soup_and_root(
        self, extractor: object | None, diagnostics: dict[str, object]
    ) -> tuple[BeautifulSoup, Tag | BeautifulSoup]:
        """Return a fresh deep-copy of the parsed soup with root selected."""
        soup = copy.deepcopy(self.original_soup)
        root = _pick_root(soup, extractor)
        root = _maybe_apply_schema_fallback(soup, root, diagnostics)
        return soup, root


def _extract_generic(html: str, url: str) -> ExtractedWebContent:
    """Run the generic extraction pipeline (no resolver match).

    Args:
        html: Raw HTML content.
        url: Source URL.

    Returns:
        Extracted web content with cleaned HTML and derived Markdown.
    """
    ctx = _ExtractionContext(html, url)

    # Prune mobile-hidden elements before scoring.
    # This mutates original_soup but that's OK — mobile style pruning
    # should persist across all retry levels.
    from markitai.webextract.mobile_styles import apply_mobile_style_pruning

    apply_mobile_style_pruning(ctx.original_soup)

    extractor = find_extractor(url)
    diagnostics: dict[str, object] = {
        "extractor": extractor.name if extractor is not None else "generic",
        "schema_fallback_used": False,
        "adaptive_retry_used": False,
        "removed_partial_selectors": False,
    }

    # Use fresh_soup_and_root for Level 1 too — _extract_once mutates
    # root in place (removals, standardize, sanitize), so operating on
    # original_soup directly would corrupt it for subsequent retries.
    _, root = ctx.fresh_soup_and_root(extractor, diagnostics)

    # Multi-level extraction with adaptive retry
    result = _extract_with_retry(
        ctx,
        root,
        diagnostics,
        extractor=extractor,
    )

    clean_html = result[0]
    markdown = result[1]
    word_count = count_words(markdown)

    extractor_name = extractor.name if extractor is not None else "generic"
    content_profile = _EXTRACTOR_CONTENT_PROFILES.get(
        extractor_name, ContentProfile.GENERIC_ARTICLE
    )

    info = ExtractionInfo(
        content_profile=content_profile,
        extractor_name=extractor_name,
        word_count=word_count,
    )

    quality = assess_native_markdown(markdown, profile=content_profile.value)

    return ExtractedWebContent(
        clean_html=clean_html,
        markdown=markdown,
        metadata=ctx.metadata,
        word_count=word_count,
        info=info,
        quality=quality,
        semantic=None,
        diagnostics={**diagnostics, "metadata": asdict(ctx.metadata)},
    )


_RETRY_SPARSE_THRESHOLD = 50
_RETRY_VERY_SPARSE_THRESHOLD = 20


def _extract_once(
    root: Tag | BeautifulSoup,
    metadata: object,
    md_instance: object,
    url: str,
    *,
    use_partial_selectors: bool = True,
    use_hidden_removal: bool = True,
    use_scoring: bool = True,
) -> tuple[str, str, dict[str, int]]:
    """Run extraction pipeline once and return (clean_html, markdown, removal_stats)."""
    title = getattr(metadata, "title", None)
    removal_stats: dict[str, int] = {}
    if isinstance(root, Tag):
        removal_stats = apply_removals(
            root,
            use_partial_selectors=use_partial_selectors,
            use_hidden_removal=use_hidden_removal,
            use_scoring=use_scoring,
        )
    if isinstance(root, Tag):
        standardize_content(root, title=title, base_url=url)
    sanitize_tag_tree(root)

    # Apply markdown preprocessing directly on the parsed Tag to avoid
    # the redundant BeautifulSoup re-parse that render_markdown() performs.
    resolve_srcset(root)
    # canonicalize_embeds is a no-op here: sanitize_tag_tree already
    # stripped all <iframe> elements.
    preserve_figure_captions(root)

    clean_html = str(root)
    markdown = html_to_markdown(clean_html, md_instance)
    markdown = postprocess_markdown(markdown)
    return clean_html, markdown, removal_stats


def _extract_with_retry(
    ctx: _ExtractionContext,
    root: Tag | BeautifulSoup,
    diagnostics: dict[str, object],
    *,
    extractor: object | None,
) -> tuple[str, str]:
    """Multi-level adaptive retry extraction.

    Level 1: Full removal pipeline
    Level 2: Disable partial selectors (may be too aggressive)
    Level 3: Disable hidden element removal
    Level 4: Disable all removals (listing page)
    Fallback: Broaden to <body>
    """
    url = getattr(ctx.metadata, "canonical_url", "") or ""
    use_scoring = extractor is None

    # Level 1: Full pipeline
    clean_html, markdown, removal_stats = _extract_once(
        root,
        ctx.metadata,
        ctx.md_instance,
        url,
        use_scoring=use_scoring,
    )
    diagnostics["removal_stats"] = removal_stats
    word_count = count_words(markdown)

    # Skip retry if schema fallback already found a good match
    schema_used = diagnostics.get("schema_fallback_used", False)
    if word_count >= _RETRY_SPARSE_THRESHOLD or schema_used:
        return clean_html, markdown

    # Level 2: Retry without partial selectors
    _soup2, root2 = ctx.fresh_soup_and_root(extractor, diagnostics)
    clean2, md2, _ = _extract_once(
        root2,
        ctx.metadata,
        ctx.md_instance,
        url,
        use_partial_selectors=False,
        use_scoring=use_scoring,
    )
    wc2 = count_words(md2)
    if wc2 > word_count * 2:
        clean_html, markdown, word_count = clean2, md2, wc2
        diagnostics["adaptive_retry_used"] = True
        diagnostics["retry_level"] = 2
    if word_count >= _RETRY_SPARSE_THRESHOLD:
        return clean_html, markdown

    # Level 3: Retry without hidden element removal
    _soup3, root3 = ctx.fresh_soup_and_root(extractor, diagnostics)
    clean3, md3, _ = _extract_once(
        root3,
        ctx.metadata,
        ctx.md_instance,
        url,
        use_hidden_removal=False,
        use_scoring=use_scoring,
    )
    wc3 = count_words(md3)
    if wc3 > word_count:
        clean_html, markdown, word_count = clean3, md3, wc3
        diagnostics["adaptive_retry_used"] = True
        diagnostics["retry_level"] = 3
    if word_count >= _RETRY_SPARSE_THRESHOLD:
        return clean_html, markdown

    # Level 4: Retry with all removals disabled
    _soup4, root4 = ctx.fresh_soup_and_root(extractor, diagnostics)
    clean4, md4, _ = _extract_once(
        root4,
        ctx.metadata,
        ctx.md_instance,
        url,
        use_partial_selectors=False,
        use_hidden_removal=False,
        use_scoring=False,
    )
    wc4 = count_words(md4)
    if wc4 > word_count:
        clean_html, markdown, word_count = clean4, md4, wc4
        diagnostics["adaptive_retry_used"] = True
        diagnostics["retry_level"] = 4

    # Fallback: broaden to <body> (deep-copy to avoid mutated state)
    if word_count <= _RETRY_VERY_SPARSE_THRESHOLD:
        soup_body = copy.deepcopy(ctx.original_soup)
        body = soup_body.body
        if body is not None:
            body_html, body_md, _ = _extract_once(
                body,
                ctx.metadata,
                ctx.md_instance,
                url,
                use_partial_selectors=False,
                use_hidden_removal=False,
                use_scoring=False,
            )
            if count_words(body_md) > word_count:
                clean_html = body_html
                markdown = body_md
                diagnostics["adaptive_retry_used"] = True
                diagnostics["retry_level"] = "body_fallback"

    return clean_html, markdown


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
    candidate---e.g. paragraphs placed directly under ``<body>``.

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
    """Create a MarkItDown instance with WebExtract's custom converter.

    Registers ``WebExtractHtmlConverter`` at higher priority than the
    built-in ``HtmlConverter`` so code-block language detection and
    other enhanced rules are applied.
    """
    from markitdown import MarkItDown

    from markitai.converter.webextract_html_converter import WebExtractHtmlConverter

    md = MarkItDown()
    md.register_converter(WebExtractHtmlConverter(), priority=-1)
    return md


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
