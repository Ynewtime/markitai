"""Resolver layer: structured extractor orchestration above root selection.

The resolver layer sits between site-specific extractor lookup and the generic
pipeline extraction.  It provides a typed contract (``ResolvedPage``) that
an extractor's optional ``resolve()`` method may return.

Design rules:
- ``ResolvedPage`` lives here, NOT in ``FetchResult`` or fetch-layer types.
- A resolver may return ``content_root`` (a Tag) or ``content_html`` (a string).
  It must NEVER return final Markdown.
- If no extractor implements ``resolve()``, or no extractor matches the URL,
  ``resolve_page()`` returns ``None`` and the generic pipeline remains the
  fallback.
- Metadata overrides from ``ResolvedPage.metadata_overrides`` are merged over
  the generic metadata extracted by ``extract_metadata()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup, Tag
from loguru import logger

if TYPE_CHECKING:
    from markitai.webextract.enrichers.base import EnrichmentPolicy
    from markitai.webextract.types import SemanticExtraction


@dataclass(slots=True)
class ResolvedPage:
    """Output of a site-specific resolver.

    A resolver may populate either ``content_root`` (a parsed Tag) or
    ``content_html`` (a raw HTML string).  It must NOT set final Markdown
    in either field — the pipeline is responsible for the HTML→Markdown
    conversion step.

    Attributes:
        content_root: Parsed BeautifulSoup Tag representing the content root.
            Mutually exclusive with ``content_html`` (caller may set either).
        content_html: Raw HTML string of the primary content.
            Mutually exclusive with ``content_root``.
        metadata_overrides: Mapping of metadata fields to override (e.g.
            ``{"title": "Custom Title"}``).  Merged over generic metadata.
        semantic: Optional semantic models derived during resolution (e.g.
            a conversation thread).
        diagnostics: Internal debug information populated during resolution.
    """

    content_root: Tag | None = None
    content_html: str | None = None
    metadata_overrides: dict[str, Any] = field(default_factory=dict)
    semantic: SemanticExtraction | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_page(
    html: str,
    url: str,
    *,
    resolver: object | None = None,
) -> ResolvedPage | None:
    """Attempt structured resolution of a page via a site-specific extractor.

    Parses ``html``, finds a matching extractor (or uses the provided
    ``resolver``), and calls its ``resolve()`` method if present.  The result
    is validated to ensure it is a ``ResolvedPage`` and not raw Markdown.

    Args:
        html: Raw HTML source of the page.
        url: Canonical URL of the page (used for extractor lookup).
        resolver: Optional extractor override.  When supplied, skips the
            extractor registry lookup and uses this object directly.

    Returns:
        A ``ResolvedPage`` if a resolver matched and returned structured
        content, or ``None`` when falling back to the generic pipeline.

    Raises:
        TypeError: If the resolver's ``resolve()`` method returns something
            other than a ``ResolvedPage`` (e.g. a Markdown string).
    """
    extractor = resolver
    if extractor is None:
        from markitai.webextract.extractors.registry import find_extractor

        extractor = find_extractor(url)

    if extractor is None:
        return None

    resolve_fn = getattr(extractor, "resolve", None)
    if resolve_fn is None or not callable(resolve_fn):
        return None

    soup = BeautifulSoup(html, "html.parser")
    raw_result = resolve_fn(soup, url)

    # Validate: resolver must not return final Markdown
    _validate_resolver_result(raw_result, extractor)

    return raw_result  # type: ignore[return-value]


async def resolve_page_async(
    html: str,
    url: str,
    *,
    resolver: object | None = None,
    policy: EnrichmentPolicy | None = None,
) -> ResolvedPage | None:
    """Async variant of ``resolve_page`` that may run policy-aware enrichers.

    First runs the sync resolver to obtain a baseline ``ResolvedPage``.
    Then, if the policy permits, discovers and runs any applicable enrichers.
    If an enricher returns an improved result, that result is used instead.
    Enricher failures never propagate — the sync baseline is the fallback.

    .. note::

        **Not yet wired into the production extraction path.**  The main
        ``extract_web_content()`` function currently calls the sync
        ``resolve_page()`` only.  This async variant and its enricher
        infrastructure are available for explicit use but are not
        automatically invoked during URL conversion.

    Args:
        html: Raw HTML source of the page.
        url: Canonical URL of the page.
        resolver: Optional extractor override (forwarded to ``resolve_page``).
        policy: Enrichment policy controlling whether enrichers may run.
            Defaults to a permissive policy when ``None``.

    Returns:
        The best available ``ResolvedPage``, or ``None`` if neither the sync
        resolver nor any enricher produced a result.
    """
    from markitai.webextract.enrichers import get_enrichers_for_url
    from markitai.webextract.enrichers.base import EnrichmentPolicy as _Policy

    if policy is None:
        policy = _Policy()

    # Step 1: run the sync resolver as the baseline
    sync_result = resolve_page(html, url, resolver=resolver)

    # Step 2: skip enrichers if the policy forbids async or network
    if not policy.allow_async or not policy.allow_network:
        return sync_result

    # Step 3: find enrichers applicable to this URL
    candidates = get_enrichers_for_url(url)
    if not candidates:
        return sync_result

    # Step 4: run each enricher; use the first one that returns a result
    for enricher in candidates:
        if not enricher.should_run(url, policy):
            continue
        try:
            enriched = await enricher.enrich(url, sync_result)
        except Exception as exc:
            logger.debug(
                "[Resolver] enricher '{}' raised unexpectedly: {}",
                enricher.name,
                exc,
            )
            enriched = None

        if enriched is not None:
            logger.debug("[Resolver] enricher '{}' improved result", enricher.name)
            return enriched

    return sync_result


def _validate_resolver_result(result: object, extractor: object) -> None:
    """Validate that a resolver did not return Markdown or another illegal type.

    Args:
        result: The raw return value from the extractor's ``resolve()`` method.
        extractor: The extractor object (used in error messages).

    Raises:
        TypeError: If ``result`` is not a ``ResolvedPage`` instance.
    """
    if isinstance(result, ResolvedPage):
        return

    extractor_name = getattr(extractor, "name", repr(extractor))
    raise TypeError(
        f"Extractor '{extractor_name}'.resolve() must return a ResolvedPage, "
        f"not {type(result).__name__!r}. "
        "Returning final Markdown from a resolver is not allowed; "
        "the pipeline handles HTML→Markdown conversion."
    )
