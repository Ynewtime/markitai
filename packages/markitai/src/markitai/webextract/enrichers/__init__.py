"""Async enricher registry and public API.

Enrichers are optional pipeline components that run after the sync resolver
to improve extraction quality using async network sources.  The registry
provides a simple function-based API for discovering enrichers by URL.

Exports:
    ``BaseEnricher``: Protocol that enrichers must satisfy.
    ``EnrichmentPolicy``: Policy dataclass controlling enricher execution.
    ``get_enrichers_for_url``: Discover enrichers applicable to a URL.
"""

from __future__ import annotations

from markitai.webextract.enrichers.base import BaseEnricher, EnrichmentPolicy
from markitai.webextract.enrichers.x_oembed import XOEmbedEnricher

# Global enricher registry — ordered by priority (most specific first)
_ENRICHERS: tuple[XOEmbedEnricher, ...] = (XOEmbedEnricher(),)


def get_enrichers_for_url(url: str) -> list[XOEmbedEnricher]:
    """Return enrichers that are potentially applicable to the given URL.

    This does NOT consult the policy — policy checks are deferred to the
    caller so that ``should_run`` semantics remain in the enricher itself.

    Args:
        url: The page URL to match against.

    Returns:
        Ordered list of enrichers whose ``should_run`` check would pass for
        a permissive policy (i.e. URL pattern matches).
    """
    permissive = EnrichmentPolicy(allow_network=True, allow_async=True)
    return [e for e in _ENRICHERS if e.should_run(url, permissive)]


__all__ = [
    "BaseEnricher",
    "EnrichmentPolicy",
    "get_enrichers_for_url",
]
