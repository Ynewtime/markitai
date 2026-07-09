"""Async enricher public API.

Enrichers are optional pipeline components that improve extraction quality
using async network sources (e.g. the X oEmbed enricher used by the
Playwright fetch path).

Exports:
    ``BaseEnricher``: Protocol that enrichers must satisfy.
    ``EnrichmentPolicy``: Policy dataclass controlling enricher execution.
"""

from __future__ import annotations

from markitai.webextract.enrichers.base import BaseEnricher, EnrichmentPolicy

__all__ = [
    "BaseEnricher",
    "EnrichmentPolicy",
]
