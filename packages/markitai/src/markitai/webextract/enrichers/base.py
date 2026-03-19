"""Base types for policy-aware async enrichers.

Enrichers are optional components that run after the sync resolver to improve
extraction quality using async network sources (e.g. oEmbed, APIs).

Design rules:
- Enrichers NEVER hard-fail a page.  Failures return ``None`` silently.
- Enrichers only run when the ``EnrichmentPolicy`` explicitly permits them.
- An enricher returns ``None`` to signal "not better than sync result";
  the resolver then keeps the sync result unchanged.
- ``enricher_name`` is recorded in ``ResolvedPage.diagnostics`` so the
  pipeline can propagate it to ``ExtractionInfo.enricher_name``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from markitai.webextract.resolver import ResolvedPage


@dataclass(slots=True)
class EnrichmentPolicy:
    """Policy controlling whether async enrichers may run.

    Attributes:
        allow_network: Whether enrichers are permitted to make network
            requests.  Defaults to ``True``.
        allow_async: Whether async enrichers are permitted to run at all.
            Defaults to ``True``.
        preferred_enrichers: Optional ordered list of enricher names to
            prefer.  When non-empty, only named enrichers are considered.
    """

    allow_network: bool = True
    allow_async: bool = True
    preferred_enrichers: list[str] = field(default_factory=list)


@runtime_checkable
class BaseEnricher(Protocol):
    """Protocol that all async enrichers must satisfy.

    Enrichers are discovered by the resolver, checked via ``should_run``,
    and then awaited via ``enrich``.  They operate at the resolver layer,
    not the fetch layer.
    """

    name: str

    async def enrich(
        self,
        url: str,
        sync_result: ResolvedPage | None,
    ) -> ResolvedPage | None:
        """Try to improve the extraction using an async source.

        Args:
            url: The page URL being resolved.
            sync_result: The result from the sync resolver (may be ``None``).

        Returns:
            An improved ``ResolvedPage`` if enrichment succeeded and produced
            something better than ``sync_result``, otherwise ``None``.
            Must never raise — failures must be caught internally and
            returned as ``None``.
        """
        ...

    def should_run(self, url: str, policy: EnrichmentPolicy) -> bool:
        """Check if this enricher should attempt to run for the given URL.

        Args:
            url: The page URL.
            policy: The enrichment policy for this request.

        Returns:
            ``True`` if this enricher is applicable and the policy permits it,
            ``False`` otherwise.
        """
        ...
