"""Tests for enricher policy and the enricher contract.

These tests verify that:
- ``EnrichmentPolicy`` defaults are permissive and toggles work
- The X oEmbed enricher satisfies the ``BaseEnricher`` contract
- Enrichers respect policy toggles in ``should_run``
"""

from __future__ import annotations

from markitai.webextract.enrichers.base import EnrichmentPolicy
from markitai.webextract.enrichers.x_oembed import XOEmbedEnricher

_X_URL = "https://x.com/ixiaowenz/status/1899069961745801250"
_GENERIC_URL = "https://example.com/blog/post"


# ---------------------------------------------------------------------------
# EnrichmentPolicy tests
# ---------------------------------------------------------------------------


class TestEnrichmentPolicy:
    """Tests for the EnrichmentPolicy dataclass."""

    def test_default_policy_allows_network_and_async(self) -> None:
        policy = EnrichmentPolicy()
        assert policy.allow_network is True
        assert policy.allow_async is True

    def test_local_only_policy_disables_network(self) -> None:
        policy = EnrichmentPolicy(allow_network=False)
        assert policy.allow_network is False

    def test_no_async_policy_disables_async_enrichers(self) -> None:
        policy = EnrichmentPolicy(allow_async=False)
        assert policy.allow_async is False

    def test_preferred_enrichers_defaults_to_empty(self) -> None:
        policy = EnrichmentPolicy()
        assert policy.preferred_enrichers == []


# ---------------------------------------------------------------------------
# BaseEnricher protocol tests
# ---------------------------------------------------------------------------


class TestBaseEnricher:
    """Tests for the BaseEnricher protocol and contract."""

    def test_x_oembed_enricher_has_required_attributes(self) -> None:
        enricher = XOEmbedEnricher()
        assert hasattr(enricher, "name")
        assert isinstance(enricher.name, str)
        assert enricher.name == "x_oembed"

    def test_x_oembed_enricher_implements_should_run(self) -> None:
        enricher = XOEmbedEnricher()
        assert callable(getattr(enricher, "should_run", None))

    def test_x_oembed_enricher_implements_enrich(self) -> None:
        enricher = XOEmbedEnricher()
        assert callable(getattr(enricher, "enrich", None))

    def test_x_oembed_should_run_for_x_url_with_network_allowed(self) -> None:
        enricher = XOEmbedEnricher()
        policy = EnrichmentPolicy(allow_network=True, allow_async=True)
        assert enricher.should_run(_X_URL, policy) is True

    def test_x_oembed_should_not_run_when_network_disallowed(self) -> None:
        enricher = XOEmbedEnricher()
        policy = EnrichmentPolicy(allow_network=False)
        assert enricher.should_run(_X_URL, policy) is False

    def test_x_oembed_should_not_run_when_async_disallowed(self) -> None:
        enricher = XOEmbedEnricher()
        policy = EnrichmentPolicy(allow_async=False)
        assert enricher.should_run(_X_URL, policy) is False

    def test_x_oembed_should_not_run_for_generic_url(self) -> None:
        enricher = XOEmbedEnricher()
        policy = EnrichmentPolicy()
        assert enricher.should_run(_GENERIC_URL, policy) is False
