"""Tests for policy-aware async enrichers at the resolver layer.

These tests verify that:
- ``resolve_page_async()`` runs the sync resolver first, then applies enrichers
- Enrichers are skipped when the policy disallows network/async
- Enrichers never hard-fail the page (failures fall back to sync result)
- The enricher_name is recorded in ``ResolvedPage.diagnostics``
- The X oEmbed enricher stub demonstrates the enricher contract
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from markitai.webextract.enrichers import get_enrichers_for_url
from markitai.webextract.enrichers.base import EnrichmentPolicy
from markitai.webextract.enrichers.x_oembed import XOEmbedEnricher
from markitai.webextract.resolver import ResolvedPage, resolve_page_async

_X_URL = "https://x.com/ixiaowenz/status/1899069961745801250"
_GENERIC_URL = "https://example.com/blog/post"

# Minimal HTML for an X page that would produce a low-quality sync extraction
_X_HTML_MINIMAL = """
<html>
<head><title>X / Twitter</title></head>
<body>
  <div data-testid="primaryColumn">
    <article data-testid="tweet">
      <span>some short text</span>
    </article>
  </div>
</body>
</html>
"""

_GENERIC_HTML = """
<html>
<head><title>Blog Post</title></head>
<body>
  <article>
    <p>This is a blog post with enough content to pass quality checks.</p>
  </article>
</body>
</html>
"""


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


# ---------------------------------------------------------------------------
# Enricher registry tests
# ---------------------------------------------------------------------------


class TestEnricherRegistry:
    """Tests for the enricher discovery function."""

    def test_get_enrichers_for_x_url_returns_x_oembed(self) -> None:
        enrichers = get_enrichers_for_url(_X_URL)
        names = [e.name for e in enrichers]
        assert "x_oembed" in names

    def test_get_enrichers_for_generic_url_returns_empty(self) -> None:
        enrichers = get_enrichers_for_url(_GENERIC_URL)
        assert len(enrichers) == 0


# ---------------------------------------------------------------------------
# resolve_page_async() tests
# ---------------------------------------------------------------------------


class TestResolvePageAsync:
    """Tests for the async resolver orchestration."""

    async def test_async_resolver_returns_resolved_page_or_none(self) -> None:
        """resolve_page_async must return ResolvedPage or None."""
        result = await resolve_page_async(_GENERIC_HTML, _GENERIC_URL)
        assert result is None or isinstance(result, ResolvedPage)

    async def test_async_resolver_falls_back_to_sync_when_no_enrichers(
        self,
    ) -> None:
        """For a generic URL, no enrichers run; result equals sync result."""
        result = await resolve_page_async(_GENERIC_HTML, _GENERIC_URL)
        assert result is None  # no extractor matches generic URL

    async def test_resolver_prefers_async_for_x_when_policy_allows_and_sync_quality_is_low(
        self,
    ) -> None:
        """When policy allows async and enricher finds better data, enricher wins."""
        policy = EnrichmentPolicy(allow_network=True, allow_async=True)

        # Mock the oEmbed HTTP call at the system boundary
        mock_oembed_data = {
            "author_name": "@ixiaowenz",
            "title": "Post by @ixiaowenz",
            "html": "<blockquote>tweet content</blockquote>",
        }

        with patch(
            "markitai.webextract.enrichers.x_oembed.XOEmbedEnricher._fetch_oembed",
            new=AsyncMock(return_value=mock_oembed_data),
        ):
            result = await resolve_page_async(_X_HTML_MINIMAL, _X_URL, policy=policy)

        assert result is not None
        assert result.diagnostics.get("enricher_name") == "x_oembed"
        assert result.metadata_overrides.get("title") == "Post by @ixiaowenz"

    async def test_async_enricher_is_skipped_when_policy_disallows_network(
        self,
    ) -> None:
        """When policy forbids network, enrichers are not invoked."""
        local_only_policy = EnrichmentPolicy(allow_network=False)

        with patch(
            "markitai.webextract.enrichers.x_oembed.XOEmbedEnricher._fetch_oembed",
            new=AsyncMock(side_effect=AssertionError("should not be called")),
        ):
            result = await resolve_page_async(
                _X_HTML_MINIMAL, _X_URL, policy=local_only_policy
            )

        # Result is whatever the sync resolver produced (no enricher tag)
        if result is not None:
            assert result.diagnostics.get("enricher_name") is None

    async def test_enricher_failure_does_not_hard_fail_page(self) -> None:
        """If an enricher raises, resolve_page_async returns the sync result."""
        policy = EnrichmentPolicy(allow_network=True, allow_async=True)

        with patch(
            "markitai.webextract.enrichers.x_oembed.XOEmbedEnricher._fetch_oembed",
            new=AsyncMock(side_effect=RuntimeError("network error")),
        ):
            # Should NOT raise
            result = await resolve_page_async(_X_HTML_MINIMAL, _X_URL, policy=policy)

        # Falls back to sync result; enricher_name must not be set
        if result is not None:
            assert result.diagnostics.get("enricher_name") is None

    async def test_enricher_returning_none_falls_back_to_sync(self) -> None:
        """If an enricher returns None (not better), sync result is used."""
        policy = EnrichmentPolicy(allow_network=True, allow_async=True)

        with patch(
            "markitai.webextract.enrichers.x_oembed.XOEmbedEnricher._fetch_oembed",
            new=AsyncMock(return_value=None),
        ):
            result = await resolve_page_async(_X_HTML_MINIMAL, _X_URL, policy=policy)

        if result is not None:
            assert result.diagnostics.get("enricher_name") is None

    async def test_default_policy_is_permissive(self) -> None:
        """Calling resolve_page_async without policy= uses permissive defaults."""
        # Should not raise; enrichers may or may not run depending on URL
        result = await resolve_page_async(_GENERIC_HTML, _GENERIC_URL)
        assert result is None or isinstance(result, ResolvedPage)
