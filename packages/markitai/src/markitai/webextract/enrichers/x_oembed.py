"""X/Twitter oEmbed enricher.

Uses the X oEmbed API to retrieve structured tweet metadata when the sync
HTML-based extraction produces a low-quality result.

The actual HTTP call is isolated in ``_fetch_oembed`` so that tests can
patch at the system boundary without mocking internal resolver logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.webextract.enrichers.base import EnrichmentPolicy

if TYPE_CHECKING:
    from markitai.webextract.resolver import ResolvedPage

# X oEmbed endpoint (public, no auth required for public tweets)
_OEMBED_URL = "https://publish.twitter.com/oembed"


class XOEmbedEnricher:
    """Enrich X/Twitter pages using the public oEmbed API.

    This enricher is a stub that demonstrates the enricher contract.
    The ``_fetch_oembed`` method is designed to be patched in tests so
    that no real HTTP requests are made.
    """

    name = "x_oembed"

    def should_run(self, url: str, policy: EnrichmentPolicy) -> bool:
        """Return True if this enricher should run for the given URL and policy.

        Args:
            url: The page URL.
            policy: Enrichment policy for this request.

        Returns:
            ``True`` when the URL is an X/Twitter status page and the policy
            permits both network access and async enrichers.
        """
        if not policy.allow_network:
            return False
        if not policy.allow_async:
            return False
        return ("x.com/" in url or "twitter.com/" in url) and "/status/" in url

    async def enrich(
        self,
        url: str,
        sync_result: ResolvedPage | None,
    ) -> ResolvedPage | None:
        """Attempt to improve extraction quality using the oEmbed API.

        Args:
            url: The tweet URL.
            sync_result: The result from the sync resolver.

        Returns:
            An improved ``ResolvedPage`` if the oEmbed API returned usable
            data, otherwise ``None``.  Never raises.
        """
        try:
            data = await self._fetch_oembed(url)
        except Exception as exc:
            logger.debug("[XOEmbedEnricher] fetch failed, falling back: {}", exc)
            return None

        if data is None:
            return None

        return self._build_resolved_page(data)

    async def _fetch_oembed(self, url: str) -> dict[str, Any] | None:
        """Fetch oEmbed data for the given URL.

        This method is intentionally isolated so tests can patch it without
        touching the rest of the enricher logic.

        Args:
            url: The tweet URL to look up.

        Returns:
            Parsed oEmbed JSON as a dict, or ``None`` on failure.
        """
        try:
            import httpx
        except ImportError:
            logger.debug("[XOEmbedEnricher] httpx not available, skipping oEmbed")
            return None

        params = {"url": url, "omit_script": "true"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_OEMBED_URL, params=params)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

    def _build_resolved_page(self, data: dict[str, Any]) -> ResolvedPage | None:
        """Build a ``ResolvedPage`` from oEmbed API data.

        Args:
            data: Parsed oEmbed response dict.

        Returns:
            A ``ResolvedPage`` with metadata overrides, or ``None`` if the
            data does not provide enough signal.
        """
        from markitai.webextract.resolver import ResolvedPage

        author_name: str = data.get("author_name", "")
        title: str = data.get("title", "") or (
            f"Post by {author_name}" if author_name else ""
        )
        html_embed: str = data.get("html", "")

        if not title and not html_embed:
            return None

        metadata_overrides: dict[str, object] = {}
        if title:
            metadata_overrides["title"] = title
        if author_name:
            metadata_overrides["author"] = author_name
        metadata_overrides["site"] = "X (Twitter)"

        return ResolvedPage(
            content_html=html_embed or None,
            metadata_overrides=metadata_overrides,
            diagnostics={"enricher_name": self.name, "source": "oembed"},
        )
