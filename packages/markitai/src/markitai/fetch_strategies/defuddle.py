"""Defuddle content extraction API fetch strategy (https://defuddle.md).

Returns clean Markdown with YAML frontmatter from any URL.

NOTE: Rate limit is undocumented — using same conservative limiter as Jina.
NOTE: JS rendering capability is unconfirmed. SPA sites may need playwright.
TODO: Migrate defuddle's core content extraction logic to markitai native.
      Defuddle is open-source (https://github.com/kepano/defuddle) and its
      HTML cleaning/article extraction could replace the external API dependency.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import DEFAULT_DEFUDDLE_BASE_URL, DEFAULT_DEFUDDLE_RPM
from markitai.fetch_session import _SlidingWindowRateLimiter, get_default_session
from markitai.fetch_types import FetchError, FetchResult, FetchStrategy
from markitai.utils.text import format_error_message

if TYPE_CHECKING:
    from markitai.fetch_strategies import StrategyContext


def _get_defuddle_rate_limiter(rpm: int) -> _SlidingWindowRateLimiter:
    """Get or create the global Defuddle rate limiter."""
    return get_default_session().get_defuddle_rate_limiter(rpm)


def _get_defuddle_client(timeout: int = 30) -> Any:
    """Get or create the shared httpx.AsyncClient for Defuddle fetching."""
    return get_default_session().get_defuddle_client(timeout)


async def fetch_with_defuddle(
    url: str,
    timeout: int = 30,
    rpm: int = DEFAULT_DEFUDDLE_RPM,
) -> FetchResult:
    """Fetch URL using Defuddle content extraction API.

    Defuddle extracts the main article content from web pages, removing clutter
    like ads, sidebars, headers, and footers. Returns clean Markdown with rich
    YAML frontmatter (title, author, published, description, word_count).

    API: GET https://defuddle.md/<url> → Markdown with YAML frontmatter

    NOTE: Rate limit is undocumented — we use a conservative default (20 RPM).
    NOTE: JS rendering capability of the API is unconfirmed. SPA-heavy sites
          may still need playwright as a fallback strategy.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        rpm: Requests per minute limit (conservative; actual limit undocumented)

    Returns:
        FetchResult with clean markdown content and extracted metadata

    Raises:
        FetchError: If fetch fails or returns empty content
    """
    limiter = _get_defuddle_rate_limiter(rpm)
    await limiter.acquire()

    import httpx

    logger.debug(f"[Defuddle] Fetching: {url}")

    from urllib.parse import quote

    defuddle_url = f"{DEFAULT_DEFUDDLE_BASE_URL}/{quote(url, safe='')}"

    try:
        client = _get_defuddle_client(timeout)
        response = await client.get(defuddle_url)

        if response.status_code == 429:
            raise FetchError(
                "Defuddle rate limit exceeded. "
                "Try again later or use --playwright for local rendering."
            )
        elif response.status_code >= 400:
            raise FetchError(
                f"Defuddle API returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        content = response.text
        if not content or not content.strip():
            raise FetchError(f"No content returned from Defuddle: {url}")

        # Parse YAML frontmatter for metadata extraction
        import yaml

        title: str | None = None
        metadata: dict[str, Any] = {"api": "defuddle"}
        source_frontmatter: dict[str, Any] = {}

        frontmatter_match = re.match(
            r"^\s*---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL
        )
        if frontmatter_match:
            try:
                fm_data = yaml.safe_load(frontmatter_match.group(1))
                if isinstance(fm_data, dict):
                    if fm_data.get("title"):
                        title = str(fm_data["title"])
                    # Preserve all source frontmatter fields for output
                    for key, value in fm_data.items():
                        if value is not None:
                            source_frontmatter[key] = value
            except yaml.YAMLError:
                logger.debug(
                    "[Defuddle] Failed to parse YAML frontmatter, using raw content"
                )
        if source_frontmatter and frontmatter_match:
            metadata["source_frontmatter"] = source_frontmatter

            # Strip frontmatter, keep only markdown body
            content = content[frontmatter_match.end() :].strip()

        if not content:
            raise FetchError(f"Defuddle returned empty content for: {url}")

        return FetchResult(
            content=content,
            strategy_used="defuddle",
            title=title,
            url=url,
            metadata=metadata,
        )

    except FetchError:
        raise
    except httpx.TimeoutException:
        raise FetchError(f"Defuddle request timed out after {timeout}s: {url}")
    except Exception as e:
        raise FetchError(f"Defuddle fetch failed: {format_error_message(e)}")


class DefuddleRunner:
    """Defuddle content extraction API fetch."""

    strategy: FetchStrategy = FetchStrategy.DEFUDDLE
    requires_remote_consent: bool = True

    def unavailable_reason(self, ctx: StrategyContext) -> str | None:
        return None

    async def fetch(self, url: str, ctx: StrategyContext) -> FetchResult:
        return await fetch_with_defuddle(
            url,
            ctx.config.defuddle.timeout,
            ctx.config.defuddle.rpm,
        )
