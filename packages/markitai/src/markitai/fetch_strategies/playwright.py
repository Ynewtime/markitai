"""Playwright strategy runner (local headless browser).

The actual fetch implementation (``fetch_with_playwright``) lives in
``markitai.fetch_playwright``; this module only hosts the runner that
adapts it to the strategy interface. The runner imports
``fetch_with_playwright`` lazily inside ``fetch`` so test patches on
``markitai.fetch_playwright.fetch_with_playwright`` keep intercepting it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.fetch_support import _get_playwright_fetch_kwargs
from markitai.fetch_types import FetchError, FetchResult, FetchStrategy

if TYPE_CHECKING:
    from markitai.fetch_strategies import StrategyContext


def _playwright_strategy_label(metadata: dict[str, Any]) -> str:
    """Build ``strategy_used`` label for playwright, including enricher source.

    Returns ``"playwright(fxtwitter)"``, ``"playwright(oembed)"``, or
    plain ``"playwright"`` when no enricher was used.
    """
    source = metadata.get("_enricher_source", "")
    if source:
        return f"playwright({source})"
    return "playwright"


class PlaywrightRunner:
    """Local headless-browser fetch via Playwright."""

    strategy: FetchStrategy = FetchStrategy.PLAYWRIGHT
    requires_remote_consent: bool = False

    def unavailable_reason(self, ctx: StrategyContext) -> str | None:
        if ctx.explicit:
            # Explicit dispatch raises actionable guidance errors in fetch()
            return None

        from markitai.fetch_playwright import (
            is_playwright_available,
            is_playwright_browser_installed,
        )

        if not is_playwright_available():
            logger.debug("playwright not available, trying next strategy")
            return "playwright package not installed"
        if not is_playwright_browser_installed():
            logger.debug(
                "[Fetch] playwright installed but Chromium browser missing; "
                "skipping playwright in auto chain "
                "(run 'markitai doctor --fix' to install it)"
            )
            return "Chromium browser missing (run 'markitai doctor --fix')"
        return None

    async def fetch(self, url: str, ctx: StrategyContext) -> FetchResult:
        from markitai.fetch_playwright import (
            fetch_with_playwright,
            is_playwright_available,
            is_playwright_browser_installed,
        )

        if ctx.explicit:
            from markitai.utils.guidance import (
                playwright_browser_missing_error,
                playwright_package_missing_error,
            )

            if not is_playwright_available():
                raise FetchError(playwright_package_missing_error())
            if not is_playwright_browser_installed():
                raise FetchError(playwright_browser_missing_error())

        pw_result = await fetch_with_playwright(
            url,
            **_get_playwright_fetch_kwargs(
                url,
                ctx.config,
                screenshot_config=ctx.screenshot_kwargs.get("screenshot_config"),
                output_dir=ctx.screenshot_kwargs.get("screenshot_dir"),
                renderer=ctx.screenshot_kwargs.get("renderer"),
            ),
            remote_consent=ctx.config.remote_consent,
        )

        return FetchResult(
            content=pw_result.content,
            strategy_used=_playwright_strategy_label(pw_result.metadata),
            title=pw_result.title,
            url=url,
            final_url=pw_result.final_url,
            metadata=pw_result.metadata,
            screenshot_path=pw_result.screenshot_path,
        )
