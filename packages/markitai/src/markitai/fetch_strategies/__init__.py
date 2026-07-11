"""Per-strategy fetch implementations behind a uniform runner interface.

Each strategy module hosts its ``fetch_with_*`` implementation plus the
``*Runner`` adapter used by ``markitai.fetch`` for both explicit dispatch
and the AUTO fallback chain:

- ``static``: direct HTTP via httpx/curl-cffi (+ conditional revalidation)
- ``playwright``: local headless browser (implementation stays in
  ``markitai.fetch_playwright``; only the runner lives here)
- ``defuddle``: Defuddle content extraction API
- ``jina``: Jina Reader API
- ``cloudflare``: Cloudflare Browser Rendering API

This package must never import ``markitai.fetch`` (enforced by
import-linter): orchestration depends on strategies, not the reverse.

NOTE: runner ``fetch`` bodies call the ``fetch_with_*`` functions through
their own module's globals, so tests must patch the strategy module (e.g.
``markitai.fetch_strategies.jina.fetch_with_jina``) to intercept runner
calls; patching ``markitai.fetch.fetch_with_jina`` only affects direct
calls through that re-export binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from markitai.fetch_types import FetchResult, FetchStrategy

if TYPE_CHECKING:
    from markitai.config import FetchConfig
    from markitai.fetch_session import FetchSession


@dataclass
class StrategyContext:
    """One fetch attempt's shared inputs for strategy runners.

    ``explicit`` distinguishes the two dispatch surfaces: True when a single
    strategy is dispatched directly (``_dispatch_strategy``), False when the
    strategy runs as one hop of the AUTO fallback chain
    (``_fetch_with_fallback``). Runners branch on it where the two surfaces
    historically diverged (static conditional variant, cloudflare strict
    credential resolution, playwright availability errors vs. silent skips).
    """

    config: FetchConfig
    session: FetchSession
    explicit: bool
    screenshot_kwargs: dict[str, Any]
    cached_etag: str | None = None
    cached_last_modified: str | None = None


class StrategyRunner(Protocol):
    """Uniform interface over the per-strategy fetch implementations.

    ``unavailable_reason`` returns None when the strategy can run, or a skip
    reason recorded by the AUTO chain as ``f"{strategy}: {reason}"``; an
    empty string skips without recording an error (historical silent skip
    when config has no cloudflare section). Explicit dispatch never skips —
    runners raise actionable errors from ``fetch`` instead (see
    ``ctx.explicit`` branches).
    """

    strategy: FetchStrategy
    requires_remote_consent: bool

    def unavailable_reason(self, ctx: StrategyContext) -> str | None: ...

    async def fetch(self, url: str, ctx: StrategyContext) -> FetchResult: ...


from markitai.fetch_strategies.cloudflare import (
    CloudflareRunner as CloudflareRunner,
)
from markitai.fetch_strategies.cloudflare import (
    fetch_with_cloudflare as fetch_with_cloudflare,
)
from markitai.fetch_strategies.cloudflare import (
    get_cf_semaphore as get_cf_semaphore,
)
from markitai.fetch_strategies.defuddle import (
    DefuddleRunner as DefuddleRunner,
)
from markitai.fetch_strategies.defuddle import (
    fetch_with_defuddle as fetch_with_defuddle,
)
from markitai.fetch_strategies.jina import (
    JinaRunner as JinaRunner,
)
from markitai.fetch_strategies.jina import (
    fetch_with_jina as fetch_with_jina,
)
from markitai.fetch_strategies.playwright import (
    PlaywrightRunner as PlaywrightRunner,
)
from markitai.fetch_strategies.static import (
    StaticRunner as StaticRunner,
)
from markitai.fetch_strategies.static import (
    fetch_with_static as fetch_with_static,
)
from markitai.fetch_strategies.static import (
    fetch_with_static_conditional as fetch_with_static_conditional,
)

_STRATEGY_RUNNERS: dict[FetchStrategy, StrategyRunner] = {
    FetchStrategy.STATIC: StaticRunner(),
    FetchStrategy.PLAYWRIGHT: PlaywrightRunner(),
    FetchStrategy.DEFUDDLE: DefuddleRunner(),
    FetchStrategy.JINA: JinaRunner(),
    FetchStrategy.CLOUDFLARE: CloudflareRunner(),
}

# AUTO-chain lookup by policy-order strategy name. Unknown names are skipped
# by the loop, exactly as the old if/elif chain fell through them silently.
_STRATEGY_RUNNERS_BY_NAME: dict[str, StrategyRunner] = {
    strategy.value: runner for strategy, runner in _STRATEGY_RUNNERS.items()
}


def get_runner(strategy: FetchStrategy | str) -> StrategyRunner | None:
    """Look up the runner for a strategy enum or policy-order name.

    Returns None for unknown strategies/names (callers decide whether that
    is an error — explicit dispatch — or a silent skip — the AUTO chain).
    """
    if isinstance(strategy, str):
        return _STRATEGY_RUNNERS_BY_NAME.get(strategy)
    return _STRATEGY_RUNNERS.get(strategy)
