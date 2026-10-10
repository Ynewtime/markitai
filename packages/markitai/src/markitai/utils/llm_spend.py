"""What a work item spent on LLM requests, per model.

Kept outside the ``markitai.llm`` package on purpose: importing anything
under it loads LiteLLM, instructor and the OpenAI client (about a second on
Linux, several on Windows), and the CLI and the web server track spend around
every URL and file, also when no model is used.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpendTally:
    """Every LLM request a work item paid for, per model.

    The per-context usage of the processor is cleared by each stage, and a
    stage that fails clears it without handing it on, so a rejected answer
    (dropped placeholders, a refusal, unparseable JSON, a truncation) was
    paid for but reported as free. This tally sees every answer the moment
    it is accounted, whatever happens to it afterwards, in the shape of
    ``LLMProcessor.get_context_usage``.
    """

    usage: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        """USD of the priced requests (unpriced ones are counted apart)."""
        return sum(stats.get("cost_usd", 0.0) for stats in self.usage.values())


# The spend tallies of the work items enclosing the current task (a URL list
# inside a batch nests). Context variables for the same reason as
# _cache_tally: the tasks an item spawns inherit its context.
_spend_tallies: ContextVar[tuple[SpendTally, ...]] = ContextVar(
    "markitai_llm_spend_tallies", default=()
)
_spend_lock = threading.Lock()


@contextmanager
def track_llm_spend() -> Iterator[SpendTally]:
    """Collect the LLM spend of one work item, failed stages included.

    Yields:
        The tally, updated in place until the block exits.
    """
    tally = SpendTally()
    token = _spend_tallies.set((*_spend_tallies.get(), tally))
    try:
        yield tally
    finally:
        _spend_tallies.reset(token)


def record_llm_spend(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float | None,
    cached_tokens: int = 0,
) -> None:
    """Add one answered request to the enclosing spend tallies (if any).

    ``cost`` None is an unknown price: counted in ``unpriced_requests``.
    """
    tallies = _spend_tallies.get()
    if not tallies:
        return
    with _spend_lock:
        for tally in tallies:
            stats = _usage_entry(tally, model)
            stats["requests"] += 1
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            stats["cached_input_tokens"] += cached_tokens
            if cost is None:
                stats["unpriced_requests"] = stats.get("unpriced_requests", 0) + 1
            else:
                stats["cost_usd"] += cost


def record_llm_spend_marker(name: str) -> None:
    """Add an all-zero pseudo-model entry (a tripped budget) to the tallies."""
    with _spend_lock:
        for tally in _spend_tallies.get():
            _usage_entry(tally, name)


def _usage_entry(tally: SpendTally, model: str) -> dict[str, Any]:
    return tally.usage.setdefault(
        model,
        {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cost_usd": 0.0,
        },
    )
