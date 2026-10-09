"""Common type definitions for Markitai."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict


class _ModelUsageStatsRequired(TypedDict):
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ModelUsageStats(_ModelUsageStatsRequired, total=False):
    """Statistics for a single LLM model's usage.

    ``cached_input_tokens`` is present once prompt caching was observed on
    any call (cache-read tokens, billed at the provider's cache rate); older
    aggregators and hand-built dicts may omit it, so readers use ``.get``.

    ``unpriced_requests`` is present once a request's price was unknown (a
    model LiteLLM has no price for): ``cost_usd`` is then only the priced
    subtotal and leaves those requests out, so the model's real cost is
    unknown, not what ``cost_usd`` says.
    """

    cached_input_tokens: int
    unpriced_requests: int


# Type alias for LLM usage by model
# Format: {"model_name": {"requests": N, "input_tokens": N, "output_tokens": N, "cost_usd": F}}
LLMUsageByModel = dict[str, ModelUsageStats]


def request_usage(
    input_tokens: int, output_tokens: int, cost_usd: float | None
) -> ModelUsageStats:
    """Usage of one request; ``cost_usd`` None means its price is unknown."""
    stats = ModelUsageStats(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd or 0.0,
    )
    if cost_usd is None:
        stats["unpriced_requests"] = 1
    return stats


def usage_has_unpriced(usage: Mapping[str, Mapping[str, Any]] | None) -> bool:
    """Whether any request in *usage* had no known price."""
    return any(stats.get("unpriced_requests", 0) for stats in (usage or {}).values())
