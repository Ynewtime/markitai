"""Common type definitions for Markitai."""

from __future__ import annotations

from typing import TypedDict


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
    """

    cached_input_tokens: int


# Type alias for LLM usage by model
# Format: {"model_name": {"requests": N, "input_tokens": N, "output_tokens": N, "cost_usd": F}}
LLMUsageByModel = dict[str, ModelUsageStats]
