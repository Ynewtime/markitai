"""Common type definitions for Markitai."""

from __future__ import annotations

from typing import TypedDict


class ModelUsageStats(TypedDict):
    """Statistics for a single LLM model's usage."""

    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


# Type alias for LLM usage by model
# Format: {"model_name": {"requests": N, "input_tokens": N, "output_tokens": N, "cost_usd": F}}
LLMUsageByModel = dict[str, ModelUsageStats]
