"""Model entries markitai ships for models its pinned LiteLLM may not list.

A default whose model is missing from LiteLLM's cost table is priced as
unknown: its requests land in ``unpriced_requests``, the batch summary says
"cost unknown", and ``max_cost_per_document_usd`` cannot count them. LiteLLM
1.104.2 (the newest stable release on 2026-10-10) has no entry for
``claude-haiku-5-5``, the Anthropic default since 1.2.1, nor for OpenRouter's
``anthropic/claude-haiku-5.5``. Its entry for OpenRouter's moving alias
``~anthropic/claude-haiku-latest``, the OpenRouter default, is kept as it is
(Claude Haiku 4.5's limits and prices): on 2026-10-10 OpenRouter's catalog
named Claude Haiku 5.5 as the alias's target, but requests to the alias were
answered and billed as Claude Haiku 4.5, so the smaller limits are the safe
ones, and OpenRouter requests are charged at the amount OpenRouter reports
(``markitai.llm.models.get_response_cost``). The copy here only fills in for
a LiteLLM that no longer lists the alias (upstream has dropped its OpenRouter
entries).

Each entry is checked against the provider's own pricing (the Anthropic
entry is LiteLLM's upstream one) and registered only when the installed
LiteLLM has no entry of that name, so a LiteLLM release that adds the model
wins and the copy here goes dormant.
"""

from __future__ import annotations

from typing import Any

# https://platform.claude.com/docs/en/models/haiku-5-5/overview (released
# 2026-10-07): $0.10 / $0.50 per MTok for prompts up to 100k tokens, $0.50 /
# $2.50 above; 1M context, 128k output; adaptive thinking; sampling
# parameters (temperature, top_p, top_k) are rejected with a 400.
_MISSING_FROM_LITELLM: dict[str, dict[str, Any]] = {
    "claude-haiku-5-5": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "max_input_tokens": 1000000,
        "max_output_tokens": 128000,
        "max_tokens": 128000,
        "input_cost_per_token": 1e-07,  # nosec B105 - a price, not a credential
        "output_cost_per_token": 5e-07,
        "input_cost_per_token_above_100k_tokens": 5e-07,
        "output_cost_per_token_above_100k_tokens": 2.5e-06,
        "input_cost_per_token_batches": 5e-08,
        "output_cost_per_token_batches": 2.5e-07,
        "input_cost_per_token_above_100k_tokens_batches": 2.5e-07,
        "output_cost_per_token_above_100k_tokens_batches": 1.25e-06,
        "cache_creation_input_token_cost": 1.25e-07,
        "cache_creation_input_token_cost_above_100k_tokens": 6.25e-07,
        "cache_creation_input_token_cost_above_1hr": 2e-07,
        "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-06,
        "cache_read_input_token_cost": 1e-08,
        "cache_read_input_token_cost_above_100k_tokens": 5e-08,
        "prompt_cache_min_tokens": 512,
        "supports_adaptive_thinking": True,
        "supports_assistant_prefill": False,
        "supports_forced_tool_use": True,
        "supports_function_calling": True,
        "supports_native_structured_output": True,
        "supports_pdf_input": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_sampling_params": False,
        "supports_tool_choice": True,
        "supports_vision": True,
        "source": "https://platform.claude.com/docs/en/models/haiku-5-5/overview",
    },
}

# https://openrouter.ai/api/v1/models on 2026-10-10: anthropic/claude-haiku-5.5
# (the target the catalog names for ~anthropic/claude-haiku-latest) lists $0.10 / $0.50
# per MTok ($0.50 / $2.50 for prompts from 100k tokens), cache reads $0.01,
# writes $0.125 (1 h: $0.20), 1M context, 128k output, image input, tools,
# tool_choice and response_format, and no sampling parameters.
_OPENROUTER_CLAUDE_HAIKU_5_5: dict[str, Any] = {
    "litellm_provider": "openrouter",
    "mode": "chat",
    "max_input_tokens": 1000000,
    "max_output_tokens": 128000,
    "max_tokens": 128000,
    "input_cost_per_token": 1e-07,  # nosec B105 - a price, not a credential
    "output_cost_per_token": 5e-07,
    "input_cost_per_token_above_100k_tokens": 5e-07,
    "output_cost_per_token_above_100k_tokens": 2.5e-06,
    "cache_read_input_token_cost": 1e-08,
    "cache_read_input_token_cost_above_100k_tokens": 5e-08,
    "cache_creation_input_token_cost": 1.25e-07,
    "cache_creation_input_token_cost_above_100k_tokens": 6.25e-07,
    "cache_creation_input_token_cost_above_1hr": 2e-07,
    "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-06,
    "supports_function_calling": True,
    "supports_pdf_input": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_tool_choice": True,
    "supports_vision": True,
    "source": "https://openrouter.ai/api/v1/models",
}
_MISSING_FROM_LITELLM["openrouter/anthropic/claude-haiku-5.5"] = (
    _OPENROUTER_CLAUDE_HAIKU_5_5
)

# LiteLLM 1.104.2's own entry for the alias: Claude Haiku 4.5's figures, which
# is what the alias was served as on 2026-10-10 (see the module docstring).
_MISSING_FROM_LITELLM["openrouter/~anthropic/claude-haiku-latest"] = {
    "litellm_provider": "openrouter",
    "mode": "chat",
    "max_input_tokens": 200000,
    "max_output_tokens": 64000,
    "max_tokens": 64000,
    "input_cost_per_token": 1e-06,  # nosec B105 - a price, not a credential
    "output_cost_per_token": 5e-06,
    "cache_read_input_token_cost": 1e-07,
    "cache_creation_input_token_cost": 1.25e-06,
    "cache_creation_input_token_cost_above_1hr": 2e-06,
    "supports_function_calling": True,
    "supports_pdf_input": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_tool_choice": True,
    "supports_vision": True,
    "source": "https://openrouter.ai/api/v1/models",
}

_registered = False


def register_missing_models() -> list[str]:
    """Add the entries the installed LiteLLM lacks; return the names added.

    Idempotent and cheap after the first call. Never replaces an entry
    LiteLLM already has.
    """
    global _registered
    if _registered:
        return []
    import litellm

    missing = {
        name: dict(entry)
        for name, entry in _MISSING_FROM_LITELLM.items()
        if name not in litellm.model_cost
    }
    if missing:
        litellm.register_model(missing)
    _registered = True
    return sorted(missing)
