"""Model entries markitai ships for models its pinned LiteLLM does not list.

A default whose model is missing from LiteLLM's cost table is priced as
unknown: its requests land in ``unpriced_requests``, the batch summary says
"cost unknown", and ``max_cost_per_document_usd`` cannot count them. LiteLLM
1.104.2 (the newest stable release on 2026-10-10) has no entry for
``claude-haiku-5-5``, the Anthropic default since 1.2.1.

Each entry here is LiteLLM's own upstream entry (main branch, 2026-10-10),
checked against the provider's pricing page. It is registered only when the
installed LiteLLM has no entry of that name, so a LiteLLM release that adds
the model wins and the copy here goes dormant.
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
