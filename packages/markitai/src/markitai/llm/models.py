"""Model information and cost tracking utilities.

This module provides utilities for:
- Model information caching and lookup
- Model pool fingerprinting for cache scoping
- Cost calculation for LLM responses
- LiteLLM callback logging
"""

from __future__ import annotations

import hashlib
from typing import Any

import litellm
from litellm import completion_cost
from loguru import logger

from markitai.constants import DEFAULT_MAX_OUTPUT_TOKENS
from markitai.model_catalog import register_missing_models

# Every LLM path imports this module before it prices or sizes a request.
register_missing_models()

# Cache for model info to avoid repeated litellm queries
_model_info_cache: dict[str, dict[str, Any]] = {}


def model_list_fingerprint(model_list: list[Any]) -> str:
    """Stable fingerprint of the configured model pool for cache scoping.

    Persistent LLM cache entries are scoped by the model pool that could
    have produced them (the router picks the concrete model only after the
    cache lookup, so a single real model name is not available as a key).
    Only the real model identifiers (``litellm_params.model``) of routable
    entries participate: ``weight == 0`` means "disabled (excluded from
    routing)" and cannot produce results, so disabled entries neither keep
    their stale results servable nor invalidate the pool when added or
    removed. Order and duplicates are normalized away, and credentials/
    endpoints (api_key, api_base) are deliberately excluded so key rotation
    does not invalidate the cache.

    Args:
        model_list: Model configurations (``ModelConfig``-shaped objects
            exposing ``litellm_params.model`` and optionally
            ``litellm_params.weight``, missing weight = enabled).

    Returns:
        ``"pool:<16-hex-char sha256 prefix>"``, or ``"pool:none"`` for an
        empty pool.
    """
    models = sorted(
        {
            item.litellm_params.model
            for item in model_list
            if getattr(item.litellm_params, "weight", 1) > 0
        }
    )
    if not models:
        return "pool:none"
    digest = hashlib.sha256("\n".join(models).encode("utf-8")).hexdigest()[:16]
    return f"pool:{digest}"


def get_model_info_cached(model: str) -> dict[str, Any]:
    """Get model info from litellm with caching.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-3.1-flash-lite-preview")

    Returns:
        Dict with keys:
            - max_input_tokens: int (context window size)
            - max_output_tokens: int (max output tokens)
            - supports_vision: bool (whether model supports images)
        Returns defaults if litellm info unavailable.
    """
    if model in _model_info_cache:
        return _model_info_cache[model]

    # Check if it's a local provider model first
    from markitai.providers import get_local_provider_model_info

    local_info = get_local_provider_model_info(model)
    if local_info is not None:
        # Local provider models have known specs, no need to query litellm
        result = {
            "max_input_tokens": local_info["max_input_tokens"],
            "max_output_tokens": local_info["max_output_tokens"],
            "supports_vision": local_info["supports_vision"],
        }
        _model_info_cache[model] = result
        return result

    # Defaults for non-local providers
    result = {
        "max_input_tokens": 128000,  # Conservative default
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "supports_vision": False,
    }

    try:
        info = litellm.get_model_info(model)
        if info.get("max_input_tokens"):
            result["max_input_tokens"] = info["max_input_tokens"]
        if info.get("max_output_tokens"):
            result["max_output_tokens"] = info["max_output_tokens"]
        supports_vision = info.get("supports_vision")
        if supports_vision is not None:
            result["supports_vision"] = bool(supports_vision)
    except Exception:
        logger.trace(f"[ModelInfo] Could not get info for {model}, using defaults")

    _model_info_cache[model] = result
    return result


def get_model_max_output_tokens(model: str) -> int:
    """Get max_output_tokens for a model using litellm.get_model_info().

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-3.1-flash-lite-preview")

    Returns:
        max_output_tokens value, or DEFAULT_MAX_OUTPUT_TOKENS if unavailable
    """
    return get_model_info_cached(model)["max_output_tokens"]


def cleanup_reasoning_params(model_id: str) -> dict[str, Any]:
    """Request parameters that keep a reasoning model from thinking.

    Every markitai request is a cleanup, a transcription or an image
    description: work without a reasoning step, which a reasoning model
    still bills as output. ``deepseek/deepseek-flash`` thought before every
    answer (70,393 output tokens for a 33k-token PDF, $0.049), and
    ``openai/gpt-6-luna`` spent up to half of its output on reasoning.

    - DeepSeek thinking models: ``thinking: {"type": "disabled"}``.
    - OpenAI models LiteLLM lists as accepting ``reasoning_effort="none"``.
    - Claude Haiku 5.5 thinks adaptively unless told not to and accepts
      ``thinking: {"type": "disabled"}`` at its default effort. Not other
      Claude models: Opus 5.5, Sonnet 5.5 and Fable reject it (HTTP 400),
      and older ones do not think unless asked.
    - Gemini and OpenRouter get nothing: the defaults answer without
      thinking (no thought tokens in their usage), and LiteLLM 1.104 would
      send Gemini 3's ``thinkingLevel`` field to a model alias it cannot
      verify, and refuses ``reasoning_effort`` for OpenRouter.

    Args:
        model_id: Deployment id (e.g. ``"deepseek/deepseek-flash"``).

    Returns:
        LiteLLM parameters to add to the deployment (empty when none apply).
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model_id)
    except Exception:
        return {}
    name = model_id.split("/", 1)[1] if "/" in model_id else model_id
    try:
        if provider == "deepseek":
            if "reasoner" not in name and litellm.supports_reasoning(model_id):
                return {"thinking": {"type": "disabled"}}
        elif provider == "openai":
            from litellm.utils import supports_none_reasoning_effort

            if supports_none_reasoning_effort(model_id):
                return {"reasoning_effort": "none"}
        elif provider == "anthropic" and name.startswith("claude-haiku-5-5"):
            return {"thinking": {"type": "disabled"}}
    except Exception:
        logger.trace(f"[ModelInfo] No reasoning metadata for {model_id}")
    return {}


def get_response_cost(response: Any) -> float | None:
    """Get cost from a LiteLLM ModelResponse.

    For local providers (claude-agent, copilot), the cost is stored in
    _hidden_params["total_cost_usd"] by the provider. For other providers,
    uses litellm.completion_cost().

    Args:
        response: LiteLLM ModelResponse object

    Returns:
        Cost in USD, or None when the model has no known price (an unknown
        price is not a zero one)
    """
    # Check for cost from local provider (stored in _hidden_params)
    hidden_params = getattr(response, "_hidden_params", None)
    if hidden_params and isinstance(hidden_params, dict):
        provider_cost = hidden_params.get("total_cost_usd")
        if provider_cost is not None:
            return float(provider_cost)
        # OpenRouter bills every request itself and returns the amount
        # (usage.cost, kept by LiteLLM as response_cost). The response names
        # the routed model, not the alias asked for (anthropic/claude-haiku-5.5
        # for ~anthropic/claude-haiku-latest), and LiteLLM's table may not
        # list it, so the provider's own figure is the one to charge.
        if hidden_params.get("custom_llm_provider") == "openrouter":
            reported = hidden_params.get("response_cost")
            if isinstance(reported, int | float) and reported >= 0:
                return float(reported)

    # DeepSeek's off-peak hours include the Chinese public holidays, which
    # LiteLLM does not know (markitai.llm.deepseek_hours)
    deepseek = _deepseek_entry(response, hidden_params)
    if deepseek is not None:
        import time

        from markitai.llm.deepseek_hours import deepseek_cost

        return deepseek_cost(
            deepseek,
            getattr(response, "usage", None),
            _sent_at(response, hidden_params),
            time.time(),
        )

    # Fall back to litellm's cost calculation
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        return None
    if cost:
        return cost
    return 0.0 if _has_listed_price(response, hidden_params) else None


#: Where the engine records, on a response, when its request was sent
SENT_AT_PARAM = "markitai_sent_at"


def stamp_sent_at(response: Any, sent_at: float) -> None:
    """Record when *response*'s request was sent (Unix seconds)."""
    hidden_params = getattr(response, "_hidden_params", None)
    if isinstance(hidden_params, dict):
        hidden_params[SENT_AT_PARAM] = sent_at


def _sent_at(response: Any, hidden_params: Any) -> float | None:
    """When the request was sent: the engine's stamp, else the provider's
    ``created`` (DeepSeek's own clock, set when it received the request)."""
    if isinstance(hidden_params, dict):
        stamped = hidden_params.get(SENT_AT_PARAM)
        if isinstance(stamped, int | float):
            return float(stamped)
    created = getattr(response, "created", None)
    return float(created) if isinstance(created, int | float) and created > 0 else None


def _deepseek_entry(response: Any, hidden_params: Any) -> dict[str, Any] | None:
    """The cost entry of a DeepSeek model billed by peak and off-peak hours."""
    if not isinstance(hidden_params, dict):
        return None
    if hidden_params.get("custom_llm_provider") != "deepseek":
        return None
    model = (getattr(response, "model", None) or "").removeprefix("deepseek/")
    entry = litellm.model_cost.get(f"deepseek/{model}") or litellm.model_cost.get(model)
    if isinstance(entry, dict) and isinstance(entry.get("off_peak_pricing"), dict):
        return entry
    return None


def _has_listed_price(response: Any, hidden_params: Any) -> bool:
    """Whether litellm's cost table lists a price for the response's model.

    ``completion_cost`` answers $0 for a model whose entry has no price at
    all: one registered only for its ``model_info`` (``supports_vision``,
    context size) from the config. Only an entry with an actual price, zero
    included (a local model), makes $0 the cost.
    """
    import litellm

    model = getattr(response, "model", None) or ""
    provider = (
        hidden_params.get("custom_llm_provider")
        if isinstance(hidden_params, dict)
        else None
    )
    for key in (f"{provider}/{model}" if provider else None, model):
        entry = litellm.model_cost.get(key) if key else None
        if isinstance(entry, dict) and any(
            "cost_per" in name and value is not None for name, value in entry.items()
        ):
            return True
    return False


def format_cost(cost: float | None) -> str:
    """A request's cost for log lines: ``$0.000123``, or ``unknown``."""
    return "unknown" if cost is None else f"${cost:.6f}"


def context_display_name(context: str) -> str:
    """Extract display name from context for logging.

    Converts full paths to filenames while preserving suffixes like ':images'.
    Examples:
        'C:/path/to/file.pdf:images' -> 'file.pdf:images'
        'file.pdf' -> 'file.pdf'
        '' -> ''

    Args:
        context: Full context path

    Returns:
        Display-friendly name
    """
    if not context:
        return context
    # Split context into path part and suffix (e.g., ':images')
    if (
        ":" in context and context[1:3] != ":\\" and context[1:3] != ":/"
    ):  # Avoid splitting Windows drive letters (C:\ and C:/)
        # Find the last colon that's not part of a Windows path
        parts = context.rsplit(":", 1)
        if len(parts) == 2 and not parts[1].startswith("\\"):
            path_part, suffix = parts
            # Handle both Unix and Windows separators
            name = path_part.replace("\\", "/").strip("/").rsplit("/", 1)[-1]
            return f"{name}:{suffix}"
    # Handle both Unix and Windows separators for cross-platform support
    return context.replace("\\", "/").strip("/").rsplit("/", 1)[-1]
