"""Model information and cost tracking utilities.

This module provides utilities for:
- Model information caching and lookup
- Cost calculation for LLM responses
- LiteLLM callback logging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import litellm
from litellm import completion_cost
from litellm.integrations.custom_logger import CustomLogger
from loguru import logger

from markitai.constants import DEFAULT_MAX_OUTPUT_TOKENS

# Cache for model info to avoid repeated litellm queries
_model_info_cache: dict[str, dict[str, Any]] = {}


def get_model_info_cached(model: str) -> dict[str, Any]:
    """Get model info from litellm with caching.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-2.5-flash")

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
        logger.debug(f"[ModelInfo] Could not get info for {model}, using defaults")

    _model_info_cache[model] = result
    return result


def get_model_max_output_tokens(model: str) -> int:
    """Get max_output_tokens for a model using litellm.get_model_info().

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-2.5-flash")

    Returns:
        max_output_tokens value, or DEFAULT_MAX_OUTPUT_TOKENS if unavailable
    """
    return get_model_info_cached(model)["max_output_tokens"]


def get_response_cost(response: Any) -> float:
    """Get cost from a LiteLLM ModelResponse.

    For local providers (claude-agent, copilot), the cost is stored in
    _hidden_params["total_cost_usd"] by the provider. For other providers,
    uses litellm.completion_cost().

    Args:
        response: LiteLLM ModelResponse object

    Returns:
        Cost in USD, or 0.0 if unavailable
    """
    # Check for cost from local provider (stored in _hidden_params)
    hidden_params = getattr(response, "_hidden_params", None)
    if hidden_params and isinstance(hidden_params, dict):
        provider_cost = hidden_params.get("total_cost_usd")
        if provider_cost is not None:
            return float(provider_cost)

    # Fall back to litellm's cost calculation
    try:
        return completion_cost(completion_response=response) or 0.0
    except Exception:
        return 0.0


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
        ":" in context and context[1:3] != ":\\"
    ):  # Avoid splitting Windows drive letters
        # Find the last colon that's not part of a Windows path
        parts = context.rsplit(":", 1)
        if len(parts) == 2 and not parts[1].startswith("\\"):
            path_part, suffix = parts
            return f"{Path(path_part).name}:{suffix}"
    return Path(context).name


class MarkitaiLLMLogger(CustomLogger):
    """Custom LiteLLM callback logger for capturing additional call details."""

    def __init__(self) -> None:
        self.last_call_details: dict[str, Any] = {}

    def log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Capture details from successful LLM calls."""
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "response_time": slo.get("response_time"),
            "cache_hit": kwargs.get("cache_hit", False),
            "model_id": slo.get("model_id"),
        }

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async version of success event logging."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Capture details from failed LLM calls."""
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "error_code": slo.get("error_code"),
            "error_class": slo.get("error_class"),
        }

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async version of failure event logging."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)
