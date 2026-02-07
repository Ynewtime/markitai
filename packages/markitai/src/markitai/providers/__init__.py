"""Custom LLM providers for markitai.

This module provides custom LiteLLM providers that integrate with local
LLM clients like Claude Code CLI and GitHub Copilot.

Usage:
    # In markitai.json configuration:
    {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "claude-agent/sonnet"
                    }
                }
            ]
        }
    }

Provider prefixes:
    - claude-agent/<model>: Uses Claude Agent SDK (Claude Code CLI authentication)
        Supported models:
        - Aliases (recommended): sonnet, opus, haiku, inherit
          (automatically resolves to latest version via LiteLLM database)
        - Full model strings: claude-sonnet-4-5-20250929, claude-opus-4-6, etc.
    - copilot/<model>: Uses GitHub Copilot SDK (Copilot CLI authentication)
        Use direct model names: gpt-4.1, claude-sonnet-4.5, gemini-2.5-pro, etc.
"""

from __future__ import annotations

import importlib.util
import re
from typing import TYPE_CHECKING

from loguru import logger

from markitai.constants import (
    CLAUDE_CODE_ALIASES,
    COPILOT_MODEL_PRICING,
    LOCAL_PROVIDER_DEFAULT_MODEL_INFO,
)

# Import auth module for public API
from markitai.providers.auth import (
    AuthManager,
    AuthStatus,
    get_auth_resolution_hint,
)

# Import error classes for public API
from markitai.providers.errors import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    QuotaError,
    SDKNotAvailableError,
)

# Import JSON mode module for public API
from markitai.providers.json_mode import (
    StructuredOutputHandler,
    clean_control_characters,
)

# Import timeout module for public API
from markitai.providers.timeout import (
    TimeoutConfig,
    calculate_timeout,
    calculate_timeout_from_messages,
)

if TYPE_CHECKING:
    from litellm.llms.custom_llm import CustomLLM

# Registry of custom providers
_providers: dict[str, CustomLLM] = {}
_registered: bool = False


def _is_tiktoken_available() -> bool:
    """Check if tiktoken is available."""
    return importlib.util.find_spec("tiktoken") is not None


# Cache for tiktoken encodings to avoid repeated initialization
_tiktoken_encodings: dict[str, object] = {}


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text for a given model.

    Uses tiktoken for OpenAI models (accurate), falls back to character
    estimation for other models or when tiktoken is unavailable.

    Args:
        text: Text to count tokens for
        model: Model name (e.g., "gpt-4.1", "claude-sonnet-4.5")

    Returns:
        Estimated token count
    """
    # Try tiktoken for OpenAI models
    if _is_tiktoken_available() and model.startswith(("gpt-", "o1", "o3")):
        try:
            import tiktoken

            # Determine encoding based on model
            # GPT-4+ uses cl100k_base, GPT-5+ uses o200k_base
            if model.startswith(("gpt-5", "o1", "o3")):
                encoding_name = "o200k_base"
            else:
                encoding_name = "cl100k_base"

            # Use cached encoding or create new one
            if encoding_name not in _tiktoken_encodings:
                _tiktoken_encodings[encoding_name] = tiktoken.get_encoding(
                    encoding_name
                )

            encoding = _tiktoken_encodings[encoding_name]
            return len(encoding.encode(text))  # type: ignore[union-attr]
        except Exception:
            pass  # Fall through to estimation

    # Fallback: character-based estimation
    # Rough estimate: 1 token ≈ 4 characters for English
    # This is less accurate but works for all models
    return len(text) // 4


# Cache for fuzzy model matching to avoid repeated LiteLLM database scans
_litellm_model_match_cache: dict[str, str | None] = {}


def _find_litellm_model_fuzzy(model: str) -> str | None:
    """Find the best matching model in LiteLLM's database using fuzzy matching.

    Uses component-based matching to find models with similar naming patterns.
    For example, 'claude-haiku-4.5' will match 'claude-haiku-4-5' because they
    share the same components: ['claude', 'haiku', '4', '5'].

    Args:
        model: Model name to search for (e.g., "claude-haiku-4.5")

    Returns:
        Best matching LiteLLM model name, or None if no good match found
    """
    # Check cache first
    if model in _litellm_model_match_cache:
        return _litellm_model_match_cache[model]

    try:
        import litellm

        # Split model name into components (split on both - and .)
        components = re.split(r"[-.]", model.lower())
        components = [c for c in components if c]  # Remove empty strings

        if not components:
            return None

        best_match: str | None = None
        best_score = 0

        for litellm_model in litellm.model_cost:
            # Skip models with provider prefixes (azure/, bedrock/, etc.)
            # We want the base model names for pricing lookup
            if "/" in litellm_model:
                continue

            model_lower = litellm_model.lower()
            model_components = re.split(r"[-.]", model_lower)

            # Calculate component match score
            score = sum(1 for c in components if c in model_components)

            # Prefer higher scores, and shorter names for ties (more generic)
            if score > best_score or (
                score == best_score
                and best_match
                and len(litellm_model) < len(best_match)
            ):
                best_score = score
                best_match = litellm_model

        # Require at least half of the components to match
        result = best_match if best_score >= len(components) / 2 else None

        # Cache the result
        _litellm_model_match_cache[model] = result

        if result and result != model:
            logger.debug(f"[Providers] Fuzzy matched '{model}' -> '{result}'")

        return result

    except Exception as e:
        logger.debug(f"[Providers] Fuzzy match failed for '{model}': {e}")
        return None


class CopilotCostResult:
    """Result of Copilot cost calculation with estimation metadata."""

    __slots__ = ("cost_usd", "is_estimated", "source", "matched_model")

    def __init__(
        self,
        cost_usd: float,
        is_estimated: bool,
        source: str,
        matched_model: str | None = None,
    ) -> None:
        self.cost_usd = cost_usd
        self.is_estimated = is_estimated
        self.source = source  # "litellm", "litellm_fuzzy", "fallback", "none"
        self.matched_model = matched_model


def calculate_copilot_cost(
    model: str, input_tokens: int, output_tokens: int
) -> CopilotCostResult:
    """Calculate estimated cost for Copilot API usage.

    Attempts to find pricing in the following order:
    1. Exact match in LiteLLM's model pricing database
    2. Fuzzy match in LiteLLM's database (for naming convention differences)
    3. Fallback to COPILOT_MODEL_PRICING constants

    Note: All costs are ESTIMATED. Copilot is subscription-based, so actual
    costs are included in your subscription fee, not billed per token.

    Args:
        model: Model name (e.g., "gpt-4.1", "claude-sonnet-4.5")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        CopilotCostResult with cost_usd and estimation metadata
    """
    import litellm

    # 1. Try exact match in LiteLLM
    try:
        info = litellm.get_model_info(model)
        input_cost = info.get("input_cost_per_token")
        output_cost = info.get("output_cost_per_token")

        if input_cost is not None and output_cost is not None:
            cost = input_tokens * input_cost + output_tokens * output_cost
            return CopilotCostResult(
                cost_usd=cost,
                is_estimated=True,  # Always estimated for Copilot (subscription)
                source="litellm",
                matched_model=model,
            )
    except Exception:
        pass

    # 2. Try fuzzy match in LiteLLM
    fuzzy_match = _find_litellm_model_fuzzy(model)
    if fuzzy_match:
        try:
            info = litellm.get_model_info(fuzzy_match)
            input_cost = info.get("input_cost_per_token")
            output_cost = info.get("output_cost_per_token")

            if input_cost is not None and output_cost is not None:
                cost = input_tokens * input_cost + output_tokens * output_cost
                return CopilotCostResult(
                    cost_usd=cost,
                    is_estimated=True,
                    source="litellm_fuzzy",
                    matched_model=fuzzy_match,
                )
        except Exception:
            pass

    # 3. Fallback to hardcoded pricing table
    pricing = COPILOT_MODEL_PRICING.get(model)
    if pricing is None:
        # Try prefix matching for versioned models like "gpt-5.1-codex-mini"
        for prefix in sorted(COPILOT_MODEL_PRICING, key=len, reverse=True):
            if model.startswith(prefix):
                pricing = COPILOT_MODEL_PRICING[prefix]
                break

    if pricing is not None:
        input_price, output_price = pricing
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
        return CopilotCostResult(
            cost_usd=cost,
            is_estimated=True,
            source="fallback",
            matched_model=None,
        )

    # 4. No pricing found
    return CopilotCostResult(
        cost_usd=0.0,
        is_estimated=True,
        source="none",
        matched_model=None,
    )


# Models deprecated on 2025-02-13
# Key: deprecated model name, Value: recommended replacement
DEPRECATED_MODELS: dict[str, str] = {
    "gpt-4o": "gpt-5.2",
    "gpt-4.1": "gpt-5.2",
    "gpt-4.1-mini": "gpt-5.2",
    "o4-mini": "gpt-5.2",
    "gpt-5": "gpt-5.2",
}


def check_deprecated_models(models: list[str]) -> list[str]:
    """Check for deprecated models and return warning messages.

    Args:
        models: List of model identifiers (e.g., ["copilot/gpt-4o", "openai/gpt-4.1"])

    Returns:
        List of deprecation warning messages
    """
    warnings: list[str] = []
    seen: set[str] = set()

    for model in models:
        # Extract the actual model name (strip provider prefix)
        if "/" in model:
            model_name = model.split("/", 1)[1]
        else:
            model_name = model

        # Check if model is deprecated
        if model_name in DEPRECATED_MODELS and model_name not in seen:
            seen.add(model_name)
            replacement = DEPRECATED_MODELS[model_name]
            warnings.append(
                f"⚠️  Model '{model_name}' was retired on February 13, 2025."
                f"\n   Please migrate to: {replacement}"
            )

    return warnings


def validate_local_provider_deps(models: list[str]) -> list[str]:
    """Validate that required SDKs are installed for configured local providers.

    This function should be called during startup to provide friendly error
    messages before attempting to use local providers.

    Args:
        models: List of model identifiers (e.g., ["claude-agent/sonnet", "copilot/gpt-4.1"])

    Returns:
        List of warning/error messages (empty if all dependencies are satisfied)
    """
    import shutil

    warnings: list[str] = []

    # Check for claude-agent models
    uses_claude_agent = any(m.startswith("claude-agent/") for m in models)
    if uses_claude_agent:
        if not importlib.util.find_spec("claude_agent_sdk"):
            warnings.append(
                "⚠️  claude-agent/ models require Claude Agent SDK."
                "\n   Install: uv tool install 'markitai[claude-agent]' --upgrade  # or: uv add claude-agent-sdk (dev mode)"
            )
        elif not shutil.which("claude"):
            warnings.append(
                "⚠️  Claude Code CLI not installed. claude-agent/ models require CLI auth."
                "\n   Install: pnpm add -g @anthropic-ai/claude-code"
                "\n   Auth: claude auth login"
            )

    # Check for copilot models
    uses_copilot = any(m.startswith("copilot/") for m in models)
    if uses_copilot:
        if not importlib.util.find_spec("copilot"):
            warnings.append(
                "⚠️  copilot/ models require GitHub Copilot SDK."
                "\n   Install: uv tool install 'markitai[copilot]' --upgrade  # or: uv add github-copilot-sdk (dev mode)"
            )
        elif not shutil.which("copilot"):
            warnings.append(
                "⚠️  Copilot CLI not installed. copilot/ models require CLI auth."
                "\n   Install: pnpm add -g @github/copilot"
                "\n   Auth: copilot auth login"
            )

    return warnings


def register_providers() -> None:
    """Register all custom providers with LiteLLM.

    This function should be called once before using custom provider models.
    It's safe to call multiple times - subsequent calls are no-ops.

    Only registers providers whose underlying SDKs are available.
    """
    global _registered
    if _registered:
        return

    import litellm

    # Try to register Claude Agent provider (only if SDK is available)
    try:
        from markitai.providers.auth import _is_claude_agent_sdk_available
        from markitai.providers.claude_agent import ClaudeAgentProvider

        if _is_claude_agent_sdk_available():
            provider = ClaudeAgentProvider()
            litellm.custom_provider_map.append(
                {"provider": "claude-agent", "custom_handler": provider}
            )
            _providers["claude-agent"] = provider
            logger.debug("[Providers] Registered claude-agent provider")
        else:
            logger.debug(
                "[Providers] claude-agent provider skipped: claude_agent_sdk not installed"
            )
    except ImportError as e:
        logger.debug(f"[Providers] claude-agent provider not available: {e}")
    except Exception as e:
        logger.warning(f"[Providers] Failed to register claude-agent provider: {e}")

    # Try to register Copilot provider (only if SDK is available)
    try:
        from markitai.providers.auth import _is_copilot_sdk_available
        from markitai.providers.copilot import CopilotProvider

        if _is_copilot_sdk_available():
            provider = CopilotProvider()
            litellm.custom_provider_map.append(
                {"provider": "copilot", "custom_handler": provider}
            )
            _providers["copilot"] = provider
            logger.debug("[Providers] Registered copilot provider")
        else:
            logger.debug(
                "[Providers] copilot provider skipped: copilot SDK not installed"
            )
    except ImportError as e:
        logger.debug(f"[Providers] copilot provider not available: {e}")
    except Exception as e:
        logger.warning(f"[Providers] Failed to register copilot provider: {e}")

    _registered = True


def get_provider(name: str) -> CustomLLM | None:
    """Get a registered provider by name.

    Args:
        name: Provider name (e.g., "claude-agent", "copilot")

    Returns:
        CustomLLM instance or None if not registered
    """
    return _providers.get(name)


def is_local_provider_model(model: str) -> bool:
    """Check if a model uses a local provider.

    Local providers (claude-agent/, copilot/) use CLI authentication
    and support vision-capable models like Claude and GPT-4o/5.2.

    Args:
        model: Model identifier (e.g., "claude-agent/sonnet")

    Returns:
        True if model uses a local provider
    """
    return model.startswith(("claude-agent/", "copilot/"))


def is_local_provider_available(model: str) -> bool:
    """Check if a local provider model's SDK is available.

    For non-local provider models, always returns True.
    For local providers, checks if the underlying SDK is installed.

    Args:
        model: Model identifier (e.g., "claude-agent/sonnet", "copilot/gpt-4.1")

    Returns:
        True if the model can be used (SDK available or not a local provider)
    """
    if model.startswith("claude-agent/"):
        return importlib.util.find_spec("claude_agent_sdk") is not None
    if model.startswith("copilot/"):
        return importlib.util.find_spec("copilot") is not None
    # Non-local provider models are always "available" (API key validity checked elsewhere)
    return True


# Cache for dynamic Claude model lookup to avoid repeated LiteLLM database scans
_claude_model_cache: dict[str, str] = {}


def _find_latest_claude_model(alias: str) -> str | None:
    """Dynamically find the latest Claude model in LiteLLM for a given alias.

    This function searches LiteLLM's model database to find the newest version
    of a Claude model matching the alias (haiku, sonnet, opus). This ensures
    automatic compatibility with new Claude versions without code changes.

    Args:
        alias: Claude Code alias (e.g., "haiku", "sonnet", "opus")

    Returns:
        LiteLLM model name (e.g., "claude-haiku-4-5") or None if not found
    """
    # Check cache first
    if alias in _claude_model_cache:
        return _claude_model_cache[alias]

    # Handle "inherit" alias - defaults to sonnet
    if alias == "inherit":
        result = _find_latest_claude_model("sonnet")
        if result:
            _claude_model_cache["inherit"] = result
        return result

    try:
        import litellm

        model_cost = litellm.model_cost
        candidates: list[
            tuple[int, int, bool, str]
        ] = []  # (major, minor, no_date, name)

        for model in model_cost:
            # Only look at native Anthropic models (no provider prefix)
            if "/" in model or "anthropic." in model:
                continue
            if not model.startswith("claude-"):
                continue
            if alias.lower() not in model.lower():
                continue

            # Parse version: remove date suffix first
            has_date = bool(re.search(r"-\d{8}$", model))
            model_no_date = re.sub(r"-\d{8}$", "", model)

            # Pattern 1: claude-{model}-{major}-{minor} (e.g., claude-haiku-4-5)
            match1 = re.match(r"^claude-\w+-(\d+)-(\d+)$", model_no_date)
            if match1:
                major, minor = int(match1.group(1)), int(match1.group(2))
                candidates.append((major, minor, not has_date, model))
                continue

            # Pattern 2: claude-{model}-{major} (e.g., claude-opus-4)
            match2 = re.match(r"^claude-\w+-(\d+)$", model_no_date)
            if match2:
                major = int(match2.group(1))
                candidates.append((major, 0, not has_date, model))
                continue

            # Pattern 3: claude-{major}-{minor}-{model} (e.g., claude-3-5-haiku)
            match3 = re.match(r"^claude-(\d+)-(\d+)-\w+$", model_no_date)
            if match3:
                major, minor = int(match3.group(1)), int(match3.group(2))
                candidates.append((major, minor, not has_date, model))

        if not candidates:
            return None

        # Sort: highest version first, prefer no-date suffix
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        result = candidates[0][3]

        # Cache the result
        _claude_model_cache[alias] = result
        logger.debug(f"[Providers] Resolved Claude alias '{alias}' → '{result}'")
        return result

    except Exception as e:
        logger.debug(f"[Providers] Failed to find Claude model for '{alias}': {e}")
        return None


def _resolve_litellm_model(model: str) -> str | None:
    """Resolve local provider model to LiteLLM model name.

    Args:
        model: Local provider model (e.g., "claude-agent/sonnet", "copilot/gpt")

    Returns:
        LiteLLM model name or None if not a local provider model
    """
    if not is_local_provider_model(model):
        return None

    if model.startswith("claude-agent/"):
        # Extract model name after prefix
        model_name = model.replace("claude-agent/", "")
        # Check if it's a known alias, use dynamic lookup
        if model_name in CLAUDE_CODE_ALIASES:
            return _find_latest_claude_model(model_name)
        # Otherwise use as-is (for full model strings like claude-sonnet-4-5-20250929)
        return model_name

    if model.startswith("copilot/"):
        # Copilot models: strip prefix and use directly
        # e.g., "copilot/gpt-4.1" → "gpt-4.1"
        return model.replace("copilot/", "")

    return None


def get_local_provider_model_info(model: str) -> dict[str, int | bool] | None:
    """Get model info for local provider models.

    Inherits model information from LiteLLM's model database by resolving
    local provider aliases to their corresponding LiteLLM model names.

    Args:
        model: Model identifier (e.g., "claude-agent/sonnet", "copilot/gpt-4o")

    Returns:
        Dict with max_input_tokens, max_output_tokens, supports_vision
        or None if not a local provider model
    """
    litellm_model = _resolve_litellm_model(model)
    if litellm_model is None:
        return None

    # Try to get info from LiteLLM's model database
    try:
        import litellm

        info = litellm.get_model_info(litellm_model)
        max_input = info.get("max_input_tokens")
        max_output = info.get("max_output_tokens")
        supports_vision = info.get("supports_vision")
        return {
            "max_input_tokens": max_input if isinstance(max_input, int) else 128000,
            "max_output_tokens": max_output if isinstance(max_output, int) else 8192,
            "supports_vision": bool(supports_vision)
            if supports_vision is not None
            else True,
        }
    except Exception:
        # LiteLLM lookup failed, use defaults
        logger.debug(
            f"[Providers] Could not get LiteLLM info for {litellm_model}, using defaults"
        )
        return dict(LOCAL_PROVIDER_DEFAULT_MODEL_INFO)


__all__ = [
    # Core provider functions
    "register_providers",
    "validate_local_provider_deps",
    "check_deprecated_models",
    "DEPRECATED_MODELS",
    "get_provider",
    "is_local_provider_model",
    "is_local_provider_available",
    "get_local_provider_model_info",
    "count_tokens",
    "calculate_copilot_cost",
    "CopilotCostResult",
    # Error classes
    "ProviderError",
    "AuthenticationError",
    "QuotaError",
    "ProviderTimeoutError",
    "SDKNotAvailableError",
    # Auth module
    "AuthManager",
    "AuthStatus",
    "get_auth_resolution_hint",
    # Timeout module
    "TimeoutConfig",
    "calculate_timeout",
    "calculate_timeout_from_messages",
    # JSON mode module
    "StructuredOutputHandler",
    "clean_control_characters",
]
