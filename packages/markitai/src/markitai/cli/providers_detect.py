"""Shared LLM provider detection module.

Provides auto-detection of available LLM providers via CLI tools,
OAuth authentication, and environment variables.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markitai.config import ModelConfig


@dataclass
class ProviderDetectionResult:
    """Result of LLM provider auto-detection."""

    provider: str
    model: str
    authenticated: bool
    source: str  # "cli", "env", "config"


def _check_claude_auth() -> bool:
    """Check if Claude CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("claude-agent"))
        return status.authenticated
    except Exception:
        return False


def _check_copilot_auth() -> bool:
    """Check if Copilot CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("copilot"))
        return status.authenticated
    except Exception:
        return False


def _check_chatgpt_auth() -> bool:
    """Check if ChatGPT provider is authenticated."""
    from markitai.providers.auth import _check_chatgpt_auth as check_fn

    try:
        status = check_fn()
        return status.authenticated
    except Exception:
        return False


def _check_gemini_cli_auth() -> bool:
    """Check if Gemini CLI is authenticated."""
    from markitai.providers.auth import _check_gemini_cli_auth as check_fn

    try:
        status = check_fn()
        return status.authenticated
    except Exception:
        return False


def get_active_models_from_config(
    model_list: list[dict[str, Any]],
) -> list[str]:
    """Extract active model names (weight > 0) from config model_list.

    Args:
        model_list: Raw model_list dicts from config (each has litellm_params).

    Returns:
        List of model identifiers with positive weight.
    """
    active: list[str] = []
    for entry in model_list:
        params = entry.get("litellm_params", {})
        model = params.get("model", "")
        weight = params.get("weight", 1)  # default weight is 1 (enabled)
        if model and weight > 0:
            active.append(model)
    return active


def providers_to_model_configs(
    providers: list[ProviderDetectionResult],
) -> list[ModelConfig]:
    """Convert detected providers to ModelConfig list for LLM router.

    Args:
        providers: Detected provider results from detect_all_providers().

    Returns:
        List of ModelConfig instances ready for cfg.llm.model_list.
    """
    from markitai.config import LiteLLMParams, ModelConfig

    return [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model=p.model),
        )
        for p in providers
    ]


def detect_all_providers() -> list[ProviderDetectionResult]:
    """Auto-detect all available LLM providers.

    Checks each provider independently and returns all that are available,
    ordered by priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ChatGPT (if authenticated via OAuth)
    4. Gemini CLI (if authenticated via OAuth)
    5. ANTHROPIC_API_KEY environment variable
    6. OPENAI_API_KEY environment variable
    7. GEMINI_API_KEY environment variable
    8. DEEPSEEK_API_KEY environment variable
    9. OPENROUTER_API_KEY environment variable

    Returns:
        List of all detected providers (may be empty).
    """
    results: list[ProviderDetectionResult] = []

    # 1. Check Claude CLI
    if shutil.which("claude"):
        if _check_claude_auth():
            results.append(
                ProviderDetectionResult(
                    provider="claude-agent",
                    model="claude-agent/sonnet",
                    authenticated=True,
                    source="cli",
                )
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot"):
        if _check_copilot_auth():
            results.append(
                ProviderDetectionResult(
                    provider="copilot",
                    model="copilot/claude-sonnet-4.5",
                    authenticated=True,
                    source="cli",
                )
            )

    # 3. Check ChatGPT (OAuth)
    if _check_chatgpt_auth():
        results.append(
            ProviderDetectionResult(
                provider="chatgpt",
                model="chatgpt/gpt-5.2",
                authenticated=True,
                source="cli",
            )
        )

    # 4. Check Gemini CLI (OAuth)
    if _check_gemini_cli_auth():
        results.append(
            ProviderDetectionResult(
                provider="gemini-cli",
                model="gemini-cli/gemini-2.5-pro",
                authenticated=True,
                source="cli",
            )
        )

    # 5-9. Check environment variables
    env_providers = [
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic/claude-sonnet-4-5-20250929"),
        ("OPENAI_API_KEY", "openai", "openai/gpt-5.2"),
        ("GEMINI_API_KEY", "gemini", "gemini/gemini-2.5-flash"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek/deepseek-chat"),
        ("OPENROUTER_API_KEY", "openrouter", "openrouter/google/gemini-2.5-flash"),
    ]
    for env_var, provider, model in env_providers:
        if os.environ.get(env_var):
            results.append(
                ProviderDetectionResult(
                    provider=provider,
                    model=model,
                    authenticated=True,
                    source="env",
                )
            )

    return results


def detect_first_provider() -> ProviderDetectionResult | None:
    """Auto-detect the highest-priority available LLM provider.

    Returns:
        ProviderDetectionResult for the best provider, or None.
    """
    providers = detect_all_providers()
    return providers[0] if providers else None


def format_model_list(models: list[str], max_show: int = 3) -> str:
    """Format a list of model names for display.

    Shows up to *max_show* names, with a "+N more" suffix if there are extras.
    """
    shown = ", ".join(models[:max_show])
    extra = len(models) - max_show
    if extra > 0:
        shown += f" (+{extra} more)"
    return shown
