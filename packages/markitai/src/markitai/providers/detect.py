"""Shared LLM provider detection module.

Provides auto-detection of available LLM providers via CLI tools,
OAuth authentication, and environment variables, and the default-model
resolution built on it (``MODEL`` env var first, then detection).

Lives below ``markitai.cli`` so every entry point resolves models the same
way: the CLI, the Python API (``markitai.convert``/``aconvert``) and the
MCP server (through the API). The import-linter contracts forbid the API
and MCP layers from importing ``markitai.cli``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from markitai.constants import (
    DEFAULT_MODEL_WEIGHT,
    PROVIDER_API_KEY_ENV,
    PROVIDER_DEFAULT_MODELS,
)

if TYPE_CHECKING:
    from markitai.config import ModelConfig


@dataclass
class ProviderDetectionResult:
    """Result of LLM provider auto-detection."""

    provider: str
    model: str
    authenticated: bool
    source: str  # "cli", "env", "config"


def _provider_authenticated(provider: str) -> bool:
    """Check one provider's auth status via ``AuthManager``, never raising."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth(provider))
        return status.authenticated
    except Exception:
        return False


def _check_claude_auth() -> bool:
    """Check if Claude CLI is authenticated."""
    return _provider_authenticated("claude-agent")


def _check_copilot_auth() -> bool:
    """Check if Copilot CLI is authenticated."""
    return _provider_authenticated("copilot")


def _check_chatgpt_auth() -> bool:
    """Check if ChatGPT provider is authenticated."""
    from markitai.providers.auth import _check_chatgpt_auth as check_fn

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
        weight = params.get("weight", DEFAULT_MODEL_WEIGHT)
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


def _runtime_dependency_available(module: str) -> bool:
    """Check the current Markitai environment, not the global CLI install."""
    return importlib.util.find_spec(module) is not None


def detect_all_providers() -> list[ProviderDetectionResult]:
    """Auto-detect all available LLM providers.

    Checks each provider independently and returns all that are available,
    ordered by priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ChatGPT (if authenticated via OAuth)
    4. ANTHROPIC_API_KEY environment variable
    5. OPENAI_API_KEY environment variable
    6. GEMINI_API_KEY environment variable
    7. DEEPSEEK_API_KEY environment variable
    8. OPENROUTER_API_KEY environment variable

    Returns:
        List of all detected providers (may be empty).
    """
    results: list[ProviderDetectionResult] = []

    # 1. Check Claude CLI
    if shutil.which("claude") and _runtime_dependency_available("claude_agent_sdk"):
        if _check_claude_auth():
            results.append(
                ProviderDetectionResult(
                    provider="claude-agent",
                    model=PROVIDER_DEFAULT_MODELS["claude-agent"],
                    authenticated=True,
                    source="cli",
                )
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot") and _runtime_dependency_available("copilot"):
        if _check_copilot_auth():
            results.append(
                ProviderDetectionResult(
                    provider="copilot",
                    model=PROVIDER_DEFAULT_MODELS["copilot"],
                    authenticated=True,
                    source="cli",
                )
            )

    # 3. Check ChatGPT (OAuth)
    if _check_chatgpt_auth():
        results.append(
            ProviderDetectionResult(
                provider="chatgpt",
                model=PROVIDER_DEFAULT_MODELS["chatgpt"],
                authenticated=True,
                source="cli",
            )
        )

    # 4-8. Check environment variables
    for provider, env_var in PROVIDER_API_KEY_ENV.items():
        model = PROVIDER_DEFAULT_MODELS[provider]
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


@dataclass
class AutoModelResolution:
    """Models chosen for an LLM run whose ``llm.model_list`` is empty.

    Attributes:
        model_list: Model entries to install as ``llm.model_list`` (empty
            when nothing was found).
        source: ``"env"`` for the ``MODEL`` environment variable,
            ``"detected"`` for provider auto-detection, None when neither
            yielded a model.
        detected: The detected providers (empty unless ``source`` is
            ``"detected"``).
    """

    model_list: list[ModelConfig] = field(default_factory=list)
    source: Literal["env", "detected"] | None = None
    detected: list[ProviderDetectionResult] = field(default_factory=list)

    @property
    def pooled(self) -> bool:
        """Whether several detected providers share one router pool."""
        return len(self.detected) > 1


def resolve_auto_models() -> AutoModelResolution:
    """Resolve default models: ``MODEL`` env var, then provider detection.

    The single precedence rule for an empty ``llm.model_list`` shared by
    the CLI, the Python API and the MCP server. ``MODEL`` is an explicit
    single-model override and wins outright; otherwise every provider
    :func:`detect_all_providers` finds joins one pool.

    Detection may probe CLI auth with ``asyncio.run``, so call this from
    synchronous code (``asyncio.to_thread`` inside an event loop).
    """
    model_env = os.environ.get("MODEL")
    if model_env:
        from markitai.config import LiteLLMParams, ModelConfig

        return AutoModelResolution(
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model=model_env),
                )
            ],
            source="env",
        )

    detected = detect_all_providers()
    if not detected:
        return AutoModelResolution()
    return AutoModelResolution(
        model_list=providers_to_model_configs(detected),
        source="detected",
        detected=detected,
    )


def pooled_providers_notice(
    detected: list[ProviderDetectionResult],
) -> tuple[str, str]:
    """Return ``(message, fix)`` for the several-providers-pooled notice.

    Every detected provider joins one pool and the router spreads requests
    across them, so one document can be cleaned by several vendors. Each
    entry point shows this where its user looks (CLI: stderr warning;
    API/MCP: a loguru warning, stderr by default).
    """
    names = ", ".join(d.model for d in detected)
    return (
        f"Auto-detected {len(detected)} LLM providers; requests are spread "
        f"across all of them: {names}",
        "Pin one model with MODEL=<provider/model> or llm.model_list in the "
        "config file.",
    )


def format_model_list(models: list[str], max_show: int = 3) -> str:
    """Format a list of model names for display.

    Shows up to *max_show* names, with a "+N more" suffix if there are extras.
    """
    shown = ", ".join(models[:max_show])
    extra = len(models) - max_show
    if extra > 0:
        shown += f" (+{extra} more)"
    return shown
