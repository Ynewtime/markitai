"""Provider-neutral connection and model discovery for CLI and serve UIs.

Remote catalog calls use HTTP directly, so importing this module never loads
provider SDKs. Local connection readiness still verifies that the optional
runtime package is installed in the current Markitai environment.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx

if TYPE_CHECKING:
    from markitai.config import ModelConfig

_AUTH_CACHE_S = 30
_REMOTE_CACHE_S = 300
_STALE_CACHE_S = 24 * 60 * 60
_REMOTE_TIMEOUT = httpx.Timeout(10.0, connect=3.0)
_COPILOT_TIMEOUT_S = 15.0
# Public OAuth endpoint, not a credential (Bandit keys off the variable name).
_CLAUDE_OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"  # nosec B105
_CLAUDE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


@dataclass(slots=True)
class _CacheEntry:
    value: dict[str, Any]
    fetched_at: float


_cache: dict[str, _CacheEntry] = {}
_connection_cache: dict[str, _CacheEntry] = {}
_locks: dict[str, asyncio.Lock] = {}


def _local_runtime_available(provider: str) -> bool:
    """Whether this runtime can execute an authenticated local provider."""
    dependency = {
        "claude-agent": "claude_agent_sdk",
        "copilot": "copilot",
    }.get(provider)
    return dependency is None or importlib.util.find_spec(dependency) is not None


def _provider_of(model: str) -> str:
    return model.split("/", 1)[0].lower() if "/" in model else "custom"


def provider_label(provider: str) -> str:
    """Return the conventional display casing for a provider identifier."""
    return {
        "claude-agent": "Claude Agent",
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
    }.get(provider, provider.replace("_", " ").title())


_PROVIDER_DEFAULT_API_BASES = {
    "anthropic": "https://api.anthropic.com/v1",
    "claude-agent": "https://api.anthropic.com/v1",
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://127.0.0.1:11434",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def provider_default_api_base(provider: str) -> str | None:
    """Return the effective built-in API base for a known provider."""
    return _PROVIDER_DEFAULT_API_BASES.get(provider.lower())


def _credential_identity(api_key: str | None) -> str:
    if not api_key:
        return "none"
    if api_key.startswith("env:"):
        return api_key
    return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _cache_key(provider: str, api_key: str | None, api_base: str | None) -> str:
    return "|".join((provider, api_base or "", _credential_identity(api_key)))


def _candidate(
    model: str, label: str | None = None, *, vision: bool = False
) -> dict[str, Any]:
    return {
        "model": model,
        "label": label or model.split("/", 1)[-1],
        "supports_vision": vision,
    }


def _result(
    provider: str,
    *,
    status: str,
    source: str,
    authoritative: bool,
    models: list[dict[str, Any]],
    detail: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": provider,
        "status": status,
        "source": source,
        "authoritative": authoritative,
        "cached": False,
        "stale": False,
        "models": models,
    }
    if detail:
        payload["detail"] = detail[:240]
    return payload


async def detect_provider_connections(
    configured: list[ModelConfig] | None = None,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return local/env/configured/common provider connection cards."""
    from markitai.providers.auth import AuthManager

    configured_key = [
        (
            model.litellm_params.model,
            model.litellm_params.api_base,
            model.litellm_params.weight,
        )
        for model in configured or []
    ]
    env_key = [
        name
        for name in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        )
        if os.environ.get(name)
    ]
    key = hashlib.sha256(repr((configured_key, env_key)).encode()).hexdigest()
    cached = _connection_cache.get(key)
    if (
        not refresh
        and cached is not None
        and time.monotonic() - cached.fetched_at < _AUTH_CACHE_S
    ):
        return [dict(card) for card in cached.value["providers"]]

    cards: list[dict[str, Any]] = []
    auth = AuthManager()

    async def local_card(
        binary: str, provider: str, label: str, default_model: str
    ) -> dict[str, Any] | None:
        if shutil.which(binary) is None:
            return None
        try:
            status = await asyncio.wait_for(
                auth.check_auth(provider, force_refresh=True), timeout=10.0
            )
        except Exception:
            status = None
        authenticated = status is not None and status.authenticated
        runtime_ready = _local_runtime_available(provider)
        return {
            "id": provider,
            "provider": provider,
            "label": label,
            "kind": "local_cli",
            "status": (
                "ready"
                if authenticated and runtime_ready
                else "missing_dependency"
                if authenticated
                else "needs_auth"
            ),
            "source": "cli",
            "default_model": default_model,
            "supports_discovery": runtime_ready,
        }

    async def chatgpt_card() -> dict[str, Any] | None:
        try:
            status = await asyncio.wait_for(
                auth.check_auth("chatgpt", force_refresh=True), timeout=10.0
            )
        except Exception:
            return None
        if not status.authenticated:
            return None
        return {
            "id": "chatgpt",
            "provider": "chatgpt",
            "label": "ChatGPT (Codex OAuth)",
            "kind": "oauth",
            "status": "ready",
            "source": "oauth",
            "default_model": "chatgpt/gpt-5.4-mini",
            "supports_discovery": True,
        }

    detected_cards = await asyncio.gather(
        local_card("claude", "claude-agent", "Claude Code CLI", "claude-agent/sonnet"),
        local_card(
            "copilot", "copilot", "GitHub Copilot CLI", "copilot/claude-haiku-4.5"
        ),
        chatgpt_card(),
    )
    cards.extend(card for card in detected_cards if card is not None)

    env_providers = (
        ("ANTHROPIC_API_KEY", "anthropic", "Anthropic"),
        ("OPENAI_API_KEY", "openai", "OpenAI"),
        ("GEMINI_API_KEY", "gemini", "Gemini"),
        ("DEEPSEEK_API_KEY", "deepseek", "DeepSeek"),
        ("OPENROUTER_API_KEY", "openrouter", "OpenRouter"),
    )
    for env_var, provider, label in env_providers:
        if os.environ.get(env_var):
            cards.append(
                {
                    "id": f"env:{provider}",
                    "provider": provider,
                    "label": label,
                    "kind": "environment",
                    "status": "ready",
                    "source": env_var,
                    "credential": f"env:{env_var}",
                    "supports_discovery": True,
                }
            )

    local_connections = {
        str(card["provider"])
        for card in cards
        if card.get("kind") in {"local_cli", "oauth"}
    }
    seen_configured: set[tuple[str, str | None]] = set()
    for model in configured or []:
        provider = _provider_of(model.litellm_params.model)
        connection = (provider, model.litellm_params.api_base)
        if connection in seen_configured:
            continue
        seen_configured.add(connection)
        # A saved local model is not a second connection. The detected card
        # remains the one entry and can still discover/add further models.
        if provider in local_connections and model.litellm_params.api_base is None:
            continue
        cards.append(
            {
                "id": f"configured:{len(seen_configured)}",
                "provider": provider,
                "label": provider_label(provider),
                "kind": "configured",
                "status": "ready" if model.litellm_params.weight > 0 else "disabled",
                "source": "config",
                "api_base_configured": bool(model.litellm_params.api_base),
                "supports_discovery": True,
            }
        )

    present = {card["provider"] for card in cards}
    for provider, label in (
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("gemini", "Gemini"),
        ("ollama", "Ollama"),
        ("deepseek", "DeepSeek"),
        ("openrouter", "OpenRouter"),
        ("azure", "Azure OpenAI"),
        ("custom", "OpenAI-compatible endpoint"),
    ):
        if provider not in present:
            cards.append(
                {
                    "id": f"common:{provider}",
                    "provider": provider,
                    "label": label,
                    "kind": "common",
                    "status": "needs_credentials"
                    if provider != "ollama"
                    else "unknown",
                    "source": "built_in",
                    "supports_discovery": True,
                }
            )
    _connection_cache[key] = _CacheEntry(
        value={"providers": cards}, fetched_at=time.monotonic()
    )
    return cards


def startup_model_candidates(connections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ready connection cards to conservative startup defaults."""
    result: list[dict[str, Any]] = []
    defaults = {
        "anthropic": "anthropic/claude-haiku-4-5",
        "openai": "openai/gpt-5.4-nano",
        "gemini": "gemini/gemini-3.1-flash-lite-preview",
        "deepseek": "deepseek/deepseek-v4-flash",
        "openrouter": "openrouter/google/gemini-3.1-flash-lite",
    }
    for card in connections:
        if card.get("status") != "ready":
            continue
        provider = str(card.get("provider", ""))
        model = card.get("default_model") or defaults.get(provider)
        if not isinstance(model, str) or not model:
            continue
        label = card.get("label", provider)
        if card.get("kind") == "environment":
            label = f"{card.get('source', provider)} (environment)"
        result.append(
            {
                "provider": provider,
                "model": model,
                "label": label,
                "requires_api_key": False,
            }
        )
    return result


async def _copilot_models() -> dict[str, Any]:
    if importlib.util.find_spec("copilot") is None:
        return _result(
            "copilot",
            status="unavailable",
            source="copilot_sdk",
            authoritative=False,
            models=[],
            detail="GitHub Copilot SDK is required for live model discovery",
        )
    from copilot import CopilotClient  # type: ignore[import-not-found]

    client: Any = None
    try:
        client = CopilotClient()
        await client.start()
        models = await client.list_models()
        candidates = []
        for model in models:
            model_id = str(model.id)
            if re.match(r"^(?:o1|o3)(?:-|$)", model_id, re.IGNORECASE):
                continue
            policy = getattr(model, "policy", None)
            if policy is not None and getattr(policy, "state", "enabled") == "disabled":
                continue
            supports = getattr(getattr(model, "capabilities", None), "supports", None)
            candidates.append(
                _candidate(
                    f"copilot/{model_id}",
                    str(getattr(model, "name", model_id)),
                    vision=bool(getattr(supports, "vision", False)),
                )
            )
        return _result(
            "copilot",
            status="ok",
            source="copilot_sdk",
            authoritative=True,
            models=candidates,
        )
    finally:
        if client is not None:
            try:
                await asyncio.wait_for(client.stop(), timeout=3.0)
            except Exception:
                pass


async def _refresh_claude_oauth_credentials(
    credentials: dict[str, Any],
) -> dict[str, Any]:
    """Refresh an expired Claude Code token and persist the rotated credentials."""
    from markitai.providers.auth import _store_claude_oauth_credentials

    refresh_token = credentials.get("refreshToken")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise RuntimeError("Claude Code OAuth refresh token is unavailable")

    async with httpx.AsyncClient(
        timeout=_REMOTE_TIMEOUT, follow_redirects=False
    ) as client:
        response = await client.post(
            _CLAUDE_OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _CLAUDE_OAUTH_CLIENT_ID,
            },
        )
        response.raise_for_status()
        body = response.json()

    access_token = body.get("access_token") if isinstance(body, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Claude Code OAuth refresh returned no access token")
    updated = dict(credentials)
    updated["accessToken"] = access_token
    rotated = body.get("refresh_token")
    if isinstance(rotated, str) and rotated:
        updated["refreshToken"] = rotated
    expires_in = body.get("expires_in")
    if isinstance(expires_in, int | float):
        updated["expiresAt"] = int((time.time() + expires_in) * 1000)
    await asyncio.to_thread(_store_claude_oauth_credentials, updated)
    return updated


async def _oauth_models(provider: str) -> dict[str, Any]:
    """Discover account-scoped models for local OAuth provider integrations."""
    from markitai.providers.auth import (
        _chatgpt_oauth_credentials,
        _claude_oauth_credentials,
    )

    if provider == "claude-agent":
        credentials = await asyncio.to_thread(_claude_oauth_credentials)
        token = credentials.get("accessToken") if credentials is not None else None
        if credentials is None or not isinstance(token, str) or not token:
            return _result(
                provider,
                status="unavailable",
                source="live_api",
                authoritative=False,
                models=[],
                detail="Claude Code OAuth credentials are unavailable",
            )
        expires_at = credentials.get("expiresAt")
        refreshed = False
        if (
            isinstance(expires_at, int | float)
            and expires_at <= (time.time() + 30) * 1000
        ):
            credentials = await _refresh_claude_oauth_credentials(credentials)
            token = str(credentials["accessToken"])
            refreshed = True
        try:
            return await _http_models(provider, token, None)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 401 or refreshed:
                raise
            credentials = await _refresh_claude_oauth_credentials(credentials)
            return await _http_models(provider, str(credentials["accessToken"]), None)

    credentials = await asyncio.to_thread(_chatgpt_oauth_credentials)
    token = credentials.get("access_token") if credentials is not None else None
    if credentials is None or not isinstance(token, str) or not token:
        return _result(
            provider,
            status="unavailable",
            source="live_api",
            authoritative=False,
            models=[],
            detail="ChatGPT OAuth credentials are unavailable",
        )
    account_id = credentials.get("account_id")
    return await _http_models(
        provider,
        token,
        None,
        account_id=account_id if isinstance(account_id, str) else None,
    )


async def _http_models(
    provider: str,
    api_key: str | None,
    api_base: str | None,
    *,
    account_id: str | None = None,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    if provider in {"openai", "deepseek", "openrouter"}:
        default_base = provider_default_api_base(provider)
        assert default_base is not None
        url = urljoin((api_base or default_base).rstrip("/") + "/", "models")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    elif provider in {"anthropic", "claude-agent"}:
        url = urljoin(
            (api_base or provider_default_api_base(provider) or "").rstrip("/") + "/",
            "models",
        )
        if api_key:
            if provider == "claude-agent":
                headers["Authorization"] = f"Bearer {api_key}"
                headers["anthropic-beta"] = "oauth-2025-04-20"
            else:
                headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        params["limit"] = "1000"
    elif provider == "chatgpt":
        url = "https://chatgpt.com/backend-api/codex/models"
        params["client_version"] = "0.0.0"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id
    elif provider == "gemini":
        url = urljoin(
            (api_base or provider_default_api_base(provider) or "").rstrip("/")
            + "/",
            "models",
        )
        if api_key:
            params["key"] = api_key
        params["pageSize"] = "1000"
    elif provider == "ollama":
        url = urljoin(
            (api_base or provider_default_api_base(provider) or "").rstrip("/") + "/",
            "api/tags",
        )
    elif provider == "azure":
        if not api_base:
            return _result(
                provider,
                status="unavailable",
                source="live_api",
                authoritative=False,
                models=[],
                detail="Azure OpenAI endpoint is required",
            )
        base = api_base.rstrip("/")
        if base.endswith(("/openai/v1", "/v1")):
            url = urljoin(base + "/", "models")
        else:
            url = urljoin(base + "/", "openai/models")
            params["api-version"] = "2024-10-21"
        if api_key:
            headers["api-key"] = api_key
    elif provider == "custom":
        if not api_base:
            return _result(
                provider,
                status="unavailable",
                source="live_api",
                authoritative=False,
                models=[],
                detail="API base is required",
            )
        url = urljoin(api_base.rstrip("/") + "/", "models")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        return _result(
            provider,
            status="unavailable",
            source="live_api",
            authoritative=False,
            models=[],
            detail="Provider does not expose a supported live model-list endpoint",
        )

    async with httpx.AsyncClient(
        timeout=_REMOTE_TIMEOUT, follow_redirects=False
    ) as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        body = response.json()

    candidates: list[dict[str, Any]] = []
    if provider == "gemini":
        records = body.get("models", []) if isinstance(body, dict) else []
        for record in records:
            if not isinstance(record, dict):
                continue
            methods = record.get("supportedGenerationMethods", [])
            if methods and "generateContent" not in methods:
                continue
            raw_id = str(record.get("name", "")).removeprefix("models/")
            if raw_id:
                candidates.append(
                    _candidate(f"gemini/{raw_id}", record.get("displayName"))
                )
    elif provider == "ollama":
        records = body.get("models", []) if isinstance(body, dict) else []
        for record in records:
            raw_id = record.get("name") if isinstance(record, dict) else None
            if isinstance(raw_id, str) and raw_id:
                candidates.append(_candidate(f"ollama/{raw_id}"))
    elif provider == "chatgpt":
        records = body.get("models", []) if isinstance(body, dict) else []
        for record in records:
            if not isinstance(record, dict) or record.get("visibility") != "list":
                continue
            raw_id = record.get("slug")
            if isinstance(raw_id, str) and raw_id:
                candidates.append(
                    _candidate(f"chatgpt/{raw_id}", record.get("display_name"))
                )
    else:
        records = body.get("data", []) if isinstance(body, dict) else []
        prefix = {
            "custom": "openai",
            "claude-agent": "claude-agent",
        }.get(provider, provider)
        for record in records:
            raw_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(raw_id, str) or not raw_id:
                continue
            label = record.get("display_name") or record.get("name")
            vision = provider in {"anthropic", "claude-agent"}
            if provider == "openrouter":
                architecture = record.get("architecture")
                modalities = (
                    architecture.get("input_modalities", [])
                    if isinstance(architecture, dict)
                    else []
                )
                vision = "image" in modalities
            candidates.append(_candidate(f"{prefix}/{raw_id}", label, vision=vision))

    deduplicated = list({item["model"]: item for item in candidates}.values())
    authoritative = provider != "azure"
    return _result(
        provider,
        status="ok" if authoritative else "partial",
        source="live_api",
        authoritative=authoritative,
        models=deduplicated,
        detail=(
            "Azure lists regional base models; routing still requires a deployment name"
            if provider == "azure"
            else None
        ),
    )


async def discover_models(
    provider: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Discover models with single-flight caching and stale-on-error fallback."""
    provider = provider.strip().lower()

    async def loader() -> dict[str, Any]:
        if provider == "copilot":
            return await asyncio.wait_for(_copilot_models(), timeout=_COPILOT_TIMEOUT_S)
        if provider in {"claude-agent", "chatgpt"}:
            return await _oauth_models(provider)
        return await _http_models(provider, api_key, api_base)

    key = _cache_key(provider, api_key, api_base)
    now = time.monotonic()
    cached = _cache.get(key)
    observed_fetch = cached.fetched_at if cached is not None else None
    if not refresh and cached is not None and now - cached.fetched_at < _REMOTE_CACHE_S:
        result = dict(cached.value)
        result["cached"] = True
        return result

    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        cached = _cache.get(key)
        refreshed_by_peer = (
            refresh and cached is not None and cached.fetched_at != observed_fetch
        )
        if cached is not None and (
            refreshed_by_peer
            or (not refresh and now - cached.fetched_at < _REMOTE_CACHE_S)
        ):
            result = dict(cached.value)
            result["cached"] = True
            return result
        try:
            result = await asyncio.wait_for(loader(), timeout=20.0)
            _cache[key] = _CacheEntry(value=result, fetched_at=time.monotonic())
            return result
        except Exception as error:
            cached = _cache.get(key)
            if cached is not None and now - cached.fetched_at < _STALE_CACHE_S:
                result = dict(cached.value)
                result.update(
                    {
                        "cached": True,
                        "stale": True,
                        "status": "partial",
                        "detail": f"refresh failed ({type(error).__name__}); showing cached models",
                    }
                )
                return result
            return _result(
                provider,
                status="unavailable",
                source="live_api",
                authoritative=False,
                models=[],
                detail=f"model discovery failed ({type(error).__name__})",
            )
