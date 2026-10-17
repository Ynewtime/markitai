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

from markitai.constants import (
    PROVIDER_API_KEY_ENV,
    PROVIDER_API_KEY_ENV_NAMES,
    PROVIDER_DEFAULT_MODELS,
)
from markitai.providers.connectable import (
    CONNECTABLE,
    Connectable,
    connectable,
    default_base,
    installed,
)

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


def provider_default_api_base(provider: str) -> str | None:
    """Return the effective built-in API base for a known provider.

    The address in LiteLLM's form (``connectable``): Anthropic's is the host,
    to which LiteLLM adds ``/v1/messages``.
    """
    if provider.lower() == "claude-agent":
        return "https://api.anthropic.com/v1"
    return default_base(provider)


_COMMON_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "ollama": "Ollama",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
    "azure": "Azure OpenAI",
    "custom": "OpenAI-compatible endpoint",
}


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
        payload["detail"] = detail[:400]
    return payload


#: The subscription runtimes whose list can fail for a cause the user fixes on
#: the server's machine (contract v1.18): the program, the release the adapter
#: speaks and what to run. ``required`` names the CLI release the install
#: command pins (the releases the SDKs were tested with, as Rust pins them),
#: ``minimum`` the oldest ``--version`` markitai accepts.
_SUBSCRIPTIONS: dict[str, dict[str, Any]] = {
    "claude-agent": {
        "name": "Claude Code CLI",
        "program": "claude",
        "required": "2.1.284 or a later release",
        "minimum": (2, 1, 0),
        "install": "npm install -g @anthropic-ai/claude-code@2.1.284",
        "sign_in": "markitai auth claude login",
    },
    "copilot": {
        "name": "GitHub Copilot CLI",
        "program": "copilot",
        "required": "1.0.90 or a later release",
        "minimum": (1, 0, 0),
        "install": "npm install -g @github/copilot@1.0.90-2",
        "sign_in": "markitai auth copilot login",
    },
    "chatgpt": {
        "name": "ChatGPT",
        "program": None,
        "required": "a ChatGPT sign-in (Codex OAuth)",
        "minimum": None,
        "install": "",
        "sign_in": "markitai auth chatgpt login",
    },
}

#: The extras that install each provider's Python SDK, and the SDK release
_SDK_EXTRAS: dict[str, tuple[str, str, str]] = {
    "claude-agent": ("Claude Agent SDK", "claude-agent", "claude-agent-sdk>=0.1.50"),
    "copilot": ("GitHub Copilot SDK", "copilot", "github-copilot-sdk>=0.2.0"),
}

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-(\d+))?")
_VERSION_TIMEOUT_S = 4.0


class SubscriptionSignedOut(Exception):
    """A subscription login that is gone or can no longer be renewed."""


def _runtime_unavailable(
    provider: str,
    reason: str,
    detail: str,
    *,
    kind: str | None = None,
    command: str | None = None,
    runtime_name: str | None = None,
    required: str | None = None,
    installed: str | None = None,
) -> dict[str, Any]:
    """An unavailable subscription list that says its cause (contract v1.18).

    ``reason`` is ``runtime_missing``, ``runtime_unsupported``,
    ``not_signed_in``, ``runtime_failed`` or ``timeout``; ``runtime`` names
    the program, the release the adapter speaks and, when it could be asked,
    the release installed; ``next_step`` (``install`` or ``sign_in``) is what
    to run where the server runs.
    """
    row = _SUBSCRIPTIONS[provider]
    payload = _result(
        provider,
        status="unavailable",
        source="live_api",
        authoritative=False,
        models=[],
        detail=detail,
    )
    payload["reason"] = reason
    runtime = {
        "name": runtime_name or row["name"],
        "required": required or row["required"],
    }
    if installed:
        runtime["installed"] = installed
    payload["runtime"] = runtime
    if kind is not None and command:
        payload["next_step"] = {"kind": kind, "command": command}
    return payload


def _sdk_missing(provider: str) -> dict[str, Any] | None:
    """The answer when the provider's Python SDK is not installed."""
    sdk = _SDK_EXTRAS.get(provider)
    if sdk is None or _local_runtime_available(provider):
        return None
    from markitai.utils.errors import extra_install_command

    name, extra, requirement = sdk
    return _runtime_unavailable(
        provider,
        "runtime_missing",
        f"{name} is not installed on the server",
        kind="install",
        command=extra_install_command(extra),
        runtime_name=name,
        required=requirement,
    )


def _cli_version(program: str) -> str | None:
    """What ``<program> --version`` prints, or None when it does not answer."""
    import subprocess

    path = shutil.which(program)
    if path is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - the CLI the user installed
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return (done.stdout or done.stderr).strip() or None


async def _runtime_problem(provider: str) -> dict[str, Any] | None:
    """The cause when the provider's CLI is absent, broken or too old.

    A signed-out account is only told to sign in when the program can: a CLI
    that is missing, does not answer ``--version`` or is older than the
    release the adapter speaks needs installing first. None when it is fine
    (or the provider has no CLI).
    """
    row = _SUBSCRIPTIONS[provider]
    program = row["program"]
    if program is None:
        return None
    if shutil.which(program) is None:
        return _runtime_unavailable(
            provider,
            "runtime_missing",
            f"{row['name']} was not found on the server's PATH",
            kind="install",
            command=row["install"],
        )
    printed = await asyncio.to_thread(_cli_version, program)
    match = _VERSION_RE.search(printed or "")
    if match is None:
        return _runtime_unavailable(
            provider,
            "runtime_failed",
            f"{row['name']} is installed but does not answer --version",
        )
    found = tuple(int(part) for part in match.groups()[:3])
    if found < row["minimum"]:
        return _runtime_unavailable(
            provider,
            "runtime_unsupported",
            f"{row['name']} {match.group(0)} is older than markitai supports",
            kind="install",
            command=row["install"],
            installed=match.group(0),
        )
    return None


async def _signed_out(provider: str, detail: str) -> dict[str, Any]:
    """Signed out: the CLI needs installing first, or the account only a login."""
    problem = await _runtime_problem(provider)
    if problem is not None:
        return problem
    row = _SUBSCRIPTIONS[provider]
    return _runtime_unavailable(
        provider, "not_signed_in", detail, kind="sign_in", command=row["sign_in"]
    )


async def subscription_failure(
    provider: str, error: BaseException, failure: dict[str, Any]
) -> dict[str, Any]:
    """Say a subscription list's failure with its cause (contract v1.18).

    A refused token or a login that cannot be renewed is a signed-out account,
    a timeout stays one, anything else that is not about the network or the
    quota is the runtime failing.
    """
    if provider not in _SUBSCRIPTIONS:
        return failure
    row = _SUBSCRIPTIONS[provider]
    refused = isinstance(
        error, httpx.HTTPStatusError
    ) and error.response.status_code in (401, 403)
    if refused or isinstance(error, SubscriptionSignedOut):
        answer = await _signed_out(provider, f"{row['name']} is not signed in")
        return {
            key: value
            for key, value in answer.items()
            if key in ("reason", "detail", "runtime", "next_step")
        }
    runtime = {"name": row["name"], "required": row["required"]}
    if failure.get("reason") == "timeout":
        return {**failure, "runtime": runtime}
    if failure.get("reason") == "provider_error":
        return {
            "reason": "runtime_failed",
            "detail": (
                f"{row['name']} is installed but could not list models"
                if row["program"]
                else f"{row['name']} could not list models"
            ),
            "runtime": runtime,
        }
    return failure


def provider_key_variables(provider: str) -> tuple[tuple[str, ...], str | None]:
    """The variables LiteLLM reads *provider*'s key from, and the main one.

    In LiteLLM's order (the first one set is the key a request carries);
    the main one is the variable markitai names when none is set. None for a
    cloud signed in with the server's own credentials.
    """
    row = connectable(provider)
    if row is not None and row.cloud:
        return (), None
    names = PROVIDER_API_KEY_ENV_NAMES.get(provider) or (
        (row.key_variable,) if row is not None and row.key_variable else ()
    )
    return names, PROVIDER_API_KEY_ENV.get(provider) or (names[0] if names else None)


def environment_key_providers() -> dict[str, str]:
    """Every provider a key variable set on this server serves, and that variable.

    The first of the provider's variables that holds a value (the one
    LiteLLM sends), for every provider of the table whose route is here;
    local servers and clouds signed in with their own credentials have none.
    """
    from markitai.providers import is_local_provider_model

    found: dict[str, str] = {}
    for row in CONNECTABLE:
        if (
            row.cloud
            or not installed(row)
            or is_local_provider_model(f"{row.provider}/")
        ):
            continue
        names, _main = provider_key_variables(row.provider)
        variable = next(
            (name for name in names if os.environ.get(name, "").strip()), None
        )
        if variable is not None:
            found[row.provider] = variable
    return found


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
    environment = environment_key_providers()
    key = hashlib.sha256(
        repr((configured_key, sorted(environment.items()))).encode()
    ).hexdigest()
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
            "default_model": PROVIDER_DEFAULT_MODELS["chatgpt"],
            "supports_discovery": True,
        }

    detected_cards = await asyncio.gather(
        local_card(
            "claude",
            "claude-agent",
            "Claude Code CLI",
            PROVIDER_DEFAULT_MODELS["claude-agent"],
        ),
        local_card(
            "copilot",
            "copilot",
            "GitHub Copilot CLI",
            PROVIDER_DEFAULT_MODELS["copilot"],
        ),
        chatgpt_card(),
    )
    cards.extend(card for card in detected_cards if card is not None)

    # Every provider whose key variable is set, not only markitai's first
    # five: the workbench tells a user to set one and find it under Saved
    for provider, env_var in environment.items():
        row = connectable(provider)
        cards.append(
            {
                "id": f"env:{provider}",
                "provider": provider,
                "label": provider_label(provider),
                "kind": "environment",
                "status": "ready",
                "source": env_var,
                "credential": f"env:{env_var}",
                "supports_discovery": row is None or row.discovery != "manual",
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
    # A subscription runtime installed with its extra but not detected (no
    # CLI on PATH, not signed in yet) can still be connected: its runtime
    # signs in and lists the account's models. ChatGPT has no SDK to install:
    # its card is always offered, and discovery says "not signed in" with the
    # command (`markitai auth chatgpt login`) until the account is
    for provider in ("claude-agent", "copilot", "chatgpt"):
        if provider not in present and (
            provider == "chatgpt" or _local_runtime_available(provider)
        ):
            cards.append(
                {
                    "id": f"common:{provider}",
                    "provider": provider,
                    "label": provider_label(provider),
                    "kind": "common",
                    "status": "unknown",
                    "source": "built_in",
                    "supports_discovery": True,
                    "key_optional": True,
                }
            )
    for row in CONNECTABLE:
        if row.provider in present or not installed(row):
            continue
        # A local server at its default address, or a cloud's own
        # credentials, may already work; only a check tells
        local = row.key_optional and (row.base is not None or row.cloud)
        cards.append(
            {
                "id": f"common:{row.provider}",
                "provider": row.provider,
                "label": _COMMON_LABELS.get(row.provider)
                or provider_label(row.provider),
                "kind": "common",
                "status": "unknown" if local else "needs_credentials",
                "source": "built_in",
                "supports_discovery": row.discovery != "manual",
            }
        )
    _connection_cache[key] = _CacheEntry(
        value={"providers": cards}, fetched_at=time.monotonic()
    )
    return cards


def startup_model_candidates(connections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map ready connection cards to conservative startup defaults."""
    result: list[dict[str, Any]] = []
    for card in connections:
        if card.get("status") != "ready":
            continue
        provider = str(card.get("provider", ""))
        model = card.get("default_model") or PROVIDER_DEFAULT_MODELS.get(provider)
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
    missing = _sdk_missing("copilot")
    if missing is not None:
        return missing
    from markitai.providers.auth import _check_copilot_config_auth

    problem = await _runtime_problem("copilot")
    if problem is not None:
        return problem
    status = await asyncio.to_thread(_check_copilot_config_auth)
    if not status.authenticated:
        return await _signed_out("copilot", "GitHub Copilot is not signed in")
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
        raise SubscriptionSignedOut("Claude Code OAuth refresh token is unavailable")

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
        if response.status_code in (400, 401, 403):
            # invalid_grant and friends: the login can no longer be renewed
            raise SubscriptionSignedOut("Claude Code OAuth refresh was refused")
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
            return await _signed_out(provider, "Claude Code is not signed in")
        missing = _sdk_missing(provider)
        if missing is not None:
            return missing
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
        return await _signed_out(provider, "ChatGPT is not signed in")
    account_id = credentials.get("account_id")
    return await _http_models(
        provider,
        token,
        None,
        account_id=account_id if isinstance(account_id, str) else None,
    )


def _manual_list(provider: str) -> dict[str, Any]:
    """A provider that publishes no model list: no failure, no ``reason``."""
    return _result(
        provider,
        status="unavailable",
        source="manual",
        authoritative=False,
        models=[],
        detail="This provider does not publish a model list; enter model IDs manually",
    )


def _no_endpoint(provider: str) -> dict[str, Any]:
    """A list that needs an address the request does not have."""
    result = _result(
        provider,
        status="unavailable",
        source="live_api",
        authoritative=False,
        models=[],
        detail="An API endpoint is required",
    )
    result["reason"] = "provider_error"
    return result


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
        # LiteLLM takes Anthropic's host (it adds /v1/messages); an address
        # entered with its /v1 lists the same models
        base = (api_base or provider_default_api_base(provider) or "").rstrip("/")
        url = urljoin(base.removesuffix("/v1") + "/", "v1/models")
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
            (api_base or provider_default_api_base(provider) or "").rstrip("/") + "/",
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
            return _no_endpoint(provider)
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
            return _no_endpoint(provider)
        url = urljoin(api_base.rstrip("/") + "/", "models")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    elif (row := connectable(provider)) is not None and row.discovery != "builtin":
        return await _listed_models(row, api_key, api_base)
    else:
        return _manual_list(provider)

    async with httpx.AsyncClient(
        timeout=_REMOTE_TIMEOUT, follow_redirects=False
    ) as client:
        response = await client.get(url, headers=headers, params=params)
        body = _list_body(response)

    candidates: list[dict[str, Any]] = []
    if provider == "gemini":
        records = _model_records(body, "models")
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
        records = _model_records(body, "models")
        for record in records:
            raw_id = record.get("name") if isinstance(record, dict) else None
            if isinstance(raw_id, str) and raw_id:
                candidates.append(_candidate(f"ollama/{raw_id}"))
    elif provider == "chatgpt":
        records = _model_records(body, "models")
        for record in records:
            if not isinstance(record, dict) or record.get("visibility") != "list":
                continue
            raw_id = record.get("slug")
            if isinstance(raw_id, str) and raw_id:
                candidates.append(
                    _candidate(f"chatgpt/{raw_id}", record.get("display_name"))
                )
    else:
        records = _model_records(body, "data")
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


async def _listed_models(
    row: Connectable, api_key: str | None, api_base: str | None
) -> dict[str, Any]:
    """List the models of a provider from ``connectable``: the OpenAI-style
    ``GET <base>/models`` most of them serve, or Cohere's own list."""
    provider = row.provider
    if row.discovery == "manual":
        return _manual_list(provider)
    base = api_base or row.base
    if not base:
        return _no_endpoint(provider)
    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if row.discovery == "cohere":
        # Cohere's list lives beside its API, whichever chat URL LiteLLM uses
        parts = httpx.URL(base)
        url = f"{parts.scheme}://{parts.netloc.decode('ascii')}/v1/models"
        params.update(endpoint="chat", page_size="1000")
    else:
        url = urljoin(base.rstrip("/") + "/", row.models_path)

    async with httpx.AsyncClient(
        timeout=_REMOTE_TIMEOUT, follow_redirects=False
    ) as client:
        response = await client.get(url, headers=headers, params=params)
        body = _list_body(response)

    if row.discovery == "cohere":
        records = _model_records(body, "models")
        key, label_keys = "name", ()
        more = isinstance(body, dict) and bool(body.get("next_page_token"))
    else:
        # Together AI answers with the bare list
        records = body if isinstance(body, list) else _model_records(body, "data")
        key, label_keys = "id", ("display_name", "name")
        more = isinstance(body, dict) and body.get("has_more") is True
    candidates: list[dict[str, Any]] = []
    for record in records if isinstance(records, list) else []:
        raw_id = record.get(key) if isinstance(record, dict) else None
        if not isinstance(raw_id, str) or not raw_id:
            continue
        label = next(
            (record[k] for k in label_keys if isinstance(record.get(k), str)), None
        )
        candidates.append(_candidate(f"{provider}/{raw_id}", label or raw_id))
    deduplicated = list({item["model"]: item for item in candidates}.values())
    return _result(
        provider,
        status="partial" if more else "ok",
        source="live_api",
        authoritative=not more,
        models=deduplicated,
        detail=(
            "The provider has more models; only its first page is shown"
            if more
            else None
        ),
    )


#: The provider's own message in a failure detail, in characters at most
_MAX_MESSAGE = 300
#: A model list answer larger than this is not read
_MAX_LIST_BYTES = 2 * 1024 * 1024


class UnreadableModelList(ValueError):
    """An answer that is not a model list (not JSON, too large, no list)."""


def _model_records(body: Any, key: str) -> list[Any]:
    """The model records under *key*, or :class:`UnreadableModelList`."""
    records = body.get(key) if isinstance(body, dict) else None
    if not isinstance(records, list):
        raise UnreadableModelList("Model discovery response has no model list")
    return records


def _list_body(response: httpx.Response) -> Any:
    """A successful answer's JSON, bounded; non-2xx raises HTTPStatusError."""
    response.raise_for_status()
    if len(response.content) > _MAX_LIST_BYTES:
        raise UnreadableModelList("Model discovery response exceeds 2 MiB")
    try:
        return response.json()
    except ValueError:
        raise UnreadableModelList("Model discovery response is not JSON") from None


def _without_markup(text: str) -> str:
    """Text with every ``<…>`` tag replaced by a space."""
    plain: list[str] = []
    markup = False
    for character in text[: 64 * 1024]:
        if character == "<":
            markup = True
        elif character == ">" and markup:
            markup = False
            plain.append(" ")
        elif not markup:
            plain.append(character)
    return "".join(plain)


def _provider_words(response: httpx.Response, secrets: tuple[str, ...]) -> str:
    """The provider's own message in a refused answer, without credentials.

    The JSON body's ``error.message``, ``message``, ``error`` or ``detail``,
    else the text without markup; at most 300 characters, and the key that
    was sent replaced by ``…`` (contract v1.11 section 8).
    """
    try:
        body = response.json()
        parsed = True
    except ValueError:
        body, parsed = None, False
    words = ""
    if parsed:
        error = body.get("error") if isinstance(body, dict) else None
        for value in (
            error.get("message") if isinstance(error, dict) else None,
            body.get("message") if isinstance(body, dict) else None,
            error,
            body.get("detail") if isinstance(body, dict) else None,
        ):
            if isinstance(value, str):
                words = value
                break
    else:
        words = _without_markup(response.text)
    words = " ".join(words.split())
    for secret in secrets:
        if len(secret) >= 4:
            words = words.replace(secret, "…")
    words = re.sub(r"(?i)\bBearer\s+\S+", "Bearer …", words)
    words = re.sub(r"\b(?:sk|gsk|xai|key)-[A-Za-z0-9._-]{4,}\b", "…", words)
    return words[:_MAX_MESSAGE] + ("…" if len(words) > _MAX_MESSAGE else "")


def _transport_cause(error: BaseException, secrets: tuple[str, ...]) -> str:
    """Why a request got no answer, in a few words (never a key)."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.strerror:
            return current.strerror.lower()
        current = current.__cause__ or current.__context__
    cause = " ".join(str(error).split()) or type(error).__name__
    for secret in secrets:
        if len(secret) >= 4:
            cause = cause.replace(secret, "…")
    return cause[:_MAX_MESSAGE]


def discovery_failure(
    error: BaseException,
    *,
    keyless: bool,
    key_variable: str | None = None,
    secrets: tuple[str, ...] = (),
) -> dict[str, str]:
    """Why a model list could not be loaded: ``reason`` and ``detail``.

    The reasons are the model test's (API contract v1.10), so the workbench
    says what to do the same way, and the details are Rust's sentences
    (contract v1.11): ``Model discovery returned HTTP 401: <the provider's
    message>``, ``Model discovery timed out``, ``Model discovery request
    failed: <cause>``. A 401/403 to a request that carried no key
    (*keyless*) means the key is missing, not wrong: ``no_key``, with
    ``key_variable`` when known. A redirect is not followed (``network``);
    for a list, a 404 means the address has none (``provider_error``).
    """
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if 300 <= status < 400:
            return {
                "reason": "network",
                "detail": "Model discovery request failed: the address "
                f"redirected (HTTP {status}), which discovery does not follow",
            }
        words = _provider_words(error.response, secrets)
        detail = f"Model discovery returned HTTP {status}" + (
            f": {words}" if words else ""
        )
        if status in (401, 403) and keyless:
            failure = {"reason": "no_key", "detail": detail}
            if key_variable:
                failure["key_variable"] = key_variable
            return failure
        reason = (
            "key_rejected"
            if status in (401, 403)
            else "rate_limited"
            if status == 429
            else "provider_error"
        )
        return {"reason": reason, "detail": detail}
    if isinstance(error, httpx.TimeoutException | asyncio.TimeoutError | TimeoutError):
        return {"reason": "timeout", "detail": "Model discovery timed out"}
    if isinstance(error, httpx.TransportError | ConnectionError):
        return {
            "reason": "network",
            "detail": "Model discovery request failed: "
            + _transport_cause(error, secrets),
        }
    if isinstance(error, UnreadableModelList):
        return {"reason": "provider_error", "detail": str(error)}
    return {"reason": "provider_error", "detail": "Model discovery could not complete"}


async def discover_models(
    provider: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    refresh: bool = False,
    key_variable: str | None = None,
) -> dict[str, Any]:
    """Discover models with single-flight caching and stale-on-error fallback.

    A failure answers ``status: unavailable`` with a ``reason`` and a
    ``detail`` (:func:`discovery_failure`); *key_variable* names the
    variable a missing key would come from.
    """
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
            if (
                result.get("status") == "unavailable"
                and result.get("source") != "manual"
            ):
                # A list that could not be had (no runtime, no sign-in, no
                # address) says why; one the provider does not publish is
                # no failure
                result.setdefault("reason", "provider_error")
            if not (
                provider in _SUBSCRIPTIONS and result.get("status") == "unavailable"
            ):
                # A subscription's cause is asked for again every time (as
                # Rust does): the user fixes it on the machine, then reloads
                _cache[key] = _CacheEntry(value=result, fetched_at=time.monotonic())
            return result
        except Exception as error:
            failure = discovery_failure(
                error,
                # Ollama is sent no key even when one is configured
                keyless=not api_key or provider == "ollama",
                key_variable=key_variable,
                secrets=(api_key or "",),
            )
            failure = await subscription_failure(provider, error, failure)
            cached = _cache.get(key)
            # Models loaded before are shown for a failure that may pass (the
            # network, the quota, a hiccup), never for a cause the user has
            # to fix (not signed in, a runtime that is missing or too old),
            # and only when there were models to show
            if (
                cached is not None
                and cached.value.get("models")
                and failure.get("reason")
                in (
                    "network",
                    "timeout",
                    "rate_limited",
                    "runtime_failed",
                    "provider_error",
                )
                and now - cached.fetched_at < _STALE_CACHE_S
            ):
                result = dict(cached.value)
                result.update(
                    {
                        "cached": True,
                        "stale": True,
                        "status": "partial",
                        "detail": (
                            f"{failure['detail']}; showing the models loaded before"
                        )[:400],
                        "reason": failure["reason"],
                    }
                )
                return result
            result = _result(
                provider,
                status="unavailable",
                source="live_api",
                authoritative=False,
                models=[],
                detail=failure.pop("detail"),
            )
            result.update(failure)
            return result
