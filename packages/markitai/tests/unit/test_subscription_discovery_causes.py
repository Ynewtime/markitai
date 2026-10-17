"""A subscription provider that lists no models says why (contract v1.18)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from markitai.providers import auth, discovery


@pytest.fixture(autouse=True)
def clear_discovery_caches() -> None:
    discovery._cache.clear()
    discovery._connection_cache.clear()
    discovery._locks.clear()


def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "_claude_oauth_credentials", lambda: None)
    monkeypatch.setattr(auth, "_chatgpt_oauth_credentials", lambda: None)


def _on_path(monkeypatch: pytest.MonkeyPatch, *programs: str) -> None:
    monkeypatch.setattr(
        discovery.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in programs else None,
    )


async def _discover(provider: str) -> dict[str, Any]:
    return await discovery.discover_models(provider, refresh=True)


async def test_claude_without_its_program_or_login_is_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    _on_path(monkeypatch)
    result = await _discover("claude-agent")
    assert result["status"] == "unavailable"
    assert result["reason"] == "runtime_missing"
    assert result["runtime"]["name"] == "Claude Code"
    assert result["runtime"]["required"]
    assert result["next_step"] == {
        "kind": "install",
        "command": "npm install -g @anthropic-ai/claude-code",
    }


async def test_claude_installed_but_signed_out_is_not_signed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    _on_path(monkeypatch, "claude")
    result = await _discover("claude-agent")
    assert result["reason"] == "not_signed_in"
    assert result["next_step"] == {
        "kind": "sign_in",
        "command": "markitai auth claude login",
    }


async def test_claude_signed_in_without_its_sdk_is_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "_claude_oauth_credentials", lambda: {"accessToken": "t"})
    monkeypatch.setattr(discovery, "_local_runtime_available", lambda _p: False)
    result = await _discover("claude-agent")
    assert result["reason"] == "runtime_missing"
    assert result["runtime"]["name"] == "Claude Agent SDK"
    assert result["next_step"]["kind"] == "install"
    assert "markitai[claude-agent]" in result["next_step"]["command"]


async def test_a_refused_token_is_not_signed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth, "_chatgpt_oauth_credentials", lambda: {"access_token": "t"}
    )
    request = httpx.Request("GET", "https://chatgpt.com/backend-api/codex/models")

    async def refused(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise httpx.HTTPStatusError(
            "401", request=request, response=httpx.Response(401, request=request)
        )

    monkeypatch.setattr(discovery, "_http_models", refused)
    result = await _discover("chatgpt")
    assert result["reason"] == "not_signed_in"
    assert result["runtime"]["name"] == "ChatGPT"
    assert result["next_step"]["command"] == "markitai auth chatgpt login"


async def test_chatgpt_without_a_login_is_not_signed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    result = await _discover("chatgpt")
    assert result["reason"] == "not_signed_in"
    assert result["next_step"]["kind"] == "sign_in"


async def test_copilot_causes(monkeypatch: pytest.MonkeyPatch) -> None:
    from markitai.providers.auth import AuthStatus

    monkeypatch.setattr(discovery, "_local_runtime_available", lambda _p: False)
    result = await _discover("copilot")
    assert result["reason"] == "runtime_missing"
    assert result["runtime"]["name"] == "GitHub Copilot SDK"

    monkeypatch.setattr(discovery, "_local_runtime_available", lambda _p: True)
    signed_out = AuthStatus("copilot", False, None, None, "no login")
    monkeypatch.setattr(auth, "_check_copilot_config_auth", lambda: signed_out)
    _on_path(monkeypatch)
    result = await _discover("copilot")
    assert (result["reason"], result["next_step"]["kind"]) == (
        "runtime_missing",
        "install",
    )
    assert result["runtime"]["name"] == "GitHub Copilot CLI"

    _on_path(monkeypatch, "copilot")
    result = await _discover("copilot")
    assert result["reason"] == "not_signed_in"
    assert result["next_step"]["command"] == "markitai auth copilot login"


async def test_an_unexplained_failure_is_runtime_failed_and_a_timeout_stays_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auth, "_chatgpt_oauth_credentials", lambda: {"access_token": "t"}
    )

    async def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("secret internals")

    monkeypatch.setattr(discovery, "_http_models", broken)
    result = await _discover("chatgpt")
    assert result["reason"] == "runtime_failed"
    assert "next_step" not in result
    assert "secret internals" not in str(result)

    async def slow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise TimeoutError

    monkeypatch.setattr(discovery, "_http_models", slow)
    result = await _discover("chatgpt")
    assert result["reason"] == "timeout"
    assert result["runtime"]["name"] == "ChatGPT"


async def test_an_ordinary_provider_keeps_its_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("x")

    monkeypatch.setattr(discovery, "_http_models", broken)
    result = await discovery.discover_models("openai", api_key="k", refresh=True)
    assert result["reason"] == "provider_error"
    assert "runtime" not in result
