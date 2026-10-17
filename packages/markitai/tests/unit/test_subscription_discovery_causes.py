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


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    *programs: str,
    version: str | None = "claude 2.1.290",
) -> None:
    """*programs* are on PATH and answer ``--version`` with *version*."""
    monkeypatch.setattr(
        discovery.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in programs else None,
    )
    monkeypatch.setattr(discovery, "_cli_version", lambda _program: version)


async def _discover(provider: str, **kwargs: Any) -> dict[str, Any]:
    return await discovery.discover_models(provider, **kwargs)


async def test_claude_without_its_program_or_login_is_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    _cli(monkeypatch)
    result = await _discover("claude-agent")
    assert result["status"] == "unavailable"
    assert result["reason"] == "runtime_missing"
    assert result["runtime"]["name"] == "Claude Code CLI"
    assert result["runtime"]["required"]
    assert result["next_step"] == {
        "kind": "install",
        "command": "npm install -g @anthropic-ai/claude-code@2.1.284",
    }


async def test_claude_installed_but_signed_out_is_not_signed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    _cli(monkeypatch, "claude")
    result = await _discover("claude-agent")
    assert result["reason"] == "not_signed_in"
    assert result["next_step"] == {
        "kind": "sign_in",
        "command": "markitai auth claude login",
    }


async def test_an_old_cli_is_unsupported_and_says_its_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    _cli(monkeypatch, "claude", version="1.0.3 (Claude Code)")
    result = await _discover("claude-agent")
    assert result["reason"] == "runtime_unsupported"
    assert result["runtime"]["installed"] == "1.0.3"
    assert result["next_step"]["kind"] == "install"
    assert "@2.1.284" in result["next_step"]["command"]


async def test_a_cli_that_does_not_answer_is_runtime_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_credentials(monkeypatch)
    _cli(monkeypatch, "claude", version=None)
    result = await _discover("claude-agent")
    assert result["reason"] == "runtime_failed"
    assert "next_step" not in result
    assert "--version" in result["detail"]


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


async def test_an_expired_claude_login_is_not_signed_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refresh is refused (invalid_grant) or there is nothing to refresh
    with: the user must sign in again, not hear that a runtime failed."""
    _cli(monkeypatch, "claude")
    monkeypatch.setattr(discovery.time, "time", lambda: 2_000_000_000)
    for credentials in (
        {"accessToken": "old", "expiresAt": 1, "refreshToken": "r"},
        {"accessToken": "old", "expiresAt": 1},
    ):
        monkeypatch.setattr(auth, "_claude_oauth_credentials", lambda c=credentials: c)

        class Refused:
            status_code = 400

        async def post(*_args: Any, **_kwargs: Any) -> Refused:
            return Refused()

        monkeypatch.setattr(httpx.AsyncClient, "post", post)
        result = await _discover("claude-agent")
        assert result["reason"] == "not_signed_in"
        assert result["next_step"]["command"] == "markitai auth claude login"


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
    _cli(monkeypatch)
    result = await _discover("copilot", refresh=True)
    assert (result["reason"], result["next_step"]["kind"]) == (
        "runtime_missing",
        "install",
    )
    assert result["runtime"]["name"] == "GitHub Copilot CLI"
    assert result["next_step"]["command"].endswith("@github/copilot@1.0.90-2")

    _cli(monkeypatch, "copilot", version="GitHub Copilot CLI 0.0.300")
    result = await _discover("copilot", refresh=True)
    assert result["reason"] == "runtime_unsupported"
    assert result["runtime"]["installed"] == "0.0.300"

    _cli(monkeypatch, "copilot", version="GitHub Copilot CLI 1.0.90-2")
    result = await _discover("copilot", refresh=True)
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
    assert "installed" not in result["detail"]  # ChatGPT has no runtime to install

    async def slow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise TimeoutError

    monkeypatch.setattr(discovery, "_http_models", slow)
    result = await _discover("chatgpt")
    assert result["reason"] == "timeout"
    assert result["runtime"]["name"] == "ChatGPT"


async def test_a_cause_is_not_cached_so_a_fix_shows_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sign in, then reload the list: no refresh flag needed."""
    _cli(monkeypatch, "claude")
    _no_credentials(monkeypatch)
    first = await _discover("chatgpt")
    assert first["reason"] == "not_signed_in"

    monkeypatch.setattr(
        auth, "_chatgpt_oauth_credentials", lambda: {"access_token": "t"}
    )

    async def listed(provider: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return discovery._result(
            provider,
            status="ok",
            source="live_api",
            authoritative=True,
            models=[discovery._candidate("chatgpt/gpt-x")],
        )

    monkeypatch.setattr(discovery, "_http_models", listed)
    second = await _discover("chatgpt")
    assert second["status"] == "ok"
    assert second["cached"] is False


async def test_a_signed_out_account_is_not_answered_with_stale_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listed once, then the login goes: the cause, not "no_key" and not the
    old list; and a failure with nothing listed before has nothing stale."""
    monkeypatch.setattr(
        auth, "_chatgpt_oauth_credentials", lambda: {"access_token": "t"}
    )
    request = httpx.Request("GET", "https://chatgpt.com/backend-api/codex/models")

    async def listed(provider: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return discovery._result(
            provider,
            status="ok",
            source="live_api",
            authoritative=True,
            models=[discovery._candidate("chatgpt/gpt-x")],
        )

    monkeypatch.setattr(discovery, "_http_models", listed)
    assert (await _discover("chatgpt"))["status"] == "ok"

    async def refused(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise httpx.HTTPStatusError(
            "401", request=request, response=httpx.Response(401, request=request)
        )

    monkeypatch.setattr(discovery, "_http_models", refused)
    result = await _discover("chatgpt", refresh=True)
    assert result["status"] == "unavailable"
    assert result["reason"] == "not_signed_in"
    assert result["models"] == []

    # a blip after a good list still shows the models loaded before
    monkeypatch.setattr(discovery, "_http_models", listed)
    await _discover("chatgpt", refresh=True)

    async def timed_out(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise TimeoutError

    monkeypatch.setattr(discovery, "_http_models", timed_out)
    result = await _discover("chatgpt", refresh=True)
    assert (result["status"], result["stale"], result["reason"]) == (
        "partial",
        True,
        "timeout",
    )
    assert result["models"]


async def test_an_ordinary_provider_keeps_its_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("x")

    monkeypatch.setattr(discovery, "_http_models", broken)
    result = await discovery.discover_models("openai", api_key="k", refresh=True)
    assert result["reason"] == "provider_error"
    assert "runtime" not in result
