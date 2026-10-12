"""POST /api/settings/llm/test: why a failed model test failed, by category.

Owner report (2026-10-12): a DeepSeek test in Settings -> Models failed with
"AuthenticationError: litellm.AuthenticationError: AuthenticationError:
DeepseekException - Authentication Fails (governor)" on a server that had no
DEEPSEEK_API_KEY at all: the test sent the request without a key, and the
provider's 401 read like a wrong key. A test now names the unset variable
without sending anything, and a failed one carries a ``reason`` and keeps
the provider's own words as ``provider_detail``.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

import httpx
import litellm.exceptions as le

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.serve import create_app
from markitai.serve.model_test import REASONS

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

# The workbench's shapes for a model test's sentence (Rust's
# crates/markitai-cli/src/server/web/src/i18n/errors.ts, NOTE_SHAPES): a
# sentence in one of them is translated by status, the rest of it is shown
# under "Details"
_HTTP_SHAPE = re.compile(
    r"^Model connection returned HTTP (\d{3})\b(?: \([^()]*\))?(?:: (.+))?$"
)
_TIMEOUT_SHAPE = re.compile(r"^Model connection test timed out$")
_UNREACHABLE_SHAPE = re.compile(r"^Model connection request failed\b")

_KEY_VARIABLES = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
)


@pytest.fixture(autouse=True)
def _no_server_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _make_app(tmp_path: Path, models: list[ModelConfig] | None = None) -> FastAPI:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    cfg.llm.model_list = models or []
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
    )


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


def _entry(model: str, api_key: str | None = None) -> ModelConfig:
    return ModelConfig(
        model_name="default",
        litellm_params=LiteLLMParams(model=model, api_key=api_key),
    )


def _recording(
    monkeypatch: pytest.MonkeyPatch, raises: BaseException | None = None
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return {"id": "ok"}

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    return calls


async def _test_saved(
    tmp_path: Path, model: str, api_key: str | None = None
) -> dict[str, Any]:
    """Test the one saved model as the Models page does (deployment_id only)."""
    async with _client(_make_app(tmp_path, [_entry(model, api_key)])) as client:
        settings = (await client.get("/api/settings/llm")).json()
        deployment_id = settings["deployments"][0]["deployment_id"]
        resp = await client.post(
            "/api/settings/llm/test", json={"deployment_id": deployment_id}
        )
    assert resp.status_code == 200
    return resp.json()


async def test_the_owners_case_names_the_unset_variable_and_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recording(monkeypatch)
    data = await _test_saved(tmp_path, "deepseek/deepseek-flash")
    assert calls == []
    assert data["ok"] is False
    assert data["reason"] == "missing_key"
    assert data["detail"].startswith(
        "DEEPSEEK_API_KEY is not set on the server, so nothing was sent"
    )
    assert "DEEPSEEK_API_KEY=your_key" in data["detail"]
    assert data.get("provider_detail") is None


async def test_an_unset_env_reference_is_a_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recording(monkeypatch)
    data = await _test_saved(
        tmp_path, "gemini/gemini-flash-lite-latest", api_key="env:MY_GEMINI_KEY"
    )
    assert calls == []
    assert data["reason"] == "missing_key"
    assert data["detail"].startswith("MY_GEMINI_KEY is not set on the server")


async def test_any_connectable_provider_without_its_key_is_a_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond markitai's five: a provider the workbench connects (Groq)."""
    calls = _recording(monkeypatch)
    data = await _test_saved(tmp_path, "groq/llama-4-scout")
    assert calls == []
    assert data["reason"] == "missing_key"
    assert data["detail"].startswith("GROQ_API_KEY is not set on the server")


async def test_a_key_in_the_environment_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-1234567890")
    calls = _recording(monkeypatch)
    data = await _test_saved(tmp_path, "deepseek/deepseek-flash")
    assert data["ok"] is True
    assert len(calls) == 1
    assert data.get("reason") is None


async def test_a_keyless_local_server_is_not_a_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recording(monkeypatch)
    async with _client(_make_app(tmp_path)) as client:
        resp = await client.post(
            "/api/settings/llm/test",
            json={"model": "ollama/llama3", "api_base": "http://127.0.0.1:11434"},
        )
    assert resp.json()["ok"] is True
    assert len(calls) == 1


def _response(status: int) -> httpx.Response:
    return httpx.Response(
        status, request=httpx.Request("POST", "https://api.example/v1/chat")
    )


@pytest.mark.parametrize(
    ("error", "reason", "status", "cause"),
    [
        (
            le.AuthenticationError(
                message="DeepseekException - Authentication Fails (governor)",
                llm_provider="deepseek",
                model="deepseek-flash",
            ),
            "auth_rejected",
            "401",
            "the provider rejected the key in DEEPSEEK_API_KEY",
        ),
        (
            le.PermissionDeniedError(
                message="DeepseekException - Forbidden",
                llm_provider="deepseek",
                model="deepseek-flash",
                response=_response(403),
            ),
            "auth_rejected",
            "403",
            "the provider rejected the key in DEEPSEEK_API_KEY",
        ),
        (
            le.NotFoundError(
                message="DeepseekException - The model does not exist",
                model="deepseek-flash",
                llm_provider="deepseek",
            ),
            "model_not_found",
            "404",
            "the provider does not offer the model deepseek-flash",
        ),
        (
            le.BadRequestError(
                message="DeepseekException - Model Not Exist",
                model="deepseek-flash",
                llm_provider="deepseek",
            ),
            "model_not_found",
            "400",
            "the provider does not offer the model deepseek-flash",
        ),
        (
            le.RateLimitError(
                message="DeepseekException - Too Many Requests",
                llm_provider="deepseek",
                model="deepseek-flash",
            ),
            "rate_limited",
            "429",
            "the provider is limiting requests or the account is out of quota",
        ),
        (
            le.InternalServerError(
                message="DeepseekException - upstream overloaded",
                llm_provider="deepseek",
                model="deepseek-flash",
            ),
            "provider_error",
            "500",
            "the provider returned an error",
        ),
    ],
)
async def test_a_refusal_by_status_uses_the_shared_sentence_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
    status: str,
    cause: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-1234567890")
    _recording(monkeypatch, raises=error)
    data = await _test_saved(tmp_path, "deepseek/deepseek-flash")
    assert data["ok"] is False
    assert data["reason"] == reason
    match = _HTTP_SHAPE.match(data["detail"])
    assert match is not None, data["detail"]
    assert match.group(1) == status
    assert (match.group(2) or "").startswith(cause)
    # The provider's own words, without LiteLLM's class names, kept apart
    words = data["provider_detail"]
    assert words and "Exception" not in words and "litellm" not in words
    assert words in data["detail"]
    assert "sk-env-1234567890" not in str(data)


async def test_the_owners_wording_reads_as_the_provider_said_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-1234567890")
    _recording(
        monkeypatch,
        raises=le.AuthenticationError(
            message=(
                "AuthenticationError: DeepseekException - Authentication Fails "
                "(governor)"
            ),
            llm_provider="deepseek",
            model="deepseek-flash",
        ),
    )
    data = await _test_saved(tmp_path, "deepseek/deepseek-flash")
    assert data["provider_detail"] == "Authentication Fails (governor)"
    assert data["detail"] == (
        "Model connection returned HTTP 401: the provider rejected the key in "
        "DEEPSEEK_API_KEY (Authentication Fails (governor))"
    )


async def test_a_key_saved_with_the_connection_is_named_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recording(
        monkeypatch,
        raises=le.AuthenticationError(
            message="OpenAIException - Incorrect API key provided",
            llm_provider="openai",
            model="gpt-6-luna",
        ),
    )
    data = await _test_saved(
        tmp_path, "openai/gpt-6-luna", api_key="sk-saved-123456789"
    )
    assert data["reason"] == "auth_rejected"
    assert "the provider rejected the key saved with this connection" in data["detail"]
    assert "sk-saved-123456789" not in str(data)


@pytest.mark.parametrize(
    ("error", "shape"),
    [
        (
            le.Timeout(message="Request timed out", model="m", llm_provider="openai"),
            _TIMEOUT_SHAPE,
        ),
        (
            le.APIConnectionError(
                message="OpenAIException - Connection error.",
                llm_provider="openai",
                model="gpt-6-luna",
            ),
            _UNREACHABLE_SHAPE,
        ),
        (httpx.ConnectError("[Errno 111] Connection refused"), _UNREACHABLE_SHAPE),
        (TimeoutError(), _TIMEOUT_SHAPE),
    ],
)
async def test_network_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    shape: re.Pattern[str],
) -> None:
    _recording(monkeypatch, raises=error)
    data = await _test_saved(
        tmp_path, "openai/gpt-6-luna", api_key="sk-saved-123456789"
    )
    assert data["reason"] == "network"
    assert shape.match(data["detail"]), data["detail"]


async def test_a_local_runtimes_refused_login_is_auth_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.providers.errors import AuthenticationError

    class _Refusing:
        async def acompletion(self, **_kwargs: Any) -> Any:
            raise AuthenticationError("Please run /login", provider="claude-agent")

    monkeypatch.setattr(
        "markitai.providers.is_local_provider_available", lambda _m: True
    )
    monkeypatch.setattr("markitai.providers.get_provider", lambda _p: _Refusing())
    data = await _test_saved(tmp_path, "claude-agent/sonnet")
    assert data["reason"] == "auth_rejected"
    assert _HTTP_SHAPE.match(data["detail"])
    assert data["provider_detail"] == "Please run /login"


async def test_an_uncategorized_error_keeps_its_own_first_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recording(monkeypatch, raises=RuntimeError("something odd\nsecond line"))
    data = await _test_saved(
        tmp_path, "openai/gpt-6-luna", api_key="sk-saved-123456789"
    )
    assert data["reason"] == "provider_error"
    assert data["detail"] == "RuntimeError: something odd"


def test_every_reason_is_one_the_task_named() -> None:
    assert set(REASONS) == {
        "missing_key",
        "auth_rejected",
        "model_not_found",
        "rate_limited",
        "network",
        "provider_error",
    }
