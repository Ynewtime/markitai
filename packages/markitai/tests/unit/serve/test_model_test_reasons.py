"""POST /api/settings/llm/test: why a failed model test failed, by category.

Seen in practice: a DeepSeek test in Settings -> Models failed with
"AuthenticationError: litellm.AuthenticationError: AuthenticationError:
DeepseekException - Authentication Fails (governor)" on a server that had no
DEEPSEEK_API_KEY at all: the test sent the request without a key, and the
provider's 401 read like a wrong key. A test now names the unset variable
without sending anything, and a failed one answers as API contract v1.10
says (``{ok: false, detail, reason, key_variable?}``, the reasons Rust's
``service_probe`` sends), with the provider's own words in ``detail``, which
the workbench shows under "Provider's message".
"""

from __future__ import annotations

import json
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

# The sentence shapes both backends use for a model test's detail (Rust's
# markitai-core service_probe): the status, then the provider's words
_HTTP_SHAPE = re.compile(r"^Model connection returned HTTP (\d{3})(?:: (.+))?$")
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
    assert data["reason"] == "no_key"
    assert data["key_variable"] == "DEEPSEEK_API_KEY"
    assert data["detail"].startswith(
        "DEEPSEEK_API_KEY is not set on the server, so nothing was sent"
    )
    # The hint is in the server shell's own syntax ($env:NAME= on Windows).
    assert (
        "DEEPSEEK_API_KEY=your_key" in data["detail"]
        or "$env:DEEPSEEK_API_KEY=" in data["detail"]
    )
    assert set(data) == {"ok", "detail", "reason", "key_variable"}


async def test_an_unset_env_reference_is_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recording(monkeypatch)
    data = await _test_saved(
        tmp_path, "gemini/gemini-flash-lite-latest", api_key="env:MY_GEMINI_KEY"
    )
    assert calls == []
    assert data["reason"] == "no_key"
    assert data["key_variable"] == "MY_GEMINI_KEY"
    assert data["detail"].startswith("MY_GEMINI_KEY is not set on the server")


async def test_an_env_reference_holding_a_key_is_never_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-written ``env:`` with a pasted key in it names no variable."""
    calls = _recording(monkeypatch)
    data = await _test_saved(
        tmp_path, "gemini/gemini-flash-lite-latest", api_key="env:AIza-pasted.key"
    )
    assert calls == []
    assert data["reason"] == "no_key"
    assert "key_variable" not in data
    assert "AIza" not in json.dumps(data)


async def test_any_connectable_provider_without_its_key_is_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond markitai's five: a provider the workbench connects (Groq)."""
    calls = _recording(monkeypatch)
    data = await _test_saved(tmp_path, "groq/llama-4-scout")
    assert calls == []
    assert data["reason"] == "no_key"
    assert data["key_variable"] == "GROQ_API_KEY"
    assert data["detail"].startswith("GROQ_API_KEY is not set on the server")


async def test_a_key_in_the_environment_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-1234567890")
    calls = _recording(monkeypatch)
    data = await _test_saved(tmp_path, "deepseek/deepseek-flash")
    assert len(calls) == 1
    # Success is {ok, detail}: no reason, as the contract says
    assert set(data) == {"ok", "detail"}
    assert data["ok"] is True


async def test_a_keyless_local_server_is_not_no_key(
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
    ("error", "reason", "status", "words"),
    [
        (
            le.AuthenticationError(
                message="DeepseekException - Authentication Fails (governor)",
                llm_provider="deepseek",
                model="deepseek-flash",
            ),
            "key_rejected",
            "401",
            "Authentication Fails (governor)",
        ),
        (
            le.PermissionDeniedError(
                message="DeepseekException - Forbidden",
                llm_provider="deepseek",
                model="deepseek-flash",
                response=_response(403),
            ),
            "key_rejected",
            "403",
            "Forbidden",
        ),
        (
            le.NotFoundError(
                message="DeepseekException - The model does not exist",
                model="deepseek-flash",
                llm_provider="deepseek",
            ),
            "model_not_found",
            "404",
            "The model does not exist",
        ),
        (
            le.BadRequestError(
                message="DeepseekException - Model Not Exist",
                model="deepseek-flash",
                llm_provider="deepseek",
            ),
            "model_not_found",
            "400",
            "Model Not Exist",
        ),
        (
            le.RateLimitError(
                message="DeepseekException - Too Many Requests",
                llm_provider="deepseek",
                model="deepseek-flash",
            ),
            "rate_limited",
            "429",
            "Too Many Requests",
        ),
        (
            le.InternalServerError(
                message="DeepseekException - upstream overloaded",
                llm_provider="deepseek",
                model="deepseek-flash",
            ),
            "provider_error",
            "500",
            "upstream overloaded",
        ),
    ],
)
async def test_a_refusal_by_status_keeps_the_providers_words(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
    status: str,
    words: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env-1234567890")
    _recording(monkeypatch, raises=error)
    data = await _test_saved(tmp_path, "deepseek/deepseek-flash")
    assert set(data) == {"ok", "detail", "reason"}
    assert data["ok"] is False
    assert data["reason"] == reason
    match = _HTTP_SHAPE.match(data["detail"])
    assert match is not None, data["detail"]
    assert match.group(1) == status
    # The provider's own words, without LiteLLM's class names
    assert match.group(2) == words
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
    assert data == {
        "ok": False,
        "detail": "Model connection returned HTTP 401: Authentication Fails (governor)",
        "reason": "key_rejected",
    }


async def test_a_key_saved_with_the_connection_is_rejected_not_missing(
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
    assert data["reason"] == "key_rejected"
    assert data["detail"] == (
        "Model connection returned HTTP 401: Incorrect API key provided"
    )
    assert "sk-saved-123456789" not in str(data)


async def test_a_keyless_request_refused_is_no_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sent without a key (an address of its own), refused 401: no key."""
    _recording(
        monkeypatch,
        raises=le.AuthenticationError(
            message="GroqException - Invalid API Key",
            llm_provider="groq",
            model="llama-4-scout",
        ),
    )
    entry = ModelConfig(
        model_name="default",
        litellm_params=LiteLLMParams(
            model="groq/llama-4-scout", api_base="https://groq.proxy.example/v1"
        ),
    )
    async with _client(_make_app(tmp_path, [entry])) as client:
        settings = (await client.get("/api/settings/llm")).json()
        resp = await client.post(
            "/api/settings/llm/test",
            json={"deployment_id": settings["deployments"][0]["deployment_id"]},
        )
    data = resp.json()
    assert data["reason"] == "no_key"
    assert data["key_variable"] == "GROQ_API_KEY"


async def test_the_keyless_placeholder_refused_is_no_key_without_a_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draft naming an address is tested with the placeholder, not a key."""
    calls = _recording(
        monkeypatch,
        raises=le.AuthenticationError(
            message="OpenAIException - 401 Unauthorized",
            llm_provider="openai",
            model="local-model",
        ),
    )
    async with _client(_make_app(tmp_path)) as client:
        resp = await client.post(
            "/api/settings/llm/test",
            json={"model": "openai/local-model", "api_base": "http://127.0.0.1:9/v1"},
        )
    data = resp.json()
    assert calls and calls[0]["api_key"] == "markitai-no-key"
    assert data["reason"] == "no_key"
    assert "key_variable" not in data


@pytest.mark.parametrize(
    ("error", "reason", "detail"),
    [
        (
            le.Timeout(message="Request timed out", model="m", llm_provider="openai"),
            "timeout",
            re.compile(r"^Model connection test timed out$"),
        ),
        (TimeoutError(), "timeout", re.compile(r"^Model connection test timed out$")),
        (
            httpx.ReadTimeout("The read operation timed out"),
            "timeout",
            re.compile(r"^Model connection test timed out$"),
        ),
        (
            le.APIConnectionError(
                message="OpenAIException - Connection error.",
                llm_provider="openai",
                model="gpt-6-luna",
            ),
            "network",
            re.compile(r"^Model connection request failed: Connection error\.$"),
        ),
        (
            httpx.ConnectError("[Errno 111] Connection refused"),
            "network",
            re.compile(r"^Model connection request failed: .*Connection refused$"),
        ),
    ],
)
async def test_network_failures_and_timeouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    reason: str,
    detail: re.Pattern[str],
) -> None:
    _recording(monkeypatch, raises=error)
    data = await _test_saved(
        tmp_path, "openai/gpt-6-luna", api_key="sk-saved-123456789"
    )
    assert data["reason"] == reason
    assert detail.match(data["detail"]), data["detail"]


async def test_a_refused_connection_litellm_reports_as_500_is_network(
    tmp_path: Path,
) -> None:
    """LiteLLM itself, to a closed loopback port: no answer, so network.

    LiteLLM raises InternalServerError 500 "Connection error." with the
    ConnectionRefusedError several links down its chain.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    entry = ModelConfig(
        model_name="default",
        litellm_params=LiteLLMParams(
            model="openai/gpt-6-luna",
            api_key="sk-saved-123456789",
            api_base=f"http://127.0.0.1:{port}/v1",
        ),
    )
    async with _client(_make_app(tmp_path, [entry])) as client:
        settings = (await client.get("/api/settings/llm")).json()
        resp = await client.post(
            "/api/settings/llm/test",
            json={"deployment_id": settings["deployments"][0]["deployment_id"]},
        )
    data = resp.json()
    assert data["reason"] == "network", data
    assert data["detail"].startswith("Model connection request failed")


async def test_a_local_runtimes_refused_login_is_its_own_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sign-in is no API key: a provider error with what the runtime said."""
    from markitai.providers.errors import AuthenticationError

    class _Refusing:
        async def acompletion(self, **_kwargs: Any) -> Any:
            raise AuthenticationError("Please run /login", provider="claude-agent")

    monkeypatch.setattr(
        "markitai.providers.is_local_provider_available", lambda _m: True
    )
    monkeypatch.setattr("markitai.providers.get_provider", lambda _p: _Refusing())
    data = await _test_saved(tmp_path, "claude-agent/sonnet")
    assert data["reason"] == "provider_error"
    assert data["detail"] == "Please run /login"


async def test_an_uncategorized_error_keeps_its_own_first_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recording(monkeypatch, raises=RuntimeError("something odd\nsecond line"))
    data = await _test_saved(
        tmp_path, "openai/gpt-6-luna", api_key="sk-saved-123456789"
    )
    assert data["reason"] == "provider_error"
    assert data["detail"] == "RuntimeError: something odd"


def test_every_reason_is_the_contracts() -> None:
    """API contract v1.10, as Rust's service_probe sends them."""
    assert REASONS == (
        "no_key",
        "key_rejected",
        "model_not_found",
        "rate_limited",
        "network",
        "timeout",
        "provider_error",
    )


_UI = Path(__file__).parents[3] / "src/markitai/serve/workbench/ui"


def test_the_bundle_reads_these_reasons_and_translates_each() -> None:
    """The vendored workbench knows every reason Python sends.

    It reads ``reason`` and ``key_variable`` (never ``provider_detail``) and
    has a title for each reason in en (app.js), zh and ja; provider_error is
    every table's fallback.
    """
    app = (_UI / "app.js").read_text(encoding="utf-8")
    listed = re.search(r'\["no_key"(?:,"[a-z_]+")*\]', app)
    assert listed is not None
    assert tuple(json.loads(listed.group(0))) == REASONS
    assert ".key_variable" in app
    assert "provider_detail" not in app
    for name in ("app.js", "i18n-zh.js", "i18n-ja.js"):
        text = (_UI / name).read_text(encoding="utf-8")
        titles = re.search(r"testFailTitle:\w+=>\(\{([^}]*)\}\)\[\w+\]\?\?\"", text)
        assert titles is not None, name
        keys = set(re.findall(r"(\w+):\"", titles.group(1)))
        assert keys == set(REASONS) - {"provider_error"}, name
