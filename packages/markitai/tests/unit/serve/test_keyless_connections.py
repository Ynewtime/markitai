"""A keyless connection the settings API saves never carries the server's key.

Without a key, LiteLLM reads the server's own key variable (OPENAI_API_KEY,
GROQ_API_KEY, LM_STUDIO_API_KEY, ...) and sends it to whatever address the
entry names, which a settings caller can choose. Project rule (
"no env: reference aimed at a host the caller picks"): a connection or
deployment saved through the settings API without a key, or tested that way,
uses the placeholder ``markitai-no-key`` instead; entries written by hand in
the config file keep their meaning. Requests go to a stubbed transport:
nothing leaves the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from markitai.config import MarkitaiConfig
from markitai.serve.app import NO_KEY
from tests.unit.serve.test_serve_settings import _make_app, _model_entry, _serve_client

ENV_KEYS = {
    "OPENAI_API_KEY": "sk-SERVER-OPENAI-0000",
    "GROQ_API_KEY": "gsk-SERVER-GROQ-0000",
    "LM_STUDIO_API_KEY": "lms-SERVER-STUDIO-0000",
}
CALLER = "caller.example"
KEYLESS_MOVE_DETAIL = (
    "a connection with a key cannot become one without a key at another "
    "address; add a new connection for that server instead"
)

# (card provider, model, an address the caller picks)
CASES = [
    pytest.param("openai", "openai/gpt-5-mini", f"https://{CALLER}/v1", id="openai"),
    pytest.param(
        "groq", "groq/llama-3.3-70b", f"https://{CALLER}/openai/v1", id="groq"
    ),
    pytest.param(
        "lm_studio", "lm_studio/qwen3-8b", f"http://{CALLER}:1234/v1", id="lm_studio"
    ),
    pytest.param("custom", "openai/llama.cpp", f"http://{CALLER}:8080/v1", id="custom"),
]


@pytest.fixture(autouse=True)
def server_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server's environment holds every provider's key."""
    for name, value in ENV_KEYS.items():
        monkeypatch.setenv(name, value)


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Record requests to the caller's host and refuse them; any other
    request (the test's own ASGI client) goes through."""
    sent: list[httpx.Request] = []
    original = httpx.AsyncClient.send

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> Any:
        if request.url.host == CALLER:
            sent.append(request)
            raise httpx.ConnectError("stubbed", request=request)
        return await original(self, request, **kw)

    def sync_send(self: httpx.Client, request: httpx.Request, **kw: Any) -> Any:
        sent.append(request)
        raise httpx.ConnectError("stubbed", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.Client, "send", sync_send)
    return sent


def _assert_no_server_key(requests: list[httpx.Request]) -> None:
    to_caller = [r for r in requests if r.url.host == CALLER]
    assert to_caller, [str(r.url) for r in requests]
    for request in to_caller:
        headers = " ".join(f"{k}: {v}" for k, v in request.headers.items())
        for secret in ENV_KEYS.values():
            assert secret not in headers, (str(request.url), headers)
        assert request.headers.get("authorization") == f"Bearer {NO_KEY}"


async def _send_through_router(params: dict[str, Any]) -> None:
    """What a job does with the saved entry: LiteLLM's router, markitai's
    way (no retries)."""
    from litellm import Router

    router = Router(
        model_list=[{"model_name": "default", "litellm_params": dict(params)}],
        num_retries=0,
    )
    with pytest.raises(Exception):  # noqa: B017, PT011 - the stub refuses it
        await router.acompletion(
            model="default", messages=[{"role": "user", "content": "hi"}]
        )


async def _add(client: Any, deployment: dict[str, Any]) -> httpx.Response:
    revision = (await client.get("/api/settings/llm")).json()["revision"]
    return await client.post(
        "/api/settings/llm/deployments/batch",
        json={
            "expected_revision": revision,
            "deployments": [{"model_name": "default", "weight": 1, **deployment}],
        },
    )


def _saved(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))["llm"]


@pytest.mark.parametrize(("provider", "model", "address"), CASES)
async def test_a_keyless_connection_never_carries_a_server_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    address: str,
) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(
            client, {"model": model, "provider": provider, "api_base": address}
        )
        assert added.status_code == 200, added.text
        cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
        connection = next(card for card in cards if card["kind"] == "configured")
        credentials = (
            await client.get(
                f"/api/settings/llm/providers/{connection['provider_id']}/credentials"
            )
        ).json()
    llm = _saved(config_path)
    (entry,) = llm["model_list"]
    (saved,) = llm["providers"]
    assert entry["litellm_params"]["api_key"] == NO_KEY
    assert saved["api_key"] == NO_KEY
    # Shown as no key at all
    assert connection["api_key_configured"] is False
    assert credentials["api_key"] is None and credentials["api_key_masked"] is None
    assert added.json()["deployments"][0]["api_key_configured"] is False

    sent = _capture(monkeypatch)
    await _send_through_router(entry["litellm_params"])
    _assert_no_server_key(sent)


@pytest.mark.parametrize(("provider", "model", "address"), CASES)
async def test_a_keyless_test_at_a_chosen_address_carries_no_server_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    address: str,
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        sent = _capture(monkeypatch)
        response = await client.post(
            "/api/settings/llm/test", json={"model": model, "api_base": address}
        )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is False
    _assert_no_server_key(sent)


async def _keyed_connection(client: Any, base: str | None) -> str:
    deployment: dict[str, Any] = {
        "model": "groq/llama-3.3-70b",
        "provider": "groq",
        "api_key": "gsk-user-typed-0000",
    }
    if base is not None:
        deployment["api_base"] = base
    assert (await _add(client, deployment)).status_code == 200
    cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
    return next(card for card in cards if card["kind"] == "configured")["provider_id"]


async def test_a_keyed_connection_cannot_turn_keyless_at_another_host(
    tmp_path: Path,
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        provider_id = await _keyed_connection(client, None)
        revision = (await client.get("/api/settings/llm")).json()["revision"]
        refused = await client.patch(
            f"/api/settings/llm/providers/{provider_id}",
            json={
                "expected_revision": revision,
                "api_key": None,
                "api_base": f"https://{CALLER}/openai/v1",
            },
        )
    assert refused.status_code == 422
    assert refused.json()["reason"] == "invalid_settings_request"
    # Verbatim: the shared workbench translates this refusal by its wording
    # (contract v1.8), so a reworded detail would reach users untranslated.
    assert refused.json()["detail"] == KEYLESS_MOVE_DETAIL


async def test_a_keyed_deployment_cannot_turn_keyless_at_another_host(
    tmp_path: Path,
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        await _keyed_connection(client, "https://api.groq.com/openai/v1")
        settings = (await client.get("/api/settings/llm")).json()
        deployment_id = settings["deployments"][0]["deployment_id"]
        refused = await client.patch(
            f"/api/settings/llm/deployments/{deployment_id}",
            json={
                "expected_revision": settings["revision"],
                "api_key": None,
                "api_base": f"https://{CALLER}/openai/v1",
            },
        )
    assert refused.status_code == 422
    assert refused.json()["reason"] == "invalid_settings_request"
    assert refused.json()["detail"] == KEYLESS_MOVE_DETAIL


async def test_clearing_a_key_at_the_same_address_stores_the_placeholder(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    base = f"http://{CALLER}:1234/v1"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        provider_id = await _keyed_connection(client, base)
        revision = (await client.get("/api/settings/llm")).json()["revision"]
        cleared = await client.patch(
            f"/api/settings/llm/providers/{provider_id}",
            json={"expected_revision": revision, "api_key": None},
        )
    assert cleared.status_code == 200, cleared.text
    llm = _saved(config_path)
    assert llm["providers"][0]["api_key"] == NO_KEY
    assert llm["model_list"][0]["litellm_params"]["api_key"] == NO_KEY


async def test_a_keyless_connection_may_move_to_another_address(
    tmp_path: Path,
) -> None:
    """The placeholder is no credential: moving it sends nobody's key."""
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        assert (
            await _add(
                client,
                {
                    "model": "lm_studio/qwen3-8b",
                    "provider": "lm_studio",
                    "api_base": f"http://{CALLER}:1234/v1",
                },
            )
        ).status_code == 200
        cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
        provider_id = next(c for c in cards if c["kind"] == "configured")["provider_id"]
        revision = (await client.get("/api/settings/llm")).json()["revision"]
        moved = await client.patch(
            f"/api/settings/llm/providers/{provider_id}",
            json={"expected_revision": revision, "api_base": "http://10.0.0.5:1234/v1"},
        )
    assert moved.status_code == 200, moved.text
    saved = _saved(config_path)["providers"][0]
    assert (saved["api_key"], saved["api_base"]) == (NO_KEY, "http://10.0.0.5:1234/v1")


async def test_a_local_server_at_its_default_address_works_without_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LM Studio left on its documented address: stored with the
    placeholder, and LiteLLM sends a request (it refused a keyless OpenAI
    client before)."""
    monkeypatch.delenv("LM_STUDIO_API_KEY")
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        assert (
            await _add(client, {"model": "lm_studio/qwen3-8b", "provider": "lm_studio"})
        ).status_code == 200
    params = _saved(config_path)["model_list"][0]["litellm_params"]
    assert params == {
        "model": "lm_studio/qwen3-8b",
        "api_key": NO_KEY,
        "api_base": "http://localhost:1234/v1",
    }


async def test_entries_written_by_hand_keep_their_meaning(tmp_path: Path) -> None:
    """A config-file entry without a key still lets LiteLLM use the
    environment's key; editing something else leaves it so."""
    cfg = MarkitaiConfig()
    cfg.llm.model_list = [
        _model_entry("openai/gpt-5-mini", api_base="https://proxy.corp.example/v1")
    ]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {
                    "model_list": [
                        {
                            "model_name": "default",
                            "litellm_params": {
                                "model": "openai/gpt-5-mini",
                                "api_base": "https://proxy.corp.example/v1",
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    async with _serve_client(
        _make_app(tmp_path, cfg, config_path=config_path)
    ) as client:
        settings = (await client.get("/api/settings/llm")).json()
        deployment_id = settings["deployments"][0]["deployment_id"]
        patched = await client.patch(
            f"/api/settings/llm/deployments/{deployment_id}",
            json={"expected_revision": settings["revision"], "weight": 2},
        )
    assert patched.status_code == 200, patched.text
    params = _saved(config_path)["model_list"][0]["litellm_params"]
    assert "api_key" not in params
    assert params["weight"] == 2


async def test_a_subscription_runtime_takes_no_placeholder(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(
            client,
            {
                "model": "claude-agent/haiku",
                "provider": "claude-agent",
                "api_base": f"https://{CALLER}/v1",
            },
        )
    assert added.status_code == 200, added.text
    params = _saved(config_path)["model_list"][0]["litellm_params"]
    assert params == {"model": "claude-agent/haiku"}


def workbench_app_js() -> str:
    """The vendored workbench's script."""
    import markitai.serve

    return (
        Path(markitai.serve.__file__).parent / "workbench" / "ui" / "app.js"
    ).read_text(encoding="utf-8")


def workbench_detail_patterns(message_key: str) -> list[str]:
    """The patterns the bundle matches a 422 detail with to show *message_key*.

    The table is ``[[/<pattern>/, <message key or a function returning it>],
    ...]``; JavaScript's ``^``, ``(.+?)``, ``\\1`` and ``$`` read the same in
    Python.
    """
    import re

    return re.findall(
        r"\[/(\^(?:[^/\\]|\\.)*)/,(?:\(\[?[^)]*\)=>\[)?\"" + message_key + r"\"",
        workbench_app_js(),
    )


def test_the_workbench_translates_the_keyless_move_detail() -> None:
    """The vendored bundle matches the refusal by the start of its wording."""
    import re

    patterns = workbench_detail_patterns("errKeylessMove")
    assert patterns, "the bundle no longer translates the keyless-move refusal"
    assert all(re.match(pattern, KEYLESS_MOVE_DETAIL) for pattern in patterns)
