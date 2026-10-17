"""Cloud logins never go to an address a settings caller picks (contract v1.9).

LiteLLM signs Vertex AI requests with the server's Google application default
credentials, watsonx ones with ``WATSONX_TOKEN`` and GigaChat ones with
``GIGACHAT_ACCESS_TOKEN``, whatever key the entry carries, and sends them to
the entry's ``api_base``; the ``markitai-no-key`` placeholder does not stop
it. The settings API therefore refuses such an entry with an address (contract v1.9), while one
without an address keeps using the cloud's own endpoint with the server's
login, as LiteLLM does. Bedrock takes the
placeholder as its bearer key and never signs with the AWS chain, so it keeps
its address. Requests go to a stubbed transport: nothing leaves the machine.
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
from tests.unit.serve.test_keyless_connections import (
    CALLER,
    _add,
    _capture,
    _saved,
    _send_through_router,
    workbench_app_js,
    workbench_detail_patterns,
)
from tests.unit.serve.test_serve_settings import _make_app, _model_entry, _serve_client

ADDRESS = f"https://{CALLER}/v1"
AWS = {
    "AWS_ACCESS_KEY_ID": "AKIAFAKESERVER000000",
    "AWS_SECRET_ACCESS_KEY": "fake-server-aws-secret-0000",
    "AWS_SESSION_TOKEN": "fake-server-aws-session-0000",
    "AWS_BEARER_TOKEN_BEDROCK": "fake-server-bedrock-bearer-0000",
    "AWS_REGION_NAME": "us-east-1",
}
BEDROCK = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
VERTEX = "vertex_ai/gemini-2.5-flash"

# (model, card provider or None, what the refusal names)
REFUSED = [
    pytest.param(VERTEX, "vertex_ai", "Google credentials", id="vertex_ai"),
    pytest.param(
        "vertex_ai_beta/gemini-2.5-flash", None, "Google credentials", id="vertex_beta"
    ),
    # LiteLLM routes this bare name to Vertex AI
    pytest.param("gemini-2.5-pro", None, "Google credentials", id="bare-gemini"),
    pytest.param(
        "watsonx/ibm/granite-3-8b-instruct", None, "WATSONX_TOKEN", id="watsonx"
    ),
    pytest.param(
        "watsonx_text/ibm/granite-13b", None, "WATSONX_TOKEN", id="watsonx_text"
    ),
    pytest.param("gigachat/GigaChat", None, "GIGACHAT_ACCESS_TOKEN", id="gigachat"),
]


# What each refusal names: the cloud, then its login
LABELS = {
    "Google credentials": "Vertex AI",
    "WATSONX_TOKEN": "watsonx",
    "GIGACHAT_ACCESS_TOKEN": "GigaChat",
}


def _detail(credential: str) -> str:
    label = LABELS[credential]
    return (
        f"api_base: {label} models send the server's own {credential} to any "
        f"address, so here they can only use {label}'s own endpoint; set another "
        "address in the config file"
    )


def _refused(response: httpx.Response, credential: str) -> None:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["reason"] == "invalid_settings_request"
    # Verbatim: the shared workbench translates this refusal by its wording
    # (contract v1.9), so a reworded detail would reach users untranslated.
    assert body["detail"] == _detail(credential)


def _deployment(model: str, provider: str | None, **extra: Any) -> dict[str, Any]:
    deployment: dict[str, Any] = {"model": model, "api_base": ADDRESS, **extra}
    if provider is not None:
        deployment["provider"] = provider
    return deployment


@pytest.mark.parametrize(("model", "provider", "credential"), REFUSED)
async def test_a_cloud_login_model_is_not_saved_with_an_address(
    tmp_path: Path, model: str, provider: str | None, credential: str
) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        keyless = await _add(client, _deployment(model, provider))
        keyed = await _add(
            client, _deployment(model, provider, api_key="user-key-0000")
        )
    _refused(keyless, credential)
    _refused(keyed, credential)
    assert not config_path.exists() or not _saved(config_path).get("model_list")


@pytest.mark.parametrize(("model", "provider", "credential"), REFUSED)
async def test_a_cloud_login_model_is_not_tested_at_an_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    provider: str | None,
    credential: str,
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        sent = _capture(monkeypatch)
        response = await client.post(
            "/api/settings/llm/test", json={"model": model, "api_base": ADDRESS}
        )
    _refused(response, credential)
    assert sent == []


async def test_a_vertex_model_under_another_connection_with_an_address_is_refused(
    tmp_path: Path,
) -> None:
    """Saving a Vertex AI model under an OpenAI-compatible connection would
    send the Google login to that connection's host."""
    async with _serve_client(_make_app(tmp_path)) as client:
        added = await _add(
            client,
            {
                "model": "openai/llama.cpp",
                "provider": "custom",
                "api_key": "sk-user-conn-0000",
                "api_base": ADDRESS,
            },
        )
        assert added.status_code == 200, added.text
        cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
        connection = next(c for c in cards if c["kind"] == "configured")
        refused = await _add(
            client,
            {"model": VERTEX, "credential_provider_id": connection["provider_id"]},
        )
    _refused(refused, "Google credentials")


async def test_patching_a_model_or_address_into_a_cloud_login_is_refused(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        assert (
            await _add(
                client,
                {
                    "model": "openai/gpt-x",
                    "api_key": "sk-user-0000",
                    "api_base": ADDRESS,
                },
            )
        ).status_code == 200
        assert (
            await _add(client, {"model": VERTEX, "provider": "vertex_ai"})
        ).status_code == 200
        settings = (await client.get("/api/settings/llm")).json()
        by_model = {d["model"]: d["deployment_id"] for d in settings["deployments"]}
        to_vertex = await client.patch(
            f"/api/settings/llm/deployments/{by_model['openai/gpt-x']}",
            json={"expected_revision": settings["revision"], "model": VERTEX},
        )
        to_address = await client.patch(
            f"/api/settings/llm/deployments/{by_model[VERTEX]}",
            json={"expected_revision": settings["revision"], "api_base": ADDRESS},
        )
        cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
        vertex_card = next(
            c
            for c in cards
            if c["kind"] == "configured" and c["provider"] == "vertex_ai"
        )
        connection_to_address = await client.patch(
            f"/api/settings/llm/providers/{vertex_card['provider_id']}",
            json={"expected_revision": settings["revision"], "api_base": ADDRESS},
        )
    for response in (to_vertex, to_address, connection_to_address):
        _refused(response, "Google credentials")
    saved = {
        e["litellm_params"]["model"]: e["litellm_params"]
        for e in _saved(config_path)["model_list"]
    }
    assert saved[VERTEX] == {"model": VERTEX}
    assert saved["openai/gpt-x"]["api_base"] == ADDRESS


@pytest.mark.parametrize("model", [VERTEX, BEDROCK])
async def test_without_an_address_the_cloud_login_is_used_at_its_own_endpoint(
    tmp_path: Path, model: str
) -> None:
    """Neither key nor address: LiteLLM's own behaviour (the server's Google
    login, the AWS chain), at the cloud's endpoint; no placeholder."""
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(client, {"model": model, "provider": model.split("/")[0]})
    assert added.status_code == 200, added.text
    (entry,) = _saved(config_path)["model_list"]
    assert entry["litellm_params"] == {"model": model}


async def test_bedrock_at_an_address_sends_the_placeholder_not_the_aws_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in AWS.items():
        monkeypatch.setenv(name, value)
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(client, _deployment(BEDROCK, "bedrock"))
        sent = _capture(monkeypatch)
        tested = await client.post(
            "/api/settings/llm/test", json={"model": BEDROCK, "api_base": ADDRESS}
        )
    assert added.status_code == 200, added.text
    assert tested.status_code == 200 and tested.json()["ok"] is False, tested.text
    (entry,) = _saved(config_path)["model_list"]
    assert entry["litellm_params"]["api_key"] == NO_KEY
    await _send_through_router(entry["litellm_params"])  # what a conversion does
    to_caller = [r for r in sent if r.url.host == CALLER]
    assert len(to_caller) >= 2, [str(r.url) for r in sent]  # test, then router
    for request in to_caller:
        assert request.headers.get("authorization") == f"Bearer {NO_KEY}"
        assert "x-amz-security-token" not in request.headers
        everything = json.dumps(dict(request.headers)) + str(request.url)
        for value in AWS.values():
            assert value not in everything


async def test_a_hand_written_vertex_address_is_left_to_the_owner(
    tmp_path: Path,
) -> None:
    """A config-file entry with an address is the admin's choice: editing
    something else keeps it, nothing refuses it."""
    cfg = MarkitaiConfig()
    cfg.llm.model_list = [_model_entry(VERTEX, api_base="https://vertex.corp.example")]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {
                    "model_list": [
                        {
                            "model_name": "default",
                            "litellm_params": {
                                "model": VERTEX,
                                "api_base": "https://vertex.corp.example",
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
        patched = await client.patch(
            f"/api/settings/llm/deployments/{settings['deployments'][0]['deployment_id']}",
            json={"expected_revision": settings["revision"], "weight": 2},
        )
    assert patched.status_code == 200, patched.text
    params = _saved(config_path)["model_list"][0]["litellm_params"]
    assert params["api_base"] == "https://vertex.corp.example"
    assert "api_key" not in params


async def test_litellm_still_sends_the_watsonx_token_despite_the_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why watsonx is refused rather than given the placeholder. If a LiteLLM
    upgrade makes this fail, the refusal may be relaxed to the placeholder."""
    monkeypatch.setenv("WATSONX_TOKEN", "fake-server-watsonx-token-0000")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "project")
    sent = _capture(monkeypatch)
    await _send_through_router(
        {
            "model": "watsonx/ibm/granite-3-8b-instruct",
            "api_base": ADDRESS,
            "api_key": NO_KEY,
        }
    )
    to_caller = [r for r in sent if r.url.host == CALLER]
    assert to_caller
    assert to_caller[0].headers.get("authorization") == (
        "Bearer fake-server-watsonx-token-0000"
    )


@pytest.mark.parametrize("credential", sorted(LABELS))
def test_the_workbench_translates_the_cloud_login_refusal(credential: str) -> None:
    """The vendored bundle matches every wording of the refusal as one template."""
    import re

    patterns = workbench_detail_patterns("errCloudLoginAddress")
    assert patterns, "the bundle no longer translates the cloud-login refusal"
    for pattern in patterns:
        match = re.fullmatch(pattern, _detail(credential))
        assert match is not None
        assert match.groups() == (LABELS[credential], credential)


def test_the_workbench_offers_no_address_where_the_server_refuses_one() -> None:
    """Every built-in card whose models the settings API refuses with an
    address is one the bundle shows without an address field; Bedrock,
    which keeps its address, is not."""
    import re

    from markitai.providers.connectable import CONNECTABLE
    from markitai.serve.app import _AMBIENT_LOGIN

    tables = [
        set(re.findall(r"(\w+):\"", body))
        for body in re.findall(r"=\{((?:\w+:\"[^\"]*\",?)+)\}", workbench_app_js())
        if 'vertex_ai:"' in body
    ]
    assert len(tables) == 1, "the bundle's no-address providers were not found"
    (no_address,) = tables
    assert set(_AMBIENT_LOGIN) <= no_address
    cards = {row.provider for row in CONNECTABLE}
    assert cards & set(_AMBIENT_LOGIN) <= no_address
    assert "vertex_ai" in cards
    assert "bedrock" in cards and "bedrock" not in no_address
