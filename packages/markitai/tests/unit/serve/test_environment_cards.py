"""Models added from an environment card use that provider's key variable.

The workbench shows a card for each provider whose key variable the server
has (``OPENAI_API_KEY`` -> ``env:openai``); listing and adding its models
send ``env:openai`` as the connection. Python looked only among saved
connections and answered 404. As in the API contract and the Rust server,
the card stands for the provider's own key variable at the provider's own
endpoint: never another variable, never an address the caller picks.
Requests leaving the test go to a stubbed transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

pytest.importorskip("fastapi")

from markitai.constants import PROVIDER_API_KEY_ENV, PROVIDER_DEFAULT_MODELS
from tests.unit.serve.test_serve_settings import _make_app, _serve_client

SECRETS = {
    variable: f"sk-SERVER-{provider.upper()}-0000"
    for provider, variable in PROVIDER_API_KEY_ENV.items()
}
HOSTS = {
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
    "gemini": "generativelanguage.googleapis.com",
    "deepseek": "api.deepseek.com",
    "openrouter": "openrouter.ai",
}
CALLER = "caller.example"


@pytest.fixture(autouse=True)
def server_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server's environment holds each card provider's key."""
    for variable, value in SECRETS.items():
        monkeypatch.setenv(variable, value)


def _outside(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Record every request that would leave the test and answer it with an
    empty model list; the test's own ASGI client goes through."""
    sent: list[httpx.Request] = []
    original = httpx.AsyncClient.send

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> Any:
        if request.url.host == "127.0.0.1":
            return await original(self, request, **kw)
        sent.append(request)
        return httpx.Response(200, json={"data": [], "models": []}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return sent


def _carries(request: httpx.Request, secret: str) -> bool:
    headers = " ".join(request.headers.values())
    return secret in headers or secret in str(request.url)


async def _revision(client: httpx.AsyncClient) -> str:
    return (await client.get("/api/settings/llm")).json()["revision"]


async def _add(client: httpx.AsyncClient, deployment: dict[str, Any]) -> httpx.Response:
    return await client.post(
        "/api/settings/llm/deployments/batch",
        json={
            "expected_revision": await _revision(client),
            "deployments": [{"model_name": "default", "weight": 1, **deployment}],
        },
    )


async def _discover(client: httpx.AsyncClient, **body: Any) -> httpx.Response:
    return await client.post(
        "/api/settings/llm/model-discovery", json={"refresh": True, **body}
    )


async def _cards(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    return (await client.get("/api/settings/llm/providers")).json()["providers"]


def _saved(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))["llm"]


async def test_models_added_from_an_environment_card_keep_its_key_variable(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        card = next(c for c in await _cards(client) if c["id"] == "env:openai")
        assert card["kind"] == "environment"
        added = await _add(
            client,
            {
                "model": "openai/gpt-5-mini",
                "provider": "openai",
                "credential_provider_id": card["id"],
            },
        )
        assert added.status_code == 200, added.text
        # The card again (a page loaded before the save): the same connection
        again = await _add(
            client,
            {
                "model": "openai/gpt-5-nano",
                "provider": "openai",
                "credential_provider_id": card["id"],
            },
        )
        assert again.status_code == 200, again.text
        cards = await _cards(client)
    llm = _saved(config_path)
    (connection,) = llm["providers"]
    assert {k: v for k, v in connection.items() if k != "id"} == {
        "provider": "openai",
        "api_key": "env:OPENAI_API_KEY",
    }
    assert [entry["litellm_params"] for entry in llm["model_list"]] == [
        {"model": "openai/gpt-5-mini", "api_key": "env:OPENAI_API_KEY"},
        {"model": "openai/gpt-5-nano", "api_key": "env:OPENAI_API_KEY"},
    ]
    assert {entry["model_info"]["provider_id"] for entry in llm["model_list"]} == {
        connection["id"]
    }
    assert SECRETS["OPENAI_API_KEY"] not in config_path.read_text(encoding="utf-8")
    # One connection, not two cards: the saved reference replaces the card
    assert not any(c["id"] == "env:openai" for c in cards)
    (saved,) = [c for c in cards if c["kind"] == "configured"]
    assert saved["provider_id"] == connection["id"]
    assert saved["model_count"] == 2


@pytest.mark.parametrize("provider", list(PROVIDER_API_KEY_ENV))
async def test_a_detected_model_is_saved_with_its_key_variable(
    tmp_path: Path, provider: str
) -> None:
    """Saving a detected model (no config file yet) sends only the model and
    ``env:<provider>``; no address is stored, so the key stays with its
    provider even where new connections get a documented address."""
    config_path = tmp_path / "config.json"
    model = PROVIDER_DEFAULT_MODELS[provider]
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(
            client, {"model": model, "credential_provider_id": f"env:{provider}"}
        )
        assert added.status_code == 200, added.text
    (entry,) = _saved(config_path)["model_list"]
    assert entry["litellm_params"] == {
        "model": model,
        "api_key": f"env:{PROVIDER_API_KEY_ENV[provider]}",
    }


@pytest.mark.parametrize("provider", list(PROVIDER_API_KEY_ENV))
async def test_listing_a_cards_models_sends_its_key_to_its_own_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        sent = _outside(monkeypatch)
        listed = await _discover(
            client, provider=provider, provider_id=f"env:{provider}"
        )
    assert listed.status_code == 200, listed.text
    assert listed.json()["provider"] == provider
    secret = SECRETS[PROVIDER_API_KEY_ENV[provider]]
    assert [urlsplit(str(r.url)).hostname for r in sent] == [HOSTS[provider]]
    assert _carries(sent[0], secret)
    assert sent[0].url.scheme == "https"


async def test_an_address_with_the_callers_own_key_gets_only_that_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        sent = _outside(monkeypatch)
        listed = await _discover(
            client,
            provider="openai",
            provider_id="env:openai",
            api_key="sk-caller-own-1111",
            api_base=f"https://{CALLER}/v1",
        )
    assert listed.status_code == 200, listed.text
    assert [r.url.host for r in sent] == [CALLER]
    assert sent[0].headers["authorization"] == "Bearer sk-caller-own-1111"
    assert not any(_carries(sent[0], secret) for secret in SECRETS.values())


@pytest.mark.parametrize(
    "reference",
    [
        # a variable, not a card
        "env:OPENAI_API_KEY",
        "env:AWS_SECRET_ACCESS_KEY",
        "env:",
        # a provider without a card here: Python offers env cards for its
        # five key variables only
        "env:groq",
    ],
)
async def test_only_the_card_providers_are_environment_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-SERVER-SECRET")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-SERVER-GROQ")
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        sent = _outside(monkeypatch)
        listed = await _discover(client, provider="openai", provider_id=reference)
        added = await _add(
            client, {"model": "openai/gpt-5-mini", "credential_provider_id": reference}
        )
    for answer in (listed, added):
        assert answer.status_code == 422, answer.text
        assert answer.json()["reason"] == "invalid_settings_request"
    assert sent == []
    assert not config_path.exists()


@pytest.mark.parametrize(
    "change",
    [
        # another provider's card
        {"provider": "anthropic", "model": "anthropic/claude-haiku-5-5"},
        # the card's key at an address the caller picks
        {"provider": "openai", "api_base": f"https://{CALLER}/v1"},
        {"provider": "openai", "api_base": "http://127.0.0.1:9/v1"},
    ],
)
async def test_a_cards_key_never_leaves_its_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, str]
) -> None:
    config_path = tmp_path / "config.json"
    request = {"provider": "openai", "model": "openai/gpt-5-mini", **change}
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        sent = _outside(monkeypatch)
        listed = await _discover(
            client,
            provider=request["provider"],
            provider_id="env:openai",
            **({"api_base": request["api_base"]} if "api_base" in request else {}),
        )
        added = await _add(client, {**request, "credential_provider_id": "env:openai"})
    for answer in (listed, added):
        assert answer.status_code == 422, answer.text
        assert answer.json()["reason"] == "invalid_settings_request"
    assert sent == []
    assert not config_path.exists()
