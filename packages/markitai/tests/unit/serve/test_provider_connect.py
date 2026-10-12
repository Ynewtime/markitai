"""Add models offers every provider this build routes, and connects it.

The workbench's picker lists Rust's catalog; Python offers a built-in card
for each provider LiteLLM routes here (``markitai.providers.connectable``),
with the catalog's group, mark and aliases and the facts the provider page
needs: the address used when none is entered, the key variable, and whether
a key is optional. No request leaves the machine: model lists come from a
stubbed HTTP client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from markitai.config import MarkitaiConfig
from markitai.providers import connectable, discovery
from tests.unit.serve.test_serve_settings import _make_app, _model_entry, _serve_client
from tests.unit.serve.test_workbench_contract import PROVIDER_GROUP
from tests.unit.test_provider_discovery import install_http_response

CATALOG = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "src"
        / "markitai"
        / "serve"
        / "provider_catalog.json"
    ).read_text(encoding="utf-8")
)["providers"]


@pytest.fixture(autouse=True)
def no_detected_runtimes(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CLI on PATH, nobody signed in, no key in the environment; the
    subscription runtimes' extras installed (as in a full install)."""
    from markitai.providers.auth import AuthManager

    class Status:
        authenticated = False

    async def signed_out(_self: AuthManager, _provider: str, **_kw: Any) -> Status:
        return Status()

    monkeypatch.setattr(AuthManager, "check_auth", signed_out)
    monkeypatch.setattr(discovery.shutil, "which", lambda _binary: None)
    monkeypatch.setattr(discovery, "_local_runtime_available", lambda _p: True)
    for row in connectable.CONNECTABLE:
        if row.key_variable:
            monkeypatch.delenv(row.key_variable, raising=False)
    discovery._cache.clear()
    discovery._connection_cache.clear()


async def _cards(tmp_path: Path, cfg: MarkitaiConfig | None = None) -> list[dict]:
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        response = await client.get("/api/settings/llm/providers?refresh=true")
    assert response.status_code == 200, response.text
    return response.json()["providers"]


async def test_the_picker_gets_the_groups_rust_offers(tmp_path: Path) -> None:
    cards = await _cards(tmp_path)
    offered = {card["provider"]: card for card in cards if card["kind"] == "common"}
    routable = connectable.routable_providers()
    expected = routable | {"claude-agent", "copilot"}
    assert set(offered) == expected
    # Every group of the catalog has members, as in the Rust server's picker
    assert {card["group"] for card in offered.values()} == PROVIDER_GROUP
    catalog = {row["provider"]: row for row in CATALOG}
    for provider, card in offered.items():
        row = catalog[provider]
        assert card["label"] == row["label"], provider
        assert (card["group"], card["icon"], card["aliases"]) == (
            row["group"],
            row["icon"],
            row["aliases"],
        ), provider
    common = [p for p, card in offered.items() if card["group"] == "common"]
    assert sorted(common) == sorted(
        [
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "openrouter",
            "azure",
            "ollama",
            "mistral",
        ]
    )


@pytest.mark.parametrize(
    ("provider", "facts"),
    [
        (
            "groq",
            (
                "https://api.groq.com/openai/v1",
                "GROQ_API_KEY",
                False,
                True,
                "needs_credentials",
            ),
        ),
        # The documented address, where LiteLLM's own default is older
        (
            "perplexity",
            (
                "https://api.perplexity.ai/router/v1",
                "PERPLEXITYAI_API_KEY",
                False,
                False,
                "needs_credentials",
            ),
        ),
        (
            "lm_studio",
            ("http://localhost:1234/v1", "LM_STUDIO_API_KEY", True, True, "unknown"),
        ),
        # Each server has its own address: the page asks for it
        ("hosted_vllm", (None, "HOSTED_VLLM_API_KEY", True, True, "needs_credentials")),
        ("databricks", (None, "DATABRICKS_API_KEY", False, True, "needs_credentials")),
        # The server's AWS credentials, or a Bedrock API key
        ("bedrock", (None, "AWS_BEARER_TOKEN_BEDROCK", True, False, "unknown")),
        (
            "cohere",
            (
                "https://api.cohere.com/v2/chat",
                "COHERE_API_KEY",
                False,
                True,
                "needs_credentials",
            ),
        ),
        ("ollama", ("http://127.0.0.1:11434", "OLLAMA_API_KEY", True, True, "unknown")),
        ("custom", (None, None, False, True, "needs_credentials")),
        ("copilot", (None, None, True, True, "unknown")),
    ],
)
async def test_a_card_says_how_its_provider_connects(
    tmp_path: Path, provider: str, facts: tuple
) -> None:
    card = next(
        card
        for card in await _cards(tmp_path)
        if card["provider"] == provider and card["kind"] == "common"
    )
    assert (
        card["default_base"],
        card["key_variable"],
        card["key_optional"],
        card["supports_discovery"],
        card["status"],
    ) == facts


async def test_a_provider_this_litellm_cannot_route_is_not_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(connectable, "routable_providers", lambda: frozenset({"groq"}))
    offered = {
        card["provider"] for card in await _cards(tmp_path) if card["kind"] == "common"
    }
    # markitai's first eight stay, and the subscription runtimes it routes
    assert offered == connectable.BUILT_IN | {"groq", "claude-agent", "copilot"}


async def test_morph_and_gradient_ai_are_never_offered(tmp_path: Path) -> None:
    offered = {card["provider"] for card in await _cards(tmp_path)}
    assert not offered & {"morph", "gradient_ai"}


async def _add(client: Any, deployment: dict[str, Any]) -> Any:
    revision = (await client.get("/api/settings/llm")).json()["revision"]
    return await client.post(
        "/api/settings/llm/deployments/batch",
        json={
            "expected_revision": revision,
            "deployments": [{"model_name": "default", "weight": 1, **deployment}],
        },
    )


def _saved(config_path: Path) -> tuple[list[dict], list[dict]]:
    llm = json.loads(config_path.read_text(encoding="utf-8"))["llm"]
    return llm.get("providers", []), llm["model_list"]


async def test_connecting_with_a_key_saves_a_connection_on_the_default_address(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(
            client,
            {
                "model": "groq/llama-3.3-70b-versatile",
                "provider": "groq",
                "api_key": "gsk-test-0000-1111",
            },
        )
        assert added.status_code == 200, added.text
        assert added.json()["routable"] is True
        cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
    providers, models = _saved(config_path)
    assert [
        (p["provider"], p.get("api_key"), p.get("api_base")) for p in providers
    ] == [("groq", "gsk-test-0000-1111", None)]
    assert models[0]["litellm_params"] == {
        "model": "groq/llama-3.3-70b-versatile",
        "api_key": "gsk-test-0000-1111",
    }
    groq = [card for card in cards if card["provider"] == "groq"]
    # The saved connection replaces the built-in card
    assert [card["kind"] for card in groq] == ["configured"]
    assert groq[0]["label"] == "Groq"
    assert groq[0]["api_base"] == "https://api.groq.com"
    assert groq[0]["group"] == "inference"
    assert "gsk-test-0000-1111" not in json.dumps(cards)


async def test_an_entered_address_is_saved_with_the_connection(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(
            client,
            {
                "model": "nebius/meta-llama/Llama-3.3-70B-Instruct",
                "provider": "nebius",
                "api_key": "nb-test-0000-1111",
                "api_base": "https://eu.tokenfactory.example/v1",
            },
        )
    assert added.status_code == 200, added.text
    providers, models = _saved(config_path)
    assert providers[0]["api_base"] == "https://eu.tokenfactory.example/v1"
    assert (
        models[0]["litellm_params"]["api_base"] == "https://eu.tokenfactory.example/v1"
    )


@pytest.mark.parametrize(
    ("provider", "model", "api_key", "base"),
    [
        # LiteLLM's default for these prefixes is api.openai.com
        ("lm_studio", "lm_studio/qwen3-8b", None, "http://localhost:1234/v1"),
        ("xinference", "xinference/qwen3", "any-text", "http://127.0.0.1:9997/v1"),
        # ... or a host the provider has moved away from
        (
            "perplexity",
            "perplexity/sonar",
            "pplx-test-0000",
            "https://api.perplexity.ai/router/v1",
        ),
    ],
)
async def test_a_connection_left_on_the_default_stores_the_documented_address(
    tmp_path: Path, provider: str, model: str, api_key: str | None, base: str
) -> None:
    config_path = tmp_path / "config.json"
    deployment: dict[str, Any] = {"model": model, "provider": provider}
    if api_key is not None:
        deployment["api_key"] = api_key
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(client, deployment)
    assert added.status_code == 200, added.text
    assert added.json()["routable"] is True
    providers, models = _saved(config_path)
    assert models[0]["litellm_params"]["api_base"] == base
    # Keyless at that address: the placeholder, never the server's own key
    assert models[0]["litellm_params"].get("api_key") == (api_key or "markitai-no-key")
    assert providers[0]["api_base"] == base


async def test_a_cloud_model_on_the_servers_own_credentials_saves_no_connection(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    async with _serve_client(_make_app(tmp_path, config_path=config_path)) as client:
        added = await _add(
            client, {"model": "bedrock/amazon.nova-micro-v1:0", "provider": "bedrock"}
        )
    assert added.status_code == 200, added.text
    # Bedrock signs with the server's AWS credentials: jobs may use it
    assert added.json()["routable"] is True
    providers, models = _saved(config_path)
    assert providers == []
    assert models[0]["litellm_params"] == {"model": "bedrock/amazon.nova-micro-v1:0"}


async def test_a_keyed_provider_without_a_key_is_not_routable(tmp_path: Path) -> None:
    cfg = MarkitaiConfig()
    cfg.llm.model_list = [_model_entry("groq/llama-3.3-70b-versatile")]
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        settings = (await client.get("/api/settings/llm")).json()
    assert settings["routable"] is False


async def test_a_keyed_provider_with_its_key_variable_set_is_routable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-environment")
    cfg = MarkitaiConfig()
    cfg.llm.model_list = [_model_entry("groq/llama-3.3-70b-versatile")]
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        settings = (await client.get("/api/settings/llm")).json()
    assert settings["routable"] is True


async def test_a_draft_connection_lists_models_at_the_cards_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    async with _serve_client(_make_app(tmp_path)) as client:
        # Once the test's own client exists: the stub replaces httpx.AsyncClient
        install_http_response(monkeypatch, {"data": [{"id": "qwen3-8b"}]}, captured)
        response = await client.post(
            "/api/settings/llm/model-discovery",
            json={"provider": "lm_studio"},
        )
    assert response.status_code == 200, response.text
    assert captured["url"] == "http://localhost:1234/v1/models"
    assert response.json()["models"] == [
        {"model": "lm_studio/qwen3-8b", "label": "qwen3-8b", "supports_vision": False}
    ]


async def test_a_draft_key_goes_only_to_the_address_entered_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    async with _serve_client(_make_app(tmp_path)) as client:
        # Once the test's own client exists: the stub replaces httpx.AsyncClient
        install_http_response(monkeypatch, {"data": [{"id": "m"}]}, captured)
        response = await client.post(
            "/api/settings/llm/model-discovery",
            json={
                "provider": "groq",
                "api_key": "gsk-draft-0000",
                "api_base": "https://groq-proxy.example/openai/v1",
            },
        )
    assert response.status_code == 200, response.text
    assert captured["url"] == "https://groq-proxy.example/openai/v1/models"
    assert captured["headers"] == {"Authorization": "Bearer gsk-draft-0000"}


async def test_a_provider_without_a_list_leaves_model_ids_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    async with _serve_client(_make_app(tmp_path)) as client:
        # Once the test's own client exists: the stub replaces httpx.AsyncClient
        install_http_response(monkeypatch, {"data": []}, captured)
        response = await client.post(
            "/api/settings/llm/model-discovery",
            json={"provider": "perplexity", "api_key": "pplx-0000"},
        )
    assert response.status_code == 200
    body = response.json()
    assert captured == {}
    assert (body["status"], body["models"]) == ("unavailable", [])
    assert "enter model IDs manually" in body["detail"]
