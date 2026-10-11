"""The providers the serve workbench connects are ones this build routes.

``markitai.providers.connectable`` says, for every provider the workbench's
catalog lists, where a connection's requests go and which key variable
LiteLLM reads. These tests send one request per provider through a LiteLLM
``Router`` (markitai's own path, ``num_retries=0``) with the HTTP transport
stubbed, so nothing leaves the machine, and check the URL and the key the
provider would really get.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from markitai.providers.connectable import (
    BUILT_IN,
    CONNECTABLE,
    connectable,
    default_base,
    installed,
    pinned_base,
    routable_providers,
)

CATALOG = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "markitai"
    / "serve"
    / "provider_catalog.json"
)

# Routed by markitai itself, not through LiteLLM
SUBSCRIPTIONS = {"claude-agent", "copilot", "chatgpt"}
# LiteLLM 1.104.2 refuses the requests markitai sends them: every one for
# GradientAI ("does not support parameter 'max_retries'"), and any with
# max_tokens for Morph, which markitai sets for the models LiteLLM knows
NOT_ROUTED = {"gradient_ai", "morph"}

# An address for the providers where every account or server has its own
TEST_BASES = {
    "azure_ai": "https://example-resource.services.ai.azure.com/models",
    "cloudflare": "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1",
    "databricks": "https://adb-1.azuredatabricks.net/serving-endpoints",
    "heroku": "https://us.inference.heroku.com",
    "hosted_vllm": "http://127.0.0.1:8000/v1",
    "litellm_proxy": "http://127.0.0.1:4000",
}
# The model ID shape a provider's route expects
TEST_MODELS = {
    "cloudflare": "cloudflare/@cf/meta/llama-3.1-8b-instruct",
    "huggingface": "huggingface/meta-llama/Llama-3.3-70B-Instruct",
}
# Where LiteLLM adds more than /chat/completions to the address
CHAT_PATHS = {
    "cohere": "",
    "datarobot": "/api/v2/genai/llmgw/chat/completions/",
    "heroku": "/v1/chat/completions",
    "oobabooga": "/v1/chat/completions",
}
KEY = "sk-markitai-test-key"


def _catalog_providers() -> set[str]:
    rows = json.loads(CATALOG.read_text(encoding="utf-8"))["providers"]
    return {row["provider"] for row in rows}


def test_every_catalog_provider_is_routed_or_left_out_for_a_reason() -> None:
    """A newly vendored catalog makes someone decide about each new provider."""
    table = {row.provider for row in CONNECTABLE}
    assert table <= _catalog_providers()
    assert _catalog_providers() - table == SUBSCRIPTIONS | NOT_ROUTED


def test_the_providers_markitai_always_offered_stay_built_in() -> None:
    assert {
        "openai",
        "anthropic",
        "gemini",
        "deepseek",
        "openrouter",
        "azure",
        "ollama",
        "custom",
    } == BUILT_IN


def test_a_pinned_address_is_the_one_the_card_shows() -> None:
    for row in CONNECTABLE:
        assert pinned_base(row.provider) == (row.base if row.pin_base else None)
        assert default_base(row.provider) == row.base
        if row.pin_base:
            assert row.base, row.provider
    assert connectable("GROQ") is connectable("groq")
    assert connectable("gradient_ai") is None


def test_this_litellm_routes_every_installed_provider_under_its_prefix() -> None:
    routable = routable_providers()
    expected = {row.provider for row in CONNECTABLE if installed(row)}
    assert routable == expected


def _stub_transport(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Record every request; refuse the chat call, answer lookups with 404."""
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        path = request.url.path
        if path.endswith(("/chat/completions", "/chat/completions/", "/converse")) or (
            request.method == "POST"
        ):
            raise httpx.ConnectError("stubbed", request=request)
        return httpx.Response(404, json={"error": "not found"}, request=request)

    async def async_send(_self: Any, request: httpx.Request, **_: Any) -> Any:
        return handle(request)

    def sync_send(_self: Any, request: httpx.Request, **_: Any) -> Any:
        return handle(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    monkeypatch.setattr(httpx.Client, "send", sync_send)
    return sent


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No key or address from the developer's environment steers a route."""
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_API_BASE", "_BASE_URL", "_TOKEN")) or (
            name.startswith(("AWS_", "CLOUDFLARE_", "DATABRICKS_"))
        ):
            monkeypatch.delenv(name, raising=False)
    for row in CONNECTABLE:
        if row.key_variable:
            monkeypatch.delenv(row.key_variable, raising=False)


async def _send_once(model: str, params: dict[str, Any]) -> None:
    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "default", "litellm_params": {"model": model, **params}}
        ],
        num_retries=0,
    )
    with pytest.raises(Exception):  # noqa: B017, PT011 - the stub refuses the call
        await router.acompletion(
            model="default",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )


ROUTED = [row for row in CONNECTABLE if row.provider not in BUILT_IN and not row.cloud]


@pytest.mark.usefixtures("clean_environment")
@pytest.mark.parametrize("row", ROUTED, ids=lambda row: row.provider)
async def test_a_connection_reaches_the_address_its_card_shows(
    row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the workbench stores for a new connection (a key, unless the
    provider takes none; the pinned or entered address, else none) sends the
    request to the card's address, with that key."""
    sent = _stub_transport(monkeypatch)
    base = pinned_base(row.provider) or TEST_BASES.get(row.provider)
    params: dict[str, Any] = {}
    if not row.key_optional:
        params["api_key"] = KEY
    if base is not None:
        params["api_base"] = base
    await _send_once(
        TEST_MODELS.get(row.provider, f"{row.provider}/probe-model"), params
    )

    chats = [r for r in sent if r.method == "POST"]
    assert chats, f"{row.provider}: nothing was sent ({[str(r.url) for r in sent]})"
    address = (base or row.base or "").rstrip("/")
    expected = address + CHAT_PATHS.get(row.provider, "/chat/completions")
    assert str(chats[0].url).split("?")[0] == expected
    if not row.key_optional:
        assert any(KEY in value for value in chats[0].headers.values()), row.provider


@pytest.mark.usefixtures("clean_environment")
@pytest.mark.parametrize(
    "row",
    [row for row in ROUTED if row.key_variable and not row.key_optional],
    ids=lambda row: row.provider,
)
async def test_a_card_names_the_key_variable_litellm_reads(
    row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _stub_transport(monkeypatch)
    monkeypatch.setenv(row.key_variable, KEY)
    base = pinned_base(row.provider) or TEST_BASES.get(row.provider)
    params = {"api_base": base} if base is not None else {}
    await _send_once(
        TEST_MODELS.get(row.provider, f"{row.provider}/probe-model"), params
    )
    chats = [r for r in sent if r.method == "POST"]
    assert chats and any(KEY in value for value in chats[0].headers.values())


@pytest.mark.usefixtures("clean_environment")
async def test_bedrock_takes_a_bearer_key_and_no_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = connectable("bedrock")
    assert row is not None and row.cloud and row.key_optional and row.base is None
    sent = _stub_transport(monkeypatch)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    await _send_once("bedrock/amazon.nova-micro-v1:0", {"api_key": KEY})
    chats = [r for r in sent if r.method == "POST"]
    assert str(chats[0].url) == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "amazon.nova-micro-v1%3A0/converse"
    )
    assert chats[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_providers_left_out_still_refuse_markitai_requests() -> None:
    """When a LiteLLM upgrade routes them, this fails: offer them then."""
    from litellm.llms.morph.chat.transformation import MorphChatConfig

    assert "max_tokens" not in MorphChatConfig().get_supported_openai_params(
        "morph-v3-large"
    )


def test_without_a_pin_litellm_would_send_these_keys_elsewhere() -> None:
    """The reason for the pins: LiteLLM's own defaults for these prefixes."""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    for provider in ("ai21", "lm_studio", "xinference", "oobabooga"):
        _model, _provider, _key, litellm_base = get_llm_provider(f"{provider}/m")
        assert litellm_base is None, provider
        assert pinned_base(provider)
    for provider in ("perplexity", "nebius", "novita", "galadriel", "lemonade"):
        _model, _provider, _key, litellm_base = get_llm_provider(f"{provider}/m")
        assert (litellm_base or "").rstrip("/") != pinned_base(provider), provider
