"""Entries markitai ships for models its pinned LiteLLM does not list.

LiteLLM 1.104.2 has no ``claude-haiku-5-5``: without the entry the Anthropic
default would be priced as unknown, so ``max_cost_per_document_usd`` could
not count its requests.
"""

from __future__ import annotations

import importlib

import pytest

import markitai.model_catalog as catalog

# markitai.llm points LiteLLM at its bundled cost map before importing it;
# importing litellm first would fetch the upstream map for the whole run.
importlib.import_module("markitai.llm")
litellm = importlib.import_module("litellm")


@pytest.fixture
def fresh_registry(monkeypatch: pytest.MonkeyPatch):
    """LiteLLM's table without the entry, and a catalog not registered yet.

    The real table is patched key by key (and restored afterwards): LiteLLM
    keeps indexes beside ``model_cost``, so swapping the whole dict for a
    copy leaves them describing the copy.
    """
    monkeypatch.delitem(litellm.model_cost, "claude-haiku-5-5", raising=False)
    monkeypatch.setattr(catalog, "_registered", False)
    return litellm.model_cost


def test_the_anthropic_default_is_priced() -> None:
    from markitai.constants import PROVIDER_DEFAULT_MODELS

    info = litellm.get_model_info(PROVIDER_DEFAULT_MODELS["anthropic"])
    assert info["input_cost_per_token"] == pytest.approx(1e-07)
    assert info["output_cost_per_token"] == pytest.approx(5e-07)
    assert info["max_output_tokens"] == 128000
    assert info["supports_vision"] is True


def test_a_response_is_charged_at_the_listed_price() -> None:
    from markitai.llm.models import get_response_cost

    response = litellm.ModelResponse(
        id="r",
        model="claude-haiku-5-5",
        choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0}],
        usage=litellm.Usage(
            prompt_tokens=1000, completion_tokens=1000, total_tokens=2000
        ),
    )
    assert get_response_cost(response) == pytest.approx(1000 * 1e-07 + 1000 * 5e-07)


def test_missing_entries_are_registered(fresh_registry: dict) -> None:
    assert catalog.register_missing_models() == ["claude-haiku-5-5"]
    entry = litellm.model_cost["claude-haiku-5-5"]
    assert entry["litellm_provider"] == "anthropic"
    assert entry["supports_sampling_params"] is False
    # idempotent
    assert catalog.register_missing_models() == []


def test_an_entry_litellm_already_has_is_never_replaced(fresh_registry: dict) -> None:
    upstream = {"litellm_provider": "anthropic", "input_cost_per_token": 9e-07}
    fresh_registry["claude-haiku-5-5"] = upstream  # restored by monkeypatch

    assert catalog.register_missing_models() == []
    assert litellm.model_cost["claude-haiku-5-5"] is upstream
