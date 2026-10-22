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


# OpenRouter: the default is its moving alias ~anthropic/claude-haiku-latest.
# On 2026-10-10 OpenRouter's catalog named Claude Haiku 5.5 as its target, but
# the alias was answered and billed as Claude Haiku 4.5: its limits stay the
# smaller ones, and the bill is what OpenRouter reports.
_ALIAS = "openrouter/~anthropic/claude-haiku-latest"
_ROUTED = "openrouter/anthropic/claude-haiku-5.5"


def test_the_openrouter_default_keeps_the_limits_of_what_serves_it() -> None:
    from markitai.constants import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["openrouter"] == _ALIAS
    info = litellm.get_model_info(_ALIAS)
    assert info["max_output_tokens"] == 64000  # Haiku 4.5 rejects more
    assert info["max_input_tokens"] == 200000
    assert info["supports_vision"] is True  # --alt works
    assert info["supports_function_calling"] is True


def test_haiku_5_5_on_openrouter_is_priced() -> None:
    info = litellm.get_model_info(_ROUTED)
    assert info["input_cost_per_token"] == pytest.approx(1e-07)
    assert info["output_cost_per_token"] == pytest.approx(5e-07)


def test_the_alias_entry_fills_in_when_litellm_drops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(litellm.model_cost, _ALIAS, raising=False)
    monkeypatch.setattr(catalog, "_registered", False)

    assert _ALIAS in catalog.register_missing_models()
    assert litellm.model_cost[_ALIAS]["max_output_tokens"] == 64000
    assert litellm.model_cost[_ALIAS]["supports_vision"] is True


def test_an_openrouter_answer_is_charged_what_openrouter_reports() -> None:
    """Its answer names the routed model; OpenRouter's own figure is the bill."""
    from markitai.llm.models import get_response_cost

    response = litellm.ModelResponse(
        id="gen-1",
        model="anthropic/claude-haiku-5.5-20261007",  # no entry under this name
        choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0}],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    response._hidden_params = {
        "custom_llm_provider": "openrouter",
        "response_cost": 0.0123,
    }

    assert get_response_cost(response) == pytest.approx(0.0123)


# LiteLLM 1.104.2 lists neither: their requests were priced as unknown. The
# rows are LiteLLM 1.106.0.dev2's, the ones markitai-rust prices from.
@pytest.mark.parametrize(
    ("model", "rates"),
    [
        (
            "claude-sonnet-5-5",
            {
                "input_cost_per_token": 2e-06,
                "output_cost_per_token": 1e-05,
                "cache_read_input_token_cost": 1e-07,
                "cache_creation_input_token_cost": 2.5e-06,
                "cache_creation_input_token_cost_above_1hr": 4e-06,
                "input_cost_per_token_batches": 1e-06,
                "output_cost_per_token_batches": 5e-06,
            },
        ),
        (
            "gpt-6.1-sol",
            {
                "input_cost_per_token": 2e-06,
                "output_cost_per_token": 1e-05,
                "cache_read_input_token_cost": 1e-07,
                "cache_creation_input_token_cost": 2.5e-06,
                "input_cost_per_token_above_272k_tokens": 4e-06,
                "output_cost_per_token_above_272k_tokens": 1.5e-05,
                "input_cost_per_token_batches": 1e-06,
                "output_cost_per_token_batches": 5e-06,
            },
        ),
        (
            "claude-haiku-5-5",
            {
                "input_cost_per_token": 1e-07,
                "output_cost_per_token": 5e-07,
                "input_cost_per_token_above_100k_tokens": 5e-07,
                "cache_read_input_token_cost_batches": 5e-09,
                "cache_creation_input_token_cost_batches": 6.25e-08,
            },
        ),
    ],
)
def test_the_shipped_rows_carry_the_official_rates(
    model: str, rates: dict[str, float]
) -> None:
    entry = litellm.model_cost[model]
    for field, rate in rates.items():
        assert entry[field] == pytest.approx(rate), field


@pytest.mark.parametrize(
    ("model", "prompt", "cost"),
    [
        ("claude-sonnet-5-5", 10_000, 10_000 * 2e-06 + 1_000 * 1e-05),
        ("gpt-6.1-sol", 10_000, 10_000 * 2e-06 + 1_000 * 1e-05),
        # Over 272K the whole request takes the long-context rates
        ("gpt-6.1-sol", 300_000, 300_000 * 4e-06 + 1_000 * 1.5e-05),
    ],
)
def test_their_answers_are_priced(model: str, prompt: int, cost: float) -> None:
    from markitai.llm.models import get_response_cost

    response = litellm.ModelResponse(
        id="r",
        model=model,
        choices=[{"message": {"role": "assistant", "content": "ok"}, "index": 0}],
        usage=litellm.Usage(
            prompt_tokens=prompt, completion_tokens=1_000, total_tokens=prompt + 1_000
        ),
    )
    assert get_response_cost(response) == pytest.approx(cost)


def test_registering_prints_nothing() -> None:
    """LiteLLM printed "Provider List: ..." to stdout while registering
    gpt-6.1-sol, a name it cannot place yet, ahead of --json output. A fresh
    interpreter: this one's LiteLLM already knows the name. A LiteLLM that
    lists it (1.106) has its entry removed first, so the registration runs."""
    import subprocess
    import sys

    script = (
        "from markitai.utils.litellm_import import prepare_litellm_import\n"
        "prepare_litellm_import()\n"
        "import litellm\n"
        "litellm.model_cost.pop('gpt-6.1-sol', None)\n"
        "from markitai.model_catalog import register_missing_models\n"
        "assert 'gpt-6.1-sol' in register_missing_models()\n"
        "assert litellm.suppress_debug_info is False\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
