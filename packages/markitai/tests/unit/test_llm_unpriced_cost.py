"""An LLM model without a known price is reported as unknown cost, never $0.

LiteLLM prices nothing it has no table entry for, and a config that gives a
model ``model_info`` (``supports_vision``, context size) registers an entry
with no price at all, for which ``completion_cost`` answers $0. Both used to
reach ``--json`` as ``cost_usd: 0.0`` (``deepseek/deepseek-flash`` in the live
benchmark: 4 requests, ~9.9k tokens, "0.0"), with no warning.
"""

from __future__ import annotations

from typing import Any

import litellm
import pytest

from markitai.api import ConversionUsage
from markitai.config import LiteLLMParams, LLMConfig, ModelConfig
from markitai.llm.models import get_response_cost
from markitai.llm.processor import LLMProcessor
from markitai.notices import capture_task_notices
from markitai.runs.json_output import build_envelope
from markitai.runs.report import build_llm_usage_block
from markitai.runs.types import Outcome
from markitai.workflow.helpers import merge_llm_usage

UNPRICED = {
    "my-model": {
        "requests": 2,
        "input_tokens": 300,
        "output_tokens": 80,
        "cost_usd": 0.0,
        "unpriced_requests": 2,
    }
}
PRICED = {
    "gpt-4o-mini": {
        "requests": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.000045,
    }
}


def _response(model: str, provider: str) -> Any:
    response = litellm.ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": "hi"}}],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    response._hidden_params = {"custom_llm_provider": provider}
    return response


class TestResponseCost:
    def test_unmapped_model_is_unknown(self) -> None:
        assert get_response_cost(_response("no-such-model-xyz", "openai")) is None

    def test_model_registered_without_a_price_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the router registers for a config entry's ``model_info``."""
        monkeypatch.setitem(
            litellm.model_cost,
            "openai/info-only-model",
            {
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_vision": True,
                "input_cost_per_token_flex": None,
                "cache_read_input_token_cost": None,
            },
        )
        assert get_response_cost(_response("info-only-model", "openai")) is None

    def test_listed_zero_price_is_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            litellm.model_cost,
            "ollama/free-local-model",
            {
                "litellm_provider": "ollama",
                "mode": "chat",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        )
        assert get_response_cost(_response("free-local-model", "ollama")) == 0.0

    def test_priced_model_keeps_its_price(self) -> None:
        cost = get_response_cost(_response("gpt-4o-mini", "openai"))
        assert cost is not None and cost > 0


class TestTracking:
    def test_unpriced_request_is_counted_and_warned_once(self) -> None:
        processor = LLMProcessor(
            LLMConfig(
                enabled=True,
                model_list=[
                    ModelConfig(
                        model_name="default",
                        litellm_params=LiteLLMParams(
                            model="openai/gpt-4o-mini", api_key="test-key"
                        ),
                    )
                ],
            )
        )
        with capture_task_notices() as notices:
            processor._track_usage("my-model", 100, 40, None, "a.pdf")
            processor._track_usage("my-model", 200, 40, None, "a.pdf")

        usage = processor.get_context_usage("a.pdf")["my-model"]
        assert usage["requests"] == 2
        assert usage["unpriced_requests"] == 2
        assert len(notices) == 1
        assert "my-model" in notices[0] and "unknown" in notices[0]

    def test_merge_keeps_the_unpriced_count(self) -> None:
        target: dict[str, dict[str, Any]] = {}
        merge_llm_usage(target, UNPRICED)
        merge_llm_usage(target, UNPRICED)
        merge_llm_usage(target, PRICED)
        assert target["my-model"]["unpriced_requests"] == 4
        assert "unpriced_requests" not in target["gpt-4o-mini"]


class TestReporting:
    def test_json_item_and_totals_report_unknown(self) -> None:
        envelope = build_envelope(
            [
                Outcome(
                    kind="file",
                    source="a.pdf",
                    status="completed",
                    cost_usd=0.0,
                    llm_usage=UNPRICED,
                ),
                Outcome(
                    kind="file",
                    source="b.pdf",
                    status="completed",
                    cost_usd=0.000045,
                    llm_usage=PRICED,
                ),
            ]
        )
        unpriced, priced = envelope["items"]
        assert unpriced["cost_usd"] is None
        assert unpriced["llm_usage"]["my-model"]["cost_usd"] is None
        assert priced["cost_usd"] == 0.000045
        assert priced["llm_usage"]["gpt-4o-mini"]["cost_usd"] == 0.000045
        assert envelope["totals"]["cost_usd"] is None

    def test_priced_run_still_sums(self) -> None:
        envelope = build_envelope(
            [
                Outcome(
                    kind="file",
                    source="b.pdf",
                    status="completed",
                    cost_usd=0.000045,
                    llm_usage=PRICED,
                )
            ]
        )
        assert envelope["totals"]["cost_usd"] == 0.000045

    def test_report_block_reports_unknown(self) -> None:
        block = build_llm_usage_block({**UNPRICED, **PRICED}, cost_usd=0.000045)
        assert block["cost_usd"] is None
        assert block["models"]["my-model"]["cost_usd"] is None
        assert block["models"]["gpt-4o-mini"]["cost_usd"] == 0.000045
        assert block["requests"] == 3

    def test_api_usage_reports_unknown(self) -> None:
        usage = ConversionUsage.from_usage_dict(0.0, dict(UNPRICED))
        assert usage.cost_usd is None
        assert usage.requests == 2
        assert ConversionUsage.from_usage_dict(0.000045, dict(PRICED)).cost_usd == (
            0.000045
        )
