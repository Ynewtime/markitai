"""Reasoning models answer markitai's tasks without thinking.

``deepseek/deepseek-flash`` thought before every cleanup: 70,393 output
tokens for a 33k-token PDF. What LiteLLM 1.104.2 sends for each parameter
was captured offline; these pin the parameters per default model.
"""

from __future__ import annotations

import pytest

from markitai.config import LiteLLMParams, LLMConfig, ModelConfig, PromptsConfig
from markitai.llm import LLMProcessor
from markitai.llm.models import cleanup_reasoning_params


class TestCleanupReasoning:
    @pytest.mark.parametrize(
        ("model", "params"),
        [
            ("deepseek/deepseek-flash", {"thinking": {"type": "disabled"}}),
            ("openai/gpt-6-luna", {"reasoning_effort": "none"}),
            ("anthropic/claude-haiku-5-5", {"thinking": {"type": "disabled"}}),
            # Already answer without thinking; Gemini's LiteLLM mapping would
            # send a field the alias may reject, OpenRouter refuses the key
            ("gemini/gemini-flash-lite-latest", {}),
            ("openrouter/~anthropic/claude-haiku-latest", {}),
            # Rejects thinking: disabled with HTTP 400
            ("anthropic/claude-opus-5-5", {}),
            # No "none" effort (minimal is the floor)
            ("openai/gpt-5", {}),
            ("deepseek/deepseek-reasoner", {}),
            ("openai/gpt-4o-mini", {}),
        ],
    )
    def test_params_per_default_model(self, model: str, params: dict) -> None:
        assert cleanup_reasoning_params(model) == params

    def test_router_entries_carry_them(self) -> None:
        models = [
            "deepseek/deepseek-flash",
            "openai/gpt-6-luna",
            "anthropic/claude-haiku-5-5",
            "gemini/gemini-flash-lite-latest",
        ]
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(model=m, api_key="sk-test"),
                )
                for m in models
            ],
        )
        processor = LLMProcessor(config, PromptsConfig(), no_cache=True)
        entries = {
            e["litellm_params"]["model"]: e["litellm_params"]
            for e in processor._build_router_entries(config.model_list)
        }
        assert entries["deepseek/deepseek-flash"]["thinking"] == {"type": "disabled"}
        assert entries["openai/gpt-6-luna"]["reasoning_effort"] == "none"
        assert entries["anthropic/claude-haiku-5-5"]["thinking"] == {"type": "disabled"}
        assert "thinking" not in entries["gemini/gemini-flash-lite-latest"]
        assert "reasoning_effort" not in entries["gemini/gemini-flash-lite-latest"]
