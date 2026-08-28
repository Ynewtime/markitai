"""The Azure configuration the docs hand out has to actually work.

`configuration.md` shows an Azure OpenAI deployment written as model +
api_key + api_base + `api_version`. `LiteLLMParams` declared no such field
and pydantic ignores unknown keys by default, so the version was accepted
without complaint, dropped on parse, and never reached litellm — a config
copied verbatim from the docs went out with whatever version litellm
inferred, and nothing said so.
"""

from __future__ import annotations

from markitai.config import LiteLLMParams, LLMConfig, ModelConfig


def _router_entry(params: LiteLLMParams) -> dict:
    from markitai.llm import LLMProcessor

    model = ModelConfig(model_name="default", litellm_params=params)
    processor = LLMProcessor(LLMConfig(model_list=[model]))
    entries = processor._build_router_entries([model])
    assert entries, "the model was dropped from the router entirely"
    return entries[0]["litellm_params"]


class TestApiVersion:
    def test_it_survives_parsing(self) -> None:
        params = LiteLLMParams(
            model="azure/my-deployment", api_version="2025-02-01-preview"
        )
        assert params.api_version == "2025-02-01-preview"

    def test_it_reaches_litellm(self) -> None:
        entry = _router_entry(
            LiteLLMParams(
                model="azure/my-deployment",
                api_key="k",
                api_base="https://example.openai.azure.com",
                api_version="2025-02-01-preview",
            )
        )
        assert entry.get("api_version") == "2025-02-01-preview"

    def test_a_provider_without_versions_sends_none(self) -> None:
        """Passing api_version=None to a provider that has no concept of one
        would be noise in every request."""
        entry = _router_entry(LiteLLMParams(model="openai/gpt-4o-mini", api_key="k"))
        assert "api_version" not in entry
