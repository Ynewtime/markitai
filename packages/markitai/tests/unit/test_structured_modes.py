"""Capability-tiered structured output: mode selection and the fallback staircase.

Structured calls used to be MD_JSON for everyone — every model was asked to
hand-write JSON into its answer, and five separate repair layers cleaned up
after it. These tests pin the replacement:

- the tier a model starts at comes from metadata only (LiteLLM's capability
  table, or a local provider handler's own declaration) and never from a
  probe request;
- each structured call site issues the wire format of its tier (tool call /
  ``response_format`` / prompt-only);
- a failing rung falls to the next one down, and JSON repair exists only on
  the last rung, where the model is the one writing the JSON.

Everything here is mocked. What real traffic must still confirm is listed in
``TestWhatMocksCannotProve``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import instructor
import litellm
import pytest

from markitai.constants import DEFAULT_INSTRUCTOR_MAX_RETRIES
from markitai.llm.engine import (
    LLMCall,
    LLMEngine,
    LLMRequestBudgetExceededError,
    RequestBudget,
    run_structured_ladder,
)
from markitai.llm.structured import (
    MODE_JSON_SCHEMA,
    MODE_MD_JSON,
    MODE_TOOLS,
    model_structured_mode,
    router_structured_ladder,
    structured_mode_ladder,
)
from markitai.llm.types import BatchImageAnalysisResult, Frontmatter

# Model ids pinned to a tier by the ``tiered_models`` fixture. Pinning keeps
# the call-site tests independent of LiteLLM's live capability table (which
# TestCapabilityResolution covers separately).
TOOLS_MODEL = "tier/tools-model"
SCHEMA_MODEL = "tier/schema-model"
TEXT_MODEL = "tier/text-model"

TIER_MODELS = {
    MODE_TOOLS: TOOLS_MODEL,
    MODE_JSON_SCHEMA: SCHEMA_MODEL,
    MODE_MD_JSON: TEXT_MODEL,
}

# Rungs each tier's pool walks down, top first.
TIER_LADDERS = {
    MODE_TOOLS: (
        instructor.Mode.TOOLS,
        instructor.Mode.JSON_SCHEMA,
        instructor.Mode.MD_JSON,
    ),
    MODE_JSON_SCHEMA: (instructor.Mode.JSON_SCHEMA, instructor.Mode.MD_JSON),
    MODE_MD_JSON: (instructor.Mode.MD_JSON,),
}

ALL_TIERS = [MODE_TOOLS, MODE_JSON_SCHEMA, MODE_MD_JSON]

# instructor's max_retries counts *retries*, so the bottom rung makes
# DEFAULT_INSTRUCTOR_MAX_RETRIES + 1 attempts; rungs above it make exactly one.
INSTRUCTOR_ATTEMPTS = DEFAULT_INSTRUCTOR_MAX_RETRIES + 1

FRONTMATTER_PAYLOAD = {"description": "A doc", "tags": ["a", "b"]}
BATCH_PAYLOAD = {"images": [{"image_index": 1, "caption": "c", "description": "d"}]}


@pytest.fixture(autouse=True)
def tiered_models():
    """Pin the three tier fixtures, and isolate the per-model cache."""
    import markitai.llm.structured as structured_mod

    saved = dict(structured_mod._model_mode_cache)
    structured_mod._model_mode_cache.update(
        {model: mode for mode, model in TIER_MODELS.items()}
    )
    yield
    structured_mod._model_mode_cache.clear()
    structured_mod._model_mode_cache.update(saved)


# =============================================================================
# Response and router doubles
# =============================================================================


def _response(model: str, message: dict[str, Any]) -> litellm.ModelResponse:
    """Build a ModelResponse instructor can parse."""
    return litellm.ModelResponse(
        id="test-response",
        choices=[{"message": message, "finish_reason": "stop", "index": 0}],
        model=model,
        usage=litellm.Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


def _tool_call_response(model: str, payload: dict[str, Any]) -> litellm.ModelResponse:
    """A well-formed TOOLS-mode answer (arguments filled by the provider)."""
    return _response(
        model,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "response_model",
                        "arguments": json.dumps(payload),
                    },
                }
            ],
        },
    )


def _content_response(model: str, content: str) -> litellm.ModelResponse:
    """An answer whose JSON (or garbage) lives in the message text."""
    return _response(model, {"role": "assistant", "content": content})


def _good_response(
    model: str, tier: str, payload: dict[str, Any]
) -> litellm.ModelResponse:
    """The answer a provider gives when the tier's wire format works."""
    if tier == MODE_TOOLS:
        return _tool_call_response(model, payload)
    if tier == MODE_JSON_SCHEMA:
        return _content_response(model, json.dumps(payload))
    return _content_response(model, f"```json\n{json.dumps(payload)}\n```")


class _ScriptedRouter:
    """Router double that records the wire format of every request."""

    def __init__(self, model_id: str, script: list[Any]) -> None:
        self.model_list = [{"litellm_params": {"model": model_id, "weight": 1}}]
        self.calls: list[dict[str, Any]] = []
        self._script = script

    async def acompletion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step

    def wire_format(self, index: int = 0) -> str:
        """Classify what the structured layer put on the wire."""
        kwargs = self.calls[index]
        if kwargs.get("tools"):
            return MODE_TOOLS
        response_format = kwargs.get("response_format") or {}
        if response_format.get("type") == "json_schema":
            return MODE_JSON_SCHEMA
        return MODE_MD_JSON


def _engine(router: Any, budget: RequestBudget | None = None) -> LLMEngine:
    return LLMEngine(
        router=router,
        semaphore=asyncio.Semaphore(2),
        memory_cache=MagicMock(get=MagicMock(return_value=None)),
        persistent_cache=MagicMock(get=MagicMock(return_value=None)),
        track_usage=MagicMock(),
        calculate_max_tokens=MagicMock(return_value=512),
        get_primary_model=MagicMock(return_value=None),
        max_retries=0,
        request_budget=budget,
    )


def _frontmatter_call(context: str = "doc.pdf") -> LLMCall:
    """The frontmatter structured call, as document.py builds it."""
    return LLMCall(
        purpose="enhance_frontmatter",
        messages=[
            {"role": "system", "content": "You generate frontmatter."},
            {"role": "user", "content": "Summarize this document."},
        ],
        response_model=Frontmatter,
        context=context,
    )


# =============================================================================
# Capability resolution (metadata only, never a probe)
# =============================================================================


class TestCapabilityResolution:
    """Which tier a model starts at, and where that answer comes from."""

    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        """Resolution must run for real here, not hit a warm cache entry."""
        import markitai.llm.structured as structured_mod

        structured_mod._model_mode_cache.clear()
        yield
        structured_mod._model_mode_cache.clear()

    def test_function_calling_model_starts_at_tools(self, monkeypatch):
        monkeypatch.setattr(litellm, "supports_function_calling", lambda _model: True)
        monkeypatch.setattr(litellm, "supports_response_schema", lambda _model: True)

        assert model_structured_mode("vendor/fc-model") == MODE_TOOLS

    def test_response_schema_only_model_starts_at_json_schema(self, monkeypatch):
        monkeypatch.setattr(litellm, "supports_function_calling", lambda _model: False)
        monkeypatch.setattr(litellm, "supports_response_schema", lambda _model: True)

        assert model_structured_mode("vendor/schema-model") == MODE_JSON_SCHEMA

    def test_model_without_capabilities_falls_back_to_md_json(self, monkeypatch):
        monkeypatch.setattr(litellm, "supports_function_calling", lambda _model: False)
        monkeypatch.setattr(litellm, "supports_response_schema", lambda _model: False)

        assert model_structured_mode("vendor/plain-model") == MODE_MD_JSON

    def test_unknown_model_is_not_assumed_capable(self, monkeypatch):
        """LiteLLM raises for models it does not know: assume the floor."""

        def _raise(model: str) -> bool:
            raise Exception(f"unknown model {model}")

        monkeypatch.setattr(litellm, "supports_function_calling", _raise)
        monkeypatch.setattr(litellm, "supports_response_schema", _raise)

        assert model_structured_mode("vendor/never-heard-of-it") == MODE_MD_JSON

    def test_local_providers_declare_their_own_capability(self):
        """The handler is the source of truth for local providers."""
        # claude-agent constrains decoding with a real JSON Schema via the
        # SDK's output_format — the native path this migration hooked up.
        assert model_structured_mode("claude-agent/sonnet") == MODE_JSON_SCHEMA
        # Neither of these SDKs exposes tools or a JSON mode.
        assert model_structured_mode("copilot/gpt-4.1") == MODE_MD_JSON
        assert model_structured_mode("chatgpt/gpt-5.4") == MODE_MD_JSON

    def test_resolution_is_cached_per_model(self, monkeypatch):
        calls: list[str] = []

        def _counting(model: str) -> bool:
            calls.append(model)
            return True

        monkeypatch.setattr(litellm, "supports_function_calling", _counting)

        for _ in range(5):
            model_structured_mode("vendor/cached-model")

        assert calls == ["vendor/cached-model"]

    def test_resolution_issues_no_request(self, monkeypatch):
        """A capability check must never cost a request from the budget."""

        async def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("capability detection must not call the model")

        monkeypatch.setattr(litellm, "acompletion", _forbidden)
        monkeypatch.setattr(litellm, "completion", _forbidden)

        for model in ("gpt-4o", "claude-agent/sonnet", "vendor/unknown"):
            model_structured_mode(model)


class TestModeLadder:
    """The staircase a pool walks down."""

    @pytest.mark.parametrize("tier", ALL_TIERS)
    def test_single_model_pool_ladder(self, tier: str):
        assert structured_mode_ladder([TIER_MODELS[tier]]) == TIER_LADDERS[tier]

    def test_pool_is_only_as_capable_as_its_weakest_member(self):
        """The router picks the deployment after the mode is fixed."""
        assert (
            structured_mode_ladder([TOOLS_MODEL, TEXT_MODEL])
            == TIER_LADDERS[MODE_MD_JSON]
        )
        assert (
            structured_mode_ladder([TOOLS_MODEL, SCHEMA_MODEL])
            == TIER_LADDERS[MODE_JSON_SCHEMA]
        )

    def test_disabled_entries_do_not_drag_the_pool_down(self):
        """weight <= 0 is never routed to, so it never sets the tier."""
        router = MagicMock()
        router.model_list = [
            {"litellm_params": {"model": TOOLS_MODEL, "weight": 1}},
            {"litellm_params": {"model": TEXT_MODEL, "weight": 0}},
        ]

        assert router_structured_ladder(router) == TIER_LADDERS[MODE_TOOLS]

    def test_empty_pool_falls_back_to_md_json(self):
        router = MagicMock()
        router.model_list = []

        assert router_structured_ladder(router) == TIER_LADDERS[MODE_MD_JSON]


# =============================================================================
# Call site: frontmatter (LLMEngine.complete_structured)
# =============================================================================


class TestFrontmatterCallSite:
    """The frontmatter/document structured path through the engine."""

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.asyncio
    async def test_tier_wire_format(self, tier: str):
        model = TIER_MODELS[tier]
        router = _ScriptedRouter(
            model, [_good_response(model, tier, FRONTMATTER_PAYLOAD)]
        )

        result, _ = await _engine(router).complete_structured(_frontmatter_call())

        assert len(router.calls) == 1
        assert router.wire_format() == tier
        assert result.description == "A doc"

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.asyncio
    async def test_malformed_answer_walks_down_the_ladder(self, tier: str):
        """Each rung's failure hands over to the next rung's wire format."""
        model = TIER_MODELS[tier]
        ladder = TIER_LADDERS[tier]
        # Garbage until the last rung, which gets clean fenced JSON.
        script: list[Any] = [
            _content_response(model, "I am afraid I cannot do that.")
            for _ in range(len(ladder) - 1)
        ]
        script.append(_good_response(model, MODE_MD_JSON, FRONTMATTER_PAYLOAD))
        router = _ScriptedRouter(model, script)

        result, _ = await _engine(router).complete_structured(_frontmatter_call())

        assert result.description == "A doc"
        assert len(router.calls) == len(ladder)
        observed = [router.wire_format(i) for i in range(len(router.calls))]
        assert observed == [
            MODE_TOOLS
            if mode is instructor.Mode.TOOLS
            else MODE_JSON_SCHEMA
            if mode is instructor.Mode.JSON_SCHEMA
            else MODE_MD_JSON
            for mode in ladder
        ]

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.asyncio
    async def test_repair_runs_only_on_the_last_rung(self, tier: str):
        """Broken-but-repairable JSON is rescued only where the model wrote it.

        Every rung gets the same trailing-comma JSON in the message text. On
        the rungs above MD_JSON that is simply a failed answer (the provider
        was supposed to produce the value), so the ladder keeps descending;
        the MD_JSON rung repairs it.
        """
        model = TIER_MODELS[tier]
        ladder = TIER_LADDERS[tier]
        broken = '```json\n{"description": "A doc", "tags": ["a", "b"],}\n```'
        router = _ScriptedRouter(model, [_content_response(model, broken)])

        result, _ = await _engine(router).complete_structured(_frontmatter_call())

        assert result.description == "A doc"
        # Rungs above MD_JSON get one attempt each; the last rung gets the
        # full instructor validation retries before repair rescues it.
        assert len(router.calls) == (len(ladder) - 1) + INSTRUCTOR_ATTEMPTS
        assert router.wire_format(len(router.calls) - 1) == MODE_MD_JSON

    @pytest.mark.asyncio
    async def test_unrepairable_answer_on_the_last_rung_raises(self):
        model = TEXT_MODEL
        router = _ScriptedRouter(model, [_content_response(model, "no json here")])

        with pytest.raises(Exception) as excinfo:
            await _engine(router).complete_structured(_frontmatter_call())

        assert not isinstance(excinfo.value, LLMRequestBudgetExceededError)

    @pytest.mark.asyncio
    async def test_every_rung_spends_the_request_budget(self):
        """The staircase must not smuggle attempts past the circuit breaker."""
        model = TOOLS_MODEL
        router = _ScriptedRouter(model, [_content_response(model, "nope")])
        budget = RequestBudget(limit=2)

        with pytest.raises(LLMRequestBudgetExceededError):
            await _engine(router, budget).complete_structured(_frontmatter_call())

        assert len(router.calls) == 2

    @pytest.mark.asyncio
    async def test_budget_refusal_stops_the_ladder_immediately(self):
        """A tripped breaker is not a reason to try the next rung."""
        model = TOOLS_MODEL
        router = _ScriptedRouter(model, [_content_response(model, "nope")])
        budget = RequestBudget(limit=1)
        budget.spend("doc.pdf")  # exhaust before the call

        with pytest.raises(LLMRequestBudgetExceededError):
            await _engine(router, budget).complete_structured(_frontmatter_call())

        assert router.calls == []


# =============================================================================
# Call site: vision analyze_batch (instructor on the vision router)
# =============================================================================


class TestVisionBatchCallSite:
    """analyze_batch drives the same staircase, on the vision router."""

    @staticmethod
    def _messages() -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": "Analyze images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the following 1 images."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA="},
                    },
                ],
            },
        ]

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.asyncio
    async def test_tier_wire_format(self, tier: str):
        model = TIER_MODELS[tier]
        router = _ScriptedRouter(model, [_good_response(model, tier, BATCH_PAYLOAD)])
        engine = _engine(router)

        result, _ = await run_structured_ladder(
            acompletion=engine.guard_acompletion(router.acompletion, "doc.pdf:images"),
            messages=self._messages(),
            response_model=BatchImageAnalysisResult,
            ladder=router_structured_ladder(router),
            call_id="image_batch:doc.pdf",
        )

        assert len(router.calls) == 1
        assert router.wire_format() == tier
        assert result.images[0].caption == "c"

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.asyncio
    async def test_malformed_answer_walks_down_the_ladder(self, tier: str):
        model = TIER_MODELS[tier]
        ladder = TIER_LADDERS[tier]
        script: list[Any] = [
            _content_response(model, "sorry") for _ in range(len(ladder) - 1)
        ]
        script.append(_good_response(model, MODE_MD_JSON, BATCH_PAYLOAD))
        router = _ScriptedRouter(model, script)
        engine = _engine(router)

        result, _ = await run_structured_ladder(
            acompletion=engine.guard_acompletion(router.acompletion, "doc.pdf:images"),
            messages=self._messages(),
            response_model=BatchImageAnalysisResult,
            ladder=router_structured_ladder(router),
            call_id="image_batch:doc.pdf",
        )

        assert result.images[0].caption == "c"
        assert len(router.calls) == len(ladder)

    @pytest.mark.asyncio
    async def test_every_rung_spends_the_request_budget(self):
        model = TOOLS_MODEL
        router = _ScriptedRouter(model, [_content_response(model, "sorry")])
        budget = RequestBudget(limit=2)
        engine = _engine(router, budget)

        with pytest.raises(LLMRequestBudgetExceededError):
            await run_structured_ladder(
                acompletion=engine.guard_acompletion(
                    router.acompletion, "doc.pdf:images"
                ),
                messages=self._messages(),
                response_model=BatchImageAnalysisResult,
                ladder=router_structured_ladder(router),
                call_id="image_batch:doc.pdf",
            )

        assert len(router.calls) == 2

    @pytest.mark.asyncio
    async def test_messages_are_not_mutated_between_rungs(self):
        """MD_JSON appends its schema to the system message in place."""
        model = SCHEMA_MODEL
        router = _ScriptedRouter(
            model,
            [
                _content_response(model, "sorry"),
                _good_response(model, MODE_MD_JSON, BATCH_PAYLOAD),
            ],
        )
        engine = _engine(router)
        messages = self._messages()

        await run_structured_ladder(
            acompletion=engine.guard_acompletion(router.acompletion, "doc.pdf:images"),
            messages=messages,
            response_model=BatchImageAnalysisResult,
            ladder=router_structured_ladder(router),
            call_id="image_batch:doc.pdf",
        )

        assert messages == self._messages()


# =============================================================================
# Call site: the provider that emulates JSON mode
# =============================================================================


class TestJsonModeCallSite:
    """The provider whose SDK can only be asked for JSON in prose."""

    def test_copilot_declares_the_md_json_tier(self):
        """Copilot's SDK has no tools and no JSON mode: floor tier."""
        from markitai.providers.copilot import CopilotProvider

        assert CopilotProvider.STRUCTURED_OUTPUT_MODE == MODE_MD_JSON
        assert model_structured_mode("copilot/gpt-4.1") == MODE_MD_JSON

    def test_claude_agent_json_schema_tier_reaches_the_sdk(self):
        """The native path that existed but nothing ever called.

        Instructor's JSON_SCHEMA mode emits exactly the ``response_format``
        shape ``_convert_response_format`` already understood, so the tier
        selection is the whole wiring.
        """
        from markitai.providers.claude_agent import ClaudeAgentProvider

        assert ClaudeAgentProvider.STRUCTURED_OUTPUT_MODE == MODE_JSON_SCHEMA
        assert model_structured_mode("claude-agent/sonnet") == MODE_JSON_SCHEMA

        schema = Frontmatter.model_json_schema()
        provider = ClaudeAgentProvider.__new__(ClaudeAgentProvider)
        output_format = provider._convert_response_format(
            {
                "type": "json_schema",
                "json_schema": {"name": "Frontmatter", "schema": schema},
            }
        )

        assert output_format == {"type": "json_schema", "schema": schema}


class TestWhatMocksCannotProve:
    """Documentation test: the honest limit of this file's coverage.

    Every response above is scripted, so these tests prove the *plumbing*:
    which wire format each tier puts on the request, how the staircase
    descends, and that repair is confined to the bottom rung. They cannot
    prove that native modes actually lower the real-world failure rate, that
    a given provider honours ``response_format``, or how often production
    traffic reaches the MD_JSON rung. Those need real traffic.
    """

    def test_the_ladder_bottom_is_always_md_json(self):
        """The one invariant that makes repair placement well-defined."""
        for tier in ALL_TIERS:
            ladder = structured_mode_ladder([TIER_MODELS[tier]])
            assert ladder[-1] is instructor.Mode.MD_JSON
