"""Tests for the unified LLM call engine (markitai.llm.engine).

These tests wire a real instructor client against a fake router whose
``acompletion`` returns hand-built litellm ``ModelResponse`` objects (or
raises litellm exceptions). Instructor internals are NOT mocked, so the
tests exercise the actual adapter wiring: transport retries happen below
instructor, validation retries happen inside instructor, and both layers
compose.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from litellm.exceptions import APIConnectionError, RateLimitError
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

import markitai.llm.engine as engine_module
from markitai.llm.engine import LLMCall, LLMEngine
from markitai.providers.errors import AuthenticationError


class _Doc(BaseModel):
    text: str


def make_model_response(
    content: str,
    *,
    model: str = "openai/gpt-test",
    prompt_tokens: int = 200,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    cost: float = 0.000123,
) -> ModelResponse:
    """Build a real litellm ModelResponse with deterministic usage and cost."""
    response = ModelResponse(
        model=model,
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
    # get_response_cost() reads local-provider cost from _hidden_params,
    # which keeps cost deterministic for a fake model id.
    response._hidden_params = {"total_cost_usd": cost}
    return response


class FakeRouter:
    """Router double: acompletion pops queued results (response or exception).

    The last queued item is returned repeatedly so instructor-level retries
    never underflow the queue; call counts are asserted via ``calls``.
    """

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        if isinstance(result, BaseException):
            raise result
        return result


class FakeMemoryCache:
    """ContentCache double recording get/set traffic."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], Any] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.set_calls: list[tuple[str, str, Any]] = []

    def get(self, key: str, content: str) -> Any | None:
        self.get_calls.append((key, content))
        return self.store.get((key, content))

    def set(self, key: str, content: str, value: Any) -> None:
        self.set_calls.append((key, content, value))
        self.store[(key, content)] = value


class FakePersistentCache:
    """PersistentCache double recording get/set traffic."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], Any] = {}
        self.get_calls: list[tuple[str, str, str]] = []
        self.set_calls: list[tuple[str, str, Any, str]] = []

    def get(self, key: str, content: str, context: str = "") -> Any | None:
        self.get_calls.append((key, content, context))
        return self.store.get((key, content))

    def set(self, key: str, content: str, value: Any, model: str = "") -> None:
        self.set_calls.append((key, content, value, model))
        self.store[(key, content)] = value


class Harness:
    """LLMEngine wired to fakes, with recorders for injected callbacks."""

    def __init__(
        self,
        router: FakeRouter,
        *,
        calculated_max_tokens: int = 512,
    ) -> None:
        self.router = router
        self.memory = FakeMemoryCache()
        self.persistent = FakePersistentCache()
        self.track_calls: list[tuple[str, int, int, float, str]] = []
        self.max_tokens_calls: list[dict[str, Any]] = []
        self.calculated_max_tokens = calculated_max_tokens

        def track_usage(
            model: str,
            input_tokens: int,
            output_tokens: int,
            cost: float,
            context: str,
        ) -> None:
            self.track_calls.append((model, input_tokens, output_tokens, cost, context))

        def calculate_max_tokens(
            messages: list[dict[str, Any]],
            model_id: str | None,
            *,
            router: Any = None,
        ) -> int | None:
            self.max_tokens_calls.append(
                {"messages": messages, "model_id": model_id, "router": router}
            )
            return self.calculated_max_tokens

        self.engine = LLMEngine(
            router=router,
            semaphore=asyncio.Semaphore(2),
            memory_cache=self.memory,
            persistent_cache=self.persistent,
            track_usage=track_usage,
            calculate_max_tokens=calculate_max_tokens,
            get_primary_model=lambda _router: "openai/gpt-test",
        )


CACHE_KEY = "doc_test:test.md"
CACHE_CONTENT = "hello content"


def make_call(**overrides: Any) -> LLMCall:
    defaults: dict[str, Any] = {
        "purpose": "doc_test",
        "messages": [{"role": "user", "content": "hello"}],
        "response_model": _Doc,
        "context": "test.md",
        "cache_key": CACHE_KEY,
        "cache_content": CACHE_CONTENT,
    }
    defaults.update(overrides)
    return LLMCall(**defaults)


@pytest.fixture
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out transport backoff so retry tests run instantly."""
    monkeypatch.setattr(engine_module, "DEFAULT_RETRY_BASE_DELAY", 0.0)


class TestCacheLayers:
    async def test_memory_cache_hit_skips_llm(self) -> None:
        harness = Harness(FakeRouter([]))
        harness.memory.store[(CACHE_KEY, CACHE_CONTENT)] = {"text": "cached"}

        result, raw = await harness.engine.complete_structured(make_call())

        assert raw is None
        assert result.text == "cached"
        assert harness.router.calls == []
        # Memory hit must short-circuit before the persistent layer
        assert harness.persistent.get_calls == []

    async def test_persistent_hit_backfills_memory_without_llm(self) -> None:
        harness = Harness(FakeRouter([]))
        harness.persistent.store[(CACHE_KEY, CACHE_CONTENT)] = {"text": "persisted"}

        result, raw = await harness.engine.complete_structured(make_call())

        assert raw is None
        assert result.text == "persisted"
        assert harness.router.calls == []
        assert harness.persistent.get_calls == [(CACHE_KEY, CACHE_CONTENT, "test.md")]
        # Backfilled into memory for subsequent lookups
        assert harness.memory.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "persisted"})
        ]

    async def test_miss_calls_llm_once_and_writes_both_layers(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "fresh"}')]))

        result, raw = await harness.engine.complete_structured(make_call())

        assert result.text == "fresh"
        assert raw is not None
        assert len(harness.router.calls) == 1
        # Default serialize is result.model_dump()
        assert harness.memory.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "fresh"})
        ]
        assert harness.persistent.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "fresh"}, "default")
        ]

    async def test_cache_key_none_skips_cache_entirely(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "fresh"}')]))

        result, raw = await harness.engine.complete_structured(
            make_call(cache_key=None)
        )

        assert result.text == "fresh"
        assert raw is not None
        assert harness.memory.get_calls == []
        assert harness.memory.set_calls == []
        assert harness.persistent.get_calls == []
        assert harness.persistent.set_calls == []


class TestTransportRetry:
    @pytest.mark.usefixtures("no_retry_delay")
    async def test_rate_limit_retried_then_succeeds(self) -> None:
        harness = Harness(
            FakeRouter(
                [
                    RateLimitError(
                        message="rate limit hit, please slow down",
                        llm_provider="openai",
                        model="gpt-test",
                    ),
                    make_model_response('{"text": "ok"}'),
                ]
            )
        )

        result, raw = await harness.engine.complete_structured(make_call())

        assert result.text == "ok"
        assert raw is not None
        assert len(harness.router.calls) == 2

    async def test_quota_error_short_circuits_without_retry(self) -> None:
        harness = Harness(
            FakeRouter(
                [
                    APIConnectionError(
                        message="insufficient_quota: credits exhausted",
                        llm_provider="openai",
                        model="gpt-test",
                    )
                ]
            )
        )

        with pytest.raises(Exception, match="insufficient_quota"):
            await harness.engine.complete_structured(make_call())

        assert len(harness.router.calls) == 1
        assert harness.memory.set_calls == []
        assert harness.persistent.set_calls == []

    @pytest.mark.usefixtures("no_retry_delay")
    async def test_empty_response_retried(self) -> None:
        # The first response carries VALID JSON so instructor alone would
        # accept it: only the adapter's completion_tokens==0 check (with
        # prompt_tokens>100) can trigger the retry that yields "ok".
        harness = Harness(
            FakeRouter(
                [
                    make_model_response(
                        '{"text": "should-be-retried"}',
                        completion_tokens=0,
                        prompt_tokens=200,
                    ),
                    make_model_response('{"text": "ok"}'),
                ]
            )
        )

        result, _raw = await harness.engine.complete_structured(make_call())

        assert result.text == "ok"
        assert len(harness.router.calls) == 2


class TestInstructorIntegration:
    async def test_validation_retry_goes_through_transport_layer(self) -> None:
        """Instructor-level retry (bad schema -> good) still works, and each
        attempt passes through the transport adapter."""
        harness = Harness(
            FakeRouter(
                [
                    make_model_response('{"wrong_field": 1}'),
                    make_model_response('{"text": "valid"}'),
                ]
            )
        )

        result, _raw = await harness.engine.complete_structured(make_call())

        assert result.text == "valid"
        assert len(harness.router.calls) == 2

    async def test_repair_path_recovers_unparseable_json(self) -> None:
        """JSON instructor cannot parse (trailing comma) exhausts instructor
        retries, then try_repair_instructor_response recovers the payload."""
        harness = Harness(FakeRouter([make_model_response('{"text": "hi",}')]))

        result, raw = await harness.engine.complete_structured(make_call())

        assert result.text == "hi"
        assert raw is not None
        # Repaired result is cached like a normal success
        assert harness.memory.set_calls == [(CACHE_KEY, CACHE_CONTENT, {"text": "hi"})]

    async def test_fatal_provider_error_raised_directly(self) -> None:
        fatal = AuthenticationError(
            "authentication failed: invalid x-api-key",
            provider="copilot",
        )
        harness = Harness(FakeRouter([fatal]))

        with pytest.raises(AuthenticationError) as excinfo:
            await harness.engine.complete_structured(make_call())

        # The original wrapped ProviderError is unwrapped and raised as-is
        assert excinfo.value is fatal
        assert len(harness.router.calls) == 1
        assert harness.memory.set_calls == []
        assert harness.persistent.set_calls == []

    async def test_length_finish_reason_raises_value_error(self) -> None:
        harness = Harness(
            FakeRouter([make_model_response('{"text": "hi"}', finish_reason="length")])
        )

        with pytest.raises(ValueError, match="max_tokens limit"):
            await harness.engine.complete_structured(make_call())

        assert harness.memory.set_calls == []
        assert harness.persistent.set_calls == []


class TestUsageAndHooks:
    async def test_usage_tracked_once_from_raw_response(self) -> None:
        harness = Harness(
            FakeRouter(
                [
                    make_model_response(
                        '{"text": "ok"}',
                        model="openai/gpt-actual",
                        prompt_tokens=200,
                        completion_tokens=20,
                        cost=0.000123,
                    )
                ]
            )
        )

        await harness.engine.complete_structured(make_call())

        assert harness.track_calls == [
            ("openai/gpt-actual", 200, 20, 0.000123, "test.md")
        ]

    async def test_validate_hook_result_is_returned_and_cached(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "raw"}')]))

        def fix(result: _Doc) -> _Doc:
            return _Doc(text=result.text + "-fixed")

        result, _raw = await harness.engine.complete_structured(make_call(validate=fix))

        assert result.text == "raw-fixed"
        assert harness.memory.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "raw-fixed"})
        ]
        assert harness.persistent.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "raw-fixed"}, "default")
        ]

    async def test_validate_hook_raise_skips_all_cache_writes(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "raw"}')]))

        def reject(result: _Doc) -> _Doc:
            raise RuntimeError("validation rejected")

        with pytest.raises(RuntimeError, match="validation rejected"):
            await harness.engine.complete_structured(make_call(validate=reject))

        assert harness.memory.set_calls == []
        assert harness.persistent.set_calls == []

    async def test_cache_if_false_skips_cache_writes_but_returns_result(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "degenerate"}')]))
        seen: list[_Doc] = []

        def veto(result: _Doc) -> bool:
            seen.append(result)
            return False

        result, raw = await harness.engine.complete_structured(make_call(cache_if=veto))

        # Result is returned normally despite the veto
        assert result.text == "degenerate"
        assert raw is not None
        # cache_if saw the (post-validate) result
        assert seen == [result]
        # Neither cache layer was written
        assert harness.memory.set_calls == []
        assert harness.persistent.set_calls == []

    async def test_cache_if_true_writes_both_cache_layers(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "clean"}')]))

        result, _raw = await harness.engine.complete_structured(
            make_call(cache_if=lambda _result: True)
        )

        assert result.text == "clean"
        assert harness.memory.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "clean"})
        ]
        assert harness.persistent.set_calls == [
            (CACHE_KEY, CACHE_CONTENT, {"text": "clean"}, "default")
        ]


class TestMaxTokens:
    async def test_dynamic_max_tokens_callback_used(self) -> None:
        harness = Harness(
            FakeRouter([make_model_response('{"text": "ok"}')]),
            calculated_max_tokens=777,
        )

        await harness.engine.complete_structured(make_call())

        assert len(harness.max_tokens_calls) == 1
        assert harness.max_tokens_calls[0]["model_id"] == "openai/gpt-test"
        assert harness.max_tokens_calls[0]["router"] is harness.router
        assert harness.router.calls[0]["max_tokens"] == 777

    async def test_explicit_max_tokens_skips_callback(self) -> None:
        harness = Harness(FakeRouter([make_model_response('{"text": "ok"}')]))

        await harness.engine.complete_structured(make_call(max_tokens=555))

        assert harness.max_tokens_calls == []
        assert harness.router.calls[0]["max_tokens"] == 555
