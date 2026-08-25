"""Tests for the per-document LLM request circuit breaker.

The budget bounds the retry multiplication of one document (transport
retries x instructor validation retries x business fallback chains): every
request attempt spends from the document context's budget, and a tripped
context makes no further API calls — the existing per-stage fallbacks keep
the unenhanced output.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.exceptions import RateLimitError

from markitai.config import LiteLLMParams, LLMConfig, ModelConfig, RouterSettings
from markitai.llm.engine import (
    LLMCall,
    LLMEngine,
    LLMRequestBudgetExceededError,
    RequestBudget,
)
from markitai.llm.processor import REQUEST_BUDGET_EXCEEDED_MARKER, LLMProcessor
from markitai.llm.types import Frontmatter


class TestRequestBudget:
    """Unit tests for the RequestBudget breaker itself."""

    def test_spend_under_limit(self):
        """Spending within the limit passes."""
        budget = RequestBudget(limit=3)
        for _ in range(3):
            budget.spend("doc.pdf")
        assert budget.exceeded("doc.pdf") is False

    def test_spend_over_limit_raises(self):
        """The attempt beyond the limit is refused (and not counted)."""
        budget = RequestBudget(limit=2)
        budget.spend("doc.pdf")
        budget.spend("doc.pdf")
        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("doc.pdf")
        assert budget.exceeded("doc.pdf") is True

    def test_contexts_are_independent(self):
        """One document's trip does not affect another."""
        budget = RequestBudget(limit=1)
        budget.spend("a.pdf")
        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("a.pdf")
        budget.spend("b.pdf")  # must not raise
        assert budget.exceeded("b.pdf") is False

    def test_zero_limit_disables(self):
        """limit=0 means unlimited."""
        budget = RequestBudget(limit=0)
        for _ in range(100):
            budget.spend("doc.pdf")
        assert budget.exceeded("doc.pdf") is False

    def test_empty_context_untracked(self):
        """Calls without a context are never budgeted."""
        budget = RequestBudget(limit=1)
        for _ in range(10):
            budget.spend("")

    def test_on_exceeded_fires_once(self):
        """The trip callback fires exactly once per context."""
        tripped: list[str] = []
        budget = RequestBudget(limit=1, on_exceeded=tripped.append)
        budget.spend("doc.pdf")
        for _ in range(3):
            with pytest.raises(LLMRequestBudgetExceededError):
                budget.spend("doc.pdf")
        assert tripped == ["doc.pdf"]

    def test_clear_resets_context(self):
        """Clearing re-arms the budget for the next document."""
        budget = RequestBudget(limit=1)
        budget.spend("doc.pdf")
        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("doc.pdf")
        budget.clear("doc.pdf")
        budget.spend("doc.pdf")  # must not raise
        assert budget.exceeded("doc.pdf") is False


def _failing_router(exc: Exception) -> MagicMock:
    """Router double whose every acompletion call raises ``exc``."""
    router = MagicMock()
    router.model_list = [
        {"litellm_params": {"model": "openai/gpt-4o-mini", "weight": 1}}
    ]
    router.acompletion = AsyncMock(side_effect=exc)
    return router


def _engine(router: MagicMock, budget: RequestBudget, max_retries: int) -> LLMEngine:
    return LLMEngine(
        router=router,
        semaphore=asyncio.Semaphore(2),
        memory_cache=MagicMock(get=MagicMock(return_value=None)),
        persistent_cache=MagicMock(get=MagicMock(return_value=None)),
        track_usage=MagicMock(),
        calculate_max_tokens=MagicMock(return_value=64),
        get_primary_model=MagicMock(return_value=None),
        max_retries=max_retries,
        request_budget=budget,
    )


class TestEngineBudget:
    """Worst-case request bounds through the engine's call paths."""

    @pytest.mark.asyncio
    async def test_text_retries_bounded_by_budget(self, monkeypatch):
        """Even with huge max_retries, at most `limit` requests are issued."""
        import markitai.llm.engine as engine_mod

        monkeypatch.setattr(engine_mod, "DEFAULT_RETRY_BASE_DELAY", 0.0)
        rate_limit = RateLimitError(
            message="429 rate limit", llm_provider="openai", model="gpt-4o-mini"
        )
        router = _failing_router(rate_limit)
        budget = RequestBudget(limit=4)
        engine = _engine(router, budget, max_retries=10)

        with pytest.raises(LLMRequestBudgetExceededError):
            await engine.complete_text(
                model="default",
                messages=[{"role": "user", "content": "Hi"}],
                call_id="budget-test",
                context="doc.pdf",
            )

        assert router.acompletion.call_count == 4

    @pytest.mark.asyncio
    async def test_structured_attempts_bounded_by_budget(self, monkeypatch):
        """The instructor adapter's attempts also spend the budget."""
        import markitai.llm.engine as engine_mod

        monkeypatch.setattr(engine_mod, "DEFAULT_RETRY_BASE_DELAY", 0.0)
        rate_limit = RateLimitError(
            message="429 rate limit", llm_provider="openai", model="gpt-4o-mini"
        )
        router = _failing_router(rate_limit)
        budget = RequestBudget(limit=2)
        engine = _engine(router, budget, max_retries=10)

        call = LLMCall(
            purpose="test",
            messages=[{"role": "user", "content": "Hi"}],
            response_model=Frontmatter,
            context="doc.pdf",
        )
        # Depending on how instructor surfaces the refusal, either the
        # budget error or its instructor wrapper propagates
        from instructor.core.exceptions import InstructorRetryException

        with pytest.raises((LLMRequestBudgetExceededError, InstructorRetryException)):
            await engine.complete_structured(call)

        assert router.acompletion.call_count == 2

    @pytest.mark.asyncio
    async def test_tripped_context_makes_zero_requests(self):
        """After the trip, further calls fail fast without touching the router."""
        router = _failing_router(RuntimeError("boom"))
        budget = RequestBudget(limit=1)
        engine = _engine(router, budget, max_retries=0)

        with pytest.raises(RuntimeError):
            await engine.complete_text(
                model="default",
                messages=[{"role": "user", "content": "Hi"}],
                call_id="t1",
                context="doc.pdf",
            )
        assert router.acompletion.call_count == 1

        with pytest.raises(LLMRequestBudgetExceededError):
            await engine.complete_text(
                model="default",
                messages=[{"role": "user", "content": "Hi"}],
                call_id="t2",
                context="doc.pdf",
            )
        assert router.acompletion.call_count == 1  # no new request

    @pytest.mark.asyncio
    async def test_guard_acompletion_spends_budget(self):
        """Direct-router call sites (vision batch/json mode) are budgeted too."""
        inner = AsyncMock(return_value="ok")
        budget = RequestBudget(limit=1)
        engine = _engine(MagicMock(), budget, max_retries=0)

        guarded = engine.guard_acompletion(inner, "doc.pdf:images")
        assert await guarded(model="default", messages=[]) == "ok"
        with pytest.raises(LLMRequestBudgetExceededError):
            await guarded(model="default", messages=[])
        assert inner.await_count == 1


class TestProcessorBudgetEndToEnd:
    """A tripped document degrades to unenhanced output, with a report mark."""

    @staticmethod
    def _processor(limit: int) -> tuple[LLMProcessor, MagicMock]:
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini", api_key="test-key"
                    ),
                ),
            ],
            router_settings=RouterSettings(num_retries=0),
            max_requests_per_document=limit,
        )
        processor = LLMProcessor(config)
        router = _failing_router(RuntimeError("boom"))
        processor._router = router
        processor._vision_router = router
        return processor, router

    @pytest.mark.asyncio
    async def test_process_document_falls_back_to_unenhanced(self):
        """Budget trip mid-document keeps the original markdown."""
        processor, router = self._processor(limit=1)
        markdown = "# Title\n\nOriginal body."

        cleaned, frontmatter = await processor.process_document(markdown, "doc.pdf")

        # The structured call consumed the budget; the cleaner fallback was
        # refused without issuing a request, so the original text survives.
        assert cleaned == markdown
        assert frontmatter  # fallback frontmatter still generated
        assert router.acompletion.call_count == 1

    @pytest.mark.asyncio
    async def test_trip_is_marked_in_context_usage(self):
        """The usage report carries an all-zero marker entry for the trip."""
        processor, _router = self._processor(limit=1)

        await processor.process_document("# T\n\nBody.", "doc.pdf")

        usage = processor.get_context_usage("doc.pdf")
        assert REQUEST_BUDGET_EXCEEDED_MARKER in usage
        marker = usage[REQUEST_BUDGET_EXCEEDED_MARKER]
        assert marker["requests"] == 0
        assert marker["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_clear_context_usage_rearms_budget(self):
        """The next document reusing the context key gets a fresh budget."""
        processor, router = self._processor(limit=1)

        await processor.process_document("# T\n\nBody.", "doc.pdf")
        assert processor._request_budget.exceeded("doc.pdf") is True

        processor.clear_context_usage("doc.pdf")
        assert processor._request_budget.exceeded("doc.pdf") is False

        await processor.process_document("# T\n\nBody.", "doc.pdf")
        assert router.acompletion.call_count == 2  # one fresh request allowed
