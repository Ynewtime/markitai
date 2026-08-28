"""A single document cannot run away with the bill.

A many-page PDF with vision enhancement is the expensive case: pages times
image tokens times a retry allowance. The request breaker already bounded
the *count*; these bound the *money*, from the two directions available:

* **Before anything is sent** — page count is knowable in advance, so an
  oversized document costs nothing at all and drops to text-only
  enhancement.
* **After each answer** — a call's price is not knowable before making it,
  so the cost cap bounds what a document goes on to spend, never the single
  call that crosses the line.

Both default to off. A limit guessed for someone else's workload turns a
large legitimate run into a silent downgrade, which is worse than no limit;
the user opts in with a number they chose.
"""

from __future__ import annotations

import pytest

from markitai.llm.engine import LLMRequestBudgetExceededError, RequestBudget


class TestCostBudget:
    def test_spending_under_the_cap_is_never_refused(self) -> None:
        budget = RequestBudget(limit=0, cost_limit=1.0)
        for _ in range(5):
            budget.spend("a.pdf")
            budget.charge("a.pdf", 0.1)
        budget.spend("a.pdf")  # still fine at $0.50

    def test_the_call_that_crosses_the_line_keeps_its_answer(self) -> None:
        """charge() never raises: that call already happened and was paid for."""
        budget = RequestBudget(limit=0, cost_limit=0.10)
        budget.spend("a.pdf")
        budget.charge("a.pdf", 5.00)  # must not raise

    def test_the_next_request_is_refused(self) -> None:
        budget = RequestBudget(limit=0, cost_limit=0.10)
        budget.spend("a.pdf")
        budget.charge("a.pdf", 5.00)

        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("a.pdf")

    def test_one_document_going_over_does_not_stop_the_others(self) -> None:
        """A batch must not lose every later document to one runaway file."""
        budget = RequestBudget(limit=0, cost_limit=0.10)
        budget.charge("expensive.pdf", 5.00)

        budget.spend("cheap.pdf")  # unaffected

        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("expensive.pdf")

    def test_exactly_at_the_cap_is_still_allowed(self) -> None:
        budget = RequestBudget(limit=0, cost_limit=0.10)
        budget.charge("a.pdf", 0.10)
        budget.spend("a.pdf")

    def test_a_disabled_cap_never_trips(self) -> None:
        budget = RequestBudget(limit=0, cost_limit=0.0)
        budget.charge("a.pdf", 10_000.0)
        budget.spend("a.pdf")
        assert not budget.exceeded("a.pdf")

    def test_the_trip_is_reported_once(self) -> None:
        """The report marks the document; repeating it per refused call
        would bury the run's real output."""
        tripped: list[str] = []
        budget = RequestBudget(limit=0, cost_limit=0.10, on_exceeded=tripped.append)

        budget.charge("a.pdf", 5.00)
        for _ in range(3):
            with pytest.raises(LLMRequestBudgetExceededError):
                budget.spend("a.pdf")

        assert tripped == ["a.pdf"]

    def test_clearing_a_context_forgets_its_spend(self) -> None:
        """Contexts are reused between documents; carrying the total over
        would trip the breaker on an innocent file."""
        budget = RequestBudget(limit=0, cost_limit=0.10)
        budget.charge("a.pdf", 5.00)
        budget.clear("a.pdf")

        budget.spend("a.pdf")

    def test_an_untracked_call_is_ignored(self) -> None:
        budget = RequestBudget(limit=0, cost_limit=0.10)
        budget.charge("", 5.00)
        budget.spend("")


class TestBothBreakersTogether:
    def test_the_request_cap_still_works_with_a_cost_cap_set(self) -> None:
        budget = RequestBudget(limit=2, cost_limit=100.0)
        budget.spend("a.pdf")
        budget.spend("a.pdf")

        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("a.pdf")

    def test_a_cost_trip_refuses_even_with_requests_to_spare(self) -> None:
        budget = RequestBudget(limit=100, cost_limit=0.10)
        budget.spend("a.pdf")
        budget.charge("a.pdf", 5.00)

        with pytest.raises(LLMRequestBudgetExceededError):
            budget.spend("a.pdf")


class TestProcessorWiring:
    def test_tracked_cost_reaches_the_breaker(self) -> None:
        """_track_usage is the one place every answer's price passes; if the
        charge is not made there, the cap is decorative."""
        from markitai.config import LLMConfig
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(
            LLMConfig(model_list=[], max_cost_per_document_usd=0.10)
        )
        processor._track_usage("m", 10, 10, 5.00, context="a.pdf")

        assert processor._request_budget.exceeded("a.pdf")

    def test_the_default_config_leaves_it_disabled(self) -> None:
        from markitai.config import LLMConfig
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(LLMConfig(model_list=[]))
        processor._track_usage("m", 10, 10, 10_000.0, context="a.pdf")

        assert not processor._request_budget.exceeded("a.pdf")


class TestVisionPageCap:
    """The pre-flight half: nothing is sent for an oversized document."""

    async def _enhance(self, cap: int, pages: int) -> object:
        from unittest.mock import AsyncMock, MagicMock

        from markitai.config import LLMConfig
        from markitai.llm.document import DocumentEnhancer

        enhancer = DocumentEnhancer.__new__(DocumentEnhancer)
        enhancer._config = LLMConfig(model_list=[], max_vision_pages_per_document=cap)
        enhancer.process_document = AsyncMock(return_value=("text only", "---"))
        enhancer._enhance_with_frontmatter = AsyncMock(
            return_value=("with vision", "---")
        )
        return await DocumentEnhancer.enhance_document_complete(
            enhancer,
            "body",
            [MagicMock(name=f"page{i}") for i in range(pages)],
            source="a.pdf",
        )

    @pytest.mark.asyncio
    async def test_an_oversized_document_never_sends_its_pages(self) -> None:
        assert await self._enhance(cap=3, pages=40) == ("text only", "---")

    @pytest.mark.asyncio
    async def test_a_document_within_the_cap_still_gets_vision(self) -> None:
        assert await self._enhance(cap=10, pages=3) == ("with vision", "---")

    @pytest.mark.asyncio
    async def test_the_cap_is_off_by_default(self) -> None:
        # Within max_pages_per_batch, so this is the single-call branch; a
        # longer document takes the multi-round path, which is not what the
        # cap decides.
        assert await self._enhance(cap=0, pages=5) == ("with vision", "---")

    @pytest.mark.asyncio
    async def test_a_document_over_the_cap_is_stopped_before_the_split(self) -> None:
        """The cap must bite on the long documents too — those are the
        expensive ones it exists for."""
        assert await self._enhance(cap=10, pages=400) == ("text only", "---")
