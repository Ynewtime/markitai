"""Regression tests: batch items keep their own LLM processor share and usage.

- ``.urls`` batches built one ``LLMProcessor`` per URL, each with its own
  semaphore, so ``--llm-concurrency`` bounded every URL separately.
- Usage and the per-document request budget were keyed by the file's
  basename on the batch's shared processor: ``a/report.pdf`` and
  ``b/report.pdf`` pooled one tally and one budget, and the first to finish
  cleared the other's usage.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.batch import ProcessResult
from markitai.config import LLMConfig, MarkitaiConfig, PromptsConfig
from markitai.llm import LLMProcessor
from markitai.llm.engine import (
    LLMRequestBudgetExceededError,
    RequestBudget,
    llm_usage_scope,
)
from markitai.workflow import core
from markitai.workflow.core import ConversionContext, ConversionStepResult


class TestUsageScope:
    def test_same_context_in_two_scopes_is_tracked_apart(self) -> None:
        processor = LLMProcessor(LLMConfig(), PromptsConfig())

        with llm_usage_scope("/in/a/report.pdf"):
            processor._track_usage("m", 10, 5, 0.1, context="report.pdf")
        with llm_usage_scope("/in/b/report.pdf"):
            processor._track_usage("m", 10, 5, 0.2, context="report.pdf")

        with llm_usage_scope("/in/a/report.pdf"):
            assert processor.get_context_cost("report.pdf") == pytest.approx(0.1)
            processor.clear_context_usage("report.pdf")
            assert processor.get_context_usage("report.pdf") == {}
        with llm_usage_scope("/in/b/report.pdf"):
            assert processor.get_context_cost("report.pdf") == pytest.approx(0.2)
            assert processor.get_context_usage("report.pdf")["m"]["requests"] == 1

    def test_unscoped_tracking_is_unchanged(self) -> None:
        processor = LLMProcessor(LLMConfig(), PromptsConfig())

        processor._track_usage("m", 10, 5, 0.1, context="report.pdf")

        assert processor.get_context_cost("report.pdf") == pytest.approx(0.1)

    def test_request_budget_is_per_scope(self) -> None:
        budget = RequestBudget(limit=1)

        with llm_usage_scope("a"):
            budget.spend("report.pdf")
            with pytest.raises(LLMRequestBudgetExceededError) as exc_info:
                budget.spend("report.pdf")
        # The message names the document, not the internal scoped key
        assert "\x00" not in str(exc_info.value)
        with llm_usage_scope("b"):
            budget.spend("report.pdf")
            assert budget.exceeded("report.pdf") is False


class TestNestedUsageScope:
    def test_inner_scope_refines_the_outer_one(self) -> None:
        """An MCP batch item (outer scope) converting the same file as another
        item (same inner path scope) still gets its own tally."""
        processor = LLMProcessor(LLMConfig(), PromptsConfig())

        for item, cost in (("item-1", 0.1), ("item-2", 0.2)):
            with llm_usage_scope(item), llm_usage_scope("/in/report.pdf"):
                processor._track_usage("m", 1, 1, cost, context="report.pdf")

        with llm_usage_scope("item-1"), llm_usage_scope("/in/report.pdf"):
            assert processor.get_context_cost("report.pdf") == pytest.approx(0.1)


class TestSameNamedFilesInOneBatch:
    async def test_each_file_reports_only_its_own_cost(self, tmp_path: Path) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        processor = LLMProcessor(cfg.llm, cfg.prompts)
        costs = {"a": 0.1, "b": 0.2}
        both_tracked = asyncio.Barrier(2)

        async def fake_standard_llm(ctx: ConversionContext) -> ConversionStepResult:
            # Mirrors SingleFileWorkflow: usage is keyed by the basename and
            # cleared in a finally once the file is done
            source = ctx.input_path.name
            assert ctx.shared_processor is processor
            processor._track_usage("m", 1, 1, costs[ctx.input_path.parent.name], source)
            await both_tracked.wait()
            ctx.llm_cost = processor.get_context_cost(source)
            processor.clear_context_usage(source)
            return ConversionStepResult(success=True)

        contexts = []
        for folder in ("a", "b"):
            (tmp_path / folder).mkdir()
            path = tmp_path / folder / "report.pdf"
            path.write_bytes(b"%PDF")
            ctx = ConversionContext(
                input_path=path,
                output_dir=tmp_path / "out",
                config=cfg,
                shared_processor=processor,
            )
            ctx.conversion_result = MagicMock(metadata={})
            contexts.append(ctx)

        with patch.object(core, "process_with_standard_llm", fake_standard_llm):
            await asyncio.gather(*(core.run_llm_enhancement(c) for c in contexts))

        assert [c.llm_cost for c in contexts] == [
            pytest.approx(0.1),
            pytest.approx(0.2),
        ]


class TestUrlListBatchSharesOneProcessor:
    async def test_every_url_uses_the_batch_processor(self, tmp_path: Path) -> None:
        class Entry:
            def __init__(self, url: str) -> None:
                self.url = url
                self.output_name = None

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.cache.enabled = False
        shared = MagicMock(name="shared processor")
        factory_kwargs: dict[str, Any] = {}

        def fake_create_url_processor(**kwargs: Any) -> Any:
            factory_kwargs.update(kwargs)
            return AsyncMock(return_value=(ProcessResult(success=False, error="x"), {}))

        from markitai.cli.processors import url as url_module

        with (
            patch.object(url_module, "create_llm_processor", return_value=shared),
            patch(
                "markitai.cli.processors.batch.create_url_processor",
                fake_create_url_processor,
            ),
            # The fake URLs fail, so the batch exits with the batch failure code
            pytest.raises(SystemExit),
        ):
            await url_module.process_url_batch(
                url_entries=[
                    Entry("https://example.com/1"),
                    Entry("https://example.com/2"),
                ],
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
                quiet=True,
            )

        assert factory_kwargs["shared_processor"] is shared
