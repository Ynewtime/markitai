"""A failed LLM item leaves nothing stale behind.

Regressions:
- ``_write_base_md_fallback`` skipped the fallback whenever a base ``.md``
  existed on disk. In LLM mode (no ``--keep-base``) this run writes no base,
  so under ``on_conflict=overwrite`` the previous run's ``X.md`` survived
  next to the failed item, and its ``X.llm.md`` too.
- ``SingleFileWorkflow`` cleared the per-file usage context only on
  success, so a same-named file later in the batch (the context is the
  basename) inherited the failed file's usage and tripped request budget.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from markitai.config import MarkitaiConfig
from markitai.converter.base import ConvertResult
from markitai.llm.engine import LLMEnhancementDegradedError
from markitai.workflow.core import (
    ConversionContext,
    _write_base_md_fallback,
    convert_document_core,
)
from markitai.workflow.single import SingleFileWorkflow

MAX_SIZE = 100 * 1024 * 1024


def _degraded() -> LLMEnhancementDegradedError:
    return LLMEnhancementDegradedError(
        "AuthenticationError: invalid api key",
        cleaned_markdown="unenhanced",
        frontmatter="llm_enhanced: false",
    )


def _failing_processor() -> MagicMock:
    processor = MagicMock()
    processor.process_document = AsyncMock(side_effect=_degraded())
    return processor


def _config(on_conflict: str, *, keep_base: bool = False) -> MarkitaiConfig:
    config = MarkitaiConfig()
    config.llm.enabled = True
    config.llm.on_failure = "fail"  # the failed item's leftovers under test
    config.llm.keep_base = keep_base
    config.output.on_conflict = on_conflict  # type: ignore[assignment]
    return config


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "doc.txt"
    path.write_text("# Fresh\n\nThis run's content.", encoding="utf-8")
    return path


class TestFallbackRewritesStaleBase:
    async def test_overwrite_replaces_old_base_and_drops_old_llm_md(
        self, source_file: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        old_base = output_dir / "doc.txt.md"
        old_llm = output_dir / "doc.txt.llm.md"
        old_base.write_text("OLD BASE FROM LAST RUN", encoding="utf-8")
        old_llm.write_text("OLD LLM FROM LAST RUN", encoding="utf-8")

        ctx = ConversionContext(
            input_path=source_file,
            output_dir=output_dir,
            config=_config("overwrite"),
            shared_processor=_failing_processor(),
        )
        result = await convert_document_core(ctx, MAX_SIZE)

        assert result.success is False
        assert ctx.output_file == old_base
        base = old_base.read_text(encoding="utf-8")
        assert "OLD BASE" not in base
        assert "This run's content." in base
        assert not old_llm.exists()

    async def test_rename_keeps_the_other_outputs_llm_md(
        self, source_file: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        other_base = output_dir / "doc.txt.md"
        other_llm = output_dir / "doc.txt.llm.md"
        other_base.write_text("OTHER BASE", encoding="utf-8")
        other_llm.write_text("OTHER LLM", encoding="utf-8")

        ctx = ConversionContext(
            input_path=source_file,
            output_dir=output_dir,
            config=_config("rename"),
            shared_processor=_failing_processor(),
        )
        result = await convert_document_core(ctx, MAX_SIZE)

        assert result.success is False
        renamed = output_dir / "doc.txt.v2.md"
        assert ctx.output_file == renamed
        assert "This run's content." in renamed.read_text(encoding="utf-8")
        assert other_base.read_text(encoding="utf-8") == "OTHER BASE"
        assert other_llm.read_text(encoding="utf-8") == "OTHER LLM"

    def test_rename_never_deletes_a_same_named_llm_md(self, tmp_path: Path) -> None:
        ctx = ConversionContext(
            input_path=tmp_path / "doc.txt",
            output_dir=tmp_path,
            config=_config("rename"),
        )
        ctx.output_file = tmp_path / "doc.txt.v2.md"
        ctx.conversion_result = ConvertResult(markdown="body", images=[], metadata={})
        sibling = tmp_path / "doc.txt.v2.llm.md"
        sibling.write_text("someone else's", encoding="utf-8")

        _write_base_md_fallback(ctx)

        assert sibling.read_text(encoding="utf-8") == "someone else's"
        assert ctx.output_file.exists()

    async def test_keep_base_written_this_run_is_not_rewritten(
        self, source_file: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        ctx = ConversionContext(
            input_path=source_file,
            output_dir=output_dir,
            config=_config("overwrite", keep_base=True),
            shared_processor=_failing_processor(),
        )
        result = await convert_document_core(ctx, MAX_SIZE)

        assert result.success is False
        assert ctx.base_written is True
        assert ctx.output_file is not None
        assert "This run's content." in ctx.output_file.read_text(encoding="utf-8")


class TestFailedDocumentClearsItsContext:
    """The usage context (and request budget) is dropped on failure too."""

    async def test_process_document_with_llm(self, tmp_path: Path) -> None:
        processor = MagicMock()
        processor.process_document = AsyncMock(side_effect=_degraded())
        workflow = SingleFileWorkflow(MarkitaiConfig(), processor=processor)

        with pytest.raises(LLMEnhancementDegradedError):
            await workflow.process_document_with_llm(
                "# Doc", "report.pdf", tmp_path / "report.pdf.md"
            )

        processor.clear_context_usage.assert_called_once_with("report.pdf")

    async def test_process_document_pure(self, tmp_path: Path) -> None:
        processor = MagicMock()
        processor.clean_document_pure = AsyncMock(side_effect=_degraded())
        workflow = SingleFileWorkflow(MarkitaiConfig(), processor=processor)

        with pytest.raises(LLMEnhancementDegradedError):
            await workflow.process_document_pure(
                "# Doc", "report.pdf", tmp_path / "report.pdf.md"
            )

        processor.clear_context_usage.assert_called_once_with("report.pdf")

    async def test_success_still_reports_then_clears(self, tmp_path: Path) -> None:
        processor = MagicMock()
        processor.clean_document_pure = AsyncMock(return_value="# Cleaned")
        processor.get_context_cost.return_value = 0.5
        processor.get_context_usage.return_value = {"m": {"requests": 1}}
        workflow = SingleFileWorkflow(MarkitaiConfig(), processor=processor)

        _, cost, usage = await workflow.process_document_pure(
            "# Doc", "report.pdf", tmp_path / "report.pdf.md"
        )

        assert cost == 0.5
        assert usage == {"m": {"requests": 1}}
        processor.clear_context_usage.assert_called_once_with("report.pdf")

    async def test_same_named_file_starts_with_a_fresh_budget(
        self, tmp_path: Path
    ) -> None:
        """End to end on a real processor: the tripped budget is not inherited."""
        from markitai.llm import LLMProcessor

        config = MarkitaiConfig()
        config.llm.max_requests_per_document = 1
        processor = LLMProcessor(config.llm, config.prompts, no_cache=True)
        budget = processor._request_budget
        # The first same-named file exhausts its budget, then fails
        budget.spend("report.pdf")
        with pytest.raises(Exception):  # noqa: B017 - any refusal
            budget.spend("report.pdf")
        assert budget.exceeded("report.pdf")

        processor.documents.process_document = AsyncMock(side_effect=_degraded())  # type: ignore[method-assign]
        workflow = SingleFileWorkflow(config, processor=processor)
        with pytest.raises(LLMEnhancementDegradedError):
            await workflow.process_document_with_llm(
                "# Doc", "report.pdf", tmp_path / "report.pdf.md"
            )

        assert not budget.exceeded("report.pdf")
        assert processor.get_context_usage("report.pdf") == {}
