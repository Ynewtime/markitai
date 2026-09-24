"""The shared core-result -> ProcessResult mapping serve's file items use.

``workflow.results.document_process_result`` is the one place that decides
skip semantics and which file is an item's output; serve's
``process_file_item`` must go through it rather than a private copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from markitai.config import MarkitaiConfig
from markitai.workflow.core import ConversionContext, ConversionStepResult
from markitai.workflow.results import document_process_result


def _ctx(tmp_path: Path, *, llm: bool = False) -> ConversionContext:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = llm
    return ConversionContext(
        input_path=tmp_path / "in" / "doc.pdf", output_dir=tmp_path, config=cfg
    )


class TestDocumentProcessResult:
    def test_failure_carries_the_core_error(self, tmp_path: Path) -> None:
        result = document_process_result(
            _ctx(tmp_path), ConversionStepResult(success=False, error="bad pdf")
        )
        assert result.success is False
        assert result.error == "bad pdf"

    def test_skip_exists_points_at_the_existing_output(self, tmp_path: Path) -> None:
        result = document_process_result(
            _ctx(tmp_path), ConversionStepResult(success=True, skip_reason="exists")
        )
        assert result.success is True
        assert result.error == "skipped (exists)"
        assert result.output_path == str(tmp_path / "doc.pdf.md")

    def test_skip_image_only_has_no_output(self, tmp_path: Path) -> None:
        result = document_process_result(
            _ctx(tmp_path), ConversionStepResult(success=True, skip_reason="image_only")
        )
        assert result.success is True
        assert result.error == "skipped (image_only)"
        assert result.output_path is None

    def test_base_output_with_tallies(self, tmp_path: Path) -> None:
        ctx = _ctx(tmp_path)
        ctx.output_file = tmp_path / "doc.pdf.md"
        ctx.output_file.write_text("# doc", encoding="utf-8")
        ctx.embedded_images_count = 3
        ctx.screenshots_count = 2
        ctx.cache_hit = True
        result = document_process_result(ctx, ConversionStepResult(success=True))
        assert result.success is True
        assert result.error is None
        assert result.output_path == str(ctx.output_file)
        assert (result.images, result.screenshots) == (3, 2)
        assert result.cache_hit is True
        assert result.llm_enhanced is False

    def test_llm_output_is_selected_only_after_a_real_llm_write(
        self, tmp_path: Path
    ) -> None:
        ctx = _ctx(tmp_path, llm=True)
        ctx.output_file = tmp_path / "doc.pdf.md"
        ctx.output_file.write_text("# base", encoding="utf-8")
        # LLM enabled but the enhanced write never happened: no silent
        # fallback to the base file passed off as the LLM result.
        missing = document_process_result(ctx, ConversionStepResult(success=True))
        assert missing.success is False
        assert missing.error == "No output was produced for doc.pdf"

        ctx.llm_output_file = tmp_path / "doc.pdf.llm.md"
        ctx.llm_output_file.write_text("# llm", encoding="utf-8")
        ctx.llm_cost = 0.02
        result = document_process_result(ctx, ConversionStepResult(success=True))
        assert result.success is True
        assert result.output_path == str(ctx.llm_output_file)
        assert result.llm_enhanced is True
        assert result.cost_usd == 0.02


async def test_serve_file_items_use_the_shared_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.serve import jobs

    seen: list[Any] = []

    async def fake_core(ctx: ConversionContext, max_size: int) -> ConversionStepResult:
        return ConversionStepResult(success=True, skip_reason="image_only")

    def fake_mapping(ctx: ConversionContext, result: ConversionStepResult) -> Any:
        seen.append((ctx.input_path, result.skip_reason))
        return "mapped"

    monkeypatch.setattr("markitai.workflow.core.convert_document_core", fake_core)
    monkeypatch.setattr(
        "markitai.workflow.results.document_process_result", fake_mapping
    )
    source = tmp_path / "pic.png"
    result = await jobs.process_file_item(source, MarkitaiConfig(), tmp_path, None)
    assert result == "mapped"
    assert seen == [(source, "image_only")]
