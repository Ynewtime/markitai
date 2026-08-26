"""Tests for cli/processors/batch_llm.py — Batch-API directory enhancement."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.cli.processors.batch_llm import (
    BATCH_COST_FACTOR,
    _prepare_pending,
    _single_openai_model,
)
from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.llm.batch_api import BatchDocItem, BatchRunState
from markitai.utils.errors import ConversionError


def _cfg_with_model(model: str | None, *, llm_enabled: bool = True) -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = llm_enabled
    if model is not None:
        cfg.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model=model, api_key="env:TEST_KEY"),
            )
        ]
    return cfg


class TestSingleOpenaiModel:
    def test_openai_pool_resolves(self) -> None:
        assert _single_openai_model(_cfg_with_model("openai/gpt-5.6-luna")) == (
            "gpt-5.6-luna",
            "openai",
        )

    def test_empty_pool_refused(self) -> None:
        with pytest.raises(ConversionError, match="needs a configured model"):
            _single_openai_model(_cfg_with_model(None))

    def test_multi_model_pool_refused(self) -> None:
        cfg = _cfg_with_model("openai/gpt-5.6-luna")
        cfg.llm.model_list.append(
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="openai/gpt-5.6-mini", api_key="env:TEST_KEY"
                ),
            )
        )
        with pytest.raises(ConversionError, match="single-model pool"):
            _single_openai_model(cfg)

    def test_non_openai_pool_refused_with_guidance(self) -> None:
        with pytest.raises(ConversionError, match="OpenAI pools only"):
            _single_openai_model(_cfg_with_model("gemini/gemini-flash-latest"))

    def test_local_provider_pool_refused(self) -> None:
        with pytest.raises(ConversionError, match="OpenAI pools only"):
            _single_openai_model(_cfg_with_model("claude-agent/sonnet"))


class TestRunState:
    def test_roundtrip(self, tmp_path: Path) -> None:
        state = BatchRunState(
            batch_id="batch_abc",
            model="gpt-5.6-luna",
            mode="tool_call",
            provider="openai",
            created_at="2026-08-26T01:00:00+08:00",
            items=[
                BatchDocItem(
                    custom_id="doc::0::a.md",
                    source="a.md",
                    input_md="inputs/0.md",
                    base_md="a.md",
                )
            ],
        )
        state_dir = state.save(tmp_path / "state").parent
        loaded = BatchRunState.load(state_dir)
        assert loaded.batch_id == "batch_abc"
        assert loaded.items[0].base_md == "a.md"

    def test_state_dir_layout(self, tmp_path: Path) -> None:
        assert BatchRunState.state_dir_for(tmp_path, "batch_x") == (
            tmp_path / ".markitai" / "batch-batch_x"
        )


class TestPreparePending:
    def test_collects_uncached_and_serves_cache_hits(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "a.md").write_text("# Doc A\n\nbody text a", encoding="utf-8")
        (out / "b.md").write_text("# Doc B\n\nbody text b", encoding="utf-8")
        (out / "a.llm.md").write_text("already enhanced", encoding="utf-8")
        (out / ".markitai").mkdir()
        (out / ".markitai" / "skip.md").write_text("internal", encoding="utf-8")

        processor = MagicMock()
        engine = processor._engine
        engine.try_cached.return_value = None
        plan_a = MagicMock()
        plan_b = MagicMock()
        processor.documents._prepare_document_plan.side_effect = [plan_a, plan_b]

        pending, cached = _prepare_pending(processor, out)

        assert cached == 0
        assert len(pending) == 2
        assert pending[0][0].base_md == "a.md"
        assert pending[1][0].base_md == "b.md"
        assert pending[0][0].custom_id.startswith("doc::0::")

    def test_cache_hit_finalizes_immediately(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "a.md").write_text("# Doc A\n\nbody", encoding="utf-8")

        processor = MagicMock()
        processor.format_llm_output.side_effect = lambda cleaned, fm: (
            f"{fm}\n\n{cleaned}\n"
        )
        hit_result = MagicMock()
        processor._engine.try_cached.return_value = hit_result
        processor.documents.finalize_document_plan.return_value = (
            "cleaned body",
            "---\ntitle: Doc A\n---",
        )

        pending, cached = _prepare_pending(processor, out)

        assert cached == 1
        assert pending == []
        llm_md = out / "a.llm.md"
        assert llm_md.exists()
        assert "cleaned body" in llm_md.read_text(encoding="utf-8")


class TestFinishBatch:
    async def test_batch_result_writes_llm_md_and_accounts_half_price(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import _finish_batch

        out = tmp_path / "out"
        out.mkdir()
        (out / "a.md").write_text("# Doc A\n\nbody", encoding="utf-8")

        state = BatchRunState(
            batch_id="batch_x",
            model="gpt-5.6-luna",
            mode="tool_call",
            provider="openai",
            created_at="2026-08-26T01:00:00+08:00",
            items=[
                BatchDocItem(
                    custom_id="doc::0::a.md",
                    source="a.md",
                    input_md="inputs/0.md",
                    base_md="a.md",
                )
            ],
        )
        state_dir = BatchRunState.state_dir_for(out, "batch_x")
        (state_dir / "inputs").mkdir(parents=True)
        (state_dir / "inputs" / "0.md").write_text("# Doc A\n\nbody", encoding="utf-8")
        state.save(state_dir)

        payload = json.dumps({"cleaned_markdown": "# Clean A", "summary": "s"})
        body = {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "DocumentProcessResult",
                                    "arguments": payload,
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
            },
        }

        processor = MagicMock()
        processor.format_llm_output.side_effect = lambda cleaned, fm: (
            f"{fm}\n\n{cleaned}\n"
        )
        plan = MagicMock()
        plan.call.validate = None
        processor.documents._prepare_document_plan.return_value = plan
        processor.documents.finalize_document_plan.return_value = (
            "# Clean A",
            "---\ntitle: Doc A\n---",
        )

        with (
            patch(
                "markitai.cli.processors.batch_llm.download_openai_batch_output",
                new_callable=AsyncMock,
            ) as mock_dl,
            patch(
                "markitai.cli.processors.batch_llm.read_openai_batch_output"
            ) as mock_read,
            patch("markitai.cli.processors.batch_llm.parse_batch_result") as mock_parse,
            patch(
                "markitai.llm.models.get_response_cost",
                return_value=0.01,
            ),
        ):
            mock_dl.return_value = state_dir / "output.jsonl"
            line = MagicMock()
            line.custom_id = "doc::0::a.md"
            line.error = None
            line.body = body
            mock_read.return_value = iter([line])
            mock_parse.return_value = MagicMock()

            code = await _finish_batch(
                _cfg_with_model("openai/gpt-5.6-luna"),
                processor,
                out,
                state,
                quiet=True,
            )

        assert code == 0
        assert (out / "a.llm.md").exists()
        # usage accounted at half the list price
        track = processor._track_usage.call_args
        assert track.args[0] == "gpt-5.6-luna"
        assert track.args[1] == 1000
        assert track.args[2] == 200
        assert track.args[3] == pytest.approx(0.01 * BATCH_COST_FACTOR)
        assert track.args[4] == "a.md"

    async def test_failed_line_reruns_live(self, tmp_path: Path) -> None:
        from markitai.cli.processors.batch_llm import _finish_batch

        out = tmp_path / "out"
        out.mkdir()
        (out / "a.md").write_text("# Doc A\n\nbody", encoding="utf-8")

        state = BatchRunState(
            batch_id="batch_x",
            model="gpt-5.6-luna",
            mode="tool_call",
            provider="openai",
            created_at="2026-08-26T01:00:00+08:00",
            items=[
                BatchDocItem(
                    custom_id="doc::0::a.md",
                    source="a.md",
                    input_md="inputs/0.md",
                    base_md="a.md",
                )
            ],
        )
        state_dir = BatchRunState.state_dir_for(out, "batch_x")
        (state_dir / "inputs").mkdir(parents=True)
        (state_dir / "inputs" / "0.md").write_text("# Doc A\n\nbody", encoding="utf-8")
        state.save(state_dir)

        processor = MagicMock()
        processor.format_llm_output.side_effect = lambda cleaned, fm: (
            f"{fm}\n\n{cleaned}\n"
        )
        processor.documents._prepare_document_plan.return_value = MagicMock()
        processor.documents.process_document = AsyncMock(
            return_value=("# Live A", "---\ntitle: Doc A\n---")
        )

        with (
            patch(
                "markitai.cli.processors.batch_llm.download_openai_batch_output",
                new_callable=AsyncMock,
            ),
            patch(
                "markitai.cli.processors.batch_llm.read_openai_batch_output"
            ) as mock_read,
        ):
            line = MagicMock()
            line.custom_id = "doc::0::a.md"
            line.error = "rate limited"
            line.body = None
            mock_read.return_value = iter([line])

            code = await _finish_batch(
                _cfg_with_model("openai/gpt-5.6-luna"),
                processor,
                out,
                state,
                quiet=True,
            )

        assert code == 0
        llm_md = (out / "a.llm.md").read_text(encoding="utf-8")
        assert "# Live A" in llm_md  # live re-run output, not a lost document
