"""Tests for cli/processors/batch_llm.py — Batch-API directory enhancement."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import instructor
import pytest

from markitai.cli.processors.batch_llm import (
    BATCH_COST_FACTOR,
    _anthropic_max_tokens,
    _batch_custom_id,
    _batch_usage,
    _prepare_pending,
    _single_batch_model,
    shell_arg,
)
from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.llm.batch_api import (
    BatchDocItem,
    BatchRunState,
    build_openai_batch_request,
    parse_batch_result,
)
from markitai.llm.types import ImageAnalysisResult
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


class TestSingleBatchModel:
    def test_openai_pool_resolves(self) -> None:
        assert _single_batch_model(_cfg_with_model("openai/gpt-5.6-luna")) == (
            "gpt-5.6-luna",
            "openai",
        )

    def test_anthropic_pool_resolves(self) -> None:
        assert _single_batch_model(_cfg_with_model("anthropic/claude-haiku-4-5")) == (
            "claude-haiku-4-5",
            "anthropic",
        )

    def test_empty_pool_refused(self) -> None:
        with pytest.raises(ConversionError, match="needs a configured model"):
            _single_batch_model(_cfg_with_model(None))

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
            _single_batch_model(cfg)

    def test_provider_without_a_batch_api_refused_with_guidance(self) -> None:
        with pytest.raises(ConversionError, match="openai and anthropic pools"):
            _single_batch_model(_cfg_with_model("gemini/gemini-flash-latest"))

    def test_local_provider_pool_refused(self) -> None:
        """claude-agent is a CLI subscription, not the Anthropic API."""
        with pytest.raises(ConversionError, match="openai and anthropic pools"):
            _single_batch_model(_cfg_with_model("claude-agent/sonnet"))


class TestBatchCustomId:
    """Anthropic validates custom_id against ^[a-zA-Z0-9_-]{1,64}$."""

    @pytest.mark.parametrize(
        "source",
        ["note1.md", "A Report (final).pdf", "报告.docx", "x" * 200, "a::b::c"],
    )
    def test_is_accepted_by_the_stricter_provider(self, source: str) -> None:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", _batch_custom_id(0, source))

    def test_index_keeps_squashed_names_distinct(self) -> None:
        """Two sources can flatten to the same characters."""
        assert _batch_custom_id(0, "a b") != _batch_custom_id(1, "a.b")


class TestAnthropicMaxTokens:
    def test_reads_the_model_ceiling(self) -> None:
        assert _anthropic_max_tokens("claude-haiku-4-5") > 8192

    def test_unknown_model_falls_back_to_a_usable_cap(self) -> None:
        assert _anthropic_max_tokens("claude-not-a-model") == 8192


class TestBatchUsage:
    def test_anthropic_usage_keys_are_read_and_priced(self) -> None:
        """Anthropic reports input_tokens/output_tokens, not prompt/completion."""
        body = {"usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}

        tokens_in, tokens_out, cost = _batch_usage(
            body, "claude-haiku-4-5", "anthropic"
        )

        assert (tokens_in, tokens_out) == (1_000_000, 1_000_000)
        assert cost > 0

    def test_openai_usage_keys_are_read(self) -> None:
        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.4-nano",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        tokens_in, tokens_out, _ = _batch_usage(body, "gpt-5.4-nano", "openai")

        assert (tokens_in, tokens_out) == (10, 5)


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
        plan_a = MagicMock(chunk_calls=[])
        plan_b = MagicMock(chunk_calls=[])
        processor.documents._prepare_document_plan.side_effect = [plan_a, plan_b]

        pending, cached, _oversized = _prepare_pending(processor, out)

        assert cached == 0
        assert len(pending) == 2
        assert pending[0][0].base_md == "a.md"
        assert pending[1][0].base_md == "b.md"
        assert pending[0][0].custom_id == "doc_0_a"

    def test_cache_hit_finalizes_immediately(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "a.md").write_text("# Doc A\n\nbody", encoding="utf-8")

        processor = MagicMock()
        processor.format_llm_output.side_effect = lambda cleaned, fm: (
            f"{fm}\n\n{cleaned}\n"
        )
        processor.documents._prepare_document_plan.return_value.chunk_calls = []
        hit_result = MagicMock()
        processor._engine.try_cached.return_value = hit_result
        processor.documents.finalize_document_plan.return_value = (
            "cleaned body",
            "---\ntitle: Doc A\n---",
        )

        pending, cached, _oversized = _prepare_pending(processor, out)

        assert cached == 1
        assert pending == []
        llm_md = out / "a.llm.md"
        assert llm_md.exists()
        assert "cleaned body" in llm_md.read_text(encoding="utf-8")

    def test_chunked_document_is_enhanced_live_not_batched(
        self, tmp_path: Path
    ) -> None:
        """A document past the per-call size is cleaned chunk by chunk. The
        batch carries one request per document, so submitting it would have
        enhanced only the first chunk."""
        out = tmp_path / "out"
        out.mkdir()
        (out / "long.md").write_text("# Long\n\nbody", encoding="utf-8")
        processor = MagicMock()
        processor._engine.try_cached.return_value = None
        processor.documents._prepare_document_plan.return_value = MagicMock(
            chunk_calls=[MagicMock()]
        )

        pending, cached, oversized = _prepare_pending(processor, out)

        assert pending == []
        assert cached == 0
        assert oversized == [(out / "long.md", "long", [])]
        processor._engine.try_cached.assert_not_called()


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
        plan = MagicMock(chunk_calls=[])
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
        processor.documents._prepare_document_plan.return_value = MagicMock(
            chunk_calls=[]
        )
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


async def test_run_manifest_controls_batch_scope_and_final_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
    from markitai.runs.json_output import render
    from markitai.runs.types import Outcome

    current = tmp_path / "current.txt.md"
    current.write_text("Current input")
    unrelated = tmp_path / "unrelated.md"
    unrelated.write_text("Do not send this document")
    processor = MagicMock()
    processor.documents._prepare_document_plan.return_value.chunk_calls = []
    processor._engine.try_cached.return_value = object()
    processor.documents.finalize_document_plan.return_value = (
        "ENHANCED",
        "title: Current",
    )
    processor.format_llm_output.side_effect = lambda cleaned, fm: (
        f"---\n{fm}\n---\n{cleaned}"
    )
    processor.get_context_cost.return_value = 0.125
    processor.get_context_usage.return_value = {
        "test-model": {
            "requests": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.125,
        }
    }
    monkeypatch.setattr(
        "markitai.workflow.helpers.create_llm_processor", lambda _cfg: processor
    )
    items = [
        Outcome(
            kind="file",
            source="current.txt",
            status="completed",
            output_path=current,
            duration=0.1,
        )
    ]
    handoff = await run_batch_llm_enhancement(
        _cfg_with_model("openai/test"), tmp_path, items=items, quiet=True
    )
    assert handoff is None
    assert processor.documents._prepare_document_plan.call_count == 1
    assert processor.documents._prepare_document_plan.call_args.args == (
        "Current input",
        "current.txt",
    )
    assert not unrelated.with_suffix(".llm.md").exists()
    envelope = json.loads(render(items))
    assert envelope["items"][0]["output"] == str(current.with_suffix(".llm.md"))
    assert envelope["totals"]["cost_usd"] == 0.125
    assert envelope["items"][0]["duration_s"] >= 0.1
    assert envelope["items"][0]["llm_usage"]["test-model"]["input_tokens"] == 10


class TestBatchCredentials:
    """The batch transport must use the pool entry's own key and endpoint."""

    def test_env_reference_and_api_base_are_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.cli.processors.batch_llm import _batch_credentials

        monkeypatch.setenv("TEST_KEY", "sk-from-config")
        cfg = _cfg_with_model("openai/gpt-5.6-luna")
        cfg.llm.model_list[0].litellm_params.api_base = "http://proxy.test/v1"

        creds = _batch_credentials(cfg, "gpt-5.6-luna", "openai")

        assert creds.api_key == "sk-from-config"
        assert creds.api_base == "http://proxy.test/v1"

    def test_unset_env_reference_is_one_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.cli.processors.batch_llm import _batch_credentials

        monkeypatch.delenv("TEST_KEY", raising=False)

        with pytest.raises(ConversionError, match=r"\$TEST_KEY, which is not set"):
            _batch_credentials(
                _cfg_with_model("openai/gpt-5.6-luna"), "gpt-5.6-luna", "openai"
            )

    def test_no_matching_entry_falls_back_to_the_environment(self) -> None:
        """A collect run whose config lacks the batch's model."""
        from markitai.cli.processors.batch_llm import _batch_credentials
        from markitai.llm.batch_api import BatchCredentials

        creds = _batch_credentials(
            _cfg_with_model("openai/other-model"), "gpt-5.6-luna", "openai"
        )

        assert creds == BatchCredentials()


# ---------------------------------------------------------------------------
# End-to-end over a real LLMProcessor, with only the network calls replaced
# ---------------------------------------------------------------------------


def _fake_answer(request: dict) -> dict:
    """A chat-completion body answering one batched TOOLS request."""
    body = request["body"]
    function = body["tools"][0]["function"]
    if "caption" in function["parameters"]["properties"]:
        args: dict = {
            "caption": "  A revenue chart ",
            "description": "Quarterly revenue bars.",
            "extracted_text": None,
        }
    else:
        text = body["messages"][-1]["content"]
        if isinstance(text, list):
            text = "\n".join(b.get("text", "") for b in text)
        args = {
            "cleaned_markdown": text.split("\n---\n", 1)[-1].strip(),
            "frontmatter": {"description": "A report", "tags": ["a", "b", "c"]},
        }
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
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
                                "name": function["name"],
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


async def _fake_download(batch_id: str, output_path: Path, **_kwargs: object) -> Path:
    """Answer every request in the batch's own requests.jsonl."""
    requests = [
        json.loads(line)
        for line in (output_path.parent / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    output_path.write_text(
        "".join(
            json.dumps(
                {
                    "custom_id": r["custom_id"],
                    "response": {"status_code": 200, "body": _fake_answer(r)},
                }
            )
            + "\n"
            for r in requests
        ),
        encoding="utf-8",
    )
    return output_path


def _real_cfg() -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.image.alt_enabled = True
    cfg.image.desc_enabled = True
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model="openai/gpt-4o-mini", api_key="sk-t"),
        )
    ]
    return cfg


def _doc_with_image(out: Path, png: bytes) -> tuple[Path, Path]:
    from markitai.constants import ASSETS_REL_PATH

    assets = out / ASSETS_REL_PATH
    assets.mkdir(parents=True)
    image = assets / "report.pdf.0001.png"
    image.write_bytes(png)
    base = out / "report.pdf.md"
    base.write_text(
        "# Quarterly report\n\nRevenue grew in every quarter.\n\n"
        f"![author alt]({ASSETS_REL_PATH}/{image.name})\n",
        encoding="utf-8",
    )
    return base, image


class TestRealProcessorCollect:
    """The shipped mocks returned MagicMock plans, so ``analysis.llm_usage``
    always existed; the real ``finalize_image_plan`` handed back the parsed
    ``ImageAnalysisResult``, which has none, and every --alt/--desc batch
    crashed at collect time — after the batch had been paid for. These run
    the real processor and only replace the network calls."""

    async def test_alt_and_desc_batch_collects_completely(
        self, tmp_path: Path, create_test_image
    ) -> None:
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        base, image = _doc_with_image(out, create_test_image(8, 8, "blue"))
        items = [
            Outcome(
                kind="file", source="report.pdf", status="completed", output_path=base
            )
        ]

        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                return_value="batch_real",
            ),
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                return_value="completed",
            ),
            patch(
                "markitai.cli.processors.batch_llm.download_openai_batch_output",
                side_effect=_fake_download,
            ),
        ):
            handoff = await run_batch_llm_enhancement(
                _real_cfg(), out, items=items, quiet=True
            )

        assert handoff is None
        llm_md = base.with_suffix(".llm.md").read_text(encoding="utf-8")
        assert f"![A revenue chart](.markitai/assets/{image.name})" in llm_md
        images_json = json.loads(
            (out / ".markitai" / "assets" / "images.json").read_text(encoding="utf-8")
        )
        assert images_json["images"][0]["desc"] == "Quarterly revenue bars."
        assert items[0].status == "completed"
        assert items[0].output_path == base.with_suffix(".llm.md")
        # both requests billed at the batch rate
        assert items[0].llm_usage["gpt-4o-mini"]["requests"] == 2
        state = BatchRunState.load(BatchRunState.state_dir_for(out, "batch_real"))
        assert state.collected_at is not None

    async def test_handoff_then_collect_then_collect_again(
        self, tmp_path: Path, create_test_image
    ) -> None:
        from markitai.cli.processors.batch_llm import (
            collect_batch_llm,
            run_batch_llm_enhancement,
        )
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        base, _image = _doc_with_image(out, create_test_image(8, 8, "green"))
        items = [
            Outcome(
                kind="file", source="report.pdf", status="completed", output_path=base
            )
        ]
        cfg = _real_cfg()

        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                return_value="batch_2p",
            ),
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                side_effect=TimeoutError("batch batch_2p still 'in_progress'"),
            ),
        ):
            handoff = await run_batch_llm_enhancement(cfg, out, items=items, quiet=True)

        assert handoff is not None
        assert handoff.batch_id == "batch_2p"
        assert handoff.to_json()["collect_command"] == (
            f"markitai --llm-batch-collect batch_2p -o {shell_arg(str(out))}"
        )
        assert items[0].status == "pending", "in flight is not failed"
        assert items[0].error is None
        state = BatchRunState.load(BatchRunState.state_dir_for(out, "batch_2p"))
        image_item = next(i for i in state.items if i.kind == "image")
        assert image_item.document_context.startswith("# Quarterly report")

        download = AsyncMock(side_effect=_fake_download)
        with (
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                return_value="completed",
            ),
            patch(
                "markitai.cli.processors.batch_llm.download_openai_batch_output",
                download,
            ),
        ):
            assert await collect_batch_llm(cfg, out, "batch_2p", quiet=True) == 0
            llm_md = base.with_suffix(".llm.md")
            assert "![A revenue chart]" in llm_md.read_text(encoding="utf-8")
            llm_md.write_text("edited by hand\n", encoding="utf-8")

            assert await collect_batch_llm(cfg, out, "batch_2p", quiet=True) == 0

        assert download.await_count == 1, "a collected batch is not fetched again"
        assert llm_md.read_text(encoding="utf-8") == "edited by hand\n"

    async def test_a_failed_image_keeps_the_author_alt_and_is_reported(
        self, tmp_path: Path, create_test_image
    ) -> None:
        """The Anthropic batch failed every image, and the collector wrote
        "Image" over the author's alt text while the run reported ok."""
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        base, _image = _doc_with_image(out, create_test_image(8, 8, "blue"))
        items = [
            Outcome(
                kind="file", source="report.pdf", status="completed", output_path=base
            )
        ]

        async def images_fail(batch_id: str, output_path: Path, **kwargs: object):
            await _fake_download(batch_id, output_path)
            lines = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            for line in lines:
                if line["custom_id"].startswith("img_"):
                    line.pop("response")
                    line["error"] = "invalid_request_error: image_url"
            output_path.write_text(
                "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
            )
            return output_path

        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                return_value="batch_img",
            ),
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                return_value="completed",
            ),
            patch(
                "markitai.cli.processors.batch_llm.download_openai_batch_output",
                side_effect=images_fail,
            ),
        ):
            await run_batch_llm_enhancement(_real_cfg(), out, items=items, quiet=True)

        llm_md = base.with_suffix(".llm.md").read_text(encoding="utf-8")
        assert "![author alt]" in llm_md
        assert "![Image]" not in llm_md
        assert not (out / ".markitai" / "assets" / "images.json").exists()
        assert items[0].status == "completed"
        assert items[0].warnings == [
            "image analysis failed for report.pdf.0001.png; its original alt "
            "text was kept"
        ]

    def test_finalize_image_plan_returns_the_live_type(
        self, tmp_path: Path, create_test_image
    ) -> None:
        """A batch hands finalize the bare parsed model; callers read the
        live path's ImageAnalysis fields (llm_usage) off whatever it returns."""
        from markitai.llm.types import ImageAnalysis
        from markitai.workflow.helpers import create_llm_processor

        image = tmp_path / "a.png"
        image.write_bytes(create_test_image(8, 8, "red"))
        vision = create_llm_processor(_real_cfg()).vision
        plan = vision.prepare_image_plan(image, context="a", document_context="x")

        analysis = vision.finalize_image_plan(
            plan,
            ImageAnalysisResult(caption=" A chart ", description="Bars."),
        )

        assert isinstance(analysis, ImageAnalysis)
        assert analysis.caption == "A chart"
        assert analysis.llm_usage is None

    async def test_no_cache_submits_what_the_cache_would_serve(
        self, tmp_path: Path, create_test_image
    ) -> None:
        """--no-cache skips cache reads for the batch plan too."""
        from markitai.cli.processors.batch_llm import _prepare_pending
        from markitai.workflow.helpers import create_llm_processor

        out = tmp_path / "out"
        base, _image = _doc_with_image(out, create_test_image(8, 8, "red"))
        cfg = _real_cfg()

        # A first processor fills the persistent cache for both requests
        warm = create_llm_processor(cfg)
        pending, cached, _ = _prepare_pending(
            warm, out, analyze_images=True, base_files=[base]
        )
        assert (len(pending), cached) == (2, 0)
        for item, plan in pending:
            answer = _fake_answer({"body": _tools_body(item, plan)})
            if item.kind == "image":
                result = parse_batch_result(
                    answer,
                    response_model=ImageAnalysisResult,
                    mode=instructor.Mode.TOOLS,
                )
                warm.vision.finalize_image_plan(plan, result)
            else:
                result = parse_batch_result(
                    answer,
                    response_model=plan.call.response_model,
                    mode=instructor.Mode.TOOLS,
                )
                warm.engine.write_cache(plan.call, result)

        base.with_suffix(".llm.md").unlink(missing_ok=True)
        _pending, cached, _ = _prepare_pending(
            create_llm_processor(cfg), out, analyze_images=True, base_files=[base]
        )
        assert cached == 2, "control: the cache does serve both"

        cfg.cache.no_cache = True
        pending, cached, _ = _prepare_pending(
            create_llm_processor(cfg), out, analyze_images=True, base_files=[base]
        )
        assert cached == 0
        assert [item.kind for item, _ in pending] == ["doc", "image"]


def _tools_body(item: BatchDocItem, plan: Any) -> dict:
    """The TOOLS request body the batch would send for one pending item."""
    if item.kind == "image":
        messages, model = plan.messages, ImageAnalysisResult
    else:
        messages, model = plan.call.messages, plan.call.response_model
    return build_openai_batch_request(
        item.custom_id,
        messages=messages,
        response_model=model,
        model="gpt-4o-mini",
        mode=instructor.Mode.TOOLS,
    )["body"]


class TestApiFailures:
    """Network/API errors end in one actionable line, never a traceback."""

    def _setup(self, tmp_path: Path) -> tuple[Path, list]:
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        out.mkdir()
        base = out / "a.txt.md"
        base.write_text("# A\n\nbody", encoding="utf-8")
        return out, [
            Outcome(kind="file", source="a.txt", status="completed", output_path=base)
        ]

    async def test_failed_submission_cleans_up_and_keeps_the_base(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement

        out, items = self._setup(tmp_path)
        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                side_effect=ConnectionError("connection refused"),
            ),
            pytest.raises(
                ConversionError, match="submission failed.*connection refused"
            ),
        ):
            await run_batch_llm_enhancement(_real_cfg(), out, items=items, quiet=True)

        assert not (out / ".markitai" / "batch-pending").exists()
        assert (out / "a.txt.md").exists()
        assert items[0].status == "failed"
        assert "Nothing was submitted" in (items[0].error or "")

    async def test_lost_contact_while_polling_is_a_handoff(
        self, tmp_path: Path
    ) -> None:
        """The batch was accepted and keeps running; the run must hand it off
        (exit 2, collect command) rather than fail it."""
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement

        out, items = self._setup(tmp_path)
        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                return_value="batch_lost",
            ),
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                side_effect=ConnectionError("network unreachable"),
            ),
        ):
            handoff = await run_batch_llm_enhancement(
                _real_cfg(), out, items=items, quiet=True
            )

        assert handoff is not None
        assert handoff.status == "unknown"
        assert "network unreachable" in handoff.reason
        assert items[0].status == "pending"
        assert (BatchRunState.state_dir_for(out, "batch_lost") / "state.json").exists()

    async def test_failed_download_keeps_the_state_for_a_retry(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement

        out, items = self._setup(tmp_path)
        with (
            patch(
                "markitai.cli.processors.batch_llm.submit_openai_batch",
                new_callable=AsyncMock,
                return_value="batch_dl",
            ),
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                return_value="completed",
            ),
            patch(
                "markitai.cli.processors.batch_llm.download_openai_batch_output",
                new_callable=AsyncMock,
                side_effect=TimeoutError("read timed out"),
            ),
            pytest.raises(
                ConversionError, match="download failed.*--llm-batch-collect batch_dl"
            ),
        ):
            await run_batch_llm_enhancement(_real_cfg(), out, items=items, quiet=True)

        state = BatchRunState.load(BatchRunState.state_dir_for(out, "batch_dl"))
        assert state.collected_at is None

    async def test_collect_status_check_failure_is_one_line(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import collect_batch_llm

        out = tmp_path / "out"
        state_dir = BatchRunState.state_dir_for(out, "batch_c")
        BatchRunState(
            batch_id="batch_c",
            model="gpt-4o-mini",
            mode="tool_call",
            provider="openai",
            created_at="2026-09-24T00:00:00+00:00",
        ).save(state_dir)

        with (
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                side_effect=ConnectionError("dns failure"),
            ),
            pytest.raises(ConversionError, match="status check failed.*dns failure"),
        ):
            await collect_batch_llm(_real_cfg(), out, "batch_c", quiet=True)

    async def test_collect_passes_the_configured_credentials(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import collect_batch_llm
        from markitai.llm.batch_api import BatchCredentials

        out = tmp_path / "out"
        BatchRunState(
            batch_id="batch_k",
            model="gpt-4o-mini",
            mode="tool_call",
            provider="openai",
            created_at="2026-09-24T00:00:00+00:00",
        ).save(BatchRunState.state_dir_for(out, "batch_k"))
        cfg = _real_cfg()
        cfg.llm.model_list[0].litellm_params.api_base = "http://proxy.test/v1"

        with patch(
            "markitai.cli.processors.batch_llm.poll_openai_batch",
            new_callable=AsyncMock,
            return_value="in_progress",
        ) as poll:
            assert await collect_batch_llm(cfg, out, "batch_k", quiet=True) == 2

        assert poll.call_args.kwargs["credentials"] == BatchCredentials(
            api_key="sk-t", api_base="http://proxy.test/v1"
        )


def test_resume_adds_documents_converted_but_never_enhanced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed --llm-batch run only converts what is left; documents the
    earlier run converted (and whose submit failed) used to be dropped, so
    the resume reported everything cached and enhanced nothing."""
    import json

    from markitai.cli.processors.batch_llm import resumed_unenhanced_outcomes
    from markitai.runs.types import Outcome

    out = tmp_path / "out"
    states = out / ".markitai" / "states"
    states.mkdir(parents=True)
    for name in ("a.txt.md", "b.txt.md", "c.txt.md"):
        (out / name).write_text("# base\n", encoding="utf-8")
    (out / "b.txt.llm.md").write_text("# enhanced\n", encoding="utf-8")
    (states / "markitai.abc123.state.json").write_text(
        json.dumps(
            {
                "documents": {
                    "a.txt": {"status": "completed", "output": str(out / "a.txt.md")},
                    "b.txt": {"status": "completed", "output": str(out / "b.txt.md")},
                    "c.txt": {"status": "completed", "output": str(out / "c.txt.md")},
                    "d.txt": {"status": "failed", "output": None},
                }
            }
        ),
        encoding="utf-8",
    )
    known = [
        Outcome(
            kind="file",
            source="c.txt",
            status="completed",
            output_path=out / "c.txt.md",
        )
    ]

    extra = resumed_unenhanced_outcomes(out, known)

    assert [(item.source, item.output_path) for item in extra] == [
        ("a.txt", out / "a.txt.md")
    ]
    # A relative output dir keeps the relative spelling this run's outcomes use
    monkeypatch.chdir(tmp_path)
    relative = resumed_unenhanced_outcomes(Path("out"), known)
    assert [item.output_path for item in relative] == [Path("out") / "a.txt.md"]
    assert all(item.status == "completed" for item in extra)


# ---------------------------------------------------------------------------
# Review fixes: cached image answers, long documents, resume, odd outputs
# ---------------------------------------------------------------------------


def _patched_batch(
    batch_id: str, *, poll: Any = "completed", download: Any = _fake_download
) -> Any:
    """Patch the three network calls of one --llm-batch run."""
    from contextlib import ExitStack

    stack = ExitStack()
    submit = stack.enter_context(
        patch(
            "markitai.cli.processors.batch_llm.submit_openai_batch",
            new_callable=AsyncMock,
            return_value=batch_id,
        )
    )
    poll_kwargs: dict[str, Any] = (
        {"side_effect": poll}
        if isinstance(poll, BaseException) or callable(poll)
        else {"return_value": poll}
    )
    stack.enter_context(
        patch(
            "markitai.cli.processors.batch_llm.poll_openai_batch",
            new_callable=AsyncMock,
            **poll_kwargs,
        )
    )
    stack.enter_context(
        patch(
            "markitai.cli.processors.batch_llm.download_openai_batch_output",
            side_effect=download,
        )
    )
    stack.submit = submit  # type: ignore[attr-defined]
    return stack


def _report_outcome(base: Path) -> Any:
    from markitai.runs.types import Outcome

    return Outcome(
        kind="file", source="report.pdf", status="completed", output_path=base
    )


class TestCachedImageAnswers:
    """A cache-served image answer must reach the output like a batched one."""

    async def test_fully_cached_rerun_keeps_alt_text_and_images_json(
        self, tmp_path: Path, create_test_image
    ) -> None:
        """An overwrite re-run served everything from the cache, rewrote the
        .llm.md without alt text, and returned before any answer was
        applied: the alt text reverted and images.json was not written."""
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement

        out = tmp_path / "out"
        base, image = _doc_with_image(out, create_test_image(8, 8, "blue"))
        cfg = _real_cfg()
        with _patched_batch("batch_warm"):
            await run_batch_llm_enhancement(
                cfg, out, items=[_report_outcome(base)], quiet=True
            )
        images_json = out / ".markitai" / "assets" / "images.json"
        images_json.unlink()

        items = [_report_outcome(base)]
        with _patched_batch("batch_again") as batch:
            handoff = await run_batch_llm_enhancement(cfg, out, items=items, quiet=True)

        assert handoff is None
        batch.submit.assert_not_awaited()  # everything came from the cache
        llm_md = base.with_suffix(".llm.md").read_text(encoding="utf-8")
        assert f"![A revenue chart](.markitai/assets/{image.name})" in llm_md
        assert "author alt" not in llm_md
        entries = json.loads(images_json.read_text(encoding="utf-8"))["images"]
        assert [entry["desc"] for entry in entries] == ["Quarterly revenue bars."]
        assert items[0].output_path == base.with_suffix(".llm.md")

    async def test_cached_image_of_a_batched_document_waits_for_collect(
        self, tmp_path: Path, create_test_image
    ) -> None:
        """The image is cached but its document is not: the answer has to
        survive the handoff, since collect writes the .llm.md it goes in."""
        from markitai.cli.processors.batch_llm import (
            collect_batch_llm,
            run_batch_llm_enhancement,
        )

        out = tmp_path / "out"
        base, _image = _doc_with_image(out, create_test_image(8, 8, "red"))
        # Longer than the image's context snippet (200 characters)
        base.write_text(
            base.read_text(encoding="utf-8") + "\n" + "Revenue detail. " * 20 + "\n",
            encoding="utf-8",
        )
        cfg = _real_cfg()
        with _patched_batch("batch_warm"):
            await run_batch_llm_enhancement(
                cfg, out, items=[_report_outcome(base)], quiet=True
            )
        # New text past the snippet the image's cache key reads: the
        # document misses the cache, the image still hits it.
        base.write_text(
            base.read_text(encoding="utf-8") + "\n" + "More findings. " * 40 + "\n",
            encoding="utf-8",
        )
        (out / ".markitai" / "assets" / "images.json").unlink()

        with _patched_batch("batch_doc", poll=TimeoutError("still running")) as b:
            handoff = await run_batch_llm_enhancement(
                cfg, out, items=[_report_outcome(base)], quiet=True
            )
        assert handoff is not None
        requests = (
            (BatchRunState.state_dir_for(out, "batch_doc") / "requests.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(requests) == 1, "only the document is asked for"
        b.submit.assert_awaited_once()
        state = BatchRunState.load(BatchRunState.state_dir_for(out, "batch_doc"))
        cached = [item for item in state.items if item.answer is not None]
        assert [item.kind for item in cached] == ["image"]

        with _patched_batch("batch_doc"):
            assert await collect_batch_llm(cfg, out, "batch_doc", quiet=True) == 0

        llm_md = base.with_suffix(".llm.md").read_text(encoding="utf-8")
        assert "![A revenue chart]" in llm_md
        assert (out / ".markitai" / "assets" / "images.json").is_file()


class TestLongDocuments:
    async def test_a_chunked_document_gets_its_alt_text(
        self, tmp_path: Path, create_test_image, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A document too long for one request was enhanced live, and its
        images were never analyzed: no alt text, no images.json."""
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.workflow.helpers import create_llm_processor

        out = tmp_path / "out"
        base, image = _doc_with_image(out, create_test_image(8, 8, "blue"))
        cfg = _real_cfg()
        processor = create_llm_processor(cfg)
        real_plan = processor.documents._prepare_document_plan

        def chunked(markdown: str, source: str) -> Any:
            plan = real_plan(markdown, source)
            plan.chunk_calls = [plan.call, plan.call]
            return plan

        live = AsyncMock(
            return_value=(
                f"# Quarterly report\n\n![author alt](.markitai/assets/{image.name})\n",
                "title: Report",
            )
        )
        monkeypatch.setattr(processor.documents, "_prepare_document_plan", chunked)
        monkeypatch.setattr(processor.documents, "process_document", live)
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor", lambda _cfg: processor
        )
        items = [_report_outcome(base)]

        with _patched_batch("batch_long"):
            handoff = await run_batch_llm_enhancement(cfg, out, items=items, quiet=True)

        assert handoff is None
        live.assert_awaited_once()
        llm_md = base.with_suffix(".llm.md").read_text(encoding="utf-8")
        assert f"![A revenue chart](.markitai/assets/{image.name})" in llm_md
        assert (out / ".markitai" / "assets" / "images.json").is_file()
        assert items[0].status == "completed"

    async def test_a_live_failure_is_not_pending_after_a_handoff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live re-run of a long document failed, then the batch for the
        others timed out: the handoff marked the failed one pending, as if
        a collect would still produce it."""
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.runs.types import Outcome
        from markitai.workflow.helpers import create_llm_processor

        out = tmp_path / "out"
        out.mkdir()
        (out / "long.txt.md").write_text("# Long\n\nbody", encoding="utf-8")
        (out / "short.txt.md").write_text("# Short\n\nbody", encoding="utf-8")
        cfg = _real_cfg()
        cfg.image.alt_enabled = cfg.image.desc_enabled = False
        processor = create_llm_processor(cfg)
        real_plan = processor.documents._prepare_document_plan

        def plan_for(markdown: str, source: str) -> Any:
            plan = real_plan(markdown, source)
            if source == "long.txt":
                plan.chunk_calls = [plan.call, plan.call]
            return plan

        monkeypatch.setattr(processor.documents, "_prepare_document_plan", plan_for)
        monkeypatch.setattr(
            processor.documents,
            "process_document",
            AsyncMock(side_effect=RuntimeError("provider down")),
        )
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor", lambda _cfg: processor
        )
        items = [
            Outcome(kind="file", source=name, status="completed", output_path=path)
            for name, path in (
                ("long.txt", out / "long.txt.md"),
                ("short.txt", out / "short.txt.md"),
            )
        ]

        with _patched_batch("batch_mixed", poll=TimeoutError("still running")):
            handoff = await run_batch_llm_enhancement(cfg, out, items=items, quiet=True)

        assert handoff is not None
        statuses = {item.source: item.status for item in items}
        assert statuses == {"long.txt": "failed", "short.txt": "pending"}
        assert "provider down" in (items[0].error or "")


class TestResumeAndUncollectedBatches:
    def _converted(self, out: Path) -> None:
        states = out / ".markitai" / "states"
        states.mkdir(parents=True)
        for name in ("a.txt.md", "b.txt.md"):
            (out / name).write_text("# base\n", encoding="utf-8")
        (states / "markitai.abc123.state.json").write_text(
            json.dumps(
                {
                    "documents": {
                        "a.txt": {
                            "status": "completed",
                            "output": str(out / "a.txt.md"),
                        },
                        "b.txt": {
                            "status": "completed",
                            "output": str(out / "b.txt.md"),
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    def _submitted(self, out: Path, batch_id: str, **fields: Any) -> BatchRunState:
        state = BatchRunState(
            batch_id=batch_id,
            model="gpt-4o-mini",
            mode="tool_call",
            provider="openai",
            created_at="2026-09-24T00:00:00+00:00",
            items=[
                BatchDocItem(
                    custom_id="doc_0_a_txt",
                    source="a.txt",
                    input_md="inputs/0.md",
                    base_md="a.txt.md",
                )
            ],
            **fields,
        )
        state.save(BatchRunState.state_dir_for(out, batch_id))
        return state

    def test_a_document_in_an_uncollected_batch_is_not_resubmitted(
        self, tmp_path: Path
    ) -> None:
        """--resume saw a base .md without .llm.md and submitted it again,
        paying twice for results already waiting in the first batch."""
        from markitai.cli.processors.batch_llm import resumed_unenhanced_outcomes

        out = tmp_path / "out"
        self._converted(out)
        self._submitted(out, "batch_open")

        extra = resumed_unenhanced_outcomes(out, [])

        assert [item.source for item in extra] == ["b.txt"]

    @pytest.mark.parametrize(
        "fields",
        [
            {"collected_at": "2026-09-24T01:00:00+00:00"},
            {"ended_status": "expired"},
        ],
    )
    def test_a_finished_batch_no_longer_holds_its_documents(
        self, tmp_path: Path, fields: dict[str, Any]
    ) -> None:
        from markitai.cli.processors.batch_llm import resumed_unenhanced_outcomes

        out = tmp_path / "out"
        self._converted(out)
        self._submitted(out, "batch_done", **fields)

        extra = resumed_unenhanced_outcomes(out, [])

        assert sorted(item.source for item in extra) == ["a.txt", "b.txt"]

    def test_the_collect_command_is_printed(self, tmp_path: Path) -> None:
        from io import StringIO

        from rich.console import Console

        from markitai.cli.processors.batch_llm import report_uncollected_batches

        out = tmp_path / "out"
        self._submitted(out, "batch_open")
        self._submitted(out, "batch_done", collected_at="2026-09-24T01:00:00+00:00")
        buffer = StringIO()

        with patch(
            "markitai.cli.processors.batch_llm.get_stderr_console",
            return_value=Console(file=buffer, width=400),
        ):
            count = report_uncollected_batches(out, config_path=Path("my.json"))

        assert count == 1
        text = buffer.getvalue()
        assert (
            f"markitai --llm-batch-collect batch_open -o {shell_arg(str(out))} "
            "-c my.json"
        ) in text
        assert "batch_done" not in text

    async def test_a_batch_that_ended_without_results_is_marked(
        self, tmp_path: Path
    ) -> None:
        from markitai.cli.processors.batch_llm import collect_batch_llm

        out = tmp_path / "out"
        self._submitted(out, "batch_x")

        with (
            patch(
                "markitai.cli.processors.batch_llm.poll_openai_batch",
                new_callable=AsyncMock,
                return_value="expired",
            ),
            pytest.raises(ConversionError, match="expired"),
        ):
            await collect_batch_llm(_real_cfg(), out, "batch_x", quiet=True)

        state = BatchRunState.load(BatchRunState.state_dir_for(out, "batch_x"))
        assert state.ended_status == "expired"

    async def test_ctrl_c_while_waiting_leaves_the_items_pending(
        self, tmp_path: Path
    ) -> None:
        """Ctrl-C during the wait left the items "completed" at their base .md
        and the batch id unprinted under --quiet/--json."""
        import asyncio

        from markitai.cli.processors.batch_llm import (
            BatchHandoff,
            run_batch_llm_enhancement,
        )
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        out.mkdir()
        (out / "a.txt.md").write_text("# A\n\nbody", encoding="utf-8")
        items = [
            Outcome(
                kind="file",
                source="a.txt",
                status="completed",
                output_path=out / "a.txt.md",
            )
        ]
        seen: list[BatchHandoff] = []

        with (
            _patched_batch("batch_cc", poll=asyncio.CancelledError()),
            pytest.raises(asyncio.CancelledError),
        ):
            await run_batch_llm_enhancement(
                _real_cfg(),
                out,
                items=items,
                quiet=True,
                config_path=Path("my.json"),
                on_submitted=seen.append,
            )

        assert [handoff.batch_id for handoff in seen] == ["batch_cc"]
        assert seen[0].collect_command.endswith("-c my.json")
        assert items[0].status == "pending"
        assert items[0].output_path == out / "a.txt.md"


class TestItemsWithoutMarkdown:
    async def test_a_screenshot_output_is_skipped_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A --screenshot-only URL's output is its .jpg; reading it as the
        document to enhance raised UnicodeDecodeError and ended the run."""
        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        shot = out / ".markitai" / "screenshots" / "example.com.full.jpg"
        shot.parent.mkdir(parents=True)
        shot.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")
        items = [
            Outcome(
                kind="url",
                source="https://example.com",
                status="completed",
                output_path=shot,
            )
        ]
        factory = MagicMock()
        monkeypatch.setattr("markitai.workflow.helpers.create_llm_processor", factory)

        handoff = await run_batch_llm_enhancement(
            _real_cfg(), out, items=items, quiet=True
        )

        assert handoff is None
        factory.assert_not_called()
        assert items[0].status == "completed"
        assert items[0].output_path == shot
        assert len(items[0].warnings) == 1
        assert "cannot enhance example.com.full.jpg" in items[0].warnings[0]

    async def test_nothing_converted_means_no_enhancement_and_no_cache_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every conversion failed, yet the run built a processor and said
        "All documents already cached"."""
        from io import StringIO

        from rich.console import Console

        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.runs.types import Outcome

        items = [
            Outcome(kind="file", source="broken.pdf", status="failed", error="bad")
        ]
        factory = MagicMock()
        monkeypatch.setattr("markitai.workflow.helpers.create_llm_processor", factory)
        buffer = StringIO()

        with patch(
            "markitai.cli.processors.batch_llm.get_stderr_console",
            return_value=Console(file=buffer),
        ):
            handoff = await run_batch_llm_enhancement(
                _real_cfg(), tmp_path, items=items, quiet=False
            )

        assert handoff is None
        factory.assert_not_called()
        assert "cached" not in buffer.getvalue()
        assert items[0].status == "failed"

    async def test_no_cache_line_when_nothing_came_from_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run whose only document was enhanced live has nothing to submit,
        but nothing was cached either."""
        from io import StringIO

        from rich.console import Console

        from markitai.cli.processors.batch_llm import run_batch_llm_enhancement
        from markitai.runs.types import Outcome

        out = tmp_path / "out"
        out.mkdir()
        (out / "a.txt.md").write_text("# A\n\nbody", encoding="utf-8")
        processor = MagicMock()
        processor.documents._prepare_document_plan.return_value.chunk_calls = [1, 2]
        processor.documents.process_document = AsyncMock(return_value=("A", "t: a"))
        processor.format_llm_output.side_effect = lambda cleaned, _fm: cleaned
        processor.get_context_cost.return_value = 0.0
        processor.get_context_usage.return_value = {}
        monkeypatch.setattr(
            "markitai.workflow.helpers.create_llm_processor", lambda _cfg: processor
        )
        cfg = _real_cfg()
        cfg.image.alt_enabled = cfg.image.desc_enabled = False
        buffer = StringIO()

        with patch(
            "markitai.cli.processors.batch_llm.get_stderr_console",
            return_value=Console(file=buffer),
        ):
            await run_batch_llm_enhancement(
                cfg,
                out,
                items=[
                    Outcome(
                        kind="file",
                        source="a.txt",
                        status="completed",
                        output_path=out / "a.txt.md",
                    )
                ],
                quiet=False,
            )

        assert "already cached" not in buffer.getvalue()
        assert (out / "a.txt.llm.md").is_file()


async def test_collect_applies_images_the_way_the_submitting_run_asked(
    tmp_path: Path, create_test_image
) -> None:
    """--alt/--desc given as flags are not in the -c file the printed collect
    command passes, so collect read them as off and dropped every paid-for
    image answer: no alt text, no images.json."""
    from markitai.cli.processors.batch_llm import (
        collect_batch_llm,
        run_batch_llm_enhancement,
    )

    out = tmp_path / "out"
    base, image = _doc_with_image(out, create_test_image(8, 8, "blue"))
    with _patched_batch("batch_flags", poll=TimeoutError("still running")):
        handoff = await run_batch_llm_enhancement(
            _real_cfg(), out, items=[_report_outcome(base)], quiet=True
        )
    assert handoff is not None

    collect_cfg = _real_cfg()  # the -c file: no --alt / --desc in it
    collect_cfg.image.alt_enabled = collect_cfg.image.desc_enabled = False
    with _patched_batch("batch_flags"):
        assert await collect_batch_llm(collect_cfg, out, "batch_flags", quiet=True) == 0

    llm_md = base.with_suffix(".llm.md").read_text(encoding="utf-8")
    assert f"![A revenue chart](.markitai/assets/{image.name})" in llm_md
    assert (out / ".markitai" / "assets" / "images.json").is_file()


class TestShellArg:
    """The printed collect command must paste into the user's own shell."""

    def test_posix_quotes_only_what_the_shell_would_split(self) -> None:
        assert shell_arg("/tmp/out", windows=False) == "/tmp/out"
        assert shell_arg("/tmp/my out", windows=False) == "'/tmp/my out'"

    def test_windows_uses_double_quotes_cmd_understands(self) -> None:
        # shlex.quote gave 'C:\out', whose single quotes cmd.exe keeps
        assert shell_arg(r"C:\Users\me\out", windows=True) == r"C:\Users\me\out"
        assert shell_arg(r"C:\My Docs\out", windows=True) == r'"C:\My Docs\out"'
