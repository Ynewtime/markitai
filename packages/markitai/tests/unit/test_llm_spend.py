"""Rejected LLM answers are billed: cost, usage and the cost cap count them.

A 1.2.1 check: ``default-table.pdf --llm`` with Gemini Flash-Lite and
OpenRouter's Claude Haiku alias failed (the answer dropped a placeholder),
kept the unenhanced output, and reported ``cost_usd: 0.0`` for a request
that was billed: the failing stage cleared its usage without handing it on.
Answers instructor rejected (unparseable JSON) were never counted at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import litellm
import pytest

from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    MarkitaiConfig,
    ModelConfig,
    PromptsConfig,
)
from markitai.llm import LLMProcessor
from markitai.llm.engine import LLMEnhancementDegradedError, track_llm_spend
from markitai.workflow.core import ConversionContext, convert_document_core
from markitai.workflow.results import document_process_result

P0, P1, IMG = "__MARKITAI_PAGENUM_0__", "__MARKITAI_PAGENUM_1__", "__MARKITAI_IMG_0__"
MODEL = "deepseek/deepseek-flash"  # JSON in the prompt (MD_JSON): a text answer

TABLE_PAGE = (
    "<!-- Page number: 1 -->\n\n# Language comparison\n\n"
    "A table in the browser's default style, its cells a few pixels apart.\n\n"
    "|Language|Year|Typing|Primary Use|\n|---|---|---|---|\n"
    "|Python|1991|Dynamic|General purpose|\n|Rust|2015|Static|Systems programming|\n"
    "|TypeScript|2012|Static|Web development|\n\n"
    "Each language has its own trade-offs."
)
PLAIN = (
    "# The budget\n\n"
    + "The committee reviewed the annual budget and approved the new library wing. " * 3
    + "\n\n"
    + "A pilot for late-night buses starts in spring and runs until the autumn. " * 3
)
REFUSAL = "I'm sorry, but I can't help with that request."
TWO_PAGES = (
    "<!-- Page number: 1 -->\n\n# The budget\n\n"
    + "The committee reviewed the annual budget and approved the new library wing. " * 3
    + "\n\n<!-- Page number: 2 -->\n\n# The buses\n\n"
    + "A pilot for late-night buses starts in spring and runs until the autumn. " * 3
)


def _body(markdown: str) -> str:
    """The text a model sees: markers as placeholders."""
    return (
        markdown.replace("<!-- Page number: 1 -->", P0)
        .replace("<!-- Page number: 2 -->", P1)
        .strip()
    )


def _answer(cleaned: str, *, cost: float = 0.01) -> litellm.ModelResponse:
    """A billed JSON-in-text answer, as DeepSeek gives one."""
    payload = {
        "cleaned_markdown": cleaned,
        "frontmatter": {"description": "d", "tags": ["t"]},
    }
    response = litellm.ModelResponse(
        id="r",
        model="deepseek-flash",
        object="chat.completion",
        created=0,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"```json\n{json.dumps(payload)}\n```",
                },
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 1000, "completion_tokens": 400, "total_tokens": 1400},
    )
    response._hidden_params = {"total_cost_usd": cost}
    return response


def _text(content: str, *, cost: float = 0.01) -> litellm.ModelResponse:
    response = _answer("", cost=cost)
    response.choices[0].message.content = content  # type: ignore[union-attr]
    return response


class _Router:
    """Answers in order (the last one repeats) and records every request."""

    def __init__(self, *answers: litellm.ModelResponse) -> None:
        self.answers = list(answers)
        self.requests: list[list[dict[str, Any]]] = []
        self.model_list = [
            {"model_name": "default", "litellm_params": {"model": MODEL}}
        ]

    async def acompletion(self, model: str, messages: list[Any], **kwargs: Any) -> Any:
        self.requests.append(messages)
        index = min(len(self.requests), len(self.answers)) - 1
        return self.answers[index]


def _config(**llm: Any) -> LLMConfig:
    return LLMConfig(
        enabled=True,
        model_list=[
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model=MODEL, api_key="sk-test"),
            )
        ],
        **llm,
    )


def _processor(router: _Router, **llm: Any) -> LLMProcessor:
    processor = LLMProcessor(_config(**llm), PromptsConfig(), no_cache=True)
    processor._router = router  # type: ignore[assignment]
    processor._vision_router = router  # type: ignore[assignment]
    return processor


class TestRejectedAnswersAreBilled:
    async def test_unparseable_answers_are_counted_and_charged(self) -> None:
        """Instructor rejects answers that are not JSON and asks again: every
        answer was billed, counts in the usage and toward the cost cap."""
        router = _Router(_text("Sorry, here is the text instead of JSON."))
        processor = _processor(router, max_cost_per_document_usd=0.015)

        with (
            track_llm_spend() as spend,
            pytest.raises(LLMEnhancementDegradedError),
        ):
            await processor.process_document(TABLE_PAGE, "default-table.pdf")

        # Two answers (0.02) pass the $0.015 cap: the third is never sent
        assert len(router.requests) == 2
        assert spend.usage["deepseek-flash"]["requests"] == 2
        assert spend.cost == pytest.approx(0.02)
        assert processor._request_budget.exceeded("default-table.pdf")

    async def test_refusal_is_billed_on_the_failed_item(self, tmp_path: Path) -> None:
        source = tmp_path / "notes.md"
        source.write_text(PLAIN, encoding="utf-8")
        router = _Router(_answer(REFUSAL))
        cfg = MarkitaiConfig()
        cfg.llm = _config(on_failure="fail")
        ctx = ConversionContext(
            input_path=source,
            output_dir=tmp_path / "out",
            config=cfg,
            shared_processor=_processor(router),
        )

        result = await convert_document_core(ctx, 10_000_000)

        assert not result.success
        assert "not a cleanup" in (result.error or "")
        processed = document_process_result(ctx, result)
        assert processed.cost_usd == pytest.approx(0.01)
        assert processed.llm_usage["deepseek-flash"]["requests"] == 1

    async def test_fallback_item_reports_what_it_paid(self, tmp_path: Path) -> None:
        source = tmp_path / "notes.md"
        source.write_text(PLAIN, encoding="utf-8")
        router = _Router(_answer(REFUSAL, cost=0.004))
        cfg = MarkitaiConfig()
        cfg.llm = _config()  # on_failure: fallback
        ctx = ConversionContext(
            input_path=source,
            output_dir=tmp_path / "out",
            config=cfg,
            shared_processor=_processor(router),
        )

        result = await convert_document_core(ctx, 10_000_000)

        assert result.success and ctx.llm_fell_back
        assert ctx.llm_cost == pytest.approx(0.004)
        assert ctx.llm_usage["deepseek-flash"]["requests"] == 1

    async def test_vision_enhancement_refusal_keeps_its_cost(
        self, tmp_path: Path
    ) -> None:
        from markitai.workflow.single import SingleFileWorkflow

        page = tmp_path / "doc.page0001.png"
        page.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
        router = _Router(_text("I can't help with that.", cost=0.03))
        processor = _processor(router)
        cfg = MarkitaiConfig()
        cfg.llm = processor.config
        workflow = SingleFileWorkflow(cfg, processor=processor)
        pages = [{"path": str(page), "page": i} for i in (1, 2)]

        with track_llm_spend() as spend:
            with pytest.raises(LLMEnhancementDegradedError):
                # Two batches of one page: the second is a plain vision call
                await processor.enhance_document_complete(
                    PLAIN, [page, page], source="doc.pdf", max_pages_per_batch=1
                )
            first = spend.cost
            with pytest.raises(LLMEnhancementDegradedError):
                # The workflow clears the failed stage's usage from its context
                await workflow.enhance_with_vision(PLAIN, pages, source="doc.pdf")

        assert first > 0
        assert spend.cost > first
        assert processor.get_context_usage("doc.pdf") == {}
