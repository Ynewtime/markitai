"""Unit tests for the LLM enhancement A/B evaluation harness.

Every test uses a stub/monkeypatched judge -- never a real model call. Covers
the position-swap x2 debiasing rule, per-format aggregation, jsonl
resume/checkpointing, offline Batches API request/response (de)serialization,
cost estimation, and (via monkeypatched ``litellm``) that the real-call and
real-batch code paths are wired correctly without ever touching the network.
See ``benchmarks/llm_ab_eval.py`` for the harness itself -- it is dev
tooling, not shipped in the wheel, imported the same way
``test_webextract_quality_benchmark.py`` imports ``benchmarks.webextract_quality``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# benchmarks/ is dev tooling next to src/, not an installed package.
_PKG_DIR = Path(__file__).parents[2]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from benchmarks import llm_ab_eval as mod
from benchmarks.llm_ab_eval import (
    DocumentJudgement,
    DocumentPair,
    JudgeVerdict,
    SwapResult,
)


def _pair(
    doc_id: str = "a.pdf",
    fmt: str = "pdf",
    base: str = "BASE TEXT",
    enhanced: str = "ENHANCED TEXT",
) -> DocumentPair:
    return DocumentPair(
        doc_id=doc_id,
        fmt=fmt,
        source=doc_id,
        base_markdown=base,
        enhanced_markdown=enhanced,
    )


def _scripted_judge(
    *winners: str,
) -> tuple[Callable[[str, str], Awaitable[JudgeVerdict]], list[tuple[str, str]]]:
    """A stub JudgeFn that returns ``winners`` in order and records every call's args."""
    calls: list[tuple[str, str]] = []
    it: Iterator[str] = iter(winners)

    async def judge(doc_a: str, doc_b: str) -> JudgeVerdict:
        calls.append((doc_a, doc_b))
        return JudgeVerdict(winner=it.__next__())  # type: ignore[arg-type]

    return judge, calls


def _raising_judge() -> Callable[[str, str], Awaitable[JudgeVerdict]]:
    async def judge(doc_a: str, doc_b: str) -> JudgeVerdict:
        raise AssertionError(
            "judge_fn must not be called for already-checkpointed documents"
        )

    return judge


class TestCombineSwapResults:
    """The position-swap x2 debiasing rule (see combine_swap_results docstring)."""

    def _swap(self, translated: str) -> SwapResult:
        return SwapResult(order="base_first", winner="A", translated=translated)  # type: ignore[arg-type]

    def test_both_agree_base(self) -> None:
        verdict, agreement = mod.combine_swap_results(
            [self._swap("base"), self._swap("base")]
        )
        assert (verdict, agreement) == ("base", True)

    def test_both_agree_enhanced(self) -> None:
        verdict, agreement = mod.combine_swap_results(
            [self._swap("enhanced"), self._swap("enhanced")]
        )
        assert (verdict, agreement) == ("enhanced", True)

    def test_both_tie(self) -> None:
        verdict, agreement = mod.combine_swap_results(
            [self._swap("tie"), self._swap("tie")]
        )
        assert (verdict, agreement) == ("tie", True)

    def test_opposite_winners_is_position_bias_and_collapses_to_tie(self) -> None:
        verdict, agreement = mod.combine_swap_results(
            [self._swap("base"), self._swap("enhanced")]
        )
        assert (verdict, agreement) == ("tie", False)

    def test_one_tie_one_definite_is_not_confident(self) -> None:
        verdict, agreement = mod.combine_swap_results(
            [self._swap("tie"), self._swap("base")]
        )
        assert (verdict, agreement) == ("tie", False)


class TestTranslate:
    def test_base_first_a_is_base_b_is_enhanced(self) -> None:
        assert mod._translate("base_first", "A") == "base"
        assert mod._translate("base_first", "B") == "enhanced"

    def test_enhanced_first_a_is_enhanced_b_is_base(self) -> None:
        assert mod._translate("enhanced_first", "A") == "enhanced"
        assert mod._translate("enhanced_first", "B") == "base"

    def test_tie_is_tie_regardless_of_order(self) -> None:
        assert mod._translate("base_first", "tie") == "tie"
        assert mod._translate("enhanced_first", "tie") == "tie"


class TestJudgeDocumentPair:
    async def test_calls_judge_twice_with_swapped_order(self) -> None:
        judge, calls = _scripted_judge("A", "A")  # A wins both times
        pair = _pair(base="BASE TEXT", enhanced="ENHANCED TEXT")

        judgement = await mod.judge_document_pair(pair, judge)

        assert calls == [("BASE TEXT", "ENHANCED TEXT"), ("ENHANCED TEXT", "BASE TEXT")]
        # "A" both times means base_first->base, enhanced_first->enhanced: disagreement.
        assert judgement.verdict == "tie"
        assert judgement.agreement is False

    async def test_consistent_preference_for_enhanced_wins(self) -> None:
        # base_first: B (=enhanced) wins. enhanced_first: A (=enhanced) wins. Agrees.
        judge, _ = _scripted_judge("B", "A")
        judgement = await mod.judge_document_pair(_pair(), judge)
        assert judgement.verdict == "enhanced"
        assert judgement.agreement is True

    async def test_consistent_preference_for_base_wins(self) -> None:
        # base_first: A (=base) wins. enhanced_first: B (=base) wins. Agrees.
        judge, _ = _scripted_judge("A", "B")
        judgement = await mod.judge_document_pair(_pair(), judge)
        assert judgement.verdict == "base"
        assert judgement.agreement is True

    async def test_preserves_doc_id_and_format(self) -> None:
        judge, _ = _scripted_judge("tie", "tie")
        judgement = await mod.judge_document_pair(
            _pair(doc_id="report.docx", fmt="docx"), judge
        )
        assert judgement.doc_id == "report.docx"
        assert judgement.fmt == "docx"
        assert len(judgement.swap_results) == 2


class TestCheckpointRoundtrip:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        judgement = DocumentJudgement(
            doc_id="x.pdf",
            fmt="pdf",
            verdict="enhanced",
            agreement=True,
            swap_results=[
                SwapResult(
                    order="base_first",
                    winner="B",
                    translated="enhanced",
                    reason="clearer",
                ),  # type: ignore[arg-type]
                SwapResult(
                    order="enhanced_first",
                    winner="A",
                    translated="enhanced",
                    reason=None,
                ),  # type: ignore[arg-type]
            ],
        )
        restored = DocumentJudgement.from_dict(
            json.loads(json.dumps(judgement.to_dict()))
        )
        assert restored == judgement


class TestRunAbEvalResume:
    async def test_writes_one_jsonl_line_per_document(self, tmp_path: Path) -> None:
        judge, calls = _scripted_judge(*(["tie"] * 6))  # 3 docs x 2 swaps
        pairs = [_pair(f"{i}.pdf") for i in range(3)]
        checkpoint = tmp_path / "ab.jsonl"

        results = await mod.run_ab_eval(pairs, judge, checkpoint)

        assert len(results) == 3
        assert len(calls) == 6
        lines = checkpoint.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert {json.loads(line)["doc_id"] for line in lines} == {
            "0.pdf",
            "1.pdf",
            "2.pdf",
        }

    async def test_resume_skips_already_checkpointed_documents(
        self, tmp_path: Path
    ) -> None:
        checkpoint = tmp_path / "ab.jsonl"
        first_judge, first_calls = _scripted_judge(*(["tie"] * 4))  # 2 docs
        pairs = [_pair("0.pdf"), _pair("1.pdf")]
        await mod.run_ab_eval(pairs, first_judge, checkpoint)
        assert len(first_calls) == 4

        # Rerun on the exact same pairs with a judge that raises if invoked:
        # nothing should be judged again.
        results = await mod.run_ab_eval(
            pairs, _raising_judge(), checkpoint, resume=True
        )
        assert len(results) == 2
        assert {r.doc_id for r in results} == {"0.pdf", "1.pdf"}

    async def test_resume_only_judges_new_documents(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "ab.jsonl"
        first_judge, _ = _scripted_judge(*(["tie"] * 2))
        await mod.run_ab_eval([_pair("0.pdf")], first_judge, checkpoint)

        second_judge, second_calls = _scripted_judge(*(["tie"] * 2))
        results = await mod.run_ab_eval(
            [_pair("0.pdf"), _pair("1.pdf")], second_judge, checkpoint, resume=True
        )

        assert len(results) == 2
        assert len(second_calls) == 2  # only the new document was judged
        assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 2

    async def test_no_resume_reevaluates_everything(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "ab.jsonl"
        first_judge, _ = _scripted_judge(*(["tie"] * 2))
        await mod.run_ab_eval([_pair("0.pdf")], first_judge, checkpoint)

        second_judge, second_calls = _scripted_judge(*(["tie"] * 2))
        results = await mod.run_ab_eval(
            [_pair("0.pdf")], second_judge, checkpoint, resume=False
        )

        assert len(results) == 1
        assert (
            len(second_calls) == 2
        )  # re-judged (2 swap calls) despite already being in the file

    def test_load_checkpoint_last_line_wins_for_a_doc_id(self, tmp_path: Path) -> None:
        # A --no-resume rerun appends rather than truncating (see run_ab_eval's
        # docstring): the file can carry >1 line per doc_id. Readers must take
        # the latest.
        checkpoint = tmp_path / "ab.jsonl"
        stale = DocumentJudgement("x.pdf", "pdf", "base", True, [])
        fresh = DocumentJudgement("x.pdf", "pdf", "enhanced", True, [])
        with checkpoint.open("w", encoding="utf-8") as f:
            f.write(json.dumps(stale.to_dict()) + "\n")
            f.write(json.dumps(fresh.to_dict()) + "\n")

        loaded = mod.load_checkpoint(checkpoint)
        assert loaded["x.pdf"].verdict == "enhanced"

    def test_load_checkpoint_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert mod.load_checkpoint(tmp_path / "nope.jsonl") == {}


class TestAggregateByFormat:
    def _judgement(self, doc_id: str, fmt: str, verdict: str) -> DocumentJudgement:
        return DocumentJudgement(doc_id, fmt, verdict, agreement=True, swap_results=[])  # type: ignore[arg-type]

    def test_empty_input(self) -> None:
        summary = mod.aggregate_by_format([])
        assert summary["document_count"] == 0
        assert summary["overall"]["total"] == 0
        assert summary["overall"]["enhanced_win_rate"] is None
        assert summary["by_format"] == {}
        assert summary["macro_avg_enhanced_win_rate"] is None

    def test_overall_and_per_format_counts(self) -> None:
        judgements = [
            self._judgement("1.pdf", "pdf", "enhanced"),
            self._judgement("2.pdf", "pdf", "base"),
            self._judgement("3.docx", "docx", "enhanced"),
            self._judgement("4.docx", "docx", "enhanced"),
        ]
        summary = mod.aggregate_by_format(judgements)

        assert summary["document_count"] == 4
        assert summary["overall"] == {
            "total": 4,
            "base_wins": 1,
            "enhanced_wins": 3,
            "ties": 0,
            "enhanced_win_rate": 0.75,
        }
        assert summary["by_format"]["pdf"]["enhanced_win_rate"] == 0.5
        assert summary["by_format"]["docx"]["enhanced_win_rate"] == 1.0

    def test_macro_average_weighs_formats_equally_not_by_doc_count(self) -> None:
        # pdf: 1/1 enhanced (rate 1.0). docx: 1 enhanced of 3 (rate 1/3).
        # Macro average = mean(1.0, 1/3), NOT the doc-count-weighted 2/4.
        judgements = [
            self._judgement("1.pdf", "pdf", "enhanced"),
            self._judgement("2.docx", "docx", "enhanced"),
            self._judgement("3.docx", "docx", "base"),
            self._judgement("4.docx", "docx", "base"),
        ]
        summary = mod.aggregate_by_format(judgements)
        assert summary["macro_avg_enhanced_win_rate"] == pytest.approx(
            (1.0 + 1 / 3) / 2, abs=1e-4
        )
        assert (
            summary["overall"]["enhanced_win_rate"] == 0.5
        )  # micro/doc-count-weighted, for contrast


class TestParseJudgeResponse:
    def test_valid_json(self) -> None:
        verdict = mod.parse_judge_response(
            '{"winner": "A", "reason": "clearer headings"}'
        )
        assert verdict.winner == "A"
        assert verdict.reason == "clearer headings"
        assert verdict.raw == {"winner": "A", "reason": "clearer headings"}

    def test_json_normalizes_lowercase_winner(self) -> None:
        assert mod.parse_judge_response('{"winner": "b"}').winner == "B"

    def test_extracts_json_from_surrounding_prose(self) -> None:
        text = 'Sure, here is my verdict:\n```json\n{"winner": "B", "reason": "ok"}\n```\nThanks!'
        verdict = mod.parse_judge_response(text)
        assert verdict.winner == "B"
        assert verdict.reason == "ok"

    def test_unparseable_text_is_a_safe_tie(self) -> None:
        verdict = mod.parse_judge_response("I refuse to answer in JSON.")
        assert verdict.winner == "tie"
        assert verdict.reason is not None and "unparseable" in verdict.reason

    def test_unexpected_winner_value_normalizes_to_tie(self) -> None:
        assert mod.parse_judge_response('{"winner": "C"}').winner == "tie"

    def test_missing_winner_key_is_tie(self) -> None:
        assert mod.parse_judge_response('{"reason": "no winner field"}').winner == "tie"


class TestEstimateJudgeCost:
    def test_matches_hand_computed_formula(self) -> None:
        pair = _pair(base="a" * 100, enhanced="b" * 100)
        estimate = mod.estimate_judge_cost(
            [pair],
            input_price_per_mtok=1.0,
            output_price_per_mtok=2.0,
            max_chars=None,
            chars_per_token=4.0,
            output_tokens_per_call=10,
        )
        # 2 calls; each call sends both docs (100 + 100 chars) = 200 chars/call,
        # 400 chars total -> /4 chars-per-token = 100, + 2*100 overhead = 300.
        assert estimate.document_count == 1
        assert estimate.judge_calls == 2
        assert estimate.estimated_input_tokens == 300
        assert estimate.estimated_output_tokens == 20  # 2 calls * 10
        expected_cost = (300 / 1e6) * 1.0 + (20 / 1e6) * 2.0
        assert estimate.estimated_cost_usd == pytest.approx(round(expected_cost, 4))

    def test_max_chars_truncates_before_counting(self) -> None:
        long_pair = _pair(base="x" * 100_000, enhanced="y" * 100_000)
        untruncated = mod.estimate_judge_cost(
            [long_pair],
            input_price_per_mtok=1.0,
            output_price_per_mtok=1.0,
            max_chars=None,
        )
        truncated = mod.estimate_judge_cost(
            [long_pair],
            input_price_per_mtok=1.0,
            output_price_per_mtok=1.0,
            max_chars=100,
        )
        assert truncated.estimated_input_tokens < untruncated.estimated_input_tokens

    def test_scales_linearly_with_document_count(self) -> None:
        pairs = [_pair(f"{i}.pdf", base="x" * 50, enhanced="y" * 50) for i in range(5)]
        estimate = mod.estimate_judge_cost(
            pairs, input_price_per_mtok=1.0, output_price_per_mtok=1.0
        )
        assert estimate.judge_calls == 10
        assert estimate.document_count == 5


class TestBatchRequestBuilding:
    def test_openai_batch_has_two_requests_per_pair(self) -> None:
        pairs = [_pair("a.pdf"), _pair("b.docx", fmt="docx")]
        requests = mod.build_openai_batch_requests(pairs, "gpt-5-mini")
        assert len(requests) == 4
        custom_ids = {r["custom_id"] for r in requests}
        assert custom_ids == {
            "a.pdf::base_first",
            "a.pdf::enhanced_first",
            "b.docx::base_first",
            "b.docx::enhanced_first",
        }
        first = requests[0]
        assert first["method"] == "POST"
        assert first["url"] == "/v1/chat/completions"
        assert first["body"]["model"] == "gpt-5-mini"
        assert first["body"]["messages"][0]["role"] == "system"

    def test_anthropic_batch_uses_params_shape(self) -> None:
        requests = mod.build_anthropic_batch_requests(
            [_pair("a.pdf")], "claude-haiku-4-5"
        )
        assert len(requests) == 2
        assert requests[0]["custom_id"] == "a.pdf::base_first"
        assert requests[0]["params"]["model"] == "claude-haiku-4-5"
        assert requests[0]["params"]["system"] == mod._JUDGE_SYSTEM_PROMPT
        assert requests[0]["params"]["messages"][0]["role"] == "user"

    def test_dispatch_by_provider(self) -> None:
        pairs = [_pair("a.pdf")]
        openai_requests = mod.build_batch_requests(pairs, "m", provider="openai")
        anthropic_requests = mod.build_batch_requests(pairs, "m", provider="anthropic")
        assert "method" in openai_requests[0]
        assert "params" in anthropic_requests[0]

    def test_dispatch_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="unknown batch provider"):
            mod.build_batch_requests([_pair()], "m", provider="bogus")  # type: ignore[arg-type]

    def test_write_batch_jsonl_roundtrip(self, tmp_path: Path) -> None:
        requests = mod.build_openai_batch_requests([_pair("a.pdf")], "gpt-5-mini")
        path = tmp_path / "batch.jsonl"
        mod.write_batch_jsonl(requests, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(line) for line in lines] == requests


class TestBatchResultParsing:
    def test_parse_openai_batch_output(self, tmp_path: Path) -> None:
        path = tmp_path / "output.jsonl"
        entries = [
            {
                "custom_id": "a.pdf::base_first",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": '{"winner": "A", "reason": "r1"}'}}
                        ]
                    }
                },
            },
            {
                "custom_id": "a.pdf::enhanced_first",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": '{"winner": "B", "reason": "r2"}'}}
                        ]
                    }
                },
            },
        ]
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
        )

        parsed = mod.parse_openai_batch_output(path)
        assert parsed["a.pdf::base_first"].winner == "A"
        assert parsed["a.pdf::enhanced_first"].winner == "B"

    def test_parse_anthropic_batch_results(self) -> None:
        results = [
            {
                "custom_id": "a.pdf::base_first",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [{"type": "text", "text": '{"winner": "A"}'}]
                    },
                },
            },
            {"custom_id": "a.pdf::enhanced_first", "result": {"type": "errored"}},
        ]
        parsed = mod.parse_anthropic_batch_results(results)
        assert parsed["a.pdf::base_first"].winner == "A"
        assert parsed["a.pdf::enhanced_first"].winner == "tie"
        assert "errored" in (parsed["a.pdf::enhanced_first"].reason or "")

    def test_judgements_from_batch_results_matches_synchronous_combine_rule(
        self,
    ) -> None:
        pairs = [_pair("a.pdf")]
        verdicts = {
            "a.pdf::base_first": JudgeVerdict(winner="B"),  # base_first B -> enhanced
            "a.pdf::enhanced_first": JudgeVerdict(
                winner="A"
            ),  # enhanced_first A -> enhanced
        }
        judgements = mod.judgements_from_batch_results(pairs, verdicts)
        assert len(judgements) == 1
        assert judgements[0].verdict == "enhanced"
        assert judgements[0].agreement is True

    def test_judgements_from_batch_results_skips_incomplete_pairs(self) -> None:
        pairs = [_pair("a.pdf"), _pair("b.pdf")]
        verdicts = {
            "a.pdf::base_first": JudgeVerdict(winner="A")
        }  # b.pdf has no results at all
        judgements = mod.judgements_from_batch_results(pairs, verdicts)
        assert judgements == []  # a.pdf is missing its enhanced_first swap too


class TestBuildDocumentPair:
    async def test_calls_public_api_with_llm_false_then_true_and_uses_llm_markdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import markitai
        from markitai.api import ConversionOutput

        calls: list[dict[str, Any]] = []

        async def fake_aconvert(source: Any, **kwargs: Any) -> ConversionOutput:
            calls.append({"source": source, **kwargs})
            if kwargs["llm"]:
                return ConversionOutput(
                    source=str(source), markdown="BASE", llm_markdown="ENHANCED"
                )
            return ConversionOutput(source=str(source), markdown="BASE")

        monkeypatch.setattr(markitai, "aconvert", fake_aconvert)

        pair = await mod.build_document_pair("report.pdf")

        assert [c["llm"] for c in calls] == [False, True]
        assert pair.doc_id == "report.pdf"
        assert pair.fmt == "pdf"
        assert pair.base_markdown == "BASE"
        assert pair.enhanced_markdown == "ENHANCED"

    async def test_falls_back_to_markdown_when_llm_markdown_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import markitai
        from markitai.api import ConversionOutput

        async def fake_aconvert(source: Any, **kwargs: Any) -> ConversionOutput:
            return ConversionOutput(
                source=str(source), markdown="ONLY BASE", llm_markdown=None
            )

        monkeypatch.setattr(markitai, "aconvert", fake_aconvert)

        pair = await mod.build_document_pair("memo.docx")
        assert pair.enhanced_markdown == "ONLY BASE"

    async def test_build_document_pairs_preserves_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import markitai
        from markitai.api import ConversionOutput

        async def fake_aconvert(source: Any, **kwargs: Any) -> ConversionOutput:
            return ConversionOutput(source=str(source), markdown=f"md:{source}")

        monkeypatch.setattr(markitai, "aconvert", fake_aconvert)

        pairs = await mod.build_document_pairs(["a.pdf", "b.docx", "c.pptx"])
        assert [p.doc_id for p in pairs] == ["a.pdf", "b.docx", "c.pptx"]
        assert [p.fmt for p in pairs] == ["pdf", "docx", "pptx"]


class TestLitellmJudgeWiring:
    """Monkeypatches litellm.acompletion -- proves the real JudgeFn is wired
    correctly without ever making a network call."""

    async def test_builds_prompt_and_parses_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        captured: dict[str, Any] = {}

        class _FakeMessage:
            content = '{"winner": "B", "reason": "more complete"}'

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        async def fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _FakeResponse()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        verdict = await mod.litellm_judge(
            "DOC A TEXT", "DOC B TEXT", model="openai/gpt-5-mini"
        )

        assert verdict.winner == "B"
        assert verdict.reason == "more complete"
        assert captured["model"] == "openai/gpt-5-mini"
        assert captured["stream"] is False
        assert "DOC A TEXT" in captured["messages"][1]["content"]
        assert "DOC B TEXT" in captured["messages"][1]["content"]

    async def test_make_litellm_judge_binds_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        class _FakeMessage:
            content = '{"winner": "tie"}'

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        captured_models = []

        async def fake_acompletion(**kwargs: Any) -> Any:
            captured_models.append(kwargs["model"])
            return _FakeResponse()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        judge_fn = mod.make_litellm_judge("anthropic/claude-haiku-4-5")
        await judge_fn("a", "b")
        assert captured_models == ["anthropic/claude-haiku-4-5"]

    async def test_retries_without_temperature_when_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reasoning models that reject temperature=0 get a no-temperature retry."""
        import litellm

        calls: list[dict[str, Any]] = []

        class _FakeMessage:
            content = '{"winner": "A", "reason": "clearer"}'

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        async def fake_acompletion(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if calls.__len__() == 1:
                raise litellm.exceptions.BadRequestError(
                    message="Unsupported value: 'temperature' does not support 0",
                    model="openai/gpt-5.6-luna",
                    llm_provider="openai",
                )
            return _FakeResponse()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        verdict = await mod.litellm_judge("DOC A", "DOC B", model="openai/gpt-5.6-luna")

        assert verdict.winner == "A"
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]

    async def test_non_temperature_error_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-temperature BadRequestError still propagates."""
        import litellm

        async def fake_acompletion(**kwargs: Any) -> Any:
            raise litellm.exceptions.BadRequestError(
                message="bad JSON body",
                model="openai/gpt-5.6-luna",
                llm_provider="openai",
            )

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        with pytest.raises(litellm.exceptions.BadRequestError):
            await mod.litellm_judge("a", "b", model="openai/gpt-5.6-luna")


class TestBatchSubmitPollWiring:
    """Monkeypatches litellm's batch functions -- proves submit/poll
    orchestration is correct without ever making a network call or
    spending money. See the module SAFETY note: these are never invoked
    by the CLI or by importing the module."""

    async def test_submit_uploads_then_creates_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        async def fake_acreate_file(**kwargs: Any) -> Any:
            class _File:
                id = "file_123"

            return _File()

        async def fake_acreate_batch(**kwargs: Any) -> Any:
            assert kwargs["input_file_id"] == "file_123"

            class _Batch:
                id = "batch_456"

            return _Batch()

        monkeypatch.setattr(litellm, "acreate_file", fake_acreate_file)
        monkeypatch.setattr(litellm, "acreate_batch", fake_acreate_batch)

        batch_id = await mod.submit_openai_batch_via_litellm(
            tmp_path / "requests.jsonl"
        )
        assert batch_id == "batch_456"

    async def test_poll_downloads_after_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        class _Batch:
            def __init__(self, status: str) -> None:
                self.status = status
                self.output_file_id = "outfile" if status == "completed" else None

        statuses = iter(["in_progress", "completed"])

        async def fake_aretrieve_batch(batch_id: str, **kwargs: Any) -> Any:
            return _Batch(next(statuses))

        class _Content:
            content = b'{"custom_id": "x", "response": {}}\n'

        async def fake_afile_content(file_id: str, **kwargs: Any) -> Any:
            assert file_id == "outfile"
            return _Content()

        monkeypatch.setattr(litellm, "aretrieve_batch", fake_aretrieve_batch)
        monkeypatch.setattr(litellm, "afile_content", fake_afile_content)
        monkeypatch.setattr(mod.asyncio, "sleep", _instant_sleep)

        output_path = tmp_path / "out.jsonl"
        result = await mod.poll_and_download_openai_batch(
            "batch_456", output_path, poll_interval_s=0.01
        )
        assert result == output_path
        assert output_path.read_bytes() == b'{"custom_id": "x", "response": {}}\n'

    async def test_poll_raises_on_failed_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        class _Batch:
            status = "failed"
            output_file_id = None

        async def fake_aretrieve_batch(batch_id: str, **kwargs: Any) -> Any:
            return _Batch()

        monkeypatch.setattr(litellm, "aretrieve_batch", fake_aretrieve_batch)

        with pytest.raises(RuntimeError, match="failed"):
            await mod.poll_and_download_openai_batch(
                "batch_456", tmp_path / "out.jsonl"
            )

    async def test_poll_raises_timeout_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm

        class _Batch:
            status = "in_progress"
            output_file_id = None

        async def fake_aretrieve_batch(batch_id: str, **kwargs: Any) -> Any:
            return _Batch()

        monkeypatch.setattr(litellm, "aretrieve_batch", fake_aretrieve_batch)
        monkeypatch.setattr(mod.asyncio, "sleep", _instant_sleep)

        with pytest.raises(TimeoutError):
            await mod.poll_and_download_openai_batch(
                "batch_456", tmp_path / "out.jsonl", poll_interval_s=1.0, timeout_s=0.0
            )


async def _instant_sleep(_seconds: float) -> None:
    """Replaces asyncio.sleep in polling tests so they run instantly."""
    return None


class TestCliDryRunAndBuildBatch:
    """The CLI never calls a judge unless --judge-model is given without
    --dry-run/--build-batch; these tests monkeypatch pair-building (no real
    conversions) and assert no judge call is ever attempted."""

    async def test_dry_run_makes_no_judge_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pairs = [_pair("a.pdf")]

        async def fake_build_pairs(sources: Any, **kwargs: Any) -> list[DocumentPair]:
            return pairs

        monkeypatch.setattr(mod, "build_document_pairs", fake_build_pairs)

        def fail_if_called(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("dry-run must not construct a judge")

        monkeypatch.setattr(mod, "make_litellm_judge", fail_if_called)

        args = mod._build_arg_parser().parse_args(
            ["--docs", "a.pdf", "--output", str(tmp_path / "out.jsonl"), "--dry-run"]
        )
        exit_code = await mod._main_async(args)
        assert exit_code == 0
        assert not (tmp_path / "out.jsonl").exists()

    async def test_build_batch_writes_file_and_skips_judging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pairs = [_pair("a.pdf")]

        async def fake_build_pairs(sources: Any, **kwargs: Any) -> list[DocumentPair]:
            return pairs

        monkeypatch.setattr(mod, "build_document_pairs", fake_build_pairs)
        monkeypatch.setattr(
            mod,
            "make_litellm_judge",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("must not judge")
            ),
        )

        batch_path = tmp_path / "batch.jsonl"
        args = mod._build_arg_parser().parse_args(
            [
                "--docs",
                "a.pdf",
                "--output",
                str(tmp_path / "out.jsonl"),
                "--build-batch",
                str(batch_path),
                "--judge-model",
                "gpt-5-mini",
            ]
        )
        exit_code = await mod._main_async(args)
        assert exit_code == 0
        assert batch_path.is_file()
        assert len(batch_path.read_text(encoding="utf-8").splitlines()) == 2

    async def test_missing_judge_model_without_dry_run_or_batch_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_build_pairs(sources: Any, **kwargs: Any) -> list[DocumentPair]:
            return [_pair("a.pdf")]

        monkeypatch.setattr(mod, "build_document_pairs", fake_build_pairs)

        args = mod._build_arg_parser().parse_args(
            ["--docs", "a.pdf", "--output", str(tmp_path / "out.jsonl")]
        )
        exit_code = await mod._main_async(args)
        assert exit_code == 1
