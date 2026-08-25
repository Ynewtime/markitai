"""LLM enhancement A/B evaluation harness (base vs. LLM-enhanced conversion).

markitai ships an LLM enhancement pipeline (``llm=True`` / ``--llm``) but
nothing has ever measured what it buys: does the enhanced ``.llm.md`` read
better than the base ``.md``, or does it just look different? This harness
runs a blind, position-debiased A/B judge over paired (base, enhanced)
conversions of the same documents and reports a per-format and overall
"enhanced win rate".

Methodology (the same shape as most LLM-judge evals; reapplied here):

- **Blind**: the judge sees two unlabeled documents ("Document A"/"Document
  B"), never told which is base/enhanced.
- **Position-swap x2**: each pair is judged twice, with the base/enhanced
  order swapped the second time. A verdict only counts as a real signal
  when both swaps agree; disagreement (including either swap returning
  "tie") folds into "tie" — this cancels the well-documented judge
  position bias instead of averaging over it (see ``combine_swap_results``).
- **Per-format aggregation**: results are grouped by source format (pdf,
  docx, ...) as well as an overall/micro summary, plus a macro average
  across formats so one format with many documents cannot dominate the
  headline number (see ``aggregate_by_format``).
- **jsonl checkpointing**: every judged document is appended to the output
  jsonl immediately; a rerun with the same ``--output`` skips documents
  already present, so an interrupted or rate-limited run resumes for free
  instead of re-paying for already-judged documents.
- **Batches API (50% of list price on both OpenAI and Anthropic)**: this
  module can build the batch request file offline (no network call) instead
  of judging synchronously — see ``build_batch_requests``/``--build-batch``.
  Submitting/polling that batch is real, working code against each
  provider's official batch surface (``submit_openai_batch_via_litellm``,
  ``poll_and_download_openai_batch``), but it is never invoked by this
  module's CLI or tests — see the SAFETY note below.

Reuses markitai's own public API (``markitai.aconvert``) to produce the
base/enhanced pair and its config system (``MARKITAI_CONFIG``) to steer
which model the *enhanced* pass uses — both READ-ONLY call sites, exactly
like any other caller of the pipeline; this module never imports
``markitai.llm`` internals.

SAFETY: every judge call costs money once you supply a real ``judge_fn`` /
``--judge-model`` with live credentials. This module makes NO network calls
on import, in its tests (a stub ``JudgeFn`` is injected — see
``tests/unit/test_llm_ab_eval.py``), or by running this file with no
arguments (``--docs`` and ``--output`` are required). Nothing in this
repository's test suite or CI invokes a real judge model.

Usage (after wiring a real judge model + credentials)::

    # cost estimate only -- builds the pairs, makes NO judge calls:
    uv run python packages/markitai/benchmarks/llm_ab_eval.py \\
        --docs report.pdf memo.docx --output /tmp/ab.jsonl --dry-run

    # real run (costs money -- see the cost estimate printed by --dry-run
    # first). OPENAI_API_KEY / ANTHROPIC_API_KEY etc. per litellm's usual
    # resolution:
    uv run python packages/markitai/benchmarks/llm_ab_eval.py \\
        --docs report.pdf memo.docx slides.pptx \\
        --judge-model openai/gpt-5-mini \\
        --output /tmp/llm_ab_results.jsonl

    # Batches API path (50% cheaper; build the request file, no judge call):
    uv run python packages/markitai/benchmarks/llm_ab_eval.py \\
        --docs report.pdf memo.docx --output /tmp/ab.jsonl \\
        --build-batch /tmp/batch_requests.jsonl --judge-model gpt-5-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if __package__ in {None, ""}:  # executed as a script, not as a module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DocumentPair:
    """One document's base and LLM-enhanced conversion.

    Attributes:
        doc_id: Stable identifier for jsonl checkpointing -- the source
            path/URL string as given.
        fmt: Source format label (file extension, lowercase, no dot).
        source: The original input as given.
        base_markdown: Output of ``markitai.aconvert(source, llm=False)``.
        enhanced_markdown: Output of ``markitai.aconvert(source, llm=True)``.
    """

    doc_id: str
    fmt: str
    source: str
    base_markdown: str
    enhanced_markdown: str


@dataclass
class JudgeVerdict:
    """One judge call's raw verdict, before position-swap translation.

    Attributes:
        winner: "A", "B", or "tie" -- always relative to whatever was
            actually sent as Document A / Document B for that call.
        reason: The judge's stated reason, when parseable.
        raw: The judge's full parsed JSON response, for auditability.
    """

    winner: Literal["A", "B", "tie"]
    reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class SwapResult:
    """One position-swap call's verdict, translated to base/enhanced space.

    Attributes:
        order: "base_first" (A=base, B=enhanced) or "enhanced_first"
            (A=enhanced, B=base).
        winner: The judge's raw "A"/"B"/"tie" for that call.
        translated: ``winner`` translated back to "base"/"enhanced"/"tie".
        reason: The judge's stated reason, when parseable.
    """

    order: Literal["base_first", "enhanced_first"]
    winner: Literal["A", "B", "tie"]
    translated: Literal["base", "enhanced", "tie"]
    reason: str | None = None


@dataclass
class DocumentJudgement:
    """The combined, position-debiased verdict for one document.

    Attributes:
        doc_id: Matches the source ``DocumentPair.doc_id``.
        fmt: Source format label.
        verdict: "base", "enhanced", or "tie" (see ``combine_swap_results``).
        agreement: True when both position-swap calls agreed.
        swap_results: The two underlying ``SwapResult`` entries.
    """

    doc_id: str
    fmt: str
    verdict: Literal["base", "enhanced", "tie"]
    agreement: bool
    swap_results: list[SwapResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for one jsonl checkpoint line."""
        return {
            "doc_id": self.doc_id,
            "fmt": self.fmt,
            "verdict": self.verdict,
            "agreement": self.agreement,
            "swap_results": [asdict(s) for s in self.swap_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentJudgement:
        """Reconstruct from one jsonl checkpoint line."""
        return cls(
            doc_id=data["doc_id"],
            fmt=data["fmt"],
            verdict=data["verdict"],
            agreement=data["agreement"],
            swap_results=[SwapResult(**s) for s in data["swap_results"]],
        )


#: Judge call signature: (doc_a_text, doc_b_text) -> verdict. The default
#: implementation is ``litellm_judge`` (via ``make_litellm_judge``); tests
#: inject a stub instead, so no network call is ever needed to exercise the
#: position-swap/aggregation/resume logic.
JudgeFn = Callable[[str, str], Awaitable[JudgeVerdict]]


# ---------------------------------------------------------------------------
# Document pair generation -- markitai's public API, read-only.
# ---------------------------------------------------------------------------


async def build_document_pair(
    source: str | Path, *, config: MarkitaiConfig | None = None
) -> DocumentPair:
    """Convert one document twice (base, then LLM-enhanced) via the public API."""
    import markitai

    base = await markitai.aconvert(source, output_dir=None, llm=False, config=config)
    enhanced = await markitai.aconvert(source, output_dir=None, llm=True, config=config)
    fmt = Path(str(source)).suffix.lstrip(".").lower() or "unknown"
    return DocumentPair(
        doc_id=str(source),
        fmt=fmt,
        source=str(source),
        base_markdown=base.markdown,
        enhanced_markdown=enhanced.llm_markdown or enhanced.markdown,
    )


async def build_document_pairs(
    sources: Sequence[str | Path], *, config: MarkitaiConfig | None = None
) -> list[DocumentPair]:
    """Convert each source into a ``DocumentPair``, sequentially.

    Sequential by design: this is a feasibility/framework tool, not a
    high-throughput pipeline, and markitai's own converters already manage
    their own thread/process pools per call.
    """
    return [await build_document_pair(source, config=config) for source in sources]


# ---------------------------------------------------------------------------
# Judge prompt + response parsing
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert technical editor judging two Markdown conversions of "
    "the same source document. Judge which conversion better preserves the "
    "source's structure, content completeness, and Markdown formatting "
    "quality (headings, tables, lists, reading order). Ignore which one "
    "looks more 'polished' if that polish invents or drops content. "
    'Respond with strict JSON only, no other text: {"winner": "A" | "B" | '
    '"tie", "reason": "<one sentence>"}.'
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_judge_user_prompt(
    doc_a: str, doc_b: str, *, max_chars: int | None = 20000
) -> str:
    """Build the judge's user-turn prompt, truncating each document to ``max_chars``."""
    if max_chars:
        doc_a = doc_a[:max_chars]
        doc_b = doc_b[:max_chars]
    return (
        f"Document A:\n---\n{doc_a}\n---\n\n"
        f"Document B:\n---\n{doc_b}\n---\n\n"
        "Which document is the better Markdown conversion? Respond with the JSON verdict only."
    )


def parse_judge_response(content: str) -> JudgeVerdict:
    """Parse the judge's JSON verdict, tolerating minor formatting noise.

    Falls back to "tie" (with the raw text as ``reason``) when the response
    isn't parseable JSON or doesn't contain a recognizable winner -- an
    unparseable response must never silently count as a win for either side.
    """
    data: Any = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(content)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return JudgeVerdict(
            winner="tie", reason=f"unparseable judge response: {content[:200]!r}"
        )

    raw_winner = str(data.get("winner", "")).strip().upper()
    winner: Literal["A", "B", "tie"]
    if raw_winner == "A":
        winner = "A"
    elif raw_winner == "B":
        winner = "B"
    else:
        winner = "tie"
    reason = data.get("reason")
    return JudgeVerdict(
        winner=winner, reason=reason if isinstance(reason, str) else None, raw=data
    )


async def litellm_judge(
    doc_a: str,
    doc_b: str,
    *,
    model: str,
    max_chars: int | None = 20000,
    **litellm_kwargs: Any,
) -> JudgeVerdict:
    """Real judge call via ``litellm.acompletion``. NETWORK CALL, COSTS MONEY.

    This is the default ``JudgeFn`` implementation, bound to a model via
    ``make_litellm_judge``. Never invoked by this module's own tests or by
    importing this module -- tests inject a stub ``JudgeFn`` instead.
    """
    import litellm

    prompt = build_judge_user_prompt(doc_a, doc_b, max_chars=max_chars)
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        stream=False,
        **litellm_kwargs,
    )
    # Non-streaming call always returns ModelResponse; litellm's return type
    # is a broader union because **litellm_kwargs could in principle carry
    # stream=True, which the type checker can't rule out statically.
    content = cast("Any", response).choices[0].message.content or ""
    return parse_judge_response(content)


def make_litellm_judge(model: str, **litellm_kwargs: Any) -> JudgeFn:
    """Bind ``litellm_judge`` to a model, returning a ready-to-use ``JudgeFn``."""

    async def _judge(doc_a: str, doc_b: str) -> JudgeVerdict:
        return await litellm_judge(doc_a, doc_b, model=model, **litellm_kwargs)

    return _judge


# ---------------------------------------------------------------------------
# Position-swap x2 + combination rule
# ---------------------------------------------------------------------------


def _swap_orders(pair: DocumentPair) -> list[tuple[str, str, str]]:
    """Return the 2 (order_label, doc_a_text, doc_b_text) position-swap calls."""
    return [
        ("base_first", pair.base_markdown, pair.enhanced_markdown),
        ("enhanced_first", pair.enhanced_markdown, pair.base_markdown),
    ]


def _translate(
    order_label: str, winner: Literal["A", "B", "tie"]
) -> Literal["base", "enhanced", "tie"]:
    """Map a raw A/B/tie verdict back to base/enhanced/tie for one swap order."""
    if winner == "tie":
        return "tie"
    if order_label == "base_first":
        return "base" if winner == "A" else "enhanced"
    return "enhanced" if winner == "A" else "base"


def combine_swap_results(
    swaps: Sequence[SwapResult],
) -> tuple[Literal["base", "enhanced", "tie"], bool]:
    """Combine the 2 position-swap verdicts into one, per the debiasing rule.

    Both swaps must translate to the same non-tie winner to declare one;
    any disagreement -- opposite winners (classic position bias) or either
    swap landing on "tie" -- collapses to "tie". This is intentionally
    conservative: a confident verdict here means the judge preferred the
    same side regardless of which slot (A/B) it was shown in.

    Returns:
        (verdict, agreement) where agreement is True iff both swaps
        translated to the same value (including both being "tie").
    """
    translations: set[Literal["base", "enhanced", "tie"]] = {
        s.translated for s in swaps
    }
    if len(translations) == 1:
        return next(iter(translations)), True
    return "tie", False


async def judge_document_pair(
    pair: DocumentPair, judge_fn: JudgeFn
) -> DocumentJudgement:
    """Judge one document pair with both position-swap orders."""
    swaps: list[SwapResult] = []
    for order_label, doc_a, doc_b in _swap_orders(pair):
        verdict = await judge_fn(doc_a, doc_b)
        swaps.append(
            SwapResult(
                order=order_label,  # type: ignore[arg-type]
                winner=verdict.winner,
                translated=_translate(order_label, verdict.winner),
                reason=verdict.reason,
            )
        )
    verdict_label, agreement = combine_swap_results(swaps)
    return DocumentJudgement(
        doc_id=pair.doc_id,
        fmt=pair.fmt,
        verdict=verdict_label,
        agreement=agreement,
        swap_results=swaps,
    )


# ---------------------------------------------------------------------------
# jsonl checkpointing (resume)
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> dict[str, DocumentJudgement]:
    """Return {doc_id: DocumentJudgement} already recorded in ``path``, or {}."""
    if not path.is_file():
        return {}
    completed: dict[str, DocumentJudgement] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        judgement = DocumentJudgement.from_dict(json.loads(line))
        completed[judgement.doc_id] = judgement
    return completed


def append_checkpoint(path: Path, judgement: DocumentJudgement) -> None:
    """Append one judged document to the jsonl checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(judgement.to_dict(), sort_keys=True) + "\n")


async def run_ab_eval(
    pairs: Sequence[DocumentPair],
    judge_fn: JudgeFn,
    checkpoint_path: Path,
    *,
    resume: bool = True,
) -> list[DocumentJudgement]:
    """Judge every pair, skipping ones already in ``checkpoint_path`` when resuming."""
    completed = load_checkpoint(checkpoint_path) if resume else {}
    results = list(completed.values())
    for pair in pairs:
        if pair.doc_id in completed:
            continue
        judgement = await judge_document_pair(pair, judge_fn)
        append_checkpoint(checkpoint_path, judgement)
        results.append(judgement)
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _verdict_stats(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    enhanced = counter.get("enhanced", 0)
    return {
        "total": total,
        "base_wins": counter.get("base", 0),
        "enhanced_wins": enhanced,
        "ties": counter.get("tie", 0),
        "enhanced_win_rate": round(enhanced / total, 4) if total else None,
    }


def aggregate_by_format(judgements: Sequence[DocumentJudgement]) -> dict[str, Any]:
    """Aggregate judgements overall and per source format.

    ``macro_avg_enhanced_win_rate`` averages each format's win rate with
    equal weight (rather than by document count), so one format with many
    documents cannot dominate the headline number -- the explicit
    "aggregate by format" requested alongside the overall/micro summary.
    """
    by_format: dict[str, Counter[str]] = {}
    overall: Counter[str] = Counter()
    for judgement in judgements:
        by_format.setdefault(judgement.fmt, Counter())[judgement.verdict] += 1
        overall[judgement.verdict] += 1

    format_stats = {
        fmt: _verdict_stats(counter) for fmt, counter in sorted(by_format.items())
    }
    rates = [
        s["enhanced_win_rate"]
        for s in format_stats.values()
        if s["enhanced_win_rate"] is not None
    ]
    return {
        "document_count": len(judgements),
        "overall": _verdict_stats(overall),
        "by_format": format_stats,
        "macro_avg_enhanced_win_rate": round(sum(rates) / len(rates), 4)
        if rates
        else None,
    }


async def evaluate_documents(
    sources: Sequence[str | Path],
    judge_fn: JudgeFn,
    checkpoint_path: Path,
    *,
    resume: bool = True,
    config: MarkitaiConfig | None = None,
) -> tuple[list[DocumentJudgement], dict[str, Any]]:
    """End-to-end: build pairs, judge them, aggregate. The main library entry point."""
    pairs = await build_document_pairs(sources, config=config)
    judgements = await run_ab_eval(pairs, judge_fn, checkpoint_path, resume=resume)
    return judgements, aggregate_by_format(judgements)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


@dataclass
class CostEstimate:
    """A rough judge-pass cost estimate at synchronous (non-batch) list price.

    Attributes:
        document_count: Number of documents.
        judge_calls: Total judge calls (2 per document -- position-swap x2).
        estimated_input_tokens: Rough input token total across all calls.
        estimated_output_tokens: Rough output token total across all calls.
        estimated_cost_usd: ``input_tokens/1e6 * input_price + output_tokens/1e6 * output_price``.
    """

    document_count: int
    judge_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float


def estimate_judge_cost(
    pairs: Sequence[DocumentPair],
    *,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    max_chars: int | None = 20000,
    chars_per_token: float = 4.0,
    output_tokens_per_call: int = 60,
) -> CostEstimate:
    """Estimate the judge pass's cost: documents x pages(chars) x judge token price.

    A ~4-characters/token rule of thumb for English text -- good enough to
    size a run, not a substitute for a provider's own tokenizer. Halve
    ``estimated_cost_usd`` for a Batches API run: both OpenAI and Anthropic
    price batch inference at 50% of synchronous list price.
    """
    total_chars = 0
    for pair in pairs:
        doc_a = pair.base_markdown[:max_chars] if max_chars else pair.base_markdown
        doc_b = (
            pair.enhanced_markdown[:max_chars] if max_chars else pair.enhanced_markdown
        )
        total_chars += 2 * (
            len(doc_a) + len(doc_b)
        )  # 2 calls/doc, each sends both documents

    # +~100 prompt-overhead tokens per call (system prompt + instructions).
    input_tokens = int(total_chars / chars_per_token) + len(pairs) * 2 * 100
    output_tokens = len(pairs) * 2 * output_tokens_per_call
    cost = (input_tokens / 1e6) * input_price_per_mtok + (
        output_tokens / 1e6
    ) * output_price_per_mtok
    return CostEstimate(
        document_count=len(pairs),
        judge_calls=len(pairs) * 2,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_cost_usd=round(cost, 4),
    )


# ---------------------------------------------------------------------------
# Batches API: offline request building + result parsing (no network).
# Real submit/poll functions are further below, clearly marked.
# ---------------------------------------------------------------------------


def build_openai_batch_requests(
    pairs: Sequence[DocumentPair], model: str, *, max_chars: int | None = 20000
) -> list[dict[str, Any]]:
    """Build an OpenAI Batches API (``/v1/chat/completions``) request list.

    2 requests per pair (one per position-swap order); ``custom_id`` encodes
    ``"<doc_id>::<order>"`` so ``judgements_from_batch_results`` can
    translate results back to base/enhanced without re-reading the pairs.
    Consumable by ``submit_openai_batch_via_litellm`` or the OpenAI/litellm
    SDK's own batch upload directly.
    """
    requests: list[dict[str, Any]] = []
    for pair in pairs:
        for order_label, doc_a, doc_b in _swap_orders(pair):
            requests.append(
                {
                    "custom_id": f"{pair.doc_id}::{order_label}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": build_judge_user_prompt(
                                    doc_a, doc_b, max_chars=max_chars
                                ),
                            },
                        ],
                    },
                }
            )
    return requests


def build_anthropic_batch_requests(
    pairs: Sequence[DocumentPair],
    model: str,
    *,
    max_tokens: int = 200,
    max_chars: int | None = 20000,
) -> list[dict[str, Any]]:
    """Build an Anthropic Message Batches API request list.

    markitai does not depend on the ``anthropic`` SDK, so this only builds
    the request list -- submit it yourself::

        import anthropic
        client = anthropic.Anthropic()
        batch = client.messages.batches.create(
            requests=build_anthropic_batch_requests(pairs, "claude-haiku-4-5")
        )
    """
    requests: list[dict[str, Any]] = []
    for pair in pairs:
        for order_label, doc_a, doc_b in _swap_orders(pair):
            requests.append(
                {
                    "custom_id": f"{pair.doc_id}::{order_label}",
                    "params": {
                        "model": model,
                        "max_tokens": max_tokens,
                        "system": _JUDGE_SYSTEM_PROMPT,
                        "messages": [
                            {
                                "role": "user",
                                "content": build_judge_user_prompt(
                                    doc_a, doc_b, max_chars=max_chars
                                ),
                            }
                        ],
                    },
                }
            )
    return requests


def build_batch_requests(
    pairs: Sequence[DocumentPair],
    model: str,
    *,
    provider: Literal["openai", "anthropic"] = "openai",
    max_chars: int | None = 20000,
    anthropic_max_tokens: int = 200,
) -> list[dict[str, Any]]:
    """Dispatch to ``build_openai_batch_requests``/``build_anthropic_batch_requests``."""
    if provider == "openai":
        return build_openai_batch_requests(pairs, model, max_chars=max_chars)
    if provider == "anthropic":
        return build_anthropic_batch_requests(
            pairs, model, max_tokens=anthropic_max_tokens, max_chars=max_chars
        )
    raise ValueError(
        f"unknown batch provider: {provider!r}"
    )  # pragma: no cover - argparse gates this


def write_batch_jsonl(requests: Sequence[dict[str, Any]], path: Path) -> None:
    """Write a batch request list as jsonl (one request per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")


def parse_openai_batch_output(path: Path) -> dict[str, JudgeVerdict]:
    """Parse a completed OpenAI Batches API output file into ``{custom_id: JudgeVerdict}``."""
    results: dict[str, JudgeVerdict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        body = entry.get("response", {}).get("body", {})
        choices = body.get("choices") or []
        content = choices[0]["message"]["content"] if choices else ""
        results[entry["custom_id"]] = parse_judge_response(content)
    return results


def parse_anthropic_batch_results(
    results: Iterable[dict[str, Any]],
) -> dict[str, JudgeVerdict]:
    """Parse Anthropic Message Batches API results into ``{custom_id: JudgeVerdict}``.

    Accepts an iterable of result dicts shaped like the ``anthropic`` SDK's
    ``client.messages.batches.results(batch_id)`` entries (or their
    ``.model_dump()``).
    """
    parsed: dict[str, JudgeVerdict] = {}
    for entry in results:
        custom_id = entry["custom_id"]
        result = entry.get("result", {})
        if result.get("type") != "succeeded":
            parsed[custom_id] = JudgeVerdict(
                winner="tie",
                reason=f"batch entry not succeeded: {result.get('type')!r}",
            )
            continue
        blocks = result.get("message", {}).get("content", [])
        text = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
        parsed[custom_id] = parse_judge_response(text)
    return parsed


def judgements_from_batch_results(
    pairs: Sequence[DocumentPair], verdicts_by_custom_id: dict[str, JudgeVerdict]
) -> list[DocumentJudgement]:
    """Recombine per-swap batch verdicts into ``DocumentJudgement``s.

    Applies the same ``combine_swap_results`` rule as the synchronous path.
    A pair missing either of its 2 swap results (batch still running, or a
    failed entry not yet retried) is skipped rather than guessed at.
    """
    judgements: list[DocumentJudgement] = []
    for pair in pairs:
        swaps: list[SwapResult] = []
        for order_label, _doc_a, _doc_b in _swap_orders(pair):
            verdict = verdicts_by_custom_id.get(f"{pair.doc_id}::{order_label}")
            if verdict is None:
                continue
            swaps.append(
                SwapResult(
                    order=order_label,  # type: ignore[arg-type]
                    winner=verdict.winner,
                    translated=_translate(order_label, verdict.winner),
                    reason=verdict.reason,
                )
            )
        if len(swaps) != 2:
            continue
        verdict_label, agreement = combine_swap_results(swaps)
        judgements.append(
            DocumentJudgement(
                doc_id=pair.doc_id,
                fmt=pair.fmt,
                verdict=verdict_label,
                agreement=agreement,
                swap_results=swaps,
            )
        )
    return judgements


# ---------------------------------------------------------------------------
# Real Batches API submit/poll (OpenAI-compatible, via litellm).
#
# NETWORK + BILLING. Never invoked by this module's CLI, by importing this
# module, or by its test suite -- a caller must explicitly call these. The
# exact response-wrapper shape (``afile_content``'s return value) is
# documented from litellm's signatures but not exercised against a live
# batch here, since doing so requires a real paid Batches run.
# ---------------------------------------------------------------------------

#: custom_llm_provider values litellm accepts for acreate_file/acreate_batch.
_SubmitProvider = Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm"]
#: custom_llm_provider values litellm accepts for aretrieve_batch/afile_content
#: (a superset of _SubmitProvider -- also allows "anthropic").
_PollProvider = Literal[
    "openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "anthropic"
]


async def submit_openai_batch_via_litellm(
    input_jsonl_path: Path, *, custom_llm_provider: _SubmitProvider = "openai"
) -> str:
    """Upload + submit a batch job via litellm. NETWORK CALL, COSTS MONEY.

    Returns the provider's batch id; poll it with
    ``poll_and_download_openai_batch``.
    """
    import litellm

    file_obj = await litellm.acreate_file(
        file=input_jsonl_path, purpose="batch", custom_llm_provider=custom_llm_provider
    )
    batch = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider=custom_llm_provider,
    )
    return batch.id


async def poll_and_download_openai_batch(
    batch_id: str,
    output_path: Path,
    *,
    custom_llm_provider: _PollProvider = "openai",
    poll_interval_s: float = 30.0,
    timeout_s: float = 24 * 3600,
) -> Path:
    """Poll a submitted batch to completion, then download its output. NETWORK CALL, COSTS MONEY."""
    import litellm

    elapsed = 0.0
    while True:
        batch = await litellm.aretrieve_batch(
            batch_id, custom_llm_provider=custom_llm_provider
        )
        if batch.status == "completed":
            break
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"batch {batch_id} ended with status={batch.status!r}")
        if elapsed >= timeout_s:
            raise TimeoutError(
                f"batch {batch_id} still {batch.status!r} after {timeout_s}s"
            )
        await asyncio.sleep(poll_interval_s)
        elapsed += poll_interval_s

    if not batch.output_file_id:
        raise RuntimeError(f"batch {batch_id} completed with no output_file_id")
    content = await litellm.afile_content(
        batch.output_file_id, custom_llm_provider=custom_llm_provider
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # afile_content's return type covers a streaming variant too (only
    # reachable with stream=True, not passed above); the non-streaming
    # wrapper exposes .content (raw bytes) per litellm's HTTP file helpers.
    output_path.write_bytes(cast("Any", content).content)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind position-debiased A/B judge: markitai's base vs. "
        "LLM-enhanced conversion. See the module docstring before running "
        "with a real --judge-model -- it costs money."
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        required=True,
        metavar="PATH",
        help="local files to evaluate",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="jsonl checkpoint path (resumable)"
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="litellm model string for the judge, e.g. openai/gpt-5-mini or "
        "anthropic/claude-haiku-4-5 (required unless --dry-run/--build-batch)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing --output checkpoint",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="per-document truncation before judging",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=None,
        help="markitai config file for the enhanced pass (sets MARKITAI_CONFIG)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build pairs + print a cost estimate; make NO judge calls",
    )
    parser.add_argument(
        "--build-batch",
        type=Path,
        default=None,
        help="write a Batches API request jsonl here instead of judging synchronously (no network call)",
    )
    parser.add_argument(
        "--batch-provider", choices=["openai", "anthropic"], default="openai"
    )
    parser.add_argument(
        "--input-price-per-mtok",
        type=float,
        default=0.25,
        help="judge input $/1M tokens, for cost estimate",
    )
    parser.add_argument(
        "--output-price-per-mtok",
        type=float,
        default=2.0,
        help="judge output $/1M tokens, for cost estimate",
    )
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if args.llm_config:
        os.environ["MARKITAI_CONFIG"] = str(args.llm_config)

    print(f"Building {len(args.docs)} base/enhanced pair(s)...")
    pairs = await build_document_pairs(args.docs)
    for pair in pairs:
        print(
            f"  {pair.doc_id} [{pair.fmt}]: base {len(pair.base_markdown)} char(s), "
            f"enhanced {len(pair.enhanced_markdown)} char(s)"
        )

    if args.build_batch:
        model = args.judge_model or "REPLACE_WITH_JUDGE_MODEL"
        requests = build_batch_requests(
            pairs, model, provider=args.batch_provider, max_chars=args.max_chars
        )
        write_batch_jsonl(requests, args.build_batch)
        print(
            f"\nWrote {len(requests)} batch request(s) to {args.build_batch} "
            f"({args.batch_provider} format). No network call was made."
        )
        return 0

    estimate = estimate_judge_cost(
        pairs,
        input_price_per_mtok=args.input_price_per_mtok,
        output_price_per_mtok=args.output_price_per_mtok,
        max_chars=args.max_chars,
    )
    print(
        f"\nCost estimate ({estimate.judge_calls} judge call(s), synchronous list price): "
        f"~{estimate.estimated_input_tokens:,} input + {estimate.estimated_output_tokens:,} output "
        f"token(s), ~${estimate.estimated_cost_usd:.4f}. Halve for a Batches API run (--build-batch)."
    )

    if args.dry_run:
        print("\n--dry-run: no judge calls made.")
        return 0

    if not args.judge_model:
        print(
            "\n--judge-model is required unless --dry-run or --build-batch.",
            file=sys.stderr,
        )
        return 1

    judge_fn = make_litellm_judge(args.judge_model)
    print(
        f"\nJudging with {args.judge_model} (2 call(s)/doc, position-swap debiased)..."
    )
    judgements = await run_ab_eval(
        pairs, judge_fn, args.output, resume=not args.no_resume
    )
    summary = aggregate_by_format(judgements)
    print(json.dumps(summary, indent=2))
    print(f"\nCheckpoint: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_arg_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
