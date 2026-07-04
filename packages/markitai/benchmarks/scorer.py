"""Heuristic quality scorer for HTML -> Markdown conversion.

Simplified port of marker's heuristic scorer
(``benchmarks/overall/scorers/heuristic.py`` in VikParuchuri/marker):

1. Split the expected (ground-truth) markdown into blocks on blank lines.
2. Fuzzy-align each block inside the produced output with
   ``rapidfuzz.fuzz.partial_ratio_alignment`` (score cutoff 70).
3. Overall score = length-weighted mean block alignment * 0.8
   + order preservation (Kendall-tau on matched block positions) * 0.2,
   on a 0-100 scale.

``rapidfuzz`` is a dev-group dependency only; this module must never be
imported from ``markitai`` runtime code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

# Minimum partial-ratio for a block to count as aligned (marker uses 70).
ALIGNMENT_THRESHOLD = 70

# Weights from marker's heuristic: fuzzy match dominates, order refines.
MATCH_WEIGHT = 0.8
ORDER_WEIGHT = 0.2


@dataclass
class ScoreResult:
    """Result of scoring produced markdown against expected markdown.

    Attributes:
        score: Overall quality score, 0-100.
        match_score: Length-weighted mean block alignment, 0-100.
        order_score: Kendall-tau order preservation of matched blocks, 0-100.
        block_scores: Per-block alignment scores (same order as expected
            blocks), 0-100 each.
    """

    score: float
    match_score: float
    order_score: float
    block_scores: list[float] = field(default_factory=list)


def split_blocks(markdown: str) -> list[str]:
    """Split markdown into non-empty blocks separated by blank lines."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", markdown)]
    return [b for b in blocks if b]


def _clean(text: str) -> str:
    """Normalize markdown for fuzzy comparison.

    Simplified version of marker's MarkdownCleaner: strip common markdown
    decoration characters, collapse whitespace, lowercase.
    """
    text = re.sub(r"```[^\n]*", "", text)  # code fence markers (keep code body)
    text = re.sub(r"[#*_`>|]", "", text)  # emphasis / heading / table chrome
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _kendall_tau(correct_order: list[int], actual_order: list[int]) -> float:
    """Kendall-tau rank correlation rescaled to 0-100 (marker's variant)."""
    n = len(correct_order)
    if n <= 1:
        return 100.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            correct_sign = correct_order[i] - correct_order[j]
            actual_sign = actual_order[i] - actual_order[j]
            if correct_sign * actual_sign > 0:
                concordant += 1
            elif correct_sign * actual_sign < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    tau = (concordant - discordant) / total_pairs
    return (tau + 1) / 2 * 100  # rescale [-1, 1] -> [0, 100]


def score_markdown(expected: str, produced: str) -> ScoreResult:
    """Score produced markdown against expected markdown, 0-100.

    Args:
        expected: Ground-truth markdown (blocks separated by blank lines).
        produced: Markdown emitted by the conversion pipeline.

    Returns:
        ScoreResult with the overall score and its components.
    """
    gt_blocks = [_clean(b) for b in split_blocks(expected)]
    gt_blocks = [b for b in gt_blocks if b]
    if not gt_blocks:
        # Nothing expected: any output (or none) is a trivial pass.
        return ScoreResult(score=100.0, match_score=100.0, order_score=100.0)

    haystack = _clean(produced)
    if not haystack:
        return ScoreResult(
            score=0.0,
            match_score=0.0,
            order_score=0.0,
            block_scores=[0.0] * len(gt_blocks),
        )

    block_scores: list[float] = []
    starts: list[int] = []
    for block in gt_blocks:
        alignment = fuzz.partial_ratio_alignment(
            block, haystack, score_cutoff=ALIGNMENT_THRESHOLD
        )
        if alignment is None:
            block_scores.append(0.0)
            starts.append(0)
        else:
            block_scores.append(float(alignment.score))
            starts.append(alignment.dest_start)

    correct_order = list(range(len(gt_blocks)))
    actual_order = sorted(correct_order, key=lambda i: starts[i])
    order_score = _kendall_tau(correct_order, actual_order)

    weights = [len(b) for b in gt_blocks]
    match_score = sum(s * w for s, w in zip(block_scores, weights)) / max(
        1, sum(weights)
    )

    score = match_score * MATCH_WEIGHT + order_score * ORDER_WEIGHT
    return ScoreResult(
        score=round(score, 2),
        match_score=round(match_score, 2),
        order_score=round(order_score, 2),
        block_scores=block_scores,
    )


def score_with_llm_judge(expected: str, produced: str) -> ScoreResult:
    """Stub for LLM-judge scoring. Intentionally NOT implemented.

    How it would plug in: this function would share ``ScoreResult`` with the
    heuristic scorer, so ``webextract_quality.main`` could take a
    ``--scorer {heuristic,llm}`` flag and dispatch here. The implementation
    would send (expected, produced) pairs to a judge model with a rubric
    (content completeness, structure fidelity, noise/chrome leakage), parse a
    0-100 score per fixture, and cache responses keyed by content hash so
    reruns are free. It must remain dev-only, opt-in, and never run in the
    default test suite or CI without an explicit API key.

    Raises:
        NotImplementedError: always; heuristic scoring is the only backend.
    """
    raise NotImplementedError(
        "LLM-judge scoring is a documented extension point, not implemented."
    )
