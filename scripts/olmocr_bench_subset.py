#!/usr/bin/env python3
"""Feasibility harness: score markitai against a SUBSET of olmOCR-bench.

Opt-in, standalone, **not** part of CI or the default test suite (network
+ real OCR work). Downloads a small slice of the ``allenai/olmOCR-bench``
dataset straight from HuggingFace's HTTPS resolve API (no
``huggingface_hub``/``datasets`` dependency needed — this reuses ``httpx``,
already a markitai dependency), converts each PDF with markitai (``--ocr``,
e.g. local RapidOCR — no LLM cost), and scores the output against a
**simplified reimplementation** of a subset of the official rule types.

## Feasibility summary (investigated 2026-08-25)

- **Dataset**: `allenai/olmOCR-bench` on HuggingFace. 1,403 single-page PDFs
  + 7,010 unit-test-style "rules" across 7 splits (``arxiv_math``,
  ``headers_footers``, ``long_tiny_text``, ``multi_column``, ``old_scans``,
  ``old_scans_math``, ``table_tests``). Total size **357 MB**. A newer
  ``allenai/olmOCR-bench-1.5-preview`` extends this to 3,401 PDFs / 28,770
  rules / 15 categories — bigger, not used here.
- **License**: data is ODC-BY-1.0 (AI2 Responsible Use Guidelines apply);
  the `olmocr` toolkit itself is Apache-2.0. Both are commercially
  reusable; ODC-BY only requires attribution.
- **Rule types & official scoring** (read from the upstream
  ``olmocr/bench/tests.py`` source, 2026-08-25):
  - ``present`` / ``absent``: ``rapidfuzz.fuzz.partial_ratio(query,
    content) / 100 >= 1 - max_diffs / len(query)``, with optional
    ``case_sensitive`` and ``first_n``/``last_n`` character-window slicing
    (``content[:first_n] + content[-last_n:]`` when both are set).
    **Reimplemented exactly** below (same library, same formula).
  - ``order``: ``fuzzysearch.find_near_matches(needle, haystack,
    max_l_dist=max_diffs)``; passes if any "before" match starts before any
    "after" match. **Approximated** below using rapidfuzz's best-alignment
    position instead of enumerating every near-match pair (see
    ``_locate``'s docstring) — close but not bit-identical to the official
    result.
  - ``table``: a target cell plus its up/down/left/right/heading neighbors,
    checked against a parsed table structure (upstream
    ``table_parsing.py``). **Not implemented** here.
  - ``math``: LaTeX visual-equivalence via KaTeX DOM bounding boxes,
    rendered through Playwright. **Not implemented** here (needs a browser
    + the upstream KaTeX bundle).
  - Empirically (sampling each split's jsonl), ``old_scans``,
    ``headers_footers``, ``multi_column`` and ``long_tiny_text`` are
    present/absent/order only — fully scorable by this script.
    ``table_tests`` is all ``table``; ``arxiv_math``/``old_scans_math``
    carry ``math`` rules. So this script's honest coverage is roughly
    4 of 7 splits, not the whole benchmark.
- **Conclusion**: acquiring and scoring a small slice with
  present/absent/order rules is feasible today with what a dev checkout
  already has (httpx is a markitai dependency; rapidfuzz comes from the
  dev group, which ``uv run`` installs) — see the run this script
  performs. Reproducing the *authoritative* full-benchmark
  score requires installing AI2's own `olmocr[bench]` toolkit (its own
  dependency tree plus Playwright/Chromium for the math scorer) and
  writing a markitai runner for it; see "Manual steps" below. That is
  real, documented, and not attempted automatically here.

## Manual steps for the authoritative (full) benchmark

This script does not, and will not, install a second PDF/OCR toolkit into
markitai's environment. To get the *official* score instead of this
subset's approximation::

    conda create -n olmocr python=3.11 && conda activate olmocr
    git clone https://github.com/allenai/olmocr.git && cd olmocr
    pip install -e .[bench]
    playwright install chromium

    huggingface-cli download --repo-type dataset --resume-download \\
        allenai/olmOCR-bench --local-dir ./olmOCR-bench

    # Write olmocr/bench/runners/run_markitai.py, modeled on run_marker.py:
    #   def run_markitai(pdf_path: str, page_num: int = 1) -> str:
    #       return markitai.convert(pdf_path, output_dir=None, llm=False,
    #                                ocr=True).markdown
    # (bench PDFs are already single-page, so page_num can be ignored.)

    python -m olmocr.bench.convert markitai --dir ./olmOCR-bench/bench_data
    python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data

## Usage

::

    uv run python scripts/olmocr_bench_subset.py --split old_scans --limit 5
    uv run python scripts/olmocr_bench_subset.py --split headers_footers --limit 5 --no-ocr
    uv run python scripts/olmocr_bench_subset.py --split old_scans --limit 5 --output report.json

    # VLM mode: the --ocr --llm path (vision model reads page images) instead
    # of local RapidOCR — paid, manual-only, refused in CI.
    uv run python scripts/olmocr_bench_subset.py --split old_scans --limit 5 --vlm

PDFs and rule files are cached under ``--cache-dir`` (default: a directory
under the system temp dir) so repeat runs do not re-download. Nothing this
script downloads is ever committed to the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from rapidfuzz import fuzz

HF_DATASET = "allenai/olmOCR-bench"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

SPLITS: tuple[str, ...] = (
    "arxiv_math",
    "headers_footers",
    "long_tiny_text",
    "multi_column",
    "old_scans",
    "old_scans_math",
    "table_tests",
)


def default_cache_dir() -> Path:
    """Default local cache directory (never part of the repository)."""
    return Path(tempfile.gettempdir()) / "markitai-olmocr-bench-cache"


def _download(url: str, dest: Path, client: httpx.Client) -> Path:
    """Download ``url`` to ``dest`` unless already cached."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = client.get(url)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def fetch_split_rules(
    split: str, cache_dir: Path, client: httpx.Client
) -> list[dict[str, Any]]:
    """Download (or reuse the cached copy of) one split's rule jsonl."""
    dest = cache_dir / f"{split}.jsonl"
    _download(f"{HF_RESOLVE_BASE}/bench_data/{split}.jsonl", dest, client)
    rules = []
    for line in dest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rules.append(json.loads(line))
    return rules


def select_pdfs(rules: list[dict[str, Any]], limit: int) -> list[str]:
    """Return up to ``limit`` distinct PDF paths, in first-seen order."""
    seen: list[str] = []
    for rule in rules:
        pdf = rule["pdf"]
        if pdf not in seen:
            seen.append(pdf)
        if len(seen) >= limit:
            break
    return seen


def fetch_pdf(pdf_rel_path: str, cache_dir: Path, client: httpx.Client) -> Path:
    """Download (or reuse the cached copy of) one bench PDF."""
    dest = cache_dir / "pdfs" / pdf_rel_path
    url = f"{HF_RESOLVE_BASE}/bench_data/pdfs/{quote(pdf_rel_path)}"
    return _download(url, dest, client)


def _ci_active() -> bool:
    """Return True when running under a CI automation environment."""
    import os

    return os.environ.get(
        "GITHUB_ACTIONS", ""
    ).strip().lower() == "true" or os.environ.get("CI", "").strip().lower() in (
        "1",
        "true",
    )


def convert_pdf(pdf_path: Path, *, ocr: bool, vlm: bool = False) -> str:
    """Convert one PDF with markitai's public API.

    ``vlm=True`` runs the ``--ocr --llm`` path: page images go to the
    configured vision model (paid, remote, opt-in). ``vlm=False`` (default)
    runs the local RapidOCR path (``llm=False``) — no network.
    """
    import markitai

    output = markitai.convert(
        pdf_path,
        output_dir=None,
        llm=vlm,
        ocr=ocr,
        screenshot=False,
        alt=False,
        desc=False,
    )
    return output.markdown


# ---------------------------------------------------------------------------
# Scoring: present/absent are an exact port of the upstream formula;
# order is a documented approximation. table/math are reported as skipped.
# ---------------------------------------------------------------------------


def _threshold(text: str, max_diffs: int) -> float:
    """Upstream's per-rule pass threshold: ``1 - max_diffs / len(text)``."""
    length = len(text) or 1
    return 1.0 - (max_diffs / length)


def _windowed(content: str, first_n: int | None, last_n: int | None) -> str:
    """Slice to the rule's first_n/last_n character window (upstream semantics)."""
    if first_n and last_n:
        return content[:first_n] + content[-last_n:]
    if first_n:
        return content[:first_n]
    if last_n:
        return content[-last_n:]
    return content


def score_present_absent(rule: dict[str, Any], content: str) -> bool:
    """Exact port of upstream's ``TextPresenceTest``/``TextAbsenceTest``."""
    query = rule["text"]
    case_sensitive = rule.get("case_sensitive", True)
    text = content if case_sensitive else content.lower()
    if not case_sensitive:
        query = query.lower()
    text = _windowed(text, rule.get("first_n"), rule.get("last_n"))
    ratio = fuzz.partial_ratio(query, text) / 100.0
    is_present = ratio >= _threshold(query, rule.get("max_diffs", 0))
    return is_present if rule["type"] == "present" else not is_present


def _locate(content: str, needle: str, max_diffs: int) -> int | None:
    """Approximate position of ``needle`` in ``content``, or None if absent.

    The official scorer (``fuzzysearch.find_near_matches``) enumerates every
    near-match and accepts an order pair if *any* before-match precedes
    *any* after-match. This uses rapidfuzz's single best-aligned window
    instead — cheaper, no new dependency, but not bit-identical: a
    borderline case with multiple candidate matches could disagree with the
    official verdict. Good enough to gauge order preservation for a
    feasibility subset, not authoritative.
    """
    if not needle:
        return None
    alignment = fuzz.partial_ratio_alignment(
        needle, content, score_cutoff=_threshold(needle, max_diffs) * 100
    )
    return alignment.dest_start if alignment is not None else None


def score_order(rule: dict[str, Any], content: str) -> bool:
    """Approximate port of upstream's ``TextOrderTest`` (see ``_locate``)."""
    max_diffs = rule.get("max_diffs", 0)
    before = _locate(content, rule["before"], max_diffs)
    after = _locate(content, rule["after"], max_diffs)
    return before is not None and after is not None and before < after


@dataclass
class RuleOutcome:
    """One rule's verdict against one converted PDF.

    Attributes:
        rule_id: The rule's ``id`` field.
        rule_type: One of the official rule types (``present``, ``absent``,
            ``order``, ``table``, ``math``, ...).
        pdf: The bench PDF's relative path.
        passed: True/False when scored; None when skipped (unsupported type).
        note: Why a rule was skipped, when ``passed`` is None.
    """

    rule_id: str
    rule_type: str
    pdf: str
    passed: bool | None
    note: str | None = None


def score_rule(rule: dict[str, Any], content: str) -> RuleOutcome:
    """Score one rule, or mark it skipped when its type is unsupported."""
    rule_type = rule["type"]
    if rule_type in ("present", "absent"):
        return RuleOutcome(
            rule["id"], rule_type, rule["pdf"], score_present_absent(rule, content)
        )
    if rule_type == "order":
        return RuleOutcome(
            rule["id"], rule_type, rule["pdf"], score_order(rule, content)
        )
    return RuleOutcome(
        rule["id"],
        rule_type,
        rule["pdf"],
        passed=None,
        note=f"rule type {rule_type!r} needs the official olmocr[bench] "
        f"toolkit (KaTeX/table-structure scoring); not implemented here",
    )


def summarize(outcomes: list[RuleOutcome]) -> dict[str, Any]:
    """Aggregate outcomes by rule type: pass rate among scored rules, skip count."""
    by_type: dict[str, list[RuleOutcome]] = {}
    for outcome in outcomes:
        by_type.setdefault(outcome.rule_type, []).append(outcome)

    summary: dict[str, Any] = {}
    for rule_type, group in sorted(by_type.items()):
        scored = [o for o in group if o.passed is not None]
        passed = sum(1 for o in scored if o.passed)
        summary[rule_type] = {
            "total": len(group),
            "scored": len(scored),
            "skipped": len(group) - len(scored),
            "passed": passed,
            "pass_rate": round(passed / len(scored), 4) if scored else None,
        }
    scored_all = [o for o in outcomes if o.passed is not None]
    summary["_overall"] = {
        "total_rules": len(outcomes),
        "scored_rules": len(scored_all),
        "skipped_rules": len(outcomes) - len(scored_all),
        "pass_rate": round(sum(1 for o in scored_all if o.passed) / len(scored_all), 4)
        if scored_all
        else None,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Score markitai against a subset of olmOCR-bench "
        "(present/absent/order rules only; see module docstring)."
    )
    parser.add_argument("--split", choices=SPLITS, default="old_scans")
    parser.add_argument(
        "--limit", type=int, default=5, help="number of distinct PDFs to sample"
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable markitai's OCR path (default: on; these are scanned docs)",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        default=False,
        help="use the --ocr --llm VLM path (vision model reads page images) "
        "instead of local RapidOCR. Requires a configured vision model, costs "
        "money, and refuses to run in a CI environment.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write a JSON report here"
    )
    args = parser.parse_args(argv)

    if args.vlm and _ci_active():
        print(
            "--vlm is a paid, manual-only mode and refuses to run in a CI "
            "environment (GITHUB_ACTIONS/CI set). Run it locally with a "
            "configured vision model.",
            file=sys.stderr,
        )
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {args.cache_dir}")
    if args.vlm:
        print(
            "VLM mode: page images will be sent to your configured vision "
            "model (paid). This is the --ocr --llm path.",
            file=sys.stderr,
        )

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        rules = fetch_split_rules(args.split, args.cache_dir, client)
        all_pdfs = {rule["pdf"] for rule in rules}
        pdfs = select_pdfs(rules, args.limit)
        print(
            f"Split {args.split!r}: {len(rules)} rule(s) across {len(all_pdfs)} PDF(s) "
            f"total; sampling {len(pdfs)}"
        )

        outcomes: list[RuleOutcome] = []
        for pdf_rel in pdfs:
            pdf_path = fetch_pdf(pdf_rel, args.cache_dir, client)
            pdf_rules = [r for r in rules if r["pdf"] == pdf_rel]
            started = time.monotonic()
            try:
                content = convert_pdf(pdf_path, ocr=args.ocr, vlm=args.vlm)
            except Exception as exc:  # noqa: BLE001 - one PDF's failure must not abort the batch
                print(
                    f"  FAILED {pdf_rel}: {type(exc).__name__}: {exc}", file=sys.stderr
                )
                continue
            elapsed = time.monotonic() - started
            outcomes.extend(score_rule(rule, content) for rule in pdf_rules)
            print(
                f"  {pdf_rel}: {len(pdf_rules)} rule(s), "
                f"{elapsed:.1f}s convert, {len(content)} char(s) output"
            )

    summary = summarize(outcomes)
    print(f"\n{'type':<12} {'scored':>7} {'skipped':>8} {'passed':>7}  pass_rate")
    for rule_type, stats in summary.items():
        if rule_type == "_overall":
            continue
        rate = f"{stats['pass_rate']:.2%}" if stats["pass_rate"] is not None else "n/a"
        print(
            f"{rule_type:<12} {stats['scored']:>7} {stats['skipped']:>8} "
            f"{stats['passed']:>7}  {rate}"
        )
    overall = summary["_overall"]
    overall_rate = (
        f"{overall['pass_rate']:.2%}" if overall["pass_rate"] is not None else "n/a"
    )
    print(
        f"\nOverall: {overall['scored_rules']}/{overall['total_rules']} rule(s) scored "
        f"({overall['skipped_rules']} skipped as out of scope), pass rate {overall_rate}"
    )

    if args.output:
        report = {
            "split": args.split,
            "sampled_pdfs": pdfs,
            "ocr": args.ocr,
            "ocr_mode": "vlm" if args.vlm else "rapidocr",
            "summary": summary,
            "outcomes": [asdict(o) for o in outcomes],
        }
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
