"""Conversion-quality benchmark for markitai's HTML -> Markdown pipeline.

Scores ``extract_web_content`` output against the defuddle ground-truth
corpus (``tests/defuddle_fixtures/``) with a heuristic fuzzy-alignment
scorer (see ``scorer.py``, ported from marker's benchmark methodology).
Unlike the pytest parity suite (exact contracts, pass/fail), this harness
produces a continuous 0-100 score per fixture so quality drift is
measurable across changes.

Usage (from the repository root)::

    uv run python packages/markitai/benchmarks/webextract_quality.py

or as a module (from ``packages/markitai``)::

    uv run python -m benchmarks.webextract_quality

Options::

    [FIXTURE ...]        score only the named fixture stems
    --output PATH        where to write the JSON results
                         (default: benchmarks/results/latest.json, gitignored)
    --update-baseline    also write benchmarks/results/baseline.json
                         (committed; regenerate deliberately)
    --check              fail (exit 1) when any fixture or the corpus mean
                         drops below its guardrail floor
                         (benchmarks/guardrails.json); meant for full runs
    --update-guardrails  regenerate benchmarks/guardrails.json from this
                         run's scores (floor = score * 0.9, xberg-style;
                         committed; regenerate deliberately)
    --no-color           disable ANSI colors in the table

The table prints each fixture's score plus the delta against the committed
baseline (green/red when beyond +/-2 points), then the aggregate mean.
Exit code is 0 on success, 1 if any fixture failed to extract.

Local fixtures: ``benchmarks/local_fixtures/`` (``fixtures/`` + ``expected/``)
holds markitai-owned regression fixtures that are NOT part of the defuddle
upstream corpus (which is synced verbatim — see tests/defuddle_fixtures/
VERSION). Their expected .md files are self-baselines: generated from the
pipeline's own output at the time the fixture was added, so any later score
drop flags a regression. They are scored and reported under a separate
"local" aggregate and never included in the defuddle-corpus mean, keeping
that mean comparable across baseline generations.

This is a manual / CI-cron tool: the full-corpus run is intentionally NOT
part of the default pytest suite (a fast smoke test covers the scorer math
and runner JSON output on a few fixtures).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # executed as a script, not as a module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.scorer import ScoreResult, score_markdown
from markitai.webextract import extract_web_content

_PKG_DIR = Path(__file__).resolve().parent.parent
_FIXTURE_DIR = _PKG_DIR / "tests" / "defuddle_fixtures"
_HTML_DIR = _FIXTURE_DIR / "fixtures"
_EXPECTED_DIR = _FIXTURE_DIR / "expected"
_LOCAL_FIXTURE_DIR = Path(__file__).resolve().parent / "local_fixtures"
_LOCAL_HTML_DIR = _LOCAL_FIXTURE_DIR / "fixtures"
_LOCAL_EXPECTED_DIR = _LOCAL_FIXTURE_DIR / "expected"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_PATH = _RESULTS_DIR / "baseline.json"
LATEST_PATH = _RESULTS_DIR / "latest.json"
GUARDRAILS_PATH = Path(__file__).resolve().parent / "guardrails.json"

# Guardrail floors are 90% of the score at generation time (xberg's
# threshold_factor): loose enough to absorb scorer jitter, tight enough
# to catch real extraction regressions.
THRESHOLD_FACTOR = 0.9

# Fixtures deviating more than this from baseline are highlighted.
_DELTA_HIGHLIGHT = 2.0

_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_RESET = "\x1b[0m"


# ---------------------------------------------------------------------------
# Fixture loading helpers.
# These mirror tests/integration/test_defuddle_parity_quality.py so the
# benchmark feeds the pipeline exactly like the parity suite does.
# ---------------------------------------------------------------------------


def discover_fixtures() -> list[str]:
    """Return fixture stems that have both HTML input and expected markdown."""
    return _discover(_HTML_DIR, _EXPECTED_DIR)


def discover_local_fixtures() -> list[str]:
    """Return local (markitai-owned, self-baseline) fixture stems."""
    return _discover(_LOCAL_HTML_DIR, _LOCAL_EXPECTED_DIR)


def _discover(html_dir: Path, expected_dir: Path) -> list[str]:
    if not html_dir.is_dir() or not expected_dir.is_dir():
        return []
    html_stems = {p.stem for p in html_dir.glob("*.html")}
    expected_stems = {p.stem for p in expected_dir.glob("*.md")}
    return sorted(html_stems & expected_stems)


def _extract_comment_url(html: str) -> str | None:
    """Extract URL from a leading HTML comment like ``<!-- {"url": "..."} -->``."""
    match = re.match(r"\s*<!--\s*(\{.*?\})\s*-->", html)
    if match:
        try:
            data = json.loads(match.group(1))
            url = data.get("url")
            if isinstance(url, str) and url:
                return url
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def _extract_og_url(html: str) -> str | None:
    """Extract og:url value from raw HTML."""
    match = re.search(
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:url["\']',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _extract_canonical_url(html: str) -> str | None:
    """Extract canonical URL from ``<link rel="canonical">``."""
    match = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']canonical["\']',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _url_from_filename(stem: str) -> str:
    """Construct a plausible URL from a fixture filename."""
    parts = stem.split("--", 1)
    slug = parts[1] if len(parts) > 1 else parts[0]
    return f"https://example.com/{slug}"


def infer_url(html: str, stem: str) -> str:
    """Infer the fixture URL: comment > og:url > canonical > filename."""
    return (
        _extract_comment_url(html)
        or _extract_og_url(html)
        or _extract_canonical_url(html)
        or _url_from_filename(stem)
    )


def parse_expected_body(text: str) -> str:
    """Strip the leading ```json metadata block from an expected .md file."""
    match = re.match(r"```json\s*\n(.*?)\n```\s*\n?(.*)", text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def score_fixture(
    stem: str,
    html_dir: Path = _HTML_DIR,
    expected_dir: Path = _EXPECTED_DIR,
) -> tuple[ScoreResult, str | None]:
    """Extract one fixture and score it against the expected markdown.

    Returns:
        (result, error) — error is an exception summary if extraction blew
        up, in which case the score is 0.
    """
    html = (html_dir / f"{stem}.html").read_text(encoding="utf-8")
    expected_body = parse_expected_body(
        (expected_dir / f"{stem}.md").read_text(encoding="utf-8")
    )
    try:
        extracted = extract_web_content(html, infer_url(html, stem))
        produced = extracted.markdown or ""
    except Exception as exc:  # pragma: no cover - defensive, reported in table
        return ScoreResult(0.0, 0.0, 0.0), f"{type(exc).__name__}: {exc}"
    return score_markdown(expected_body, produced), None


def load_baseline(
    path: Path = BASELINE_PATH, key: str = "fixtures"
) -> dict[str, float]:
    """Return {fixture stem: baseline score}, or {} if no baseline exists.

    Args:
        path: Baseline JSON path.
        key: Which payload section to read ("fixtures" or "local_fixtures").
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            stem: float(entry["score"]) for stem, entry in data.get(key, {}).items()
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}


def _score_set(stems: list[str], html_dir: Path, expected_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stem in stems:
        score, error = score_fixture(stem, html_dir, expected_dir)
        entry: dict[str, Any] = {
            "score": score.score,
            "match_score": score.match_score,
            "order_score": score.order_score,
        }
        if error is not None:
            entry["error"] = error
        results[stem] = entry
    return results


def _mean(results: dict[str, Any]) -> float:
    scores = [entry["score"] for entry in results.values()]
    return round(sum(scores) / len(scores), 2) if scores else 0.0


def run_benchmark(
    fixtures: list[str], local_fixtures: list[str] | None = None
) -> dict[str, Any]:
    """Score every fixture and return the JSON-serializable results payload.

    Local fixtures are scored the same way but aggregated separately
    (``aggregate.local_mean_score``) so ``aggregate.mean_score`` stays
    comparable with historical defuddle-corpus baselines.
    """
    results = _score_set(fixtures, _HTML_DIR, _EXPECTED_DIR)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scorer": "heuristic (rapidfuzz partial-ratio, marker-style)",
        "fixture_count": len(results),
        "aggregate": {"mean_score": _mean(results)},
        "fixtures": results,
    }

    if local_fixtures:
        local_results = _score_set(local_fixtures, _LOCAL_HTML_DIR, _LOCAL_EXPECTED_DIR)
        payload["local_fixture_count"] = len(local_results)
        payload["aggregate"]["local_mean_score"] = _mean(local_results)
        payload["local_fixtures"] = local_results

    return payload


# ---------------------------------------------------------------------------
# Quality guardrails (xberg-style: per-fixture floors + corpus-mean floor).
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """One guardrail breach: a fixture (or aggregate) below its floor.

    ``score`` is ``None`` when the guardrailed fixture was not scored at
    all in this run (removed/renamed fixture, or a partial run).
    """

    name: str
    score: float | None
    floor: float

    def __str__(self) -> str:
        scored = f"{self.score:.2f}" if self.score is not None else "missing"
        return f"{self.name}: score {scored} < floor {self.floor:.2f}"


def generate_guardrails(
    payload: dict[str, Any], threshold_factor: float = THRESHOLD_FACTOR
) -> dict[str, Any]:
    """Build the guardrails document from a benchmark results payload."""
    guardrails: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold_factor": threshold_factor,
        "mean_floor": round(payload["aggregate"]["mean_score"] * threshold_factor, 2),
        "fixtures": {
            stem: {"min_score": round(entry["score"] * threshold_factor, 2)}
            for stem, entry in payload["fixtures"].items()
        },
    }
    if payload.get("local_fixtures"):
        guardrails["local_mean_floor"] = round(
            payload["aggregate"]["local_mean_score"] * threshold_factor, 2
        )
        guardrails["local_fixtures"] = {
            stem: {"min_score": round(entry["score"] * threshold_factor, 2)}
            for stem, entry in payload["local_fixtures"].items()
        }
    return guardrails


def load_guardrails(path: Path = GUARDRAILS_PATH) -> dict[str, Any] | None:
    """Return the guardrails document, or None if the file doesn't exist."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_guardrails(
    payload: dict[str, Any], guardrails: dict[str, Any]
) -> list[Violation]:
    """Compare a results payload against guardrail floors.

    Flags: any guardrailed fixture scoring below its ``min_score``, any
    guardrailed fixture missing from the run, and the corpus/local mean
    dropping below its floor. Fixtures present in the run but absent from
    the guardrails are NOT violations (they get floors on the next
    ``--update-guardrails``).
    """
    violations: list[Violation] = []
    for section in ("fixtures", "local_fixtures"):
        scored = payload.get(section, {})
        for stem, contract in guardrails.get(section, {}).items():
            floor = float(contract["min_score"])
            entry = scored.get(stem)
            if entry is None:
                violations.append(Violation(stem, None, floor))
            elif entry["score"] < floor:
                violations.append(Violation(stem, entry["score"], floor))

    mean_floor = guardrails.get("mean_floor")
    if mean_floor is not None:
        mean = payload["aggregate"]["mean_score"]
        if mean < mean_floor:
            violations.append(Violation("MEAN", mean, float(mean_floor)))

    local_floor = guardrails.get("local_mean_floor")
    local_mean = payload["aggregate"].get("local_mean_score")
    if local_floor is not None and local_mean is not None and local_mean < local_floor:
        violations.append(Violation("LOCAL MEAN", local_mean, float(local_floor)))

    return violations


def _format_delta(delta: float | None, use_color: bool) -> str:
    if delta is None:
        return "   (new)"
    text = f"{delta:+8.2f}"
    if use_color and delta > _DELTA_HIGHLIGHT:
        return f"{_GREEN}{text}{_RESET}"
    if use_color and delta < -_DELTA_HIGHLIGHT:
        return f"{_RED}{text}{_RESET}"
    return text


def print_table(
    payload: dict[str, Any],
    baseline: dict[str, float],
    use_color: bool,
    local_baseline: dict[str, float] | None = None,
) -> None:
    """Print the per-fixture score table and the aggregate line(s)."""
    all_stems = list(payload["fixtures"]) + list(payload.get("local_fixtures", {}))
    name_width = max([len(stem) for stem in all_stems] + [len("fixture")])
    header = f"{'fixture':<{name_width}}  {'score':>7}  {'order':>7}  {'delta':>8}"
    print(header)
    print("-" * len(header))
    for stem, entry in payload["fixtures"].items():
        _print_fixture_row(stem, entry, baseline, name_width, use_color)
    print("-" * len(header))

    mean = payload["aggregate"]["mean_score"]
    common = [s for s in payload["fixtures"] if s in baseline]
    if common:
        baseline_mean = sum(baseline[s] for s in common) / len(common)
        agg_delta = _format_delta(round(mean - baseline_mean, 2), use_color)
        print(
            f"{'MEAN':<{name_width}}  {mean:>7.2f}  {'':>7}  {agg_delta}"
            f"  (baseline mean over {len(common)} shared fixtures:"
            f" {baseline_mean:.2f})"
        )
    else:
        print(f"{'MEAN':<{name_width}}  {mean:>7.2f}  (no baseline for comparison)")

    local_results = payload.get("local_fixtures")
    if not local_results:
        return
    local_baseline = local_baseline or {}
    print()
    print("local fixtures (self-baseline; excluded from the corpus mean)")
    print("-" * len(header))
    for stem, entry in local_results.items():
        _print_fixture_row(stem, entry, local_baseline, name_width, use_color)
    print("-" * len(header))
    local_mean = payload["aggregate"]["local_mean_score"]
    print(f"{'LOCAL MEAN':<{name_width}}  {local_mean:>7.2f}")


def _print_fixture_row(
    stem: str,
    entry: dict[str, Any],
    baseline: dict[str, float],
    name_width: int,
    use_color: bool,
) -> None:
    delta = entry["score"] - baseline[stem] if stem in baseline else None
    line = (
        f"{stem:<{name_width}}  {entry['score']:>7.2f}  "
        f"{entry['order_score']:>7.2f}  {_format_delta(delta, use_color)}"
    )
    if "error" in entry:
        line += f"  EXTRACTION FAILED ({entry['error']})"
    print(line)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Score markitai HTML -> Markdown output against the "
        "defuddle ground-truth corpus (0-100 per fixture)."
    )
    parser.add_argument(
        "fixtures",
        nargs="*",
        help="fixture stems to score (default: full corpus)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LATEST_PATH,
        help="path for the JSON results (default: benchmarks/results/latest.json)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="also write benchmarks/results/baseline.json (committed)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when any fixture or the mean drops below its guardrail "
        "floor (benchmarks/guardrails.json); meant for full-corpus runs",
    )
    parser.add_argument(
        "--update-guardrails",
        action="store_true",
        help="regenerate benchmarks/guardrails.json from this run's scores "
        f"(floor = score * {THRESHOLD_FACTOR})",
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = parser.parse_args(argv)

    all_fixtures = discover_fixtures()
    all_local = discover_local_fixtures()
    if not all_fixtures:
        print(f"No fixtures found under {_FIXTURE_DIR}", file=sys.stderr)
        return 1
    if args.fixtures:
        known = set(all_fixtures) | set(all_local)
        unknown = sorted(set(args.fixtures) - known)
        if unknown:
            print(f"Unknown fixtures: {', '.join(unknown)}", file=sys.stderr)
            return 1
        selected = [s for s in all_fixtures if s in set(args.fixtures)]
        selected_local = [s for s in all_local if s in set(args.fixtures)]
    else:
        selected = all_fixtures
        selected_local = all_local

    payload = run_benchmark(selected, selected_local)
    use_color = not args.no_color and sys.stdout.isatty()
    print_table(
        payload,
        load_baseline(),
        use_color,
        local_baseline=load_baseline(key="local_fixtures"),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"\nResults written to {args.output}")
    if args.update_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"Baseline updated at {BASELINE_PATH}")
    if args.update_guardrails:
        guardrails = generate_guardrails(payload)
        GUARDRAILS_PATH.write_text(
            json.dumps(guardrails, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(f"Guardrails updated at {GUARDRAILS_PATH}")

    failed = [s for s, e in payload["fixtures"].items() if "error" in e]
    failed += [s for s, e in payload.get("local_fixtures", {}).items() if "error" in e]
    if failed:
        return 1

    if args.check:
        guardrails = load_guardrails(GUARDRAILS_PATH)
        if guardrails is None:
            print(
                f"No guardrails file at {GUARDRAILS_PATH}; "
                "run with --update-guardrails to generate it.",
                file=sys.stderr,
            )
            return 1
        violations = check_guardrails(payload, guardrails)
        if violations:
            print(f"\nGuardrail violations ({len(violations)}):", file=sys.stderr)
            for violation in violations:
                print(f"  {violation}", file=sys.stderr)
            return 1
        floors = len(guardrails.get("fixtures", {})) + len(
            guardrails.get("local_fixtures", {})
        )
        print(
            f"\nGuardrails check passed: {floors} fixture floors, "
            f"mean floor {guardrails.get('mean_floor', 0.0):.2f} "
            f"(threshold_factor {guardrails.get('threshold_factor', THRESHOLD_FACTOR)})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
