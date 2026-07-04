"""Smoke tests for the dev-only webextract quality benchmark.

Covers the heuristic scorer math on synthetic cases and checks that the
runner produces valid JSON on a handful of fixtures. The full-corpus run
is intentionally NOT part of the default test suite — it is a manual /
CI-cron tool (see ``benchmarks/webextract_quality.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# benchmarks/ is dev tooling next to src/, not an installed package.
_PKG_DIR = Path(__file__).parents[2]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from benchmarks import webextract_quality
from benchmarks.scorer import score_markdown, score_with_llm_judge, split_blocks

_BLOCK_A = "The quick brown fox jumps over the lazy dog near the river bank."
_BLOCK_B = "Markdown conversion quality is measured with fuzzy block alignment."
_BLOCK_C = "Order preservation is scored with a Kendall tau rank correlation."
_EXPECTED = f"{_BLOCK_A}\n\n{_BLOCK_B}\n\n{_BLOCK_C}"


class TestHeuristicScorer:
    def test_split_blocks_on_blank_lines(self) -> None:
        assert split_blocks(_EXPECTED) == [_BLOCK_A, _BLOCK_B, _BLOCK_C]
        assert split_blocks("  \n\n\n") == []

    def test_perfect_match_scores_100(self) -> None:
        result = score_markdown(_EXPECTED, _EXPECTED)
        assert result.score == 100.0
        assert result.match_score == 100.0
        assert result.order_score == 100.0
        assert result.block_scores == [100.0, 100.0, 100.0]

    def test_shuffled_blocks_lose_order_points(self) -> None:
        reversed_output = f"{_BLOCK_C}\n\n{_BLOCK_B}\n\n{_BLOCK_A}"
        result = score_markdown(_EXPECTED, reversed_output)
        # Content is fully present, so alignment stays high...
        assert result.match_score > 95.0
        # ...but the fully reversed order zeroes the Kendall-tau component.
        assert result.order_score < 50.0
        assert result.score < 90.0

    def test_missing_block_loses_length_weight(self) -> None:
        partial_output = f"{_BLOCK_A}\n\n{_BLOCK_B}"
        result = score_markdown(_EXPECTED, partial_output)
        assert result.block_scores[0] == 100.0
        assert result.block_scores[1] == 100.0
        assert result.block_scores[2] == 0.0
        assert 0.0 < result.match_score < 100.0
        assert result.score < score_markdown(_EXPECTED, _EXPECTED).score

    def test_empty_output_scores_zero(self) -> None:
        result = score_markdown(_EXPECTED, "")
        assert result.score == 0.0
        assert result.block_scores == [0.0, 0.0, 0.0]

    def test_empty_expected_is_trivial_pass(self) -> None:
        assert score_markdown("", "anything").score == 100.0

    def test_llm_judge_is_a_stub(self) -> None:
        with pytest.raises(NotImplementedError):
            score_with_llm_judge("expected", "produced")


class TestRunnerSmoke:
    _SMOKE_FIXTURES = [
        "elements--bootstrap-alerts",
        "elements--nbsp-handling",
        "footnotes--heading-notes",
    ]

    def test_runner_produces_valid_json_on_small_fixtures(self, tmp_path: Path) -> None:
        available = webextract_quality.discover_fixtures()
        fixtures = [f for f in self._SMOKE_FIXTURES if f in available]
        assert len(fixtures) >= 2, "smoke fixtures missing from corpus"

        output = tmp_path / "latest.json"
        exit_code = webextract_quality.main(
            [*fixtures, "--output", str(output), "--no-color"]
        )
        assert exit_code == 0

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["fixture_count"] == len(fixtures)
        assert set(payload["fixtures"]) == set(fixtures)
        for entry in payload["fixtures"].values():
            assert 0.0 <= entry["score"] <= 100.0
            assert 0.0 <= entry["order_score"] <= 100.0
        assert 0.0 <= payload["aggregate"]["mean_score"] <= 100.0

    def test_runner_rejects_unknown_fixture(self, tmp_path: Path) -> None:
        exit_code = webextract_quality.main(
            [
                "no-such-fixture",
                "--output",
                str(tmp_path / "out.json"),
                "--no-color",
            ]
        )
        assert exit_code == 1


def _payload(
    fixtures: dict[str, float], local: dict[str, float] | None = None
) -> dict[str, object]:
    """Build a minimal results payload from {stem: score} maps."""

    def entries(scores: dict[str, float]) -> dict[str, dict[str, float]]:
        return {
            stem: {"score": s, "match_score": s, "order_score": s}
            for stem, s in scores.items()
        }

    def mean(scores: dict[str, float]) -> float:
        return round(sum(scores.values()) / len(scores), 2) if scores else 0.0

    payload: dict[str, object] = {
        "fixture_count": len(fixtures),
        "aggregate": {"mean_score": mean(fixtures)},
        "fixtures": entries(fixtures),
    }
    if local is not None:
        payload["local_fixture_count"] = len(local)
        payload["aggregate"]["local_mean_score"] = mean(local)  # type: ignore[index]
        payload["local_fixtures"] = entries(local)
    return payload


class TestGuardrails:
    """Quality-gate logic: per-fixture floors + corpus-mean floor
    (floor = score * 0.9 at generation time, xberg-style)."""

    def test_generate_guardrails_applies_threshold_factor(self) -> None:
        payload = _payload({"a": 90.0, "b": 100.0}, local={"loc": 98.0})
        guardrails = webextract_quality.generate_guardrails(payload)

        assert guardrails["threshold_factor"] == 0.9
        assert guardrails["fixtures"]["a"]["min_score"] == 81.0
        assert guardrails["fixtures"]["b"]["min_score"] == 90.0
        assert guardrails["mean_floor"] == round(95.0 * 0.9, 2)
        assert guardrails["local_fixtures"]["loc"]["min_score"] == round(98.0 * 0.9, 2)
        assert guardrails["local_mean_floor"] == round(98.0 * 0.9, 2)

    def test_scores_at_or_above_floors_pass(self) -> None:
        payload = _payload({"a": 90.0, "b": 100.0}, local={"loc": 98.0})
        guardrails = webextract_quality.generate_guardrails(payload)
        # Scores exactly at generation time (well above 0.9x floors).
        assert webextract_quality.check_guardrails(payload, guardrails) == []

        # A drop that stays above the floor is still a pass.
        dropped = _payload({"a": 82.0, "b": 100.0}, local={"loc": 98.0})
        assert webextract_quality.check_guardrails(dropped, guardrails) == []

    def test_fixture_below_floor_is_a_violation(self) -> None:
        guardrails = webextract_quality.generate_guardrails(
            _payload({"a": 90.0, "b": 100.0})
        )
        # b collapses below its 90.0 floor; the mean (75.0) also breaches
        # its 85.5 floor.
        payload = _payload({"a": 90.0, "b": 60.0})
        violations = webextract_quality.check_guardrails(payload, guardrails)

        names = {v.name for v in violations}
        assert names == {"b", "MEAN"}
        by_name = {v.name: v for v in violations}
        assert by_name["b"].score == 60.0
        assert by_name["b"].floor == 90.0
        assert "60.00" in str(by_name["b"]) and "90.00" in str(by_name["b"])

    def test_mean_below_floor_is_a_violation_even_if_fixtures_pass(self) -> None:
        guardrails = {
            "threshold_factor": 0.9,
            "mean_floor": 95.0,
            "fixtures": {"a": {"min_score": 50.0}},
        }
        violations = webextract_quality.check_guardrails(
            _payload({"a": 90.0}), guardrails
        )
        assert [(v.name, v.score, v.floor) for v in violations] == [
            ("MEAN", 90.0, 95.0)
        ]

    def test_missing_guardrailed_fixture_is_a_violation(self) -> None:
        guardrails = webextract_quality.generate_guardrails(
            _payload({"a": 90.0, "gone": 100.0})
        )
        violations = webextract_quality.check_guardrails(
            _payload({"a": 90.0}), guardrails
        )
        missing = [v for v in violations if v.name == "gone"]
        assert len(missing) == 1
        assert missing[0].score is None
        assert "missing" in str(missing[0])

    def test_new_fixture_without_floor_is_not_a_violation(self) -> None:
        guardrails = webextract_quality.generate_guardrails(_payload({"a": 90.0}))
        payload = _payload({"a": 90.0, "brand-new": 10.0})
        # brand-new has no floor yet; only the mean could complain (50.0
        # vs floor 81.0).
        violations = webextract_quality.check_guardrails(payload, guardrails)
        assert {v.name for v in violations} == {"MEAN"}

    def test_runner_check_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--update-guardrails then --check on the same fixtures passes;
        a tampered (raised) floor makes --check exit 1."""
        stem = "elements--bootstrap-alerts"
        assert stem in webextract_quality.discover_fixtures()
        guardrails_path = tmp_path / "guardrails.json"
        monkeypatch.setattr(webextract_quality, "GUARDRAILS_PATH", guardrails_path)
        # Keep the tmp guardrails scoped to this one corpus fixture: local
        # fixtures are excluded by naming only `stem`.
        common = [stem, "--output", str(tmp_path / "latest.json"), "--no-color"]

        assert webextract_quality.main([*common, "--update-guardrails"]) == 0
        assert guardrails_path.is_file()
        assert webextract_quality.main([*common, "--check"]) == 0

        # Raise the fixture floor above any possible score -> violation.
        guardrails = json.loads(guardrails_path.read_text(encoding="utf-8"))
        guardrails["fixtures"][stem]["min_score"] = 101.0
        guardrails_path.write_text(json.dumps(guardrails), encoding="utf-8")
        assert webextract_quality.main([*common, "--check"]) == 1

    def test_runner_check_without_guardrails_file_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            webextract_quality, "GUARDRAILS_PATH", tmp_path / "nope.json"
        )
        exit_code = webextract_quality.main(
            [
                "elements--bootstrap-alerts",
                "--check",
                "--output",
                str(tmp_path / "latest.json"),
                "--no-color",
            ]
        )
        assert exit_code == 1

    def test_committed_guardrails_cover_current_corpus(self) -> None:
        """Every discoverable fixture has a floor and vice versa, so the CI
        gate can't silently skip fixtures."""
        guardrails = webextract_quality.load_guardrails()
        assert guardrails is not None, (
            "benchmarks/guardrails.json missing; run the benchmark with "
            "--update-guardrails"
        )
        assert set(guardrails["fixtures"]) == set(
            webextract_quality.discover_fixtures()
        )
        assert set(guardrails.get("local_fixtures", {})) == set(
            webextract_quality.discover_local_fixtures()
        )


class TestLocalFixtures:
    """Local (markitai-owned, self-baseline) fixtures — see
    benchmarks/local_fixtures/README.md. Kept out of the defuddle-corpus
    mean; aggregated separately as ``aggregate.local_mean_score``."""

    def test_local_fixtures_are_discovered_and_disjoint_from_corpus(self) -> None:
        local = webextract_quality.discover_local_fixtures()
        assert "github-repo--panniantong-agent-reach" in local
        assert not set(local) & set(webextract_quality.discover_fixtures())

    def test_local_fixtures_do_not_affect_corpus_mean(self, tmp_path: Path) -> None:
        corpus_stem = "elements--bootstrap-alerts"
        local_stem = "github-repo--panniantong-agent-reach"
        assert corpus_stem in webextract_quality.discover_fixtures()
        assert local_stem in webextract_quality.discover_local_fixtures()

        output = tmp_path / "latest.json"
        exit_code = webextract_quality.main(
            [corpus_stem, local_stem, "--output", str(output), "--no-color"]
        )
        assert exit_code == 0

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert set(payload["fixtures"]) == {corpus_stem}
        assert set(payload["local_fixtures"]) == {local_stem}
        assert payload["fixture_count"] == 1
        assert payload["local_fixture_count"] == 1
        # Corpus mean must equal the single corpus fixture's score — the
        # local fixture must not leak into it.
        assert payload["aggregate"]["mean_score"] == round(
            payload["fixtures"][corpus_stem]["score"], 2
        )
        assert payload["aggregate"]["local_mean_score"] == round(
            payload["local_fixtures"][local_stem]["score"], 2
        )

    def test_local_fixture_scores_near_self_baseline(self) -> None:
        """Expected .md files are self-baselines: scores must stay ~100.

        A drop below this floor signals an extraction regression on the
        locally captured pages (GitHub repo README, Jekyll blog post).
        """
        for stem in webextract_quality.discover_local_fixtures():
            result, error = webextract_quality.score_fixture(
                stem,
                webextract_quality._LOCAL_HTML_DIR,
                webextract_quality._LOCAL_EXPECTED_DIR,
            )
            assert error is None, f"{stem}: {error}"
            # 95 (not 100): the greedy block-alignment scorer loses a few
            # order points on repeated blocks even for identical text.
            assert result.score >= 95.0, f"{stem} regressed: {result.score}"
