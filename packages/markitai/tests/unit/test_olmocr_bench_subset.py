"""Unit tests for the olmOCR-bench feasibility scorer (scripts/olmocr_bench_subset.py).

Covers the pure scoring/aggregation functions only — no network, no PDF
conversion, no markitai OCR calls. ``score_present_absent`` is an exact port
of the upstream ``olmocr/bench/tests.py`` formula (see the script's module
docstring for the source); these tests pin that formula down. See
``scripts/olmocr_bench_subset.py`` itself for the opt-in, network-using
trial harness — that script is deliberately not part of this suite or CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "olmocr_bench_subset.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_olmocr_bench_subset", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load_module()


class TestThreshold:
    def test_zero_max_diffs_requires_perfect_ratio(self, mod: ModuleType) -> None:
        assert mod._threshold("hello", 0) == 1.0

    def test_max_diffs_equal_to_length_zeroes_the_threshold(
        self, mod: ModuleType
    ) -> None:
        assert mod._threshold("hello", 5) == 0.0

    def test_empty_text_treated_as_length_one(self, mod: ModuleType) -> None:
        # Upstream guards div-by-zero with `len(text) if len(text) > 0 else 1`.
        assert mod._threshold("", 0) == 1.0


class TestPresentAbsent:
    def _rule(self, mod: ModuleType, rule_type: str, **overrides: object) -> dict:
        rule = {
            "type": rule_type,
            "text": "hello world",
            "max_diffs": 0,
            "case_sensitive": True,
            "first_n": None,
            "last_n": None,
        }
        rule.update(overrides)
        return rule

    def test_present_passes_on_exact_substring(self, mod: ModuleType) -> None:
        rule = self._rule(mod, "present")
        assert mod.score_present_absent(rule, "say hello world today") is True

    def test_present_fails_when_missing(self, mod: ModuleType) -> None:
        rule = self._rule(mod, "present")
        assert mod.score_present_absent(rule, "completely unrelated text") is False

    def test_absent_passes_when_missing(self, mod: ModuleType) -> None:
        rule = self._rule(mod, "absent")
        assert mod.score_present_absent(rule, "completely unrelated text") is True

    def test_absent_fails_when_present(self, mod: ModuleType) -> None:
        rule = self._rule(mod, "absent")
        assert mod.score_present_absent(rule, "say hello world today") is False

    def test_present_tolerates_up_to_max_diffs_edits(self, mod: ModuleType) -> None:
        # "hello world" with one substitution ("hallo") should still pass
        # once max_diffs allows it, and fail at max_diffs=0.
        rule = self._rule(mod, "present", text="hello world", max_diffs=0)
        assert mod.score_present_absent(rule, "say hallo world today") is False
        rule = self._rule(mod, "present", text="hello world", max_diffs=2)
        assert mod.score_present_absent(rule, "say hallo world today") is True

    def test_case_insensitive_matches_regardless_of_case(self, mod: ModuleType) -> None:
        rule = self._rule(mod, "present", text="Hello World", case_sensitive=False)
        assert mod.score_present_absent(rule, "say HELLO WORLD today") is True

    def test_case_sensitive_default_true_rejects_case_mismatch(
        self, mod: ModuleType
    ) -> None:
        rule = self._rule(mod, "present", text="Hello World", max_diffs=0)
        assert mod.score_present_absent(rule, "say hello world today") is False

    def test_first_n_window_ignores_text_after_it(self, mod: ModuleType) -> None:
        content = "HEADER " + ("x" * 100) + " hello world"
        rule = self._rule(mod, "present", text="hello world", first_n=10)
        assert mod.score_present_absent(rule, content) is False
        rule = self._rule(mod, "absent", text="hello world", first_n=10)
        assert mod.score_present_absent(rule, content) is True

    def test_last_n_window_ignores_text_before_it(self, mod: ModuleType) -> None:
        content = "hello world " + ("x" * 100) + " FOOTER"
        rule = self._rule(mod, "present", text="hello world", last_n=10)
        assert mod.score_present_absent(rule, content) is False
        rule = self._rule(mod, "absent", text="hello world", last_n=10)
        assert mod.score_present_absent(rule, content) is True

    def test_first_n_and_last_n_combine_as_upstream_does(self, mod: ModuleType) -> None:
        # Upstream: content[:first_n] + content[-last_n:] when both are set.
        content = "hello" + ("x" * 100) + "world"
        rule = self._rule(mod, "present", text="hello", first_n=5, last_n=5)
        assert mod.score_present_absent(rule, content) is True
        rule = self._rule(mod, "present", text="world", first_n=5, last_n=5)
        assert mod.score_present_absent(rule, content) is True
        rule = self._rule(
            mod, "present", text="xxxxx", first_n=5, last_n=5, max_diffs=0
        )
        assert mod.score_present_absent(rule, content) is False


class TestOrder:
    def test_passes_when_before_precedes_after(self, mod: ModuleType) -> None:
        rule = {"before": "alpha", "after": "omega", "max_diffs": 0}
        assert mod.score_order(rule, "alpha appears then omega appears later") is True

    def test_fails_when_reversed(self, mod: ModuleType) -> None:
        rule = {"before": "alpha", "after": "omega", "max_diffs": 0}
        assert mod.score_order(rule, "omega appears then alpha appears later") is False

    def test_fails_when_before_missing(self, mod: ModuleType) -> None:
        rule = {"before": "nowhere", "after": "omega", "max_diffs": 0}
        assert mod.score_order(rule, "only omega is here") is False

    def test_fails_when_after_missing(self, mod: ModuleType) -> None:
        rule = {"before": "alpha", "after": "nowhere", "max_diffs": 0}
        assert mod.score_order(rule, "only alpha is here") is False


class TestScoreRule:
    def test_dispatches_present_and_absent(self, mod: ModuleType) -> None:
        rule = {
            "id": "r1",
            "pdf": "x.pdf",
            "type": "present",
            "text": "hi",
            "max_diffs": 0,
            "case_sensitive": True,
            "first_n": None,
            "last_n": None,
        }
        outcome = mod.score_rule(rule, "hi there")
        assert outcome == mod.RuleOutcome("r1", "present", "x.pdf", True)

    def test_dispatches_order(self, mod: ModuleType) -> None:
        rule = {
            "id": "r2",
            "pdf": "x.pdf",
            "type": "order",
            "before": "a",
            "after": "b",
            "max_diffs": 0,
        }
        outcome = mod.score_rule(rule, "a then b")
        assert outcome.passed is True

    def test_unsupported_types_are_skipped_not_failed(self, mod: ModuleType) -> None:
        for rule_type in ("table", "math"):
            rule = {"id": "r3", "pdf": "x.pdf", "type": rule_type}
            outcome = mod.score_rule(rule, "irrelevant content")
            assert outcome.passed is None
            assert outcome.note is not None and rule_type in outcome.note


class TestSelectPdfs:
    def test_dedups_preserving_first_seen_order(self, mod: ModuleType) -> None:
        rules = [{"pdf": "b.pdf"}, {"pdf": "a.pdf"}, {"pdf": "b.pdf"}, {"pdf": "c.pdf"}]
        assert mod.select_pdfs(rules, limit=10) == ["b.pdf", "a.pdf", "c.pdf"]

    def test_respects_limit(self, mod: ModuleType) -> None:
        rules = [{"pdf": "a.pdf"}, {"pdf": "b.pdf"}, {"pdf": "c.pdf"}]
        assert mod.select_pdfs(rules, limit=2) == ["a.pdf", "b.pdf"]


class TestSummarize:
    def test_aggregates_pass_rate_per_type_and_overall(self, mod: ModuleType) -> None:
        outcomes = [
            mod.RuleOutcome("1", "present", "x.pdf", True),
            mod.RuleOutcome("2", "present", "x.pdf", False),
            mod.RuleOutcome("3", "absent", "x.pdf", True),
            mod.RuleOutcome("4", "table", "x.pdf", None, note="skipped"),
        ]
        summary = mod.summarize(outcomes)

        assert summary["present"] == {
            "total": 2,
            "scored": 2,
            "skipped": 0,
            "passed": 1,
            "pass_rate": 0.5,
        }
        assert summary["absent"]["pass_rate"] == 1.0
        assert summary["table"] == {
            "total": 1,
            "scored": 0,
            "skipped": 1,
            "passed": 0,
            "pass_rate": None,
        }
        assert summary["_overall"]["total_rules"] == 4
        assert summary["_overall"]["scored_rules"] == 3
        assert summary["_overall"]["skipped_rules"] == 1
        assert summary["_overall"]["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)

    def test_all_skipped_reports_none_pass_rate(self, mod: ModuleType) -> None:
        outcomes = [mod.RuleOutcome("1", "math", "x.pdf", None, note="skipped")]
        summary = mod.summarize(outcomes)
        assert summary["_overall"]["pass_rate"] is None


class TestCacheDir:
    def test_default_cache_dir_is_outside_the_repository(self, mod: ModuleType) -> None:
        cache_dir = mod.default_cache_dir()
        assert _REPO_ROOT not in cache_dir.parents
