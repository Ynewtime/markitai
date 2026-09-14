"""A fast result cannot pass the audit after losing content or attachments."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[4] / "scripts/benchmarks/audit_quality.py"
    spec = importlib.util.spec_from_file_location("audit_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot():
    return {
        "markdown": "# Complete title\n\n![diagram](assets/image.png)",
        "metadata": {"title": "Complete title", "converter": "markitdown"},
        "images": [{"sha256": "original", "width": 20, "height": 10}],
    }


@pytest.mark.parametrize("field", ["markdown", "metadata", "images"])
def test_changed_output_fails_even_with_identical_speed(gate, snapshot, field):
    after = copy.deepcopy(snapshot)
    after.pop(field)
    assert gate.compare_snapshot("converter-docx", snapshot, after)


def test_same_image_count_but_changed_bytes_fails(gate, snapshot):
    after = copy.deepcopy(snapshot)
    after["images"][0]["sha256"] = "corrupted"
    assert gate.compare_snapshot("converter-docx", snapshot, after)


def test_only_reviewed_engine_transition_is_allowed(gate, snapshot):
    after = copy.deepcopy(snapshot)
    after["metadata"]["converter"] = "mammoth"
    assert not gate.compare_snapshot("converter-docx", snapshot, after)
    assert gate.compare_snapshot("converter-pdf", snapshot, after)
    after["metadata"].pop("title")
    assert gate.compare_snapshot("converter-docx", snapshot, after)


def test_comparison_does_not_remove_recorded_metadata(gate, snapshot):
    original = copy.deepcopy(snapshot)
    assert not gate.compare_snapshot("converter-docx", snapshot, snapshot)
    assert snapshot == original


@pytest.fixture
def scorer():
    path = (
        Path(__file__).resolve().parents[4] / "scripts/benchmarks/score_performance.py"
    )
    spec = importlib.util.spec_from_file_location("score_performance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", ["missing", "failed", "incomplete", "input"])
def test_quality_evidence_must_be_complete_and_match_workload(scorer, change):
    report = {
        "schema": 1,
        "passed": True,
        "issues": [],
        "cases": ["api-docx", "converter-docx"],
        "inputs": {"sample.docx": "abc"},
    }
    expected = set(report["cases"])
    inputs = dict(report["inputs"])
    if change == "missing":
        report = None
    elif change == "failed":
        report["issues"] = ["Image changed"]
    elif change == "incomplete":
        report["cases"].pop()
    else:
        report["inputs"]["sample.docx"] = "different"
    assert scorer.quality_issues(report, inputs, expected)


def test_complete_matching_quality_evidence_passes(scorer):
    report = {
        "schema": 1,
        "passed": True,
        "issues": [],
        "cases": ["api-docx"],
        "inputs": {"sample.docx": "abc"},
    }
    assert not scorer.quality_issues(report, report["inputs"], {"api-docx"})


def test_reviewed_xlsx_reader_keeps_every_other_metadata_field(gate, snapshot):
    before = copy.deepcopy(snapshot)
    before["metadata"]["converter"] = "pandas"
    after = copy.deepcopy(before)
    after["metadata"]["converter"] = "openpyxl"
    assert not gate.compare_snapshot("converter-xlsx", before, after)
    after["metadata"].pop("title")
    assert gate.compare_snapshot("converter-xlsx", before, after)


@pytest.mark.parametrize("previous", ["markitdown", "mammoth"])
def test_plain_docx_engine_transition_preserves_every_output_field(
    gate, snapshot, previous
):
    before = copy.deepcopy(snapshot)
    before["metadata"]["converter"] = previous
    after = copy.deepcopy(before)
    after["metadata"]["converter"] = "native-docx"
    assert not gate.compare_snapshot("converter-docx", before, after)
    after["images"][0]["sha256"] = "lost"
    assert gate.compare_snapshot("converter-docx", before, after)
