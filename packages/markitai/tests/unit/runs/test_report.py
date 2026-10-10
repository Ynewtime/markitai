"""Single-item report: its entry carries the item's warnings like --json."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from markitai.runs.report import build_single_report
from markitai.runs.types import Outcome

WARNING = "LLM enhancement failed (boom); kept the unenhanced output"


@pytest.mark.parametrize("kind", ["file", "url"])
def test_single_report_lists_item_warnings(
    kind: Literal["file", "url"], tmp_path: Path
) -> None:
    """A completed item whose LLM step failed is not a clean success."""
    source = "a.md" if kind == "file" else "https://example.com/"
    report = build_single_report(
        Outcome(
            kind=kind,
            source=source,
            status="completed",
            output_path=tmp_path / "a.md.md",
            warnings=[WARNING],
        )
    )
    entries = report["documents"] if kind == "file" else report["urls"]
    assert entries[source]["warnings"] == [WARNING]
