"""Fast guardrail: document conversion output must match its committed snapshot.

Runs markitai's default (no-LLM, no-OCR, no-screenshot) conversion pipeline
against a small, stable fixture set (PDF/DOCX/PPTX/XLSX — see
``benchmarks/docs_snapshot.py`` for why legacy/OCR formats are excluded) and
compares it to a normalized, committed snapshot. No network, no LLM calls,
sub-second per fixture, so unlike the full webextract quality benchmark this
runs in the default test suite (and therefore CI) with no extra wiring.

A failure here means either a real conversion regression, or a deliberate
change that needs ``uv run python packages/markitai/benchmarks/docs_snapshot.py
--update`` (review the diff before committing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# benchmarks/ is dev tooling next to src/, not an installed package.
_PKG_DIR = Path(__file__).parents[2]
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from benchmarks.docs_snapshot import FIXTURES, SnapshotCheck, check_fixture


@pytest.mark.parametrize("fixture_name", FIXTURES)
def test_conversion_matches_committed_snapshot(fixture_name: str) -> None:
    result: SnapshotCheck = check_fixture(fixture_name)
    if not result.matches:
        detail = result.error or result.diff
        pytest.fail(
            f"{fixture_name}: output drifted from benchmarks/docs_snapshots/"
            f"expected/{fixture_name}.md. Review and, if intentional, "
            f"regenerate with `docs_snapshot.py --update`:\n\n{detail}"
        )


def test_every_fixture_has_a_committed_snapshot() -> None:
    """Guards against a fixture being added to FIXTURES without --update."""
    from benchmarks.docs_snapshot import load_expected

    missing = [name for name in FIXTURES if load_expected(name) is None]
    assert not missing, (
        f"Missing committed snapshot(s) for {missing}; run "
        f"`uv run python packages/markitai/benchmarks/docs_snapshot.py --update`"
    )


def test_fixtures_exist_and_avoid_legacy_libreoffice_formats() -> None:
    """FIXTURES must stay on pure-Python converter paths (see module docstring)."""
    from benchmarks.docs_snapshot import _FIXTURES_DIR

    legacy_suffixes = {".doc", ".ppt", ".xls"}
    for name in FIXTURES:
        assert (_FIXTURES_DIR / name).is_file(), f"{name} missing from tests/fixtures/"
        assert Path(name).suffix not in legacy_suffixes, (
            f"{name}: legacy formats route through LibreOffice/MS Office and "
            f"are excluded from this exact-match guardrail (version drift)"
        )
