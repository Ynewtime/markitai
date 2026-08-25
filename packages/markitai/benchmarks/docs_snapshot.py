"""Document conversion snapshot guardrail (PDF/DOCX/PPTX/XLSX, no LLM).

markitai's HTML -> Markdown pipeline has a quality benchmark
(``webextract_quality.py``) with a committed baseline and CI guardrails, but
plain document conversion (PDF, Office formats, ...) had no regression
detector at all: a change to pymupdf4llm/markitdown extraction, page-marker
formatting, or frontmatter shape could silently drift and nobody would
notice until a user filed a bug.

This module freezes markitai's default (``llm=False``, no OCR/screenshot)
conversion output for a small, stable fixture set and fails when a later
change shifts it. Unlike ``webextract_quality.py`` (continuous fuzzy score,
because HTML->Markdown extraction has no single "correct" answer), document
conversion for a fixed input is deterministic, so this is a plain
normalize-then-exact-match snapshot: no scorer, no baseline/guardrail floor
math.

Fixture set (``FIXTURES``): one fixture per pure-Python converter path
(pymupdf4llm for PDF, markitdown/mammoth for DOCX, markitdown/python-pptx
for PPTX, markitdown/openpyxl for XLSX) under ``tests/fixtures/``.
Deliberately excludes the legacy ``.doc``/``.ppt``/``.xls`` fixtures: those
route through ``converter.legacy`` (LibreOffice or MS Office CLI), whose
output can vary by installed version across CI runners/OSes — a bad fit for
an exact-match snapshot. OCR (RapidOCR) and screenshot rendering are
likewise out of scope here for the same reason (model/renderer version
drift); this guardrail is scoped to markitai's own extraction logic.

Usage (from the repository root)::

    uv run python packages/markitai/benchmarks/docs_snapshot.py         # compare
    uv run python packages/markitai/benchmarks/docs_snapshot.py --update  # regenerate

or as a module (from ``packages/markitai``)::

    uv run python -m benchmarks.docs_snapshot

``tests/unit/test_docs_snapshot_guardrail.py`` runs the same comparison as a
fast, network-free, non-LLM pytest check, so it participates in the default
suite (and therefore CI) with no extra wiring.

Noise stripping (``normalize_snapshot``): a snapshot is compared *after*
stripping fields that vary across machines/time rather than across real
pipeline changes:

- CRLF is normalized to LF (Windows CI runners).
- ISO-8601-ish timestamps (the ``markitai_processed`` frontmatter field,
  and anything shaped like one) become ``<TIMESTAMP>``.
- YAML frontmatter lines whose key contains "version" become ``<VERSION>``,
  and an inline ``markitai/vX.Y.Z``-shaped token becomes ``markitai
  <VERSION>``, in case a future field embeds the running markitai version.
- Absolute filesystem paths (``/Users/...``, ``/home/...``, ``/var/...``,
  ``/tmp/...``, ``/private/...``, or a Windows ``C:\\...`` drive path)
  become ``<PATH>``. Asset references stay untouched: markitai emits them
  relative (``.markitai/assets/...``), which is itself a property worth
  guarding, not noise.

Regenerating deliberately: run with ``--update`` and review the diff like
any other code change — this is the same discipline
``webextract_quality.py --update-baseline`` uses for its baseline.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in {None, ""}:  # executed as a script, not as a module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

if TYPE_CHECKING:
    from markitai.api import ConversionOutput

_PKG_DIR = Path(__file__).resolve().parent.parent
_FIXTURES_DIR = _PKG_DIR / "tests" / "fixtures"
_EXPECTED_DIR = Path(__file__).resolve().parent / "docs_snapshots" / "expected"

# One fixture per pure-Python converter path; see the module docstring for
# why legacy/OCR/screenshot formats are excluded.
FIXTURES: tuple[str, ...] = ("sample.pdf", "sample.docx", "sample.pptx", "sample.xlsx")

_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_VERSION_LINE_RE = re.compile(r"(?im)^(\s*[\w.-]*version[\w.-]*\s*:\s*).+$")
_INLINE_VERSION_RE = re.compile(r"\bmarkitai[/ ]v?\d+\.\d+\.\d+(?:[.\-][0-9A-Za-z]+)*")
_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s\"')]+|/(?:Users|home|var|tmp|private|root)/[^\s\"')]+)"
)


def normalize_snapshot(text: str) -> str:
    """Strip environment noise (timestamps, versions, absolute paths) from a snapshot.

    See the module docstring for exactly what each pattern targets.
    """
    text = text.replace("\r\n", "\n")
    text = _TIMESTAMP_RE.sub("<TIMESTAMP>", text)
    text = _VERSION_LINE_RE.sub(lambda m: f"{m.group(1)}<VERSION>", text)
    text = _INLINE_VERSION_RE.sub("markitai <VERSION>", text)
    text = _ABS_PATH_RE.sub("<PATH>", text)
    return text


def render_snapshot(output: ConversionOutput) -> str:
    """Render a ``ConversionOutput`` as one comparable text blob.

    Uses ``ConversionOutput``'s public fields (frontmatter dict + markdown
    body) rather than reading the written ``.md`` file back, so the
    snapshot format is independent of the on-disk frontmatter serialization.
    """
    frontmatter_yaml = yaml.safe_dump(
        output.frontmatter, sort_keys=True, allow_unicode=True
    ).strip()
    return f"---\n{frontmatter_yaml}\n---\n\n{output.markdown}"


def convert_fixture(name: str) -> ConversionOutput:
    """Convert one fixture with LLM/OCR/screenshot disabled (in-memory).

    Args:
        name: Fixture filename under ``tests/fixtures/`` (e.g. ``"sample.pdf"``).

    Returns:
        The conversion result.

    Raises:
        FileNotFoundError: The fixture is not under ``tests/fixtures/``.
        ConversionError: The conversion pipeline failed.
    """
    import markitai

    path = _FIXTURES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return markitai.convert(
        path,
        output_dir=None,
        llm=False,
        ocr=False,
        screenshot=False,
        alt=False,
        desc=False,
    )


def _expected_path(name: str) -> Path:
    return _EXPECTED_DIR / f"{name}.md"


def load_expected(name: str) -> str | None:
    """Return the committed snapshot for a fixture, or None if absent."""
    path = _expected_path(name)
    return path.read_text(encoding="utf-8") if path.is_file() else None


@dataclass
class SnapshotCheck:
    """Result of comparing one fixture's live conversion against its snapshot.

    Attributes:
        name: Fixture filename.
        matches: True when the normalized live output equals the committed
            snapshot.
        diff: Unified diff (expected vs actual) when ``matches`` is False
            and both sides were available.
        error: Why the check could not run to a diff (conversion raised, or
            no committed snapshot exists yet).
    """

    name: str
    matches: bool
    diff: str | None = None
    error: str | None = None


def check_fixture(name: str) -> SnapshotCheck:
    """Convert one fixture and compare it against its committed snapshot."""
    try:
        actual = normalize_snapshot(render_snapshot(convert_fixture(name)))
    except Exception as exc:  # pragma: no cover - defensive, reported to caller
        return SnapshotCheck(name, matches=False, error=f"{type(exc).__name__}: {exc}")

    expected = load_expected(name)
    if expected is None:
        return SnapshotCheck(
            name, matches=False, error="no committed snapshot; run with --update"
        )
    if expected == actual:
        return SnapshotCheck(name, matches=True)

    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"expected/{name}.md",
            tofile=f"actual ({name})",
        )
    )
    return SnapshotCheck(name, matches=False, diff=diff)


def update_fixture(name: str) -> None:
    """Regenerate the committed snapshot for one fixture from live output."""
    actual = normalize_snapshot(render_snapshot(convert_fixture(name)))
    _EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    _expected_path(name).write_text(actual, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Compare markitai's default (no-LLM) document conversion "
        "output against committed snapshots."
    )
    parser.add_argument(
        "fixtures", nargs="*", help="fixture filenames to check (default: all)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="regenerate committed snapshots under benchmarks/docs_snapshots/expected/",
    )
    args = parser.parse_args(argv)

    selected = args.fixtures or list(FIXTURES)
    unknown = sorted(set(selected) - set(FIXTURES))
    if unknown:
        print(f"Unknown fixtures: {', '.join(unknown)}", file=sys.stderr)
        return 1

    if args.update:
        for name in selected:
            update_fixture(name)
            print(f"updated {name}")
        print(f"\n{len(selected)} snapshot(s) written under {_EXPECTED_DIR}")
        return 0

    results = [check_fixture(name) for name in selected]
    failures = [r for r in results if not r.matches]
    for result in results:
        print(f"{'OK  ' if result.matches else 'FAIL'}  {result.name}")

    if failures:
        print(f"\n{len(failures)} snapshot mismatch(es):", file=sys.stderr)
        for result in failures:
            print(f"\n--- {result.name} ---", file=sys.stderr)
            print(result.error or result.diff, file=sys.stderr)
        return 1

    print(f"\nAll {len(results)} snapshot(s) match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
