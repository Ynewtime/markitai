"""Consistency checks for the defuddle port manifest.

PORT_MANIFEST.md pins the upstream defuddle commit the port tracks; the
parity corpus records its source commit in tests/defuddle_fixtures/VERSION.
Both are rewritten by scripts/sync_defuddle_fixtures.sh — these tests fail
when the two pins drift apart again (e.g. after a partial resync).
"""

from __future__ import annotations

import re
from pathlib import Path

_PKG_DIR = Path(__file__).parents[3]
_MANIFEST_PATH = _PKG_DIR / "src" / "markitai" / "webextract" / "PORT_MANIFEST.md"
_VERSION_PATH = _PKG_DIR / "tests" / "defuddle_fixtures" / "VERSION"

_MANIFEST_PIN_RE = re.compile(r"^Pinned upstream commit: `([0-9a-f]{40})`$", re.M)
_VERSION_PIN_RE = re.compile(r"^defuddle commit: ([0-9a-f]{40})$", re.M)


def _read_pin(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"No upstream commit pin found in {path}"
    return match.group(1)


def test_manifest_pin_matches_fixture_corpus_pin() -> None:
    manifest_pin = _read_pin(_MANIFEST_PATH, _MANIFEST_PIN_RE)
    corpus_pin = _read_pin(_VERSION_PATH, _VERSION_PIN_RE)
    assert manifest_pin == corpus_pin, (
        f"PORT_MANIFEST.md pins {manifest_pin} but tests/defuddle_fixtures/VERSION "
        f"pins {corpus_pin}; re-run scripts/sync_defuddle_fixtures.sh so the "
        "algorithm manifest and the parity corpus reference the same upstream "
        "commit"
    )


def test_manifest_port_paths_exist() -> None:
    """Every .py path named in the manifest exists on disk.

    Port module paths are relative to the webextract package; other .py
    references (e.g. this test) are relative to the package root.
    """
    webextract_dir = _MANIFEST_PATH.parent
    text = _MANIFEST_PATH.read_text(encoding="utf-8")
    py_paths = set(re.findall(r"`([\w/]+\.py)`", text))
    assert py_paths, "No port module paths found in PORT_MANIFEST.md"
    missing = sorted(
        p
        for p in py_paths
        if not (webextract_dir / p).is_file() and not (_PKG_DIR / p).is_file()
    )
    assert not missing, f"PORT_MANIFEST.md references missing modules: {missing}"
