"""Guard: the built wheel must carry LICENSE and NOTICE.

``license-files`` globs are resolved against the *package* root
(``packages/markitai``), not the repository root, so a bare
``license-files = ["LICENSE"]`` pointing at the repo-root file silently
matched nothing and every published wheel shipped without a licence — an
MIT requirement, and Apache-2.0 section 4(d) for the NOTICE.

Escaping upwards (``"../../LICENSE"``) is not a fix: hatchling emits the
literal path and the wheel gains a traversal entry
(``…dist-info/licenses/../../LICENSE``). The package therefore keeps its
own copies, and these tests keep them from drifting from the canonical
repo-root files.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "packages" / "markitai"
LICENCE_NAMES = ("LICENSE", "NOTICE")


@pytest.mark.parametrize("name", LICENCE_NAMES)
def test_package_copy_matches_the_repo_root_file(name: str) -> None:
    """The packaged copy is byte-identical to the canonical repo-root file."""
    canonical = REPO_ROOT / name
    packaged = PACKAGE_ROOT / name

    assert canonical.is_file(), f"{name} is missing from the repository root"
    assert packaged.is_file(), (
        f"packages/markitai/{name} is missing; without it the wheel ships "
        f"no {name} (license-files cannot reach outside the package root)"
    )
    assert packaged.read_bytes() == canonical.read_bytes(), (
        f"packages/markitai/{name} has drifted from {name} at the repo root; "
        f"copy the root file over it"
    )


def test_license_files_declares_both_without_escaping_the_package() -> None:
    """``license-files`` names both files with package-relative paths."""
    pyproject = tomllib.loads(
        (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = pyproject["project"]["license-files"]

    assert set(declared) == set(LICENCE_NAMES)
    for entry in declared:
        assert ".." not in entry, (
            f"{entry!r} escapes the package root; hatchling would write a "
            f"path-traversal entry into the wheel's dist-info"
        )
