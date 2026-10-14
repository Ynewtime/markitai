"""Which build of markitai is running: version, commit, build time, dirty.

A wheel or sdist carries ``markitai/_build_info.py``, written by the build
hook (``hatch_build.py``). A source checkout (an editable install) has none,
so its commit and dirty state come from git, read-only; its build time is
unknown. ``serve`` answers it as ``capabilities.build`` (workbench contract
v1.16).
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import TypedDict

from markitai import __version__


class BuildIdentity(TypedDict):
    version: str
    commit: str | None  # the short commit hash
    built_at: str | None  # RFC 3339 UTC
    dirty: bool  # the tree had uncommitted changes to tracked files


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
            # Read-only: never refresh the index (no index.lock)
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _checkout_identity() -> tuple[str | None, bool]:
    """Commit and dirty state of the checkout this package is imported from.

    Only for ``<top>/packages/markitai/src/markitai``: an installed copy that
    happens to sit inside some other repository must not report its commit.
    """
    package = Path(__file__).resolve().parent
    top = _git(package, "rev-parse", "--show-toplevel")
    if not top or package != Path(top).resolve() / "packages/markitai/src/markitai":
        return None, False
    commit = _git(package, "rev-parse", "--short", "HEAD")
    if not commit:
        return None, False
    status = _git(package, "status", "--porcelain", "--untracked-files=no")
    return commit, bool(status)


@functools.cache
def build_identity() -> BuildIdentity:
    """This build's identity (computed once per process; may run git)."""
    try:
        from markitai import _build_info  # type: ignore[attr-defined]
    except ImportError:
        commit, dirty = _checkout_identity()
        return {
            "version": __version__,
            "commit": commit,
            "built_at": None,
            "dirty": dirty,
        }
    commit = getattr(_build_info, "COMMIT", None)
    built_at = getattr(_build_info, "BUILT_AT", None)
    return {
        "version": __version__,
        "commit": commit if isinstance(commit, str) and commit else None,
        "built_at": built_at if isinstance(built_at, str) and built_at else None,
        "dirty": getattr(_build_info, "DIRTY", False) is True,
    }
