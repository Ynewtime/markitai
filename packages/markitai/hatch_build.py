from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface

_README_NAME = "README.md"


def _resolve_readme(project_root: Path) -> Path:
    candidates = (
        project_root / _README_NAME,
        project_root.parent.parent / _README_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise OSError(f"Readme file does not exist: {_README_NAME}")


class CustomMetadataHook(MetadataHookInterface):
    def update(self, metadata: dict) -> None:
        readme_path = _resolve_readme(Path(self.root))
        metadata["readme"] = {
            "text": readme_path.read_text(encoding="utf-8"),
            "content-type": "text/markdown",
        }


def get_metadata_hook() -> type[CustomMetadataHook]:
    return CustomMetadataHook


# Where the build identity goes in the package (read by markitai.build_info)
_BUILD_INFO = "src/markitai/_build_info.py"


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            # Read-only: never refresh the index (no index.lock)
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _built_at() -> str:
    # SOURCE_DATE_EPOCH keeps a rebuild of one commit byte-identical
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class CustomBuildHook(BuildHookInterface):
    """Write ``markitai/_build_info.py`` (short commit, build time, dirty)
    into the wheel and the sdist, never into the source tree.

    A wheel built from an sdist keeps the sdist's file. An editable install
    gets none: ``markitai.build_info`` asks git instead.
    """

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        root = Path(self.root)
        if (root / _BUILD_INFO).is_file():
            return  # building from an sdist: its file is already in the tree
        commit = _git(root, "rev-parse", "--short", "HEAD") or None
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        source = (
            '"""Build identity, written by hatch_build.py at build time."""\n\n'
            f"COMMIT = {commit!r}\n"
            f"BUILT_AT = {_built_at()!r}\n"
            f"DIRTY = {bool(status)!r}\n"
        )
        self._scratch = Path(tempfile.mkdtemp(prefix="markitai-build-info-"))
        target = self._scratch / "_build_info.py"
        target.write_text(source, encoding="utf-8")
        inside = (
            "markitai/_build_info.py" if self.target_name == "wheel" else _BUILD_INFO
        )
        build_data["force_include"][str(target)] = inside

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        scratch = getattr(self, "_scratch", None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def get_build_hook() -> type[CustomBuildHook]:
    return CustomBuildHook
