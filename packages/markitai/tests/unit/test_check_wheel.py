"""scripts/check_wheel.py: the release gate finds every vendored workbench file.

CI and the release workflow build the wheel and run this script on it; these
tests run it on small stand-in wheels, so a check that stops looking (the
brand marks, fonts and licences the workbench manifest lists) fails here
first.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / "scripts" / "check_wheel.py"
_WORKBENCH = (
    _REPO_ROOT / "packages" / "markitai" / "src" / "markitai" / "serve" / "workbench"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_check_wheel", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check_wheel() -> ModuleType:
    return _load()


def _wheel(path: Path, check_wheel: ModuleType, *, without: str = "") -> Path:
    """A wheel with every member the check looks for, less *without*."""
    manifest = json.loads((_WORKBENCH / "manifest.json").read_text(encoding="utf-8"))
    members = {
        "markitai-1.2.1.dist-info/entry_points.txt": (
            "[console_scripts]\nmarkitai-mcp = markitai.mcp.server:main\n"
        ),
        "markitai-1.2.1.dist-info/licenses/LICENSE": "MIT",
        "markitai-1.2.1.dist-info/licenses/NOTICE": "NOTICE",
        "markitai/mcp/__init__.py": "",
        **dict.fromkeys(check_wheel.REQUIRED_FILES, ""),
        **{f"{check_wheel.WORKBENCH}{name}": "" for name in manifest["files"]},
        f"{check_wheel.WORKBENCH}manifest.json": json.dumps(manifest),
    }
    with zipfile.ZipFile(path, "w") as wheel:
        for name, text in members.items():
            if name != without:
                wheel.writestr(name, text)
    return path


def test_a_complete_wheel_passes(tmp_path: Path, check_wheel: ModuleType) -> None:
    check_wheel.check_wheel(_wheel(tmp_path / "markitai.whl", check_wheel))


@pytest.mark.parametrize(
    "name",
    [
        "ui/brands.js",
        "LICENSES/lobehub-icons-LICENSE",
        "ui/inter-latin-wght.woff2",
        "ui/math.js",
        "ui/math.css",
        "ui/KaTeX_Main-Regular.woff2",
        "LICENSES/KaTeX-LICENSE",
    ],
)
def test_every_file_the_workbench_manifest_lists_is_required(
    tmp_path: Path, check_wheel: ModuleType, name: str
) -> None:
    wheel = _wheel(
        tmp_path / "markitai.whl", check_wheel, without=f"{check_wheel.WORKBENCH}{name}"
    )
    with pytest.raises(SystemExit, match=name):
        check_wheel.check_wheel(wheel)
