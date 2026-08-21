"""What `pip install markitai` (no extras) is allowed to drag in.

The core install was 633MB. Two of the heaviest items were optional in
practice but declared as hard dependencies: `opencv-python` (121MB, already
behind a Pillow fallback) and `rapidocr` (32MB + its own opencv copy, only
reachable via `--ocr`). These tests pin the resolved core closure so neither
returns by accident — including transitively, which is how they would.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOCK = _REPO_ROOT / "uv.lock"
_PKG_PYPROJECT = _REPO_ROOT / "packages" / "markitai" / "pyproject.toml"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"


@lru_cache(maxsize=1)
def _lock() -> dict:
    return tomllib.loads(_LOCK.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _packages() -> dict[str, dict]:
    return {package["name"]: package for package in _lock()["package"]}


def _closure(roots: frozenset[str]) -> set[str]:
    """Transitively resolve dependency names from the lockfile."""
    packages = _packages()
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        entry = packages.get(name)
        if entry is None:
            continue
        stack.extend(dep["name"] for dep in entry.get("dependencies", []))
    return seen


@lru_cache(maxsize=1)
def core_closure() -> frozenset[str]:
    """Everything an extra-free `pip install markitai` resolves to."""
    markitai = _packages()["markitai"]
    roots = frozenset(dep["name"] for dep in markitai["dependencies"])
    return frozenset(_closure(roots))


def _extra_closure(extra: str) -> frozenset[str]:
    entry = _packages()["markitai"]["optional-dependencies"][extra]
    return frozenset(_closure(frozenset(dep["name"] for dep in entry)))


class TestCoreClosureExclusions:
    @pytest.mark.parametrize("package", ["opencv-python", "opencv-python-headless"])
    def test_opencv_absent_from_core(self, package: str) -> None:
        assert package not in core_closure()

    def test_rapidocr_absent_from_core(self) -> None:
        assert "rapidocr" not in core_closure()

    def test_core_keeps_the_engines_it_actually_needs(self) -> None:
        """Guard against over-trimming: these must stay reachable without extras."""
        for package in ("pillow", "pymupdf4llm", "pymupdf", "litellm", "markitdown"):
            assert package in core_closure(), package

    def test_onnxruntime_stays_because_magika_needs_it(self) -> None:
        """Documented reality: it arrives via markitdown->magika, not via OCR."""
        closure = core_closure()
        assert "magika" in closure
        assert "onnxruntime" in closure


class TestOcrExtra:
    def test_ocr_extra_exists_and_provides_rapidocr(self) -> None:
        assert "rapidocr" in _extra_closure("ocr")

    def test_ocr_extra_name_is_stable(self) -> None:
        """Install scripts and error messages hard-code `markitai[ocr]`."""
        extras = _packages()["markitai"]["optional-dependencies"]
        assert "ocr" in extras

    def test_all_extra_includes_ocr(self) -> None:
        assert "rapidocr" in _extra_closure("all")

    def test_root_workspace_forwards_the_ocr_extra(self) -> None:
        data = tomllib.loads(_ROOT_PYPROJECT.read_text(encoding="utf-8"))
        forwarded = data["project"]["optional-dependencies"]
        assert forwarded.get("ocr") == ["markitai[ocr]"]

    def test_declared_ocr_extra_matches_the_documented_floor(self) -> None:
        data = tomllib.loads(_PKG_PYPROJECT.read_text(encoding="utf-8"))
        assert data["project"]["optional-dependencies"]["ocr"] == ["rapidocr>=3.7.0"]

    def test_every_package_extra_is_forwarded_by_the_workspace_root(self) -> None:
        package_extras = set(
            tomllib.loads(_PKG_PYPROJECT.read_text(encoding="utf-8"))["project"][
                "optional-dependencies"
            ]
        )
        root_extras = set(
            tomllib.loads(_ROOT_PYPROJECT.read_text(encoding="utf-8"))["project"][
                "optional-dependencies"
            ]
        )
        assert package_extras - root_extras == set()


class TestAllExtraCompleteness:
    def test_all_extra_covers_every_other_extra(self) -> None:
        """`all` is hand-maintained; drift silently drops capabilities."""
        extras = tomllib.loads(_PKG_PYPROJECT.read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
        expected: set[str] = set()
        for name, requirements in extras.items():
            if name != "all":
                expected.update(requirements)
        assert expected - set(extras["all"]) == set()
