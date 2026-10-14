"""The build identity a wheel carries, or a source checkout's git state."""

from __future__ import annotations

import sys
import types

import pytest

from markitai import __version__, build_info


@pytest.fixture(autouse=True)
def _fresh() -> object:
    build_info.build_identity.cache_clear()
    yield
    build_info.build_identity.cache_clear()


def test_a_wheel_reports_its_build_info(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("markitai._build_info")
    module.COMMIT = "0123abc"  # type: ignore[attr-defined]
    module.BUILT_AT = "2026-10-15T08:00:00Z"  # type: ignore[attr-defined]
    module.DIRTY = False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "markitai._build_info", module)
    # A wheel never asks git
    monkeypatch.setattr(build_info, "_git", lambda *_args: pytest.fail("git"))
    assert build_info.build_identity() == {
        "version": __version__,
        "commit": "0123abc",
        "built_at": "2026-10-15T08:00:00Z",
        "dirty": False,
    }


def test_a_source_checkout_reports_its_commit_without_a_build_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    monkeypatch.setitem(sys.modules, "markitai._build_info", None)  # ImportError
    top = Path(build_info.__file__).resolve().parents[4]
    answers = {
        ("rev-parse", "--show-toplevel"): str(top),
        ("rev-parse", "--short", "HEAD"): "4567def",
        ("status", "--porcelain", "--untracked-files=no"): " M README.md",
    }
    monkeypatch.setattr(build_info, "_git", lambda _root, *args: answers[args])
    assert build_info.build_identity() == {
        "version": __version__,
        "commit": "4567def",
        "built_at": None,
        "dirty": True,
    }


def test_another_repository_around_an_installed_copy_is_not_its_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "markitai._build_info", None)
    monkeypatch.setattr(
        build_info,
        "_git",
        lambda _root, *args: "/elsewhere" if "--show-toplevel" in args else "89abcde",
    )
    assert build_info.build_identity()["commit"] is None


def test_without_build_info_or_git_everything_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "markitai._build_info", None)
    monkeypatch.setattr(build_info, "_git", lambda *_args: None)
    assert build_info.build_identity() == {
        "version": __version__,
        "commit": None,
        "built_at": None,
        "dirty": False,
    }
