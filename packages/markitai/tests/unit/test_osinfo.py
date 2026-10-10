"""The OS name without WMI on Windows (markitai.utils.osinfo)."""

from __future__ import annotations

import ast
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from markitai.utils import osinfo

_SRC = Path(osinfo.__file__).resolve().parents[1]


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """sys.platform says Windows, and any WMI query fails the test."""

    def no_wmi() -> None:
        raise AssertionError("platform.uname() runs WMI queries on Windows")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "uname", no_wmi)
    monkeypatch.setattr(platform, "win32_ver", no_wmi)


@pytest.mark.usefixtures("windows")
class TestOnWindows:
    def test_system_asks_no_wmi(self) -> None:
        assert osinfo.system() == "Windows"

    @pytest.mark.parametrize(("build", "release"), [(22631, "11"), (19045, "10")])
    def test_release_from_the_build_number(
        self, monkeypatch: pytest.MonkeyPatch, build: int, release: str
    ) -> None:
        monkeypatch.setattr(
            sys,
            "getwindowsversion",
            lambda: SimpleNamespace(major=10, build=build),
            raising=False,
        )
        assert osinfo.release() == release

    def test_machine_is_this_process_architecture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
        assert osinfo.machine() == "AMD64"

    def test_a_patched_platform_system_still_decides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tests simulate macOS or Linux by patching platform.system."""
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert osinfo.system() == "Darwin"


def test_elsewhere_it_is_platform_system() -> None:
    if sys.platform != "win32":
        assert osinfo.system() == platform.system()


def test_markitai_asks_platform_only_through_osinfo() -> None:
    """platform.system() & co. ran two WMI queries per run on Windows (the
    converter pool's size at import, the system proxy before each fetch):
    ~5 s each where the session may not query WMI (an SSH logon)."""
    wmi = {"system", "release", "machine", "uname", "version", "win32_ver"}
    offenders = []
    for path in _SRC.rglob("*.py"):
        if path.name == "osinfo.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in wmi
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "platform"
            ):
                offenders.append(f"{path.relative_to(_SRC)}:{node.lineno}")
    assert not offenders, offenders
