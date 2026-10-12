"""python -m markitai.warmup: compile what markitai imports, after an install.

The setup scripts used to have uv compile every installed file
(UV_COMPILE_BYTECODE, about 13,400, twice per setup on Windows on Arm); now
they run this in the background right after the install.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from markitai import warmup


def test_names_cover_markitai_and_what_it_imports() -> None:
    names = warmup.module_names()
    assert "markitai.cli.main" in names
    assert "markitai.cli.commands.doctor" in names
    assert "markitai.serve.app" in names
    # Read from markitai's own import statements, function-level ones too
    assert any(name.split(".")[0] == "litellm" for name in names)
    assert any(name.split(".")[0] == "pymupdf" for name in names)
    # Never the entry points that run something, never the standard library
    assert "markitai.__main__" not in names
    assert "markitai.warmup" not in names
    assert not any(name.split(".")[0] in sys.stdlib_module_names for name in names)
    assert len(names) == len(set(names))


def test_a_package_stays_in_one_process() -> None:
    names = warmup.module_names()
    shares = warmup._shares(names, 4)
    assert sorted(name for share in shares for name in share) == sorted(names)
    owner: dict[str, int] = {}
    for index, share in enumerate(shares):
        for name in share:
            top = name.split(".")[0]
            assert owner.setdefault(top, index) == index, top


def test_a_module_of_a_missing_extra_is_no_loss() -> None:
    unavailable = warmup._import_all(
        ["markitai.constants", "markitai_no_such_extra.module"]
    )
    assert unavailable == []


def test_a_module_that_fails_otherwise_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "markitai_warmup_broken.py"
    broken.write_text("raise OSError('no native library')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert warmup._import_all(["markitai_warmup_broken"]) == ["markitai_warmup_broken"]


def test_a_run_writes_its_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "markitai.warmup",
            "--jobs",
            "2",
            "--report",
            str(report),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["processes"] == 2
    assert data["crashed"] == 0
    assert data["modules"] > 100
    assert isinstance(data["unavailable"], list)
    assert data["seconds"] > 0
