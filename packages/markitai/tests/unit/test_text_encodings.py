"""markitai reads and writes text as UTF-8, whatever the system's code page.

Python's default text encoding is the locale's: cp936 (GBK) on a Chinese
Windows, where a UTF-8 file or a program's UTF-8 output with any character
GBK lacks raises UnicodeDecodeError, or reads as other characters. Every
text read, write and child-process pipe in markitai names its encoding.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "markitai"
_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}


def _keywords(node: ast.Call) -> dict[str, ast.expr]:
    return {k.arg: k.value for k in node.keywords if k.arg}


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _text_mode(node: ast.Call, position: int) -> bool:
    mode = _keywords(node).get("mode")
    if mode is None and len(node.args) > position:
        mode = node.args[position]
    if mode is None:
        return True  # "r"
    return not (isinstance(mode, ast.Constant) and "b" in str(mode.value))


def _locale_dependent_calls() -> list[str]:
    found = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = _keywords(node)
            if "encoding" in keywords:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else ""
            where = f"{path.relative_to(_SRC)}:{node.lineno}"
            if (
                name in _SUBPROCESS_CALLS
                and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and (
                    _is_true(keywords.get("text"))
                    or _is_true(keywords.get("universal_newlines"))
                )
            ):
                found.append(f"{where}: subprocess.{name}(text=True)")
            elif isinstance(func, ast.Name) and func.id == "open":
                if _text_mode(node, 1):
                    found.append(f"{where}: open() in text mode")
            elif name == "read_text" and not node.args:
                found.append(f"{where}: .read_text()")
            elif name == "write_text" and len(node.args) < 2:
                found.append(f"{where}: .write_text()")
    return found


def test_every_text_read_write_and_pipe_names_its_encoding() -> None:
    """A text call without an encoding works on Linux and macOS (UTF-8) and
    breaks on a Chinese Windows: name it (``encoding="utf-8"``, and
    ``errors="replace"`` for another program's output)."""
    assert _locale_dependent_calls() == []


def _gbk_locale() -> dict[str, str] | None:
    """An environment whose default text encoding is GBK, as on a Chinese
    Windows, where this system has that locale (``locale-gen zh_CN.GBK``)."""
    if sys.platform == "win32":
        return None
    env = {**os.environ, "LANG": "zh_CN.GBK", "LC_ALL": "zh_CN.GBK"}
    env.pop("PYTHONUTF8", None)
    env.pop("PYTHONIOENCODING", None)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import locale; print(locale.getpreferredencoding(False))",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    return env if probe.stdout.strip().upper() in ("GBK", "CP936") else None


def test_a_programs_utf8_output_is_read_as_utf8_under_a_gbk_locale(
    tmp_path: Path,
) -> None:
    """A provider CLI prints UTF-8: under cp936 its ``✓`` stopped the
    ``--version`` check with UnicodeDecodeError."""
    env = _gbk_locale()
    if env is None:
        pytest.skip("no zh_CN.GBK locale on this system")
    tool = tmp_path / "fake-cli"
    tool.write_bytes(
        "#!/bin/sh\nprintf 'fake-cli 1.2.3 \\342\\234\\223 已就绪\\n'\n".encode()
    )
    tool.chmod(0o755)
    script = textwrap.dedent(
        f"""
        from markitai.providers.discovery import _cli_version
        print(ascii(_cli_version({str(tool)!r})))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ascii("fake-cli 1.2.3 ✓ 已就绪")
