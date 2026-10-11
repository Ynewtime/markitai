"""Windows pipes and files follow the console's output code page (real consoles).

Owner rule (2026-10-11), the same in markitai-rust: redirected output gets the
Unicode marks only while the attached console's output code page is UTF-8
(65001), the ASCII set otherwise, because Windows PowerShell decodes a
program's piped output with that code page. Each run starts ``cmd`` in a
hidden console of its own (``CREATE_NO_WINDOW``), sets its code page with
``chcp`` and sends ``markitai config validate`` to a file; the test process's
own console keeps its code page. The faked cases are in tests/unit/utils.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click import unstyle

from tests.terminal_env import plain_env

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows consoles and code pages"
)

_CREATE_NO_WINDOW = 0x08000000


def _console_code_page() -> int:
    import ctypes

    return int(ctypes.WinDLL("kernel32").GetConsoleOutputCP())  # type: ignore[attr-defined]


def _validate_in_console(tmp_path: Path, code_page: int) -> str:
    out = tmp_path / f"validate-{code_page}.txt"
    script = tmp_path / f"validate-{code_page}.cmd"
    # A batch file, so cmd parses no quotes of ours; paths are ASCII on CI
    script.write_text(
        "@echo off\r\n"
        f"chcp {code_page} >nul\r\n"
        f'"{sys.executable}" -m markitai config validate > "{out}" 2>&1\r\n',
        encoding="ascii",
    )
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = plain_env(
        USERPROFILE=str(home),
        HOME=str(home),
        MARKITAI_HOME=str(home / ".markitai"),
    )
    env.pop("TERM", None)
    result = subprocess.run(
        ["cmd", "/d", "/c", str(script)],
        env=env,
        creationflags=_CREATE_NO_WINDOW,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return unstyle(out.read_bytes().decode("utf-8"))


@pytest.mark.parametrize(
    ("code_page", "marked", "not_marked"),
    [
        # The OEM code page of an English console: what PowerShell 5.1 reads
        (437, "+ Configuration is valid", "✓"),
        (850, "+ Configuration is valid", "✓"),
        # After `[Console]::OutputEncoding = [Text.Encoding]::UTF8` or chcp 65001
        (65001, "✓ Configuration is valid", "+ Configuration"),
    ],
)
def test_a_file_gets_the_marks_its_console_code_page_reads(
    tmp_path: Path, code_page: int, marked: str, not_marked: str
) -> None:
    ascii_path = str(tmp_path).isascii() and sys.executable.isascii()
    if not ascii_path:
        pytest.skip("the batch file names ASCII paths")
    before = _console_code_page()
    text = _validate_in_console(tmp_path, code_page)
    assert marked in text, text
    assert not_marked not in text, text
    # The hidden console was the child's own
    assert _console_code_page() == before
