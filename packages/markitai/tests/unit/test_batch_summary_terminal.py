"""The batch summary's warnings on a terminal and in a pipe, end to end.

A real ``markitai <dir>`` run in a fresh interpreter: on a 90-column terminal
a long warning wraps at a word with a hanging indent under its text; piped,
it stays one line. (Reported from a Windows console as a mid-sentence break
at column 89 with no indent: this pins down what markitai itself writes.)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ESCAPES = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
_WARNING = (
    "  ! 2 files skipped (image_only): english.png, sample.jpg; use --llm or --ocr "
    "for content extraction"
)


def _inputs(tmp_path: Path) -> Path:
    from PIL import Image

    folder = tmp_path / "in"
    folder.mkdir()
    Image.new("RGB", (64, 32), "white").save(folder / "english.png")
    Image.new("RGB", (64, 32), "white").save(folder / "sample.jpg")
    (folder / "note.md").write_text("# Note\n\nText.\n", encoding="utf-8")
    return folder


def _env(tmp_path: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COLUMNS", "LINES", "FORCE_COLOR", "NO_COLOR", "TTY_COMPATIBLE"}
    }
    env.update(
        HOME=str(tmp_path / "home"),
        MARKITAI_HOME=str(tmp_path / "home" / ".markitai"),
        MARKITAI_NO_REMOTE_FETCH="1",
        TERM="xterm-256color",
    )
    return env


def _command(folder: Path, out: Path) -> list[str]:
    return [sys.executable, "-m", "markitai", str(folder), "-o", str(out), "--no-cache"]


def test_piped_summary_keeps_each_warning_on_one_line(tmp_path: Path) -> None:
    folder = _inputs(tmp_path)
    result = subprocess.run(
        _command(folder, tmp_path / "out"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_env(tmp_path),
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _WARNING in result.stderr.splitlines()


@pytest.mark.skipif(sys.platform == "win32", reason="pseudo-terminals are POSIX only")
def test_terminal_summary_wraps_under_its_text(tmp_path: Path) -> None:
    import fcntl
    import pty
    import select
    import struct
    import termios

    folder = _inputs(tmp_path)
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child
        os.execvpe(sys.executable, _command(folder, tmp_path / "out"), _env(tmp_path))
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 90, 0, 0))
    output = b""
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 120)
            if not ready:
                break
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    lines = _ESCAPES.sub(b"", output).decode("utf-8", "replace").splitlines()
    start = next(i for i, line in enumerate(lines) if "files skipped" in line)
    assert lines[start : start + 2] == [
        "  ! 2 files skipped (image_only): english.png, sample.jpg; use --llm or --ocr for content",
        "    extraction",
    ]
