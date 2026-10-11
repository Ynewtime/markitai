"""The CLI's standard streams: UTF-8 on Windows, before the marks are picked."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from markitai.cli.streams import configure_standard_streams


class _Stream(io.StringIO):
    def __init__(self, encoding: str) -> None:
        super().__init__()
        self._encoding = encoding
        self.calls: list[dict[str, str]] = []

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


@pytest.mark.parametrize(
    ("platform", "encoding", "expected"),
    [
        # A Windows pipe uses the ANSI code page otherwise
        ("win32", "cp1252", [{"encoding": "utf-8", "errors": "replace"}]),
        ("win32", "utf-8", [{"encoding": "utf-8", "errors": "replace"}]),
        ("linux", "iso-8859-1", [{"errors": "replace"}]),
        ("darwin", "utf-8", []),
    ],
)
def test_streams_by_platform(
    platform: str, encoding: str, expected: list[dict[str, str]]
) -> None:
    stdout, stderr = _Stream(encoding), _Stream(encoding)
    with (
        patch.object(sys, "platform", platform),
        patch.object(sys, "stdout", stdout),
        patch.object(sys, "stderr", stderr),
    ):
        configure_standard_streams()
    assert stdout.calls == expected and stderr.calls == expected


_WINDOWS_PIPE = """
import sys
# Real Linux modules first: their own Windows branches need Windows
import asyncio, ctypes.wintypes, loguru, multiprocessing, pydantic, rich.console
import rich_click, subprocess
sys.platform = "win32"
import markitai.cli
from markitai.utils import term
print(term.MARK_SUCCESS, term.MARK_ERROR, sys.stdout.encoding)
"""


def test_a_windows_pipe_gets_utf8_before_the_marks_are_picked() -> None:
    """markitai.cli.ui imports term before cli.main ran its stream setup."""
    result = subprocess.run(
        [sys.executable, "-c", _WINDOWS_PIPE],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout.decode("utf-8").split() == ["✓", "✗", "utf-8"]
