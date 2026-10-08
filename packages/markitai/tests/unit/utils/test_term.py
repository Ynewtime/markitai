"""Tests for the shared terminal consoles."""

from __future__ import annotations

import io

import pytest

from markitai.utils.term import make_stderr_console


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class TestStderrConsoleWrapping:
    """Stderr diagnostics hard-wrap on a terminal only."""

    MESSAGE = "Error: Path '" + "/very/long/path" * 20 + "/missing.pdf' does not exist."

    def test_does_not_wrap_when_not_a_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.StringIO()
        monkeypatch.setattr("sys.stderr", buf)

        make_stderr_console().print(f"[red]{self.MESSAGE}[/red]")

        assert buf.getvalue() == self.MESSAGE + "\n"

    def test_wraps_to_width_on_a_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        buf = _TtyStringIO()
        monkeypatch.setattr("sys.stderr", buf)
        console = make_stderr_console()
        console.width = 80

        console.print(self.MESSAGE, highlight=False)

        lines = buf.getvalue().splitlines()
        assert len(lines) > 1
        assert all(len(line) <= 80 for line in lines)

    def test_explicit_soft_wrap_argument_still_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        buf = io.StringIO()
        monkeypatch.setattr("sys.stderr", buf)
        console = make_stderr_console()
        console.width = 80

        console.print(self.MESSAGE, soft_wrap=False)

        assert len(buf.getvalue().splitlines()) > 1
