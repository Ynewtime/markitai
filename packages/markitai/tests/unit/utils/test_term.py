"""Tests for the shared terminal consoles."""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest
from rich.console import Console

from markitai.utils import term
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


class TestPrintMarked:
    """Marked lines wrap under their text on a terminal, never when piped."""

    TEXT = (
        "VLM OCR: available — claude-agent/sonnet, deepseek/deepseek-flash "
        "(+2 more) reads page images (MARKITAI_NO_VLM_OCR=1 forces local RapidOCR)"
    )

    @pytest.mark.parametrize("width", [80, 40])
    def test_hanging_indent_on_a_terminal(self, width: int) -> None:
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=width, color_system=None)

        term.print_marked(self.TEXT, mark=term.MARK_SUCCESS, console=console)

        lines = buf.getvalue().splitlines()
        assert len(lines) > 1
        assert lines[0].startswith(f"  {term.MARK_SUCCESS} VLM OCR:")
        assert all(line.startswith("    ") for line in lines[1:])
        assert all(len(line) <= width for line in lines)
        assert " ".join(line.strip() for line in lines).endswith("local RapidOCR)")

    def test_one_line_when_piped(self) -> None:
        buf = io.StringIO()
        console = term._SoftWrapConsole(file=buf, width=80)

        term.print_marked(self.TEXT, mark=term.MARK_WARNING, console=console)

        assert buf.getvalue() == f"  ! {self.TEXT}\n"

    def test_long_word_is_never_split(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=40, color_system=None)
        path = "/home/someone/.local/share/uv/tools/markitai/bin/python"

        term.print_marked(
            f"Interpreter: {path} here", mark=term.MARK_INFO, console=console
        )

        assert f"    {path}" in buf.getvalue().splitlines()

    def test_text_is_never_read_as_markup(self) -> None:
        buf = io.StringIO()
        console = term._SoftWrapConsole(file=buf, width=80)

        term.print_marked('uv tool install "markitai[ocr]"', mark="", console=console)

        assert "markitai[ocr]" in buf.getvalue()

    def test_stdout_console_does_not_wrap_when_piped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """doctor and init print on stdout; piped, one line per message."""
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)
        monkeypatch.setattr(term, "_console", None)

        term.get_console().print("word " * 40)

        assert len(buf.getvalue().splitlines()) == 1


class TestAsciiMarks:
    """A stream that cannot carry the marks gets the ASCII set, not a crash.

    `PYTHONIOENCODING=latin-1 markitai doctor` raised UnicodeEncodeError on
    the title mark.
    """

    def test_latin1_stream_selects_ascii(self) -> None:
        latin1 = io.TextIOWrapper(io.BytesIO(), encoding="latin-1")
        with patch.object(sys, "stdout", latin1):
            assert term._streams_take_unicode() is False

    def test_utf8_streams_keep_unicode(self) -> None:
        utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with patch.object(sys, "stdout", utf8), patch.object(sys, "stderr", utf8):
            if sys.platform != "win32":
                assert term._streams_take_unicode() is True
