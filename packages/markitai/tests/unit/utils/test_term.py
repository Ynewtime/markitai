"""Tests for the shared terminal consoles."""

from __future__ import annotations

import io
import sys
from typing import Any
from unittest.mock import MagicMock, patch

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
            assert term._streams_take_unicode(legacy_stdout=False) is True


_VT = term._ENABLE_VIRTUAL_TERMINAL_PROCESSING


class TestWindowsMarks:
    """Marks mean the same on every platform (owner rule, as in markitai-rust).

    Windows pipes and files get the Unicode marks (the CLI writes UTF-8 to
    them); so does a console that reads escape sequences, after markitai
    switches them on; only a console that refuses (legacy conhost) gets the
    ASCII set. The console API is faked: these run on any OS.
    """

    @staticmethod
    def _legacy(mode: int | None, settable: bool = True) -> tuple[bool, list[int]]:
        calls: list[int] = []

        def set_mode(handle: int, new_mode: int) -> bool:
            calls.append(new_mode)
            return settable

        with (
            patch.object(sys, "platform", "win32"),
            patch.object(term, "_console_mode", return_value=mode),
            patch.object(term, "_set_console_mode", side_effect=set_mode),
        ):
            return term._legacy_windows_console(term._STD_OUTPUT_HANDLE), calls

    def test_a_pipe_or_file_is_not_a_legacy_console(self) -> None:
        assert self._legacy(None) == (False, [])

    def test_a_console_with_escape_sequences_on_is_modern(self) -> None:
        assert self._legacy(0x3 | _VT) == (False, [])

    def test_escape_sequences_are_switched_on_first(self) -> None:
        """Windows 10 1511+ conhost starts without them; asking turns them on."""
        assert self._legacy(0x3) == (False, [0x3 | _VT])

    def test_a_console_that_refuses_them_is_legacy(self) -> None:
        assert self._legacy(0x3, settable=False) == (True, [0x3 | _VT])

    def test_no_console_api_is_not_legacy(self) -> None:
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(term, "_console_mode", side_effect=OSError),
        ):
            assert term._legacy_windows_console(term._STD_OUTPUT_HANDLE) is False

    def test_other_platforms_never_ask(self) -> None:
        with (
            patch.object(sys, "platform", "darwin"),
            patch.object(term, "_console_mode") as mode,
        ):
            assert term._legacy_windows_console(term._STD_OUTPUT_HANDLE) is False
        mode.assert_not_called()

    def test_only_a_legacy_console_gets_ascii_marks(self) -> None:
        utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with (
            patch.object(sys, "stdout", utf8),
            patch.object(sys, "stderr", utf8),
            patch.dict("os.environ", {"NO_COLOR": "1"}),
        ):
            assert term._streams_take_unicode(legacy_stdout=False) is True
            assert term._streams_take_unicode(legacy_stdout=True) is False

    @pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
    def test_term_dumb_gets_ascii_everywhere(self, platform: str) -> None:
        utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with (
            patch.object(sys, "platform", platform),
            patch.object(sys, "stdout", utf8),
            patch.object(sys, "stderr", utf8),
            patch.dict("os.environ", {"TERM": "dumb"}),
        ):
            assert term._streams_take_unicode(legacy_stdout=False) is False

    def test_consoles_take_their_streams_answer_not_richs(self) -> None:
        """Rich calls every Windows pipe legacy: square panels, win32 colours."""
        with (
            patch("rich.console.detect_legacy_windows", return_value=True),
            patch.object(term, "_LEGACY_STDOUT", False),
            patch.object(term, "_LEGACY_STDERR", True),
        ):
            stdout = term._SoftWrapConsole(file=io.StringIO())
            stderr = term.make_stderr_console()
            explicit = term._SoftWrapConsole(file=io.StringIO(), legacy_windows=True)
        assert stdout.legacy_windows is False
        assert stderr.legacy_windows is True
        assert explicit.legacy_windows is True

    def test_the_console_api_calls(self) -> None:
        """The ctypes calls themselves, against a fake kernel32."""
        import ctypes

        class Kernel32:
            def __init__(self, mode: int | None) -> None:
                self.mode, self.set_to = mode, None

            def GetStdHandle(self, number: int) -> int:  # noqa: N802 - win32 name
                assert number == term._STD_OUTPUT_HANDLE
                return 7

            def GetConsoleMode(self, handle: int, out: Any) -> int:  # noqa: N802
                assert handle == 7
                if self.mode is None:
                    return 0
                ctypes.cast(out, ctypes.POINTER(ctypes.c_ulong))[0] = self.mode
                return 1

            def SetConsoleMode(self, handle: int, mode: int) -> int:  # noqa: N802
                self.set_to = mode
                return 1

        for mode, legacy in ((None, False), (0x3, False), (0x3 | _VT, False)):
            kernel32 = Kernel32(mode)
            with (
                patch.object(sys, "platform", "win32"),
                patch.object(
                    ctypes, "WinDLL", MagicMock(return_value=kernel32), create=True
                ),
            ):
                assert term._legacy_windows_console(term._STD_OUTPUT_HANDLE) is legacy
            assert kernel32.set_to == (0x3 | _VT if mode == 0x3 else None)
