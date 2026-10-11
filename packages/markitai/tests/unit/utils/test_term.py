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

    A console that reads escape sequences gets the Unicode marks, after
    markitai switches them on; only a console that refuses (legacy conhost)
    gets the ASCII set. Pipes and files: ``TestWindowsCodePage``. The console
    API is faked: these run on any OS.
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
            patch.object(sys, "platform", "win32"),
            patch.object(sys, "stdout", utf8),
            patch.object(sys, "stderr", utf8),
            patch.object(term, "_console_mode", return_value=0x3 | _VT),
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


class TestWindowsCodePage:
    """Windows pipes and files follow the console's output code page (owner
    rule, 2026-10-11, as in markitai-rust): Windows PowerShell decodes a
    program's piped output with ``[Console]::OutputEncoding``, so at code page
    437 the UTF-8 bytes of ``✓`` read as ``Γ£ô``. Each case is faked."""

    @pytest.mark.parametrize(
        ("kind", "vt", "code_page", "unicode"),
        [
            ("console", True, None, True),
            ("console", False, None, False),
            ("pty", True, None, True),
            ("redirected", True, 65001, True),
            ("redirected", True, 437, False),
            ("redirected", True, 850, False),
            ("redirected", True, 936, False),
            ("redirected", True, 1252, False),
            # No console attached: no code page to follow, Unicode as before
            ("redirected", True, None, True),
        ],
    )
    def test_the_rule(
        self, kind: str, vt: bool, code_page: int | None, unicode: bool
    ) -> None:
        assert term._windows_unicode(kind, vt=vt, code_page=code_page) is unicode

    @staticmethod
    def _decide(
        *,
        stdout: str,
        stderr: str = "console",
        code_page: int | None = 437,
        legacy: bool = False,
    ) -> bool:
        """``_streams_take_unicode`` on a faked Windows: each stream is a
        ``console``, a ``pipe`` or an msys ``pty``."""
        utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        kinds = {term._STD_OUTPUT_HANDLE: stdout, term._STD_ERROR_HANDLE: stderr}

        def mode(handle: int) -> int | None:
            return 0x3 | _VT if kinds[handle] == "console" else None

        with (
            patch.object(sys, "platform", "win32"),
            patch.object(sys, "stdout", utf8),
            patch.object(sys, "stderr", utf8),
            patch.object(term, "_console_mode", side_effect=mode),
            patch.object(
                term, "_msys_terminal", side_effect=lambda h: kinds[h] == "pty"
            ),
            patch.object(term, "_console_output_code_page", return_value=code_page),
        ):
            return term._streams_take_unicode(
                legacy_stdout=legacy and stdout == "console",
                legacy_stderr=legacy and stderr == "console",
            )

    def test_consoles_with_escape_sequences_keep_the_marks(self) -> None:
        assert self._decide(stdout="console") is True

    def test_a_legacy_console_gets_ascii(self) -> None:
        assert self._decide(stdout="console", legacy=True) is False

    def test_a_pipe_at_an_oem_code_page_gets_ascii(self) -> None:
        """`markitai doctor | Out-File` in Windows PowerShell 5.1 at 437."""
        assert self._decide(stdout="pipe", code_page=437) is False

    def test_a_pipe_at_utf8_keeps_the_marks(self) -> None:
        """After `[Console]::OutputEncoding = [Text.Encoding]::UTF8`."""
        assert self._decide(stdout="pipe", code_page=65001) is True

    def test_no_console_keeps_the_marks(self) -> None:
        """A service or a detached process: no code page to follow."""
        assert self._decide(stdout="pipe", stderr="pipe", code_page=None) is True

    def test_one_redirected_stream_decides_for_both(self) -> None:
        """`markitai dir -o out 2> log.txt` at 437: the log gets no mark it
        cannot decode, and the console the same set."""
        assert self._decide(stdout="console", stderr="pipe", code_page=437) is False

    def test_an_msys_terminal_keeps_the_marks(self) -> None:
        """mintty decodes UTF-8 itself, whatever the code page."""
        assert self._decide(stdout="pty", stderr="pty", code_page=437) is True

    def test_term_dumb_still_wins(self) -> None:
        with patch.dict("os.environ", {"TERM": "dumb"}):
            assert self._decide(stdout="pipe", code_page=65001) is False

    def test_the_code_page_call(self) -> None:
        import ctypes

        for page, expected in ((437, 437), (65001, 65001), (0, None)):
            kernel32 = MagicMock()
            kernel32.GetConsoleOutputCP.return_value = page
            with (
                patch.object(sys, "platform", "win32"),
                patch.object(
                    ctypes, "WinDLL", MagicMock(return_value=kernel32), create=True
                ),
            ):
                assert term._console_output_code_page() == expected
        with patch.object(sys, "platform", "linux"):
            assert term._console_output_code_page() is None

    @pytest.mark.parametrize(
        ("name", "pty"),
        [
            ("\\msys-1888ae32e00d56aa-pty0-to-master", True),
            ("\\cygwin-e022582115c10879-pty3-from-master", True),
            ("\\Device\\NamedPipe\\other", False),
        ],
    )
    def test_an_msys_pipe_is_known_by_its_name(self, name: str, pty: bool) -> None:
        import ctypes

        def file_info(handle: int, kind: int, buffer: Any, size: Any) -> int:
            encoded = name.encode("utf-16-le")
            ctypes.memmove(
                buffer, len(encoded).to_bytes(4, "little") + encoded, 4 + len(encoded)
            )
            return 1

        kernel32 = MagicMock()
        kernel32.GetFileInformationByHandleEx.side_effect = file_info
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(
                ctypes, "WinDLL", MagicMock(return_value=kernel32), create=True
            ),
        ):
            assert term._msys_terminal(term._STD_OUTPUT_HANDLE) is pty
