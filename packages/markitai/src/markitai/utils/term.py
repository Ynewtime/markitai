"""Terminal output foundation: shared Console singletons and text helpers.

This is infrastructure, not CLI logic — it lives below the presentation
layer so engines like batch.py can render without importing markitai.cli.
``markitai.cli.console`` and ``markitai.cli.ui`` re-export everything here
for their existing callers.
"""

from __future__ import annotations

import os
import sys
from typing import IO, Any

from rich.console import Console
from rich.text import Text


def is_terminal(stream: IO[str]) -> bool:
    """Whether ``stream`` is an interactive terminal (False when closed)."""
    isatty = getattr(stream, "isatty", None)
    try:
        return isatty is not None and bool(isatty())
    except ValueError:  # closed stream
        return False


# SetConsoleMode flag: the console processes ANSI escape sequences
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_STD_OUTPUT_HANDLE = -11
_STD_ERROR_HANDLE = -12


def _console_mode(std_handle: int) -> int | None:
    """The mode of a Windows standard handle, None when it is no console.

    A pipe, a file, NUL or an msys terminal (a pipe that mintty reads) has
    no console mode.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(
        kernel32.GetStdHandle(std_handle), ctypes.byref(mode)
    ):
        return None
    return mode.value


def _set_console_mode(std_handle: int, mode: int) -> bool:
    if sys.platform != "win32":
        return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")
    return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(std_handle), mode))


def _legacy_windows_console(std_handle: int) -> bool:
    """Whether a standard handle is a console that cannot read escape codes.

    Only a real console is asked (a pipe or a file follows the console's code
    page instead, see ``_windows_unicode``). A console that does not process
    escape sequences yet has them switched on first, which every console host
    since Windows 10 version 1511 allows; only one that refuses (older
    Windows, conhost's "Use legacy console") is legacy. It gets the ASCII
    marks and Rich's win32 rendering (markitai-rust follows the same rule).
    """
    if sys.platform != "win32":
        return False
    try:
        mode = _console_mode(std_handle)
        if mode is None or mode & _ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return False
        return not _set_console_mode(
            std_handle, mode | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
        )
    except (OSError, AttributeError):  # no console API to ask
        return False


_LEGACY_STDOUT = _legacy_windows_console(_STD_OUTPUT_HANDLE)
_LEGACY_STDERR = _legacy_windows_console(_STD_ERROR_HANDLE)

#: The UTF-8 code page
_UTF8_CODE_PAGE = 65001


def _console_output_code_page() -> int | None:
    """The output code page of the console this process is attached to.

    None without a console (a service, a detached process), where
    ``GetConsoleOutputCP`` returns 0, and off Windows.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        page = int(ctypes.WinDLL("kernel32").GetConsoleOutputCP())
    except (OSError, AttributeError):  # no console API to ask
        return None
    return page or None


def _msys_terminal(std_handle: int) -> bool:
    """Whether a standard handle is an msys or Cygwin pseudo-terminal.

    mintty (Git Bash, MSYS2) hands a program a named pipe called like
    ``\\msys-1888ae32e00d56aa-pty0-to-master``; it decodes the UTF-8 bytes
    itself. The same test as Rust's standard library makes for ``is_terminal``.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32")
        # FILE_NAME_INFO: a DWORD length, then the name in UTF-16
        buffer = ctypes.create_string_buffer(4 + 2 * 260)
        file_name_info = 2
        if not kernel32.GetFileInformationByHandleEx(
            kernel32.GetStdHandle(std_handle),
            file_name_info,
            buffer,
            wintypes.DWORD(len(buffer)),
        ):
            return False
        length = int.from_bytes(buffer.raw[:4], "little")
        name = buffer.raw[4 : 4 + length].decode("utf-16-le", "replace")
    except (OSError, AttributeError, ValueError):
        return False
    return ("msys-" in name or "cygwin-" in name) and "-pty" in name


def _windows_unicode(
    kind: str, *, vt: bool = True, code_page: int | None = None
) -> bool:
    """Whether one Windows output stream gets the Unicode marks (owner rule,
    2026-10-11, the same in markitai-rust's ``term::windows_unicode``).

    ``kind`` is what the stream is:

    * ``"console"``: the marks where it reads escape sequences (``vt``:
      Windows 10 version 1511 and later, Windows Terminal, VS Code), the
      ASCII set on a legacy console;
    * ``"pty"``: mintty or another msys terminal, which decodes UTF-8 itself;
    * ``"redirected"``: a pipe or a file follows the attached console's output
      code page (``code_page``), because Windows PowerShell decodes a
      program's output with it (``[Console]::OutputEncoding``): read at code
      page 437, the UTF-8 bytes of ``✓`` show as ``Γ£ô``. So the marks only
      at 65001 (UTF-8). With no console attached (``code_page`` None) there
      is no code page to follow, and the marks stay Unicode, as on macOS and
      Linux.
    """
    if kind == "console":
        return vt
    if kind == "pty":
        return True
    return code_page is None or code_page == _UTF8_CODE_PAGE


def _windows_output(
    std_handle: int, stream: IO[str], legacy: bool
) -> tuple[str, bool, int | None]:
    """``(kind, vt, code_page)`` of one Windows standard stream, for
    :func:`_windows_unicode`."""
    try:
        mode = _console_mode(std_handle)
    except (OSError, AttributeError):  # no console API to ask
        mode = None
    if mode is not None:
        return "console", not legacy, None
    if is_terminal(stream) or _msys_terminal(std_handle):
        return "pty", True, None
    return "redirected", True, _console_output_code_page()


class _SoftWrapConsole(Console):
    """Console that only hard-wraps on a real terminal.

    Off a TTY (pipes, log files, CI) Rich falls back to 80 columns and would
    break long messages mid-sentence, so ``print`` soft-wraps there instead:
    one line per message. Checked per print because ``sys.stdout`` and
    ``sys.stderr`` can be swapped after creation.

    On Windows, Rich takes every stream that is not a console with escape
    sequences already on (a pipe, a file, a console nobody switched over)
    for a legacy console: square panel corners, win32 colour calls. Its
    stream's own answer (``_legacy_windows_console``) is used instead.

    Progress bars, rules, boxes and panels are drawn with the same choice as
    the status marks: Rich draws them in ASCII for a non-UTF-8 ``encoding``,
    which this console reports while the marks are ASCII. Rich reads the
    encoding only for that choice; the stream itself stays UTF-8.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "legacy_windows" not in kwargs:
            kwargs["legacy_windows"] = (
                _LEGACY_STDERR if kwargs.get("stderr") else _LEGACY_STDOUT
            )
        super().__init__(*args, **kwargs)

    @property
    def encoding(self) -> str:
        return super().encoding if _UNICODE_MARKS else "ascii"

    @property
    def soft_wrap(self) -> bool:
        return self._soft_wrap or not is_terminal(self.file)

    @soft_wrap.setter
    def soft_wrap(self, value: bool) -> None:
        self._soft_wrap = value


# Kept for callers and tests that name the stderr flavour
_StderrConsole = _SoftWrapConsole


def make_stderr_console() -> Console:
    """Create a stderr Console that does not hard-wrap when piped."""
    return _SoftWrapConsole(stderr=True)


# Singleton instances
_console: Console | None = None
_stderr_console: Console | None = None


def get_console() -> Console:
    """Get the shared stdout Console instance."""
    global _console
    if _console is None:
        _console = _SoftWrapConsole()
    return _console


def get_stderr_console() -> Console:
    """Get the shared stderr Console instance."""
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = make_stderr_console()
    return _stderr_console


def _streams_take_unicode(
    legacy_stdout: bool = _LEGACY_STDOUT, legacy_stderr: bool = _LEGACY_STDERR
) -> bool:
    """Whether stdout and stderr can carry the Unicode status marks.

    The marks mean the same on every platform, so only output that would not
    show them gets the ASCII set (see the terminal style guide): ``TERM=dumb``;
    a stream that cannot encode them (a non-UTF-8 locale such as
    ``LANG=en_US.ISO-8859-1``, or ``PYTHONIOENCODING=ascii``: printing one
    raised UnicodeEncodeError); and on Windows, a stream
    :func:`_windows_unicode` gives the ASCII set: a legacy console, or a pipe
    or file while the console's output code page is not UTF-8. ``NO_COLOR``
    turns colour off, never the marks.

    One answer for both streams: when either needs the ASCII set, both get it,
    so one run never mixes the two sets (markitai-rust judges each stream on
    its own; the difference shows only when one stream is a console and the
    other a pipe or file at another code page, and never puts a mark a reader
    cannot decode into a pipe).
    """
    if os.environ.get("TERM") == "dumb":
        return False
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            "✓✗•◆│→○".encode(encoding)
        except (UnicodeEncodeError, LookupError):
            return False
    if sys.platform != "win32":
        return True
    for std_handle, stream, legacy in (
        (_STD_OUTPUT_HANDLE, sys.stdout, legacy_stdout),
        (_STD_ERROR_HANDLE, sys.stderr, legacy_stderr),
    ):
        if stream is None:  # pythonw: no stream, nothing printed there
            continue
        kind, vt, code_page = _windows_output(std_handle, stream, legacy)
        if not _windows_unicode(kind, vt=vt, code_page=code_page):
            return False
    return True


_UNICODE_MARKS = _streams_take_unicode()

# Status glyphs shared by every terminal surface; one meaning each:
# success, error, warning, info/list item, title, detail guide, next
# action, skipped. The colour that goes with each lives in the helpers below.
MARK_SUCCESS = "✓" if _UNICODE_MARKS else "+"  # Checkmark
MARK_ERROR = "✗" if _UNICODE_MARKS else "x"  # Cross
MARK_WARNING = "!"  # Exclamation
MARK_INFO = "•" if _UNICODE_MARKS else "-"  # Bullet
MARK_TITLE = "◆" if _UNICODE_MARKS else "*"  # Diamond
MARK_LINE = "│" if _UNICODE_MARKS else "|"  # Vertical line
MARK_ARROW = "→" if _UNICODE_MARKS else ">"  # Next action
MARK_SKIP = "○" if _UNICODE_MARKS else "o"  # Skipped on purpose
# Drawn around the marks, with the same choice
ELLIPSIS = "…" if _UNICODE_MARKS else "..."
#: Rich spinner of progress displays: braille dots, or ASCII - \ | /
SPINNER = "dots" if _UNICODE_MARKS else "line"

# Colour per mark (Rich styles): green only for success, yellow only for
# warnings, red only for errors; info, guides and skips stay neutral.
MARK_STYLES = {
    MARK_SUCCESS: "green",
    MARK_ERROR: "red",
    MARK_WARNING: "yellow",
    MARK_INFO: "dim",
    MARK_LINE: "dim",
    MARK_ARROW: "cyan",
    MARK_SKIP: "dim",
}


def print_marked(
    text: str | Text,
    *,
    mark: str,
    style: str | None = None,
    console: Console | None = None,
    indent: int = 2,
    text_style: str = "",
) -> None:
    """Print ``<indent><mark> <text>`` the same way on every surface.

    On a terminal a long line wraps at word boundaries with a hanging indent
    under the first character of the text, instead of Rich's wrap back to
    column 0. Off a terminal (pipe, file, CI) it stays one line, so every
    message can be grepped.

    Args:
        text: Plain text (never parsed as markup) or a styled ``Text``.
        mark: One of the ``MARK_*`` glyphs, or "" for an unmarked line.
        style: Style for the mark; defaults to the mark's own colour.
        console: Console to print on (defaults to the shared stdout one).
        indent: Columns before the mark.
        text_style: Style for plain ``text``.
    """
    c = console or get_console()
    body = text.copy() if isinstance(text, Text) else Text(text, style=text_style)
    mark_style = MARK_STYLES.get(mark, "") if style is None else style
    # No mark: a plain line indented like the text of a marked one
    prefix = Text.assemble(" " * indent, (mark, mark_style), " " if mark else "")
    if not c.is_terminal:
        c.print(prefix + body, soft_wrap=True, highlight=False)
        return
    hang = prefix.cell_len
    # The Windows console wraps a line that fills the window by itself, so a
    # full-width line there turns into an extra (blank) row: stop one short
    edge = 1 if sys.platform == "win32" else 0
    width = max(c.width - hang - edge, 20)
    out = prefix.copy()
    first = True
    for paragraph in body.split("\n", allow_blank=True):
        for line in _wrap_keeping_words(paragraph, width, c):
            if not first:
                out.append("\n" + " " * hang)
            first = False
            line.rstrip()
            out.append_text(line)
    c.print(out, soft_wrap=True, highlight=False)


def _wrap_keeping_words(text: Text, width: int, console: Console) -> list[Text]:
    """Wrap at spaces; a word longer than the line (a path, a URL) stays
    whole, so the terminal wraps it visually and it still copies as one."""
    try:
        from rich._wrap import divide_line
    except ImportError:  # private helper; fall back to Rich's own wrap
        return list(text.wrap(console, width))
    return list(text.divide(divide_line(text.plain, width, fold=False)))


def print_command(
    command: str, *, console: Console | None = None, indent: int = 4
) -> None:
    """Print a command to copy on its own line, in the accent colour.

    Never wrapped by markitai: a terminal narrower than the command wraps it
    visually, and copying it back still gives one line.
    """
    c = console or get_console()
    c.print(
        Text.assemble(" " * indent, (command, "cyan")),
        soft_wrap=True,
        highlight=False,
    )


def term_width(console: Console | None = None) -> int:
    """Return current terminal width."""
    c = console or get_console()
    return c.width


def truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, appending '...' if trimmed."""
    if max_len < 4:
        return text[:max_len]
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def summarize_active_items(
    items: list[str],
    *,
    max_items: int = 3,
    max_len: int | None = None,
) -> str:
    """Summarize active concurrent items for compact progress displays.

    Args:
        items: Active item labels in display order.
        max_items: Maximum number of labels to show before collapsing the rest.
        max_len: Optional maximum display length.

    Returns:
        Compact summary string such as ``"a.txt, b.txt +2"``.
    """
    unique_items = [item.strip() for item in items if item and item.strip()]
    unique_items = list(dict.fromkeys(unique_items))
    if not unique_items:
        return ""

    shown = unique_items[:max_items]
    remaining = len(unique_items) - len(shown)
    summary = ", ".join(shown)
    if remaining > 0:
        summary = f"{summary} +{remaining}"
    if max_len is not None:
        summary = truncate(summary, max_len)
    return summary


def summary(
    text: str, *, ok: bool | None = True, console: Console | None = None
) -> None:
    """Display a summary message with a status glyph and leading blank line.

    Args:
        text: The summary message to display.
        ok: True renders a green checkmark, False a red cross, and None a
            yellow warning mark.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print()
    if ok is True:
        mark = MARK_SUCCESS
    elif ok is False:
        mark = MARK_ERROR
    else:
        mark = MARK_WARNING
    print_marked(text, mark=mark, console=c, indent=0)
