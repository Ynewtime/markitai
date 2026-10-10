"""Terminal output foundation: shared Console singletons and text helpers.

This is infrastructure, not CLI logic — it lives below the presentation
layer so engines like batch.py can render without importing markitai.cli.
``markitai.cli.console`` and ``markitai.cli.ui`` re-export everything here
for their existing callers.
"""

from __future__ import annotations

import sys
from typing import IO

from rich.console import Console
from rich.text import Text


def is_terminal(stream: IO[str]) -> bool:
    """Whether ``stream`` is an interactive terminal (False when closed)."""
    isatty = getattr(stream, "isatty", None)
    try:
        return isatty is not None and bool(isatty())
    except ValueError:  # closed stream
        return False


class _SoftWrapConsole(Console):
    """Console that only hard-wraps on a real terminal.

    Off a TTY (pipes, log files, CI) Rich falls back to 80 columns and would
    break long messages mid-sentence, so ``print`` soft-wraps there instead:
    one line per message. Checked per print because ``sys.stdout`` and
    ``sys.stderr`` can be swapped after creation.
    """

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


def _streams_take_unicode() -> bool:
    """Whether stdout and stderr can carry the Unicode status marks.

    A non-UTF-8 locale (``LANG=en_US.ISO-8859-1``, ``PYTHONIOENCODING=ascii``)
    gives Python a stream that cannot encode them, and printing one raised
    UnicodeEncodeError; old Windows consoles without VT support cannot show
    them. Both get the ASCII set (see the terminal style guide).
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            "✓✗•◆│→○".encode(encoding)
        except (UnicodeEncodeError, LookupError):
            return False
    if sys.platform == "win32":
        from rich.console import detect_legacy_windows

        return not detect_legacy_windows()
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
