"""Terminal output foundation: shared Console singletons and text helpers.

This is infrastructure, not CLI logic — it lives below the presentation
layer so engines like batch.py can render without importing markitai.cli.
``markitai.cli.console`` and ``markitai.cli.ui`` re-export everything here
for their existing callers.
"""

from __future__ import annotations

from rich.console import Console

# Singleton instances
_console: Console | None = None
_stderr_console: Console | None = None


def get_console() -> Console:
    """Get the shared stdout Console instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def get_stderr_console() -> Console:
    """Get the shared stderr Console instance."""
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(stderr=True)
    return _stderr_console


# Status glyphs shared by every terminal surface
MARK_SUCCESS = "✓"  # Checkmark
MARK_ERROR = "✗"  # Cross
MARK_WARNING = "!"  # Exclamation
MARK_INFO = "•"  # Bullet
MARK_TITLE = "◆"  # Diamond
MARK_LINE = "│"  # Vertical line


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
        c.print(f"[green]{MARK_SUCCESS}[/] {text}")
    elif ok is False:
        c.print(f"[red]{MARK_ERROR}[/] {text}")
    else:
        c.print(f"[yellow]{MARK_WARNING}[/] {text}")
