"""Unified UI components for Markitai CLI.

This module provides consistent visual components for CLI output,
including status messages, titles, and progress indicators.

Usage:
    from markitai.cli.ui import title, success, error, warning, info

    title("Processing Files")
    success("File converted successfully")
    error("Conversion failed", detail="Invalid format")
"""

from __future__ import annotations

from rich.console import Console

from markitai.cli.console import get_console

# Symbol constants for visual markers
MARK_SUCCESS = "\u2713"  # Checkmark
MARK_ERROR = "\u2717"  # Cross
MARK_WARNING = "!"  # Exclamation
MARK_INFO = "\u2022"  # Bullet
MARK_TITLE = "\u25c6"  # Diamond
MARK_LINE = "\u2502"  # Vertical line


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


def title(text: str, *, console: Console | None = None) -> None:
    """Display a title with diamond symbol.

    Args:
        text: The title text to display.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"[cyan]{MARK_TITLE}[/] [bold]{text}[/]")
    c.print()


def success(text: str, *, console: Console | None = None) -> None:
    """Display a success message with checkmark.

    Args:
        text: The success message to display.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"  [green]{MARK_SUCCESS}[/] {text}")


def error(
    text: str, *, detail: str | None = None, console: Console | None = None
) -> None:
    """Display an error message with cross symbol.

    Args:
        text: The error message to display.
        detail: Optional detail text shown on a separate line.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"  [red]{MARK_ERROR}[/] {text}")
    if detail:
        c.print(f"    [dim]{MARK_LINE} {detail}[/]")


def warning(
    text: str, *, detail: str | None = None, console: Console | None = None
) -> None:
    """Display a warning message with exclamation symbol.

    Args:
        text: The warning message to display.
        detail: Optional detail text shown on a separate line.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"  [yellow]{MARK_WARNING}[/] {text}")
    if detail:
        c.print(f"    [dim]{MARK_LINE} {detail}[/]")


def info(text: str, *, console: Console | None = None) -> None:
    """Display an info message with bullet symbol.

    Args:
        text: The info message to display.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"  [dim]{MARK_INFO}[/] {text}")


def step(text: str, *, console: Console | None = None) -> None:
    """Display a step message with vertical line symbol.

    Args:
        text: The step message to display.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"  [dim]{MARK_LINE}[/] {text}")


def section(text: str, *, console: Console | None = None) -> None:
    """Display a section header in bold.

    Args:
        text: The section header text to display.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print(f"[bold]{text}[/]")


def summary(text: str, *, console: Console | None = None) -> None:
    """Display a summary message with checkmark and leading blank line.

    Args:
        text: The summary message to display.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print()
    c.print(f"[green]{MARK_SUCCESS}[/] {text}")
