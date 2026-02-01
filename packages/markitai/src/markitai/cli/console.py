"""Centralized Rich Console instances for Markitai CLI.

This module provides singleton Console instances to ensure consistent
output formatting and prevent multiple Console instances from conflicting.

Usage:
    from markitai.cli.console import get_console, get_stderr_console

    console = get_console()  # stdout console
    stderr_console = get_stderr_console()  # stderr console
"""

from __future__ import annotations

from rich.console import Console

# Singleton instances
_console: Console | None = None
_stderr_console: Console | None = None


def get_console() -> Console:
    """Get the shared stdout Console instance.

    Returns:
        The singleton Console instance for stdout output.
    """
    global _console
    if _console is None:
        _console = Console()
    return _console


def get_stderr_console() -> Console:
    """Get the shared stderr Console instance.

    Returns:
        The singleton Console instance for stderr output.
    """
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(stderr=True)
    return _stderr_console


def reset_consoles() -> None:
    """Reset console instances (for testing purposes)."""
    global _console, _stderr_console
    _console = None
    _stderr_console = None
