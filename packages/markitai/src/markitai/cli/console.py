"""Centralized Rich Console instances for Markitai CLI.

The singletons now live in :mod:`markitai.utils.term` (terminal output is
foundation infrastructure, usable below the presentation layer); this module
re-exports them for existing CLI callers.

Usage:
    from markitai.cli.console import get_console, get_stderr_console

    console = get_console()  # stdout console
    stderr_console = get_stderr_console()  # stderr console
"""

from __future__ import annotations

from markitai.utils.term import get_console, get_stderr_console

__all__ = ["get_console", "get_stderr_console"]
