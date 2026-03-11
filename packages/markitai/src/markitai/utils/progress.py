"""Progress reporting utilities.

This module provides progress reporting for CLI operations.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console

# Separate stderr console for status/progress (doesn't mix with stdout output)
# Note: Using direct Console() instead of cli.console to avoid circular import
# (utils -> cli.console -> cli.__init__ -> cli.main -> cli.processors -> utils)
stderr_console = Console(stderr=True)


class ProgressReporter:
    """Progress reporter for single file/URL conversion.

    In non-verbose mode, shows:
    1. Spinner during conversion/processing stages
    2. Completion messages after each stage
    3. Clears all output before final result

    In verbose mode, does nothing (logging handles feedback).

    When an ``output_manager`` is provided, all operations are delegated
    to it so that line-counting and erasure are handled centrally.
    """

    def __init__(
        self,
        enabled: bool = True,
        output_manager: Any = None,
    ) -> None:
        """Initialize progress reporter.

        Args:
            enabled: Whether to show progress (False in verbose mode).
            output_manager: Optional OutputManager instance. When provided,
                all output is delegated to it for centralized line tracking.
                When None, the legacy Rich Console path is used.
        """
        self.enabled = enabled
        self._om = output_manager
        self._status = None
        self._messages: list[str] = []

    def start_spinner(self, message: str) -> None:
        """Start showing a spinner with message.

        Args:
            message: Message to display with spinner
        """
        if not self.enabled:
            return
        if self._om is not None:
            self._om.start_spinner(message)
            return
        # Legacy path
        self.stop_spinner()  # Stop any existing spinner
        self._status = stderr_console.status(f"[cyan]{message}[/cyan]", spinner="dots")
        self._status.start()

    def stop_spinner(self) -> None:
        """Stop the current spinner."""
        if self._om is not None:
            self._om.stop_spinner()
            return
        # Legacy path
        if self._status is not None:
            self._status.stop()
            self._status = None

    def log(self, message: str) -> None:
        """Print a progress message.

        Args:
            message: Message to print
        """
        if not self.enabled:
            return
        if self._om is not None:
            self._om.stop_spinner()
            self._om.print(message, style="dim")
            return
        # Legacy path
        self.stop_spinner()
        self._messages.append(message)
        stderr_console.print(f"[dim]{message}[/dim]")

    def clear_and_finish(self) -> None:
        """Clear all progress output before printing final result.

        Uses ANSI escape codes to move cursor up and clear lines.
        When an OutputManager is active, delegates to ``erase_all()``.
        """
        if not self.enabled:
            return
        if self._om is not None:
            self._om.stop_spinner()
            self._om.erase_all()
            return
        # Legacy path
        self.stop_spinner()

        # Clear previous messages by moving cursor up and clearing lines
        if self._messages:
            # Move cursor up N lines and clear each line
            for _ in self._messages:
                # Move up one line and clear it
                stderr_console.file.write("\033[A\033[2K")
            stderr_console.file.flush()
            self._messages.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Context manager exit - ensure spinner is stopped."""
        self.stop_spinner()
        return False
