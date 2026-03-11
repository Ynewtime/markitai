"""Output manager for CLI stderr output with accurate line tracking.

Wraps Rich Console to track the exact number of rendered terminal lines
written to stderr, enabling clean erasure before printing final output
to stdout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console

if TYPE_CHECKING:
    from types import TracebackType


class OutputManager:
    """Manages stderr output with accurate terminal line counting.

    Tracks all lines written to stderr (both via print() and external writers)
    so they can be erased with ANSI escape codes before the final markdown
    content is written to stdout.

    Args:
        console: Rich Console instance to use for output. Defaults to a new
            stderr Console.
        enabled: Whether output and tracking are active. When False, all
            print/spinner/erase operations are no-ops.
    """

    def __init__(
        self,
        console: Console | None = None,
        enabled: bool = True,
    ) -> None:
        self._console = console or Console(stderr=True)
        self._enabled = enabled
        self._line_count = 0
        self._status = None

    @property
    def line_count(self) -> int:
        """Number of terminal lines currently tracked."""
        return self._line_count

    def print(self, text: str, *, style: str | None = None) -> None:
        """Print text to the console and track rendered line count.

        Uses Console.capture() to render the text with the same width as the
        real console, then counts newlines in the rendered output to determine
        how many terminal lines were consumed.

        Args:
            text: Text to print (may contain Rich markup).
            style: Optional Rich style to apply.
        """
        if not self._enabled:
            return

        # Capture rendered output to count lines accurately
        with self._console.capture() as capture:
            self._console.print(text, style=style)

        rendered = capture.get()

        # Count newlines in rendered output — Rich always adds a trailing
        # newline, so the number of \n equals the number of terminal lines.
        lines = rendered.count("\n")
        self._line_count += lines

        logger.debug(
            "[OutputManager] print: {} rendered line(s), total={}",
            lines,
            self._line_count,
        )

        # Write the rendered text directly to the underlying file object.
        # capture() rendered with the console's settings (TTY-aware ANSI codes),
        # so we bypass console.print() to avoid double-rendering.
        self._console.file.write(rendered)
        self._console.file.flush()

    def erase_all(self) -> None:
        """Erase all tracked lines from the terminal.

        Sends ANSI cursor-up + clear-line sequences for each tracked line,
        then resets the line count to 0.
        """
        if not self._enabled or self._line_count == 0:
            return

        self.stop_spinner()

        # Build erase sequence: move up one line + clear entire line, repeated
        erase_seq = "\033[A\033[2K" * self._line_count

        logger.debug(
            "[OutputManager] erasing {} line(s)",
            self._line_count,
        )

        self._console.file.write(erase_seq)
        self._console.file.flush()
        self._line_count = 0

    def print_persistent(self, text: str, *, style: str | None = None) -> None:
        """Print text to the console WITHOUT tracking it for erasure.

        Use for messages that should survive ``erase_all()`` — e.g., auth
        result summaries that the user needs to see alongside final output.

        Args:
            text: Text to print (may contain Rich markup).
            style: Optional Rich style to apply.
        """
        if not self._enabled:
            return
        self._console.print(text, style=style)

    def track_external_lines(self, count: int) -> None:
        """Track lines written by external code (e.g., ProgressReporter).

        Args:
            count: Number of terminal lines to add to the tracked count.
        """
        if not self._enabled:
            return
        self._line_count += count
        logger.debug(
            "[OutputManager] tracked {} external line(s), total={}",
            count,
            self._line_count,
        )

    def start_spinner(self, message: str) -> None:
        """Start an in-place spinner with a message.

        The spinner uses cursor positioning for in-place updates and does
        not add to the line count.

        Args:
            message: Message to display alongside the spinner.
        """
        if not self._enabled:
            return
        self.stop_spinner()
        self._status = self._console.status(f"[cyan]{message}[/cyan]", spinner="dots")
        self._status.start()
        logger.debug("[OutputManager] spinner started: {}", message)

    def stop_spinner(self) -> None:
        """Stop the current spinner if one is running."""
        if self._status is not None:
            self._status.stop()
            self._status = None
            logger.debug("[OutputManager] spinner stopped")

    def __enter__(self) -> OutputManager:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit — ensure spinner is stopped.

        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Exception traceback, if any.

        Returns:
            False to propagate any exception.
        """
        self.stop_spinner()
        return False
