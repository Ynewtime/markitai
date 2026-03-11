"""Subprocess output interceptor for CLI login commands.

Intercepts stdout/stderr from subprocesses like `copilot login` and
`claude auth login`, pattern-matches known message types, and reformats
output with consistent indentation and Rich markup through an OutputTarget.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    pass


@runtime_checkable
class OutputTarget(Protocol):
    """Protocol for output targets (OutputManager or mock)."""

    def print(self, text: str, *, style: str | None = None) -> None:
        """Print formatted text."""
        ...

    def start_spinner(self, message: str) -> None:
        """Start a spinner with a message."""
        ...

    def stop_spinner(self) -> None:
        """Stop the current spinner."""
        ...


# Provider-specific display labels
_PROVIDER_LABELS: dict[str, str] = {
    "copilot": "Copilot",
    "claude-agent": "Claude",
}

# Patterns for device code URLs
_URL_PATTERN = re.compile(r"(https?://\S+)")
_CODE_PATTERN = re.compile(
    r"(?:enter|code)[:\s]+([A-Z0-9]{4}-[A-Z0-9]{4})", re.IGNORECASE
)
_SUCCESS_PATTERNS = [
    re.compile(
        r"(?:signed in|logged in|authenticated)\s+(?:successfully\s+)?(?:as\s+)?(.+)",
        re.IGNORECASE,
    ),
    re.compile(r"^authenticated$", re.IGNORECASE),
]
_WAITING_PATTERN = re.compile(r"waiting\s+for\s+", re.IGNORECASE)
_ERROR_PATTERN = re.compile(r"error|failed|denied", re.IGNORECASE)

_INDENT = "    "

# ANSI escape sequence pattern (CSI sequences + OSC sequences)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[[\d;]*m")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    return _ANSI_ESCAPE.sub("", text)


class SubprocessInterceptor:
    """Intercepts and reformats CLI subprocess output.

    Captures stdout and stderr from a subprocess, strips ANSI escape
    sequences, pattern-matches known message types, and displays
    formatted output through an OutputTarget.
    """

    def __init__(self, output: OutputTarget) -> None:
        self._output = output
        self._spinner_active = False
        self.raw_lines: list[str] = []

    async def run(self, args: list[str], *, provider: str) -> int:
        """Run subprocess with output interception.

        Args:
            args: Full command args (e.g., ["/usr/bin/copilot", "login"]).
            provider: Provider name for display formatting.

        Returns:
            Subprocess exit code.
        """
        logger.debug(
            "[SubprocessInterceptor] Running: {} (provider={})",
            " ".join(args),
            provider,
        )

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        assert proc.stdout is not None  # noqa: S101

        while True:
            raw_line = await proc.stdout.readline()
            if not raw_line:
                break

            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            clean = _strip_ansi(line)
            if clean.strip():
                self.raw_lines.append(clean)
                logger.debug("[SubprocessInterceptor] Output: {}", clean)
            formatted = self._format_line(clean, provider)
            if formatted is not None:
                self._output.print(formatted)

        exit_code = await proc.wait()
        logger.debug("[SubprocessInterceptor] Process exited with code {}", exit_code)
        return exit_code

    def _format_line(self, line: str, provider: str) -> str | None:
        """Format a single line of subprocess output.

        Returns formatted string or None if handled via spinner or empty.

        Args:
            line: Raw line from subprocess output (already stripped of newline).
            provider: Provider name for label lookup.

        Returns:
            Formatted string with Rich markup, or None.
        """
        stripped = line.strip()
        if not stripped:
            return None

        label = _PROVIDER_LABELS.get(provider, provider)

        # Check waiting pattern first (uses spinner, not print)
        if _WAITING_PATTERN.search(stripped):
            self._output.start_spinner(f"{label}: {stripped}")
            self._spinner_active = True
            return None

        # Stop spinner before printing any non-waiting line
        if self._spinner_active:
            self._output.stop_spinner()
            self._spinner_active = False

        # Check success patterns
        for pattern in _SUCCESS_PATTERNS:
            m = pattern.search(stripped)
            if m:
                groups = m.groups()
                user = groups[0].strip() if groups and groups[0] else None
                msg = f"{_INDENT}[green]\u2713[/] {label} authenticated"
                if user:
                    msg += f" as [bold]{user}[/]"
                return msg

        # Check device code pattern
        code_match = _CODE_PATTERN.search(stripped)
        if code_match:
            code = code_match.group(1)
            code_line = f"{_INDENT}Code: [bold cyan]{code}[/]"
            # Also check for URL on the same line (copilot outputs both together)
            url_match = _URL_PATTERN.search(stripped)
            if url_match:
                url = url_match.group(1)
                return f"{_INDENT}Visit: [link={url}]{url}[/link]\n{code_line}"
            return code_line

        # Check URL pattern (standalone URL line)
        url_match = _URL_PATTERN.search(stripped)
        if url_match:
            url = url_match.group(1)
            return f"{_INDENT}Visit: [link={url}]{url}[/link]"

        # Check error patterns
        if _ERROR_PATTERN.search(stripped):
            return f"{_INDENT}[red]\u2717[/] {stripped}"

        # Unknown line - dim
        return f"{_INDENT}[dim]{stripped}[/]"
