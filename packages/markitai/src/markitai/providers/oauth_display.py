"""OAuth flow display utilities for provider authentication.

Provides Rich-formatted output for OAuth flows across providers
(gemini-cli, chatgpt). All output goes to stderr to avoid interfering
with stdout piping.
"""

from __future__ import annotations

import io
import re
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console


_PROVIDER_LABELS: dict[str, str] = {
    "gemini-cli": "Gemini",
    "chatgpt": "ChatGPT",
}


def _get_stderr_console() -> Console:
    """Lazy import and create stderr console."""
    from rich.console import Console

    return Console(stderr=True)


@contextmanager
def suppress_stdout() -> Generator[io.StringIO]:
    """Capture and suppress stdout writes during OAuth flows.

    Yields:
        StringIO buffer containing the captured output.
    """
    captured = io.StringIO()
    original = sys.stdout
    sys.stdout = captured
    try:
        yield captured
    finally:
        sys.stdout = original


def show_oauth_start(provider: str, *, console: Console | None = None) -> None:
    """Display OAuth flow start message on stderr.

    Args:
        provider: Provider name (e.g., "gemini-cli").
        console: Optional console override for testing.
    """
    c = console or _get_stderr_console()
    label = _PROVIDER_LABELS.get(provider, provider)
    c.print(f"\n  [bold]{label} Authentication[/]")
    c.print("  Opening browser for login...")


def show_device_code(
    url: str,
    code: str,
    *,
    console: Console | None = None,
) -> None:
    """Display device code auth instructions on stderr.

    Args:
        url: Auth URL the user should visit.
        code: Device code the user should enter.
        console: Optional console override for testing.
    """
    c = console or _get_stderr_console()
    c.print("\n  [bold]ChatGPT Authentication[/]")
    c.print(f"  1. Visit: [link={url}]{url}[/link]")
    c.print(f"  2. Enter code: [bold cyan]{code}[/]")
    c.print("  [dim]Device codes are a phishing target. Never share this code.[/]")


def show_oauth_success(
    provider: str,
    *,
    user: str | None = None,
    detail: str | None = None,
    console: Console | None = None,
) -> None:
    """Display OAuth success message on stderr.

    Args:
        provider: Provider name.
        user: Username or email if available.
        detail: Additional detail (e.g., credential path).
        console: Optional console override for testing.
    """
    c = console or _get_stderr_console()
    label = _PROVIDER_LABELS.get(provider, provider)
    msg = f"  [green]\u2713[/] {label} authenticated"
    if user:
        msg += f" as [bold]{user}[/]"
    c.print(msg)
    if detail:
        c.print(f"    [dim]\u2502 {detail}[/]")


def parse_chatgpt_device_code(output: str) -> tuple[str, str] | None:
    """Parse device code URL and code from LiteLLM's stdout output.

    Args:
        output: Captured stdout text from LiteLLM authenticator.

    Returns:
        Tuple of (url, code) or None if parsing fails.
    """
    url_match = re.search(r"Visit\s+(https?://\S+)", output)
    code_match = re.search(r"Enter code:\s*(\S+)", output)
    if url_match and code_match:
        return url_match.group(1), code_match.group(1)
    return None


class DeviceCodeInterceptor(io.TextIOBase):
    """Intercepts stdout during ChatGPT device code auth.

    Captures print() output from LiteLLM's authenticator and re-displays
    the device code information in Rich format on stderr.

    Args:
        console: Rich console for stderr output.
    """

    def __init__(self, console: Console | None = None) -> None:
        super().__init__()
        self._console = console or _get_stderr_console()
        self._buffer = ""
        self._displayed = False

    def write(self, s: str) -> int:
        """Capture write and display Rich-formatted version when ready."""
        if self._displayed:
            return len(s)
        self._buffer += s
        result = parse_chatgpt_device_code(self._buffer)
        if result:
            url, code = result
            show_device_code(url, code, console=self._console)
            self._displayed = True
        return len(s)

    def flush(self) -> None:
        """No-op flush."""

    @property
    def displayed(self) -> bool:
        """Whether device code info has been displayed."""
        return self._displayed
