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

from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.console import Console
from rich.markup import escape

from markitai.cli.console import get_console, get_stderr_console

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.status import Status

    from markitai.config import MarkitaiConfig

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


def summary(text: str, *, ok: bool = True, console: Console | None = None) -> None:
    """Display a summary message with a status glyph and leading blank line.

    Args:
        text: The summary message to display.
        ok: True renders a green checkmark, False a red cross.
        console: Optional console for output (defaults to shared console).
    """
    c = console or get_console()
    c.print()
    if ok:
        c.print(f"[green]{MARK_SUCCESS}[/] {text}")
    else:
        c.print(f"[red]{MARK_ERROR}[/] {text}")


def build_feature_str(cfg: MarkitaiConfig) -> str:
    """Build a human-readable feature summary string from config.

    Separates LLM-dependent features from local processing features.
    LLM features: LLM, alt, desc.
    Local features: OCR (RapidOCR), screenshot (Playwright).

    Args:
        cfg: The full Markitai configuration.

    Returns:
        A Rich-formatted string like "LLM alt desc | OCR screenshot".
    """
    llm_features: list[str] = []
    local_features: list[str] = []

    if cfg.llm.enabled:
        llm_features.append("[green]LLM[/green]")
    if cfg.image.alt_enabled:
        llm_features.append("[green]alt[/green]")
    if cfg.image.desc_enabled:
        llm_features.append("[green]desc[/green]")

    if cfg.ocr.enabled:
        local_features.append("[green]OCR[/green]")
    if cfg.screenshot.enabled:
        local_features.append("[green]screenshot[/green]")

    parts = []
    if llm_features:
        parts.append(" ".join(llm_features))
    if local_features:
        parts.append(" ".join(local_features))

    return " | ".join(parts) if parts else "[dim]none[/dim]"


# ---------------------------------------------------------------------------
# Live conversion status (single-URL / single-file conversions)
# ---------------------------------------------------------------------------

# Spinner shown during long-running single-input conversions.
# rich's "line" spinner is pure ASCII (- \ | /) and renders on any
# terminal/encoding; "dots" (braille) looks nicer but is non-ASCII.
# Change this constant to switch the spinner globally.
STATUS_SPINNER = "line"

# Known stage log messages -> human-readable spinner text. fetch.py and
# workflow/core.py already emit these via loguru; ConversionStatus subscribes
# with a temporary sink so stage transitions update the spinner text with
# zero changes to the fetch/convert code.
_STAGE_MESSAGE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Fetching URL with static", "Fetching (static)"),
    # Static content turned out to need JS: the auto chain moves on to a
    # browser render next.
    ("JS required", "Rendering (playwright)"),
    ("[Fetch] FxTwitter", "Fetching (fxtwitter)"),
    ("[Defuddle] Fetching", "Fetching (defuddle)"),
    ("Fetching URL with Jina Reader", "Fetching (jina)"),
    ("Fetching URL with CF Browser Rendering", "Rendering (cloudflare)"),
    ("[LLM]", "Enhancing with LLM"),
    ("Analyzing ", "Analyzing images"),
)

# Any log record emitted from these modules implies the given stage.
_STAGE_MODULES: dict[str, str] = {
    "fetch_playwright": "Rendering (playwright)",
    "fetch_fxtwitter": "Fetching (fxtwitter)",
}


def stage_from_log_record(record: Mapping[str, Any]) -> str | None:
    """Map a loguru record to a spinner stage label.

    Args:
        record: A loguru record (dict-like, with "message"/"module" keys).

    Returns:
        A stage label such as ``"Rendering (playwright)"`` or None when the
        record does not indicate a known conversion stage.
    """
    message = str(record.get("message", ""))
    for prefix, stage in _STAGE_MESSAGE_PREFIXES:
        if message.startswith(prefix):
            return stage
    module = str(record.get("module", "") or "")
    return _STAGE_MODULES.get(module)


def _is_stage_record(record: Mapping[str, Any]) -> bool:
    """Loguru filter: keep only records that map to a known stage."""
    return stage_from_log_record(record) is not None


class ConversionStatus:
    """Transient stderr spinner for long-running single-input conversions.

    Shows a dim, pure-ASCII spinner (see ``STATUS_SPINNER``) on stderr while
    fetching / converting / LLM-enhancing a single URL or file. Stage
    transitions inside the fetch chain are picked up from existing loguru
    logs via a temporary sink (``stage_from_log_record``), so fetch code
    needs no changes.

    Display gating (all must hold, otherwise every method is a no-op):

    - the caller passes ``enabled=True`` — callers disable the status for
      ``--quiet`` and for ``-v``/verbose (verbose already streams logs to
      stderr; a spinner would interleave badly with them)
    - the stderr console is a real terminal (suppressed in CI / pipes)

    The spinner only ever writes to stderr, never stdout (content channel),
    and is transient: rich erases the frame on ``stop()``, so final output
    lines are never mixed with spinner frames. Always call ``stop()`` (or
    use as a context manager) before printing final results.

    Known v1 limitation: the lazy remote-consent prompt (``click.confirm``
    inside the fetch chain) can fire while the spinner is live; the answer
    line may briefly share a line with a spinner frame, after which the
    next stage update repaints cleanly. Pausing around that prompt would
    require changes to fetch.py, which owns the prompt.
    """

    def __init__(
        self,
        initial: str,
        *,
        enabled: bool = True,
        console: Console | None = None,
    ) -> None:
        """Initialize the status.

        Args:
            initial: Initial stage text (e.g. ``"Fetching example.com..."``).
            enabled: Caller-side gate (False for quiet/verbose modes).
            console: Console to render on (defaults to the shared stderr
                console). Must be a stderr console.
        """
        self._console = console if console is not None else get_stderr_console()
        self.enabled = bool(enabled) and self._console.is_terminal
        self.stage_text = initial
        self._status: Status | None = None
        self._sink_id: int | None = None

    @property
    def active(self) -> bool:
        """Whether the spinner is currently rendering."""
        return self._status is not None

    def start(self) -> None:
        """Start the spinner and attach the loguru stage bridge."""
        if not self.enabled or self._status is not None:
            return
        self._status = self._console.status(
            f"[dim]{escape(self.stage_text)}[/dim]",
            spinner=STATUS_SPINNER,
            spinner_style="dim",
        )
        self._status.start()
        # Temporary sink: rewrites the spinner text on known stage logs.
        # DEBUG level because fetch strategy hops log at DEBUG.
        self._sink_id = logger.add(
            self._on_stage_log, level="DEBUG", filter=_is_stage_record
        )

    def update(self, text: str) -> None:
        """Update the stage text (no-op rendering-wise when not active)."""
        self.stage_text = text
        if self._status is not None:
            self._status.update(f"[dim]{escape(text)}[/dim]")

    def stop(self) -> None:
        """Detach the loguru bridge and erase the spinner. Idempotent."""
        if self._sink_id is not None:
            try:
                logger.remove(self._sink_id)
            except ValueError:  # already removed (e.g. logger reconfigured)
                pass
            self._sink_id = None
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _on_stage_log(self, message: Any) -> None:
        """Loguru sink: map a stage log record to spinner text."""
        stage = stage_from_log_record(message.record)
        if stage is not None:
            self.update(f"{stage}...")

    def __enter__(self) -> ConversionStatus:
        """Start the spinner on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Stop the spinner on context exit; never swallows exceptions."""
        self.stop()
        return False
