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

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

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
    ("[Fetch] Enriching", "Fetching (fxtwitter)"),
    ("[Defuddle] Fetching", "Fetching (defuddle)"),
    ("Fetching URL with Jina Reader", "Fetching (jina)"),
    ("Fetching URL with CF Browser Rendering", "Rendering (cloudflare)"),
    ("[LLM]", "Enhancing with LLM"),
    ("Analyzing ", "Analyzing images"),
)

# Any log record emitted from these modules implies the given stage.
_STAGE_MODULES: dict[str, str] = {
    "fetch_playwright": "Rendering (playwright)",
}

# A stage running longer than this gets an "(Ns)" elapsed-time suffix, so a
# slow-but-alive LLM call (which can legitimately take 1-2+ minutes with no
# stage transitions to bridge from) doesn't read as a hang. Short stages
# (typical fetches) never show a number, avoiding noise.
ELAPSED_SUFFIX_THRESHOLD_S = 5.0
ELAPSED_TICK_INTERVAL_S = 1.0


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


def elapsed_suffix(started_at: float | None, now: float) -> str:
    """Build the "(Ns)" spinner suffix once a stage has run long enough.

    Args:
        started_at: ``time.monotonic()`` value when the current stage began,
            or None if no stage has started yet.
        now: Current ``time.monotonic()`` value.

    Returns:
        ``" (Ns)"`` once elapsed time reaches ``ELAPSED_SUFFIX_THRESHOLD_S``,
        otherwise ``""``.
    """
    if started_at is None:
        return ""
    elapsed = now - started_at
    if elapsed < ELAPSED_SUFFIX_THRESHOLD_S:
        return ""
    return f" ({elapsed:.0f}s)"


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
        self._stage_started_at: float | None = None
        self._ticker_task: asyncio.Task[None] | None = None

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
        self._stage_started_at = time.monotonic()
        # Temporary sink: rewrites the spinner text on known stage logs.
        # DEBUG level because fetch strategy hops log at DEBUG.
        self._sink_id = logger.add(
            self._on_stage_log, level="DEBUG", filter=_is_stage_record
        )
        # Ticker: repaints with an elapsed-time suffix once a stage runs
        # long (see ELAPSED_SUFFIX_THRESHOLD_S). No running loop (e.g. a
        # sync test driving start()/stop() directly) just skips the ticker;
        # the spinner still works, only the elapsed suffix is unavailable.
        try:
            self._ticker_task = asyncio.get_running_loop().create_task(self._tick())
        except RuntimeError:
            self._ticker_task = None

    def update(self, text: str) -> None:
        """Update the stage text (no-op rendering-wise when not active)."""
        self.stage_text = text
        self._stage_started_at = time.monotonic()
        self._render()

    def stop(self) -> None:
        """Detach the loguru bridge and erase the spinner. Idempotent."""
        if self._ticker_task is not None:
            self._ticker_task.cancel()
            self._ticker_task = None
        if self._sink_id is not None:
            try:
                logger.remove(self._sink_id)
            except ValueError:  # already removed (e.g. logger reconfigured)
                pass
            self._sink_id = None
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _render(self) -> None:
        """Repaint the spinner text, adding elapsed time once a stage runs long."""
        if self._status is None:
            return
        suffix = elapsed_suffix(self._stage_started_at, time.monotonic())
        self._status.update(f"[dim]{escape(self.stage_text)}{suffix}[/dim]")

    async def _tick(self) -> None:
        """Repaint on an interval so the elapsed-time suffix keeps counting."""
        while True:
            await asyncio.sleep(ELAPSED_TICK_INTERVAL_S)
            self._render()

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


# ---------------------------------------------------------------------------
# Live multi-stage progress list (single-URL / single-file conversions)
# ---------------------------------------------------------------------------

# Known stage log messages -> (stage_key, spinner text). Same bridge idea as
# ConversionStatus, but entries carry a stage key: a record whose key matches
# the active stage only rewrites the active line's text (fetch-internal
# strategy hops), while a record with a NEW key advances the list (finalizes
# the previous stage and starts a new line). This is what lets the file
# conversion path (convert_document_core) drive the checklist with zero
# changes to its code.
_STAGE_ENTRY_MESSAGE_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("Fetching URL with static", "fetch", "Fetching (static)"),
    # Static content turned out to need JS: the auto chain moves on to a
    # browser render next.
    ("JS required", "fetch", "Rendering (playwright)"),
    ("[Fetch] Enriching", "fetch", "Fetching (fxtwitter)"),
    ("[Defuddle] Fetching", "fetch", "Fetching (defuddle)"),
    ("Fetching URL with Jina Reader", "fetch", "Fetching (jina)"),
    ("Fetching URL with CF Browser Rendering", "fetch", "Rendering (cloudflare)"),
    ("[LLM]", "llm", "Enhancing with LLM"),
    ("Analyzing ", "images", "Analyzing images"),
)

# Any log record emitted from these modules implies the given stage entry.
_STAGE_ENTRY_MODULES: dict[str, tuple[str, str]] = {
    "fetch_playwright": ("fetch", "Rendering (playwright)"),
}


def stage_entry_from_log_record(
    record: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Map a loguru record to a (stage_key, stage_text) pair.

    Args:
        record: A loguru record (dict-like, with "message"/"module" keys).

    Returns:
        A (key, text) tuple such as ``("fetch", "Rendering (playwright)")``
        or None when the record does not indicate a known conversion stage.
    """
    message = str(record.get("message", ""))
    for prefix, key, text in _STAGE_ENTRY_MESSAGE_PREFIXES:
        if message.startswith(prefix):
            return (key, text)
    module = str(record.get("module", "") or "")
    return _STAGE_ENTRY_MODULES.get(module)


def _is_stage_entry_record(record: Mapping[str, Any]) -> bool:
    """Loguru filter: keep only records that map to a known stage entry."""
    return stage_entry_from_log_record(record) is not None


@dataclass(frozen=True)
class _DoneLine:
    """A finalized line in the stage list."""

    mark: str  # MARK_SUCCESS | MARK_ERROR | MARK_INFO
    style: str  # rich style for the mark
    text: str
    duration: float | None  # seconds; None for notes
    annotation: str | None = None  # e.g. "cached" -> "(cached, 0.2s)"


@dataclass
class _ActiveStage:
    """The stage currently rendered with a spinner."""

    key: str
    text: str
    started_at: float
    pinned: bool = False


class StageList:
    """Live multi-stage progress checklist on stderr.

    Renders finalized stages as persistent lines and the current stage as a
    spinner line, via ``rich.live.Live``::

        ✓ Fetched via fxtwitter (2.1s)
        ✓ Downloaded 3 images (1.4s)
        - Enhancing with LLM... (23s)

    Event sources:

    - explicit API (URL conversion path): :meth:`advance`, :meth:`finalize`,
      :meth:`update_text`, :meth:`note`, :meth:`fail`
    - loguru bridge (file conversion path): known stage logs (see
      ``stage_entry_from_log_record``) update the active line's text when
      their key matches the active stage, and advance the list when the key
      is new. ``advance(..., pin=True)`` opts the active stage out of the
      bridge entirely (used while parallel LLM tasks interleave their logs).

    Stop semantics:

    - ``transient=True`` (stdout mode): the Live region is erased on
      ``stop()``; nothing persists on success. On :meth:`fail`, the final
      list IS printed persistently so the user keeps the "died at stage X"
      context.
    - ``transient=False`` (-o file mode): ``stop()`` prints the final list
      persistently (the Live region itself is always erased first, so the
      persistent output never contains a half-rendered spinner frame).

    Display gating:

    - ``enabled=False`` (callers pass this for --quiet and -v/verbose) makes
      every method a no-op
    - non-TTY stderr: no Live rendering. ``transient=False`` degrades to
      printing each finalized line immediately (CI-friendly);
      ``transient=True`` stays fully silent on success, but still prints the
      list when :meth:`fail` was recorded.

    Thread-safety note: Live's auto-refresh thread re-renders this object
    (via ``__rich_console__``) concurrently with main-thread state changes;
    renders take local snapshots of ``_active`` and iterate ``_done`` by
    index, which is safe under the GIL for append-only lists.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        transient: bool = False,
        console: Console | None = None,
    ) -> None:
        """Initialize the stage list.

        Args:
            enabled: Caller-side gate (False for quiet/verbose modes).
            transient: True for stdout mode (erase on stop), False for -o
                file mode (persist the final list).
            console: Console to render on (defaults to the shared stderr
                console). Must be a stderr console.
        """
        self._console = console if console is not None else get_stderr_console()
        self.enabled = bool(enabled)
        self.transient = transient
        self._is_tty = self._console.is_terminal
        self._done: list[_DoneLine] = []
        self._active: _ActiveStage | None = None
        self._live: Live | None = None
        self._sink_id: int | None = None
        self._spinner = Spinner(STATUS_SPINNER, style="dim")
        self._failed = False
        self._printed_final = False

    @property
    def active(self) -> bool:
        """Whether the Live display is currently rendering."""
        return self._live is not None

    @property
    def active_key(self) -> str | None:
        """Key of the stage currently in progress, or None."""
        active = self._active
        return active.key if active is not None else None

    def start(self) -> None:
        """Start live rendering (TTY only) and attach the loguru bridge."""
        if not self.enabled:
            return
        if self._is_tty and self._live is None:
            # transient=True always: the Live region is erased on stop and
            # persistence is handled by _print_final_list(), so the
            # persistent output never contains a spinner frame.
            self._live = Live(
                self,
                console=self._console,
                transient=True,
                refresh_per_second=8,
            )
            self._live.start()
        if self._sink_id is None:
            # DEBUG level because fetch strategy hops log at DEBUG.
            self._sink_id = logger.add(
                self._on_stage_log, level="DEBUG", filter=_is_stage_entry_record
            )

    def advance(self, key: str, text: str, *, pin: bool = False) -> None:
        """Start a new stage, finalizing the previous active one (if any).

        Args:
            key: Stage key (e.g. "fetch", "llm"); the loguru bridge compares
                incoming records against this.
            text: Active-line text (e.g. "Fetching example.com...").
            pin: When True the loguru bridge leaves this stage alone
                (no text rewrites, no auto-advance) until the next explicit
                call. Used while parallel LLM tasks interleave their logs.
        """
        if not self.enabled:
            return
        self._finalize_active()
        self._active = _ActiveStage(
            key=key, text=text, started_at=time.monotonic(), pinned=pin
        )
        self._refresh()

    def update_text(self, text: str) -> None:
        """Rewrite the active line's text without resetting its timer."""
        if not self.enabled:
            return
        active = self._active
        if active is None:
            return
        active.text = text
        self._refresh()

    def finalize(
        self, text: str | None = None, *, annotation: str | None = None
    ) -> None:
        """Finalize the active stage as a success line.

        Args:
            text: Completed-state text (e.g. "Fetched via fxtwitter").
                Defaults to the active text with trailing dots stripped.
            annotation: Optional annotation folded into the duration parens,
                e.g. "cached" renders as "(cached, 0.2s)".
        """
        if not self.enabled:
            return
        self._finalize_active(text=text, annotation=annotation)
        self._refresh()

    def note(self, text: str) -> None:
        """Append an informational line without touching the active stage."""
        if not self.enabled:
            return
        line = _DoneLine(mark=MARK_INFO, style="dim", text=text, duration=None)
        self._done.append(line)
        self._print_static_if_degraded(line)
        self._refresh()

    def fail(self, text: str | None = None) -> None:
        """Finalize the active stage as a failure line (red cross)."""
        if not self.enabled:
            return
        self._failed = True
        self._finalize_active(text=text, failed=True)
        self._refresh()

    def stop(self) -> None:
        """Detach the bridge, stop Live, persist the list if applicable.

        Idempotent. An active stage that was never finalized or failed is
        discarded (its line disappears with the Live region).
        """
        if self._sink_id is not None:
            try:
                logger.remove(self._sink_id)
            except ValueError:  # already removed (e.g. logger reconfigured)
                pass
            self._sink_id = None
        if self._live is not None:
            self._live.stop()
            self._live = None
            if (
                (not self.transient or self._failed)
                and self._done
                and not self._printed_final
            ):
                self._print_final_list()
        elif (
            self.enabled
            and not self._is_tty
            and self.transient
            and self._failed
            and self._done
            and not self._printed_final
        ):
            # Non-TTY stdout mode stays silent on success, but a failure
            # still prints the context (which stage died).
            self._print_final_list()

    def _finalize_active(
        self,
        *,
        text: str | None = None,
        annotation: str | None = None,
        failed: bool = False,
    ) -> None:
        """Convert the active stage into a done line."""
        active = self._active
        if active is None:
            return
        duration = time.monotonic() - active.started_at
        label = text if text is not None else active.text.rstrip(".")
        if failed:
            mark, style = MARK_ERROR, "red"
        else:
            mark, style = MARK_SUCCESS, "green"
        line = _DoneLine(
            mark=mark,
            style=style,
            text=label,
            duration=duration,
            annotation=annotation,
        )
        self._done.append(line)
        self._active = None
        self._print_static_if_degraded(line)

    def _print_static_if_degraded(self, line: _DoneLine) -> None:
        """Non-TTY persistent mode: print each finalized line immediately."""
        if self._is_tty or self.transient or not self.enabled:
            return
        self._console.print(self._render_done_line(line))

    def _print_final_list(self) -> None:
        """Persistently print all done lines (after the Live region is gone)."""
        for line in self._done:
            self._console.print(self._render_done_line(line))
        self._printed_final = True

    def _render_done_line(self, line: _DoneLine) -> Text:
        """Render one finalized line: '  ✓ text (2.1s)'."""
        parts: list[tuple[str, str]] = [
            ("  ", ""),
            (f"{line.mark} ", line.style),
            (line.text, "dim" if line.mark == MARK_INFO else ""),
        ]
        if line.duration is not None:
            if line.annotation:
                parts.append((f" ({line.annotation}, {line.duration:.1f}s)", "dim"))
            else:
                parts.append((f" ({line.duration:.1f}s)", "dim"))
        return Text.assemble(*parts)

    def _refresh(self) -> None:
        """Force an immediate Live repaint after a state change."""
        if self._live is not None:
            self._live.refresh()

    def _on_stage_log(self, message: Any) -> None:
        """Loguru sink: drive the stage list from known stage logs."""
        entry = stage_entry_from_log_record(message.record)
        if entry is None:
            return
        key, text = entry
        active = self._active
        if active is not None and active.pinned:
            return
        if active is not None and active.key == key:
            self.update_text(f"{text}...")
        else:
            self.advance(key, f"{text}...")

    def __rich_console__(self, console: Console, options: Any) -> Any:
        """Render done lines plus the active spinner line.

        Called by Live's refresh thread; takes a local snapshot of _active
        so a concurrent finalize can't null it mid-render. The elapsed
        suffix is computed here on every repaint, so no ticker task is
        needed (Live's auto-refresh keeps it counting).
        """
        for line in self._done:
            yield self._render_done_line(line)
        active = self._active
        if active is not None:
            suffix = elapsed_suffix(active.started_at, time.monotonic())
            self._spinner.update(text=Text(f"{active.text}{suffix}", style="dim"))
            yield Padding(self._spinner, (0, 0, 0, 2))

    def __enter__(self) -> StageList:
        """Start rendering on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Stop rendering on context exit; never swallows exceptions."""
        self.stop()
        return False
