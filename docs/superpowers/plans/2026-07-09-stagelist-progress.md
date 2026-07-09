# StageList Multi-Stage Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two parallel stderr progress facilities (ConversionStatus, ProgressReporter/OutputManager) with a single multi-stage checklist component (StageList), fixing the total silence of stdout-mode conversions (`mkai <url> -p standard` without `-o`).

**Architecture:** New `StageList` component in `cli/ui.py` renders completed stage lines (`✓ Fetched via fxtwitter (2.1s)`) plus one active spinner line via `rich.live.Live` on stderr. URL path drives it with explicit API calls; file path drives it zero-touch via the existing loguru stage bridge (same-key log → text update, new-key log → stage advance). stdout mode erases on stop; `-o` mode persists the list.

**Tech Stack:** Python 3.10+, rich (Live/Spinner/Text), loguru, pytest (asyncio_mode=auto), uv.

**Spec:** `docs/superpowers/specs/2026-07-09-stagelist-progress-design.md` (approved; includes two plan-stage corrections: OutputManager fully retired, 0-images finalizes instead of note).

## Global Constraints

- Working dir for all commands: `packages/markitai` (run as `cd packages/markitai && <cmd>`)
- Test command: `uv run pytest <path> -x -q` — asyncio_mode is `auto`, async tests need no marker
- Lint/format/type gates that must stay green after every task: `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run pyright src`
- Spinner stays pure-ASCII: `STATUS_SPINNER = "line"` (existing constant, do not change)
- Elapsed suffix threshold stays `ELAPSED_SUFFIX_THRESHOLD_S = 5.0`; active line shows ` (Ns)` only past it; completed lines ALWAYS show ` (N.Ns)` (1 decimal)
- StageList writes ONLY to stderr, never stdout
- Gating everywhere: `enabled = not quiet and not verbose` (the `not stdout_mode` term is the bug being removed); non-TTY handling lives INSIDE StageList, not at call sites
- Commit after every task; commit messages follow repo convention (`feat(cli): ...`, `refactor(cli): ...`), end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Do NOT commit files outside this feature (the repo has unrelated dirty files under `website/` — never `git add -A`; always add explicit paths)

---

### Task 1: StageList component + stage-entry bridge mapping

**Files:**
- Modify: `packages/markitai/src/markitai/cli/ui.py` (append new section; do NOT touch the existing `ConversionStatus` class yet — it still has callers)
- Create: `packages/markitai/tests/unit/cli/test_stage_list.py`

**Interfaces:**
- Consumes: existing `get_stderr_console()` from `markitai.cli.console`, existing constants `STATUS_SPINNER`, `ELAPSED_SUFFIX_THRESHOLD_S`, existing helper `elapsed_suffix()`.
- Produces (later tasks rely on these exact signatures):
  - `stage_entry_from_log_record(record: Mapping[str, Any]) -> tuple[str, str] | None` — returns `(stage_key, stage_text)`
  - `class StageList` with:
    - `__init__(self, *, enabled: bool = True, transient: bool = False, console: Console | None = None)`
    - `start() -> None`, `stop() -> None` (idempotent), `__enter__`/`__exit__`
    - `advance(key: str, text: str, *, pin: bool = False) -> None`
    - `update_text(text: str) -> None`
    - `finalize(text: str | None = None, *, annotation: str | None = None) -> None`
    - `note(text: str) -> None`
    - `fail(text: str | None = None) -> None`
    - property `active: bool` (Live running), property `active_key: str | None`

- [ ] **Step 1: Write the failing tests**

Create `packages/markitai/tests/unit/cli/test_stage_list.py` with this exact content:

```python
"""Tests for the live multi-stage progress list (cli.ui.StageList).

Covers:
- the (key, text) log-record mapping (stage_entry_from_log_record)
- explicit API: advance / update_text / finalize / note / fail
- loguru bridge: same-key text update, new-key auto-advance, pin
- transient vs persistent stop behavior
- TTY / enabled gating and non-TTY degradation
"""

from __future__ import annotations

import io
import time

import pytest
from loguru import logger
from rich.console import Console

from markitai.cli import ui

# ANSI hide-cursor sequence emitted by rich Live on a terminal;
# a reliable marker that live rendering happened on a given stream.
HIDE_CURSOR = "\x1b[?25l"


def make_tty_console() -> Console:
    """Console backed by a StringIO that claims to be a terminal."""
    return Console(file=io.StringIO(), force_terminal=True, width=80)


def make_pipe_console() -> Console:
    """Console backed by a StringIO that is NOT a terminal."""
    return Console(file=io.StringIO(), force_terminal=False, width=80)


class TestStageEntryFromLogRecord:
    """Tests for the log-record -> (key, text) mapping."""

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                "Fetching URL with static strategy: https://example.com",
                ("fetch", "Fetching (static)"),
            ),
            ("JS required: string pattern matched 'x'", ("fetch", "Rendering (playwright)")),
            (
                "[Fetch] Enriching via FxTwitter/oEmbed: https://x.com/a",
                ("fetch", "Fetching (fxtwitter)"),
            ),
            ("[Defuddle] Fetching: https://example.com", ("fetch", "Fetching (defuddle)")),
            (
                "Fetching URL with Jina Reader (JSON mode): https://example.com",
                ("fetch", "Fetching (jina)"),
            ),
            (
                "Fetching URL with CF Browser Rendering: https://example.com",
                ("fetch", "Rendering (cloudflare)"),
            ),
            ("[LLM] doc.md: Starting standard LLM processing", ("llm", "Enhancing with LLM")),
            ("Analyzing image 1/3", ("images", "Analyzing images")),
        ],
    )
    def test_known_message_prefixes(self, message: str, expected: tuple[str, str]) -> None:
        record = {"message": message, "module": "fetch"}
        assert ui.stage_entry_from_log_record(record) == expected

    def test_playwright_module_maps_to_fetch_rendering(self) -> None:
        record = {
            "message": "Created new cached Playwright context for: x",
            "module": "fetch_playwright",
        }
        assert ui.stage_entry_from_log_record(record) == ("fetch", "Rendering (playwright)")

    def test_unknown_record_returns_none(self) -> None:
        assert ui.stage_entry_from_log_record({"message": "hi", "module": "config"}) is None

    def test_missing_keys_return_none(self) -> None:
        assert ui.stage_entry_from_log_record({}) is None


class TestExplicitApi:
    """Tests for advance / update_text / finalize / note / fail."""

    def test_advance_then_finalize_produces_done_line(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console)
        stages.start()
        stages.advance("fetch", "Fetching example.com...")
        assert stages.active_key == "fetch"
        stages.finalize("Fetched via static")
        assert stages.active_key is None
        assert len(stages._done) == 1
        line = stages._done[0]
        assert line.mark == ui.MARK_SUCCESS
        assert line.text == "Fetched via static"
        assert line.duration is not None
        stages.stop()

    def test_advance_finalizes_previous_stage(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching...")
            stages.advance("llm", "Enhancing with LLM...")
            assert stages.active_key == "llm"
            assert len(stages._done) == 1
            # default finalize text = active text without trailing dots
            assert stages._done[0].text == "Fetching"

    def test_finalize_with_annotation(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching...")
            stages.finalize("Fetched via static", annotation="cached")
            rendered = stages._render_done_line(stages._done[0]).plain
            # "(cached, 0.0s)" — annotation folded into the duration parens
            assert "(cached, " in rendered

    def test_update_text_does_not_reset_timer(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching (static)...")
            started = stages._active.started_at  # type: ignore[union-attr]
            stages.update_text("Rendering (playwright)...")
            assert stages._active.started_at == started  # type: ignore[union-attr]
            assert stages._active.text == "Rendering (playwright)..."  # type: ignore[union-attr]

    def test_note_adds_line_without_touching_active(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching...")
            stages.note("Screenshot captured: shot.png")
            assert stages.active_key == "fetch"
            assert len(stages._done) == 1
            assert stages._done[0].mark == ui.MARK_INFO
            assert stages._done[0].duration is None

    def test_fail_marks_line_with_cross(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("llm", "Enhancing with LLM...")
            stages.fail()
            assert stages._done[0].mark == ui.MARK_ERROR
            assert stages._done[0].text == "Enhancing with LLM"

    def test_calls_before_start_or_when_disabled_are_noops(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console, enabled=False)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.finalize()
        stages.note("x")
        stages.fail()
        stages.stop()
        assert stages._done == []
        assert console.file.getvalue() == ""  # type: ignore[union-attr]

    def test_finalize_without_active_stage_is_noop(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.finalize()
            assert stages._done == []


class TestLoguruBridge:
    """Tests for the loguru sink driving the stage list."""

    def test_same_key_log_updates_text_only(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching example.com...")
            logger.debug("Fetching URL with static strategy: https://example.com")
            assert stages.active_key == "fetch"
            assert stages._active.text == "Fetching (static)..."  # type: ignore[union-attr]
            logger.debug("JS required: content too short")
            assert stages.active_key == "fetch"
            assert stages._active.text == "Rendering (playwright)..."  # type: ignore[union-attr]
            assert stages._done == []  # no stage was finalized

    def test_new_key_log_advances_stage(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("convert", "Converting doc.pdf...")
            logger.debug("[LLM] doc.pdf: Starting standard LLM processing")
            assert stages.active_key == "llm"
            assert len(stages._done) == 1
            assert stages._done[0].text == "Converting doc.pdf"

    def test_pinned_stage_ignores_bridge(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("llm", "Enhancing with LLM (document + images)...", pin=True)
            logger.debug("[LLM] doc.md: Starting standard LLM processing")
            logger.debug("Analyzing image 1/3")
            assert stages.active_key == "llm"
            assert (
                stages._active.text  # type: ignore[union-attr]
                == "Enhancing with LLM (document + images)..."
            )
            assert stages._done == []

    def test_bridge_with_no_active_stage_advances(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            logger.debug("[Defuddle] Fetching: https://example.com")
            assert stages.active_key == "fetch"

    def test_sink_removed_after_stop(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console)
        stages.start()
        stages.stop()
        logger.debug("[Defuddle] Fetching: https://example.com")
        assert stages.active_key is None

    def test_stop_is_idempotent(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console)
        stages.start()
        stages.stop()
        stages.stop()  # must not raise
        assert not stages.active


class TestStopBehavior:
    """Transient (stdout mode) vs persistent (-o mode) stop semantics."""

    def test_persistent_mode_prints_final_list_on_stop(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console, transient=False)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.finalize("Fetched via static")
        stages.stop()
        out = console.file.getvalue()  # type: ignore[union-attr]
        assert "Fetched via static" in out
        assert ui.MARK_SUCCESS in out

    def test_transient_mode_leaves_no_final_list_on_success(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console, transient=True)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.finalize("Fetched via static")
        stages.stop()
        # Live erases its region via ANSI, so a plain-text scan of the
        # buffer can't distinguish erased frames from persistent output;
        # assert instead that stop() never took the static re-print path.
        assert stages._printed_final is False

    def test_transient_mode_prints_list_on_failure(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console, transient=True)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.finalize("Fetched via static")
        stages.advance("llm", "Enhancing with LLM...")
        stages.fail()
        stages.stop()
        assert stages._printed_final is True
        out = console.file.getvalue()  # type: ignore[union-attr]
        assert "Fetched via static" in out
        assert ui.MARK_ERROR in out

    def test_stop_discards_unfinalized_active_stage(self) -> None:
        console = make_tty_console()
        stages = ui.StageList(console=console, transient=False)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.stop()
        # Active stage was never finalized: no done line for it
        assert stages._done == []


class TestGatingAndDegradation:
    """TTY / enabled gating and non-TTY static fallback."""

    def test_non_tty_persistent_prints_static_lines_immediately(self) -> None:
        console = make_pipe_console()
        stages = ui.StageList(console=console, transient=False)
        stages.start()
        assert not stages.active  # no Live on non-TTY
        stages.advance("fetch", "Fetching...")
        stages.finalize("Fetched via static")
        out = console.file.getvalue()  # type: ignore[union-attr]
        assert "Fetched via static" in out
        stages.stop()
        # stop() must not duplicate the already-printed line
        assert console.file.getvalue().count("Fetched via static") == 1  # type: ignore[union-attr]

    def test_non_tty_transient_is_fully_silent(self) -> None:
        console = make_pipe_console()
        stages = ui.StageList(console=console, transient=True)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.finalize("Fetched via static")
        stages.stop()
        assert console.file.getvalue() == ""  # type: ignore[union-attr]

    def test_non_tty_transient_failure_still_prints_context(self) -> None:
        console = make_pipe_console()
        stages = ui.StageList(console=console, transient=True)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.fail()
        stages.stop()
        out = console.file.getvalue()  # type: ignore[union-attr]
        assert "Fetching" in out

    def test_tty_renders_live(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            assert stages.active
        assert HIDE_CURSOR in console.file.getvalue()  # type: ignore[union-attr]


class TestRendering:
    """Rendering details: duration formatting, elapsed suffix."""

    def test_done_line_shows_one_decimal_duration(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching...")
            stages._active.started_at = time.monotonic() - 2.14  # type: ignore[union-attr]
            stages.finalize("Fetched via static")
            rendered = stages._render_done_line(stages._done[0]).plain
            assert "(2.1s)" in rendered

    def test_active_line_elapsed_suffix_past_threshold(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("llm", "Enhancing with LLM...")
            stages._active.started_at = time.monotonic() - 10.0  # type: ignore[union-attr]
            segments = list(
                stages.__rich_console__(console, console.options)
            )
            # last renderable is the active spinner line; render it to text
            with console.capture() as cap:
                console.print(segments[-1])
            assert "(10s)" in cap.get()

    def test_active_line_no_suffix_before_threshold(self) -> None:
        console = make_tty_console()
        with ui.StageList(console=console) as stages:
            stages.advance("fetch", "Fetching...")
            segments = list(stages.__rich_console__(console, console.options))
            with console.capture() as cap:
                console.print(segments[-1])
            assert "(" not in cap.get().replace("Fetching...", "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_stage_list.py -x -q`
Expected: FAIL / ERROR with `AttributeError: module 'markitai.cli.ui' has no attribute 'stage_entry_from_log_record'` (collection error on first test class).

- [ ] **Step 3: Implement StageList in ui.py**

In `packages/markitai/src/markitai/cli/ui.py`:

3a. Extend the imports at the top of the file (keep existing ones):

```python
from dataclasses import dataclass

from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text
```

(`asyncio` import becomes unused only after Task 4 removes ConversionStatus — leave it for now.)

3b. Append this section at the end of the file. IMPORTANT: do not modify the existing `_STAGE_MESSAGE_PREFIXES`, `_STAGE_MODULES`, `stage_from_log_record`, `_is_stage_record`, or `ConversionStatus` — they still serve ConversionStatus until Task 4. The new code uses parallel `_STAGE_ENTRY_*` tables:

```python
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
        line = _DoneLine(
            mark=MARK_INFO, style="dim", text=text, duration=None
        )
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
            if (not self.transient or self._failed) and self._done:
                self._print_final_list()
        elif (
            self.enabled
            and not self._is_tty
            and self.transient
            and self._failed
            and self._done
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
            self._spinner.update(
                text=Text(f"{active.text}{suffix}", style="dim")
            )
            yield Padding(self._spinner, (0, 0, 0, 2))

    def __enter__(self) -> StageList:
        """Start rendering on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Stop rendering on context exit; never swallows exceptions."""
        self.stop()
        return False
```

Note the elapsed-suffix design difference vs ConversionStatus: no asyncio ticker task. `__rich_console__` recomputes the suffix on every Live auto-refresh (8/s), which also makes StageList safe in sync (no-event-loop) contexts like `init.py`'s ThreadPoolExecutor usage.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_stage_list.py -x -q`
Expected: all PASS.

Also run the untouched old suite to prove no regression:
Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_conversion_status.py tests/unit/cli/test_ui.py -x -q`
Expected: all PASS.

- [ ] **Step 5: Lint, format, typecheck**

Run: `cd packages/markitai && uv run ruff check src tests && uv run ruff format src tests && uv run pyright src`
Expected: no errors (ruff format may rewrite the new files; that's fine).

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/cli/ui.py packages/markitai/tests/unit/cli/test_stage_list.py
git commit -m "feat(cli): StageList multi-stage live progress component

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Migrate the single-URL path to StageList

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/url.py` (function `process_url` only; `process_url_batch` keeps its rich Progress)
- Modify: `packages/markitai/src/markitai/cli/main.py:939` (drop `output_manager=om` from the `process_url` call)
- Create: `packages/markitai/tests/unit/cli/test_progress_integration.py`
- Modify: `packages/markitai/tests/unit/cli/test_conversion_status.py` (delete the URL-specific classes that the new file replaces)

**Interfaces:**
- Consumes: `ui.StageList` exactly as produced by Task 1.
- Produces: `process_url` signature loses its `output_manager: Any = None` parameter. New keyword set: `(url, output_dir, cfg, dry_run, verbose, log_file_path=None, fetch_strategy=None, explicit_fetch_strategy=False, quiet=False)`. Task 3 mirrors this for `process_single_file`.

- [ ] **Step 1: Write the failing integration tests**

Create `packages/markitai/tests/unit/cli/test_progress_integration.py`:

```python
"""Integration tests: StageList wiring inside the URL/file processors.

The URL tests live here from Task 2; Task 3 adds the file-processor tests.
Replaces the processor-level classes of test_conversion_status.py.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from loguru import logger
from rich.console import Console

from markitai.cli.processors.url import process_url
from markitai.config import MarkitaiConfig

HIDE_CURSOR = "\x1b[?25l"


def make_tty_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=80)


def make_fetch_result(content: str = "# Test Page\n\nSome content.") -> MagicMock:
    result = MagicMock()
    result.content = content
    result.cache_hit = False
    result.strategy_used = "static"
    result.screenshot_path = None
    result.title = "Test Page"
    result.static_content = None
    result.browser_content = None
    result.metadata = {}
    return result


class TestUrlProcessorGating:
    """process_url must enable StageList in BOTH file and stdout modes."""

    async def _run(self, output_dir: Path | None, **kwargs) -> MagicMock:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=make_fetch_result(),
            ),
            patch("markitai.cli.ui.StageList") as stage_cls,
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=output_dir,
                cfg=cfg,
                dry_run=False,
                verbose=kwargs.get("verbose", False),
                quiet=kwargs.get("quiet", False),
            )
        return stage_cls

    async def test_file_mode_enables_persistent_stagelist(
        self, tmp_path: Path
    ) -> None:
        stage_cls = await self._run(tmp_path)
        kwargs = stage_cls.call_args.kwargs
        assert kwargs["enabled"] is True
        assert kwargs["transient"] is False

    async def test_stdout_mode_enables_transient_stagelist(self, capsys) -> None:
        stage_cls = await self._run(None)
        capsys.readouterr()  # discard markdown on stdout
        kwargs = stage_cls.call_args.kwargs
        assert kwargs["enabled"] is True  # THE bug fix: was False before
        assert kwargs["transient"] is True

    async def test_verbose_disables_stagelist(self, tmp_path: Path) -> None:
        stage_cls = await self._run(tmp_path, verbose=True)
        assert stage_cls.call_args.kwargs["enabled"] is False

    async def test_quiet_disables_stagelist(self, tmp_path: Path) -> None:
        stage_cls = await self._run(tmp_path, quiet=True)
        assert stage_cls.call_args.kwargs["enabled"] is False


class TestUrlStdoutStaysPure:
    """Stage frames go to stderr only; stdout carries only markdown."""

    async def test_stdout_mode_output_contains_no_frames(
        self, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr(
            "markitai.cli.ui.get_stderr_console", lambda: fake_stderr
        )

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        async def fetch_with_log(*args, **kwargs):
            logger.debug("Fetching URL with static strategy: https://example.com")
            return make_fetch_result()

        with patch("markitai.fetch.fetch_url", side_effect=fetch_with_log):
            await process_url(
                url="https://example.com/test",
                output_dir=None,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        captured = capsys.readouterr()
        # Live rendered on (fake) stderr
        assert HIDE_CURSOR in fake_stderr.file.getvalue()  # type: ignore[union-attr]
        # stdout carries clean markdown only
        assert HIDE_CURSOR not in captured.out
        assert "Fetching (static)" not in captured.out
        assert "Test Page" in captured.out

    async def test_file_mode_persists_stage_lines_on_stderr(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr(
            "markitai.cli.ui.get_stderr_console", lambda: fake_stderr
        )

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=make_fetch_result(),
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        stderr_text = fake_stderr.file.getvalue()  # type: ignore[union-attr]
        assert "Fetched via static" in stderr_text
        assert (tmp_path / "test.md").exists()
        captured = capsys.readouterr()
        assert HIDE_CURSOR not in captured.out


class TestUrlErrorPath:
    """Fetch errors must finalize the active stage as a failure."""

    async def test_fetch_error_marks_stage_failed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import pytest

        from markitai.fetch import FetchError

        fake_stderr = make_tty_console()
        monkeypatch.setattr(
            "markitai.cli.ui.get_stderr_console", lambda: fake_stderr
        )

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            side_effect=FetchError("boom"),
        ):
            with pytest.raises(SystemExit):
                await process_url(
                    url="https://example.com/test",
                    output_dir=tmp_path,
                    cfg=cfg,
                    dry_run=False,
                    verbose=False,
                )

        stderr_text = fake_stderr.file.getvalue()  # type: ignore[union-attr]
        # The failed fetch stage line persisted (file mode persists on stop)
        assert "Fetching" in stderr_text
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_progress_integration.py -x -q`
Expected: FAIL — `process_url` still instantiates `ConversionStatus`/`ProgressReporter`, so `StageList` mock never gets called (`call_args` is None → AttributeError/assert failure).

- [ ] **Step 3: Rewrite process_url's progress wiring**

In `packages/markitai/src/markitai/cli/processors/url.py`, function `process_url`:

3a. Delete the import `from markitai.utils.progress import ProgressReporter` (line 32).

3b. Remove the `output_manager: Any = None` parameter from the signature and its docstring mention. Signature becomes:

```python
async def process_url(
    url: str,
    output_dir: Path | None,
    cfg: MarkitaiConfig,
    dry_run: bool,
    verbose: bool,
    log_file_path: Path | None = None,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
    quiet: bool = False,
) -> None:
```

3c. Replace the ProgressReporter + ConversionStatus creation block (current lines 192-204) with:

```python
    # Live multi-stage progress list (stderr). Enabled in BOTH file and
    # stdout modes — stdout mode renders transiently (erased before the
    # markdown hits stdout), file mode persists the final list. Suppressed
    # for --quiet and -v/verbose (verbose already streams logs to stderr).
    # StageList itself degrades on non-TTY stderr.
    stages = ui.StageList(
        enabled=not quiet and not verbose,
        transient=stdout_mode,
    )
```

3d. Replace each old call site inside `process_url` as follows (old → new):

| Old code | New code |
|---|---|
| `progress.start_spinner(f"Fetching {url}...")` + `status.start()` (lines 219-220) | `stages.start()` + `stages.advance("fetch", f"Fetching {ui.truncate(url, 60)}...")` |
| after fetch success, `cache_note = ...` / `logger.info(f"Fetched via {used_strategy}{cache_note}: {url}")` (lines 248-249) | keep the logger line, then add: `stages.finalize(f"Fetched via {used_strategy}", annotation="cached" if fetch_cache_hit else None)` |
| `status.stop()` in `except JinaRateLimitError` (line 251) | `stages.fail()` then `stages.stop()` |
| `status.stop()` in `except FetchError` (line 262) | `stages.fail()` then `stages.stop()` |
| `status.stop()` in the empty-content branch (line 267) | `stages.fail("No content extracted")` then `stages.stop()` |
| `status.stop()` in the skip-exists branch (line 281) | `stages.stop()` |
| `progress.log(f"Fetched via {used_strategy}: {url}")` (line 288) | delete (finalize above replaced it) |
| `progress.log(f"Screenshot captured: {screenshot_path.name}")` (line 299) | `stages.note(f"Screenshot captured: {screenshot_path.name}")` |
| `progress.start_spinner("Downloading images...")` + `status.update("Downloading images...")` (lines 303-304) | `stages.advance("images", "Downloading images...")` |
| `progress.log(f"Downloaded {len(downloaded_images)} images")` (line 323) | `stages.finalize(f"Downloaded {len(downloaded_images)} images")` |
| `progress.log("No images to download")` (line 325) | `stages.finalize("No images to download")` |
| `status.stop()` in screenshot-only-no-LLM branch (line 331) | `stages.stop()` |
| `status.update("Enhancing with LLM...")` (line 387) | `stages.advance("llm", "Enhancing with LLM...")` |
| `progress.start_spinner("Extracting content from screenshot...")` (line 403) | `stages.update_text("Extracting content from screenshot...")` |
| `progress.log("LLM processing complete (screenshot-only)")` (line 431) | `stages.finalize("LLM enhanced (screenshot-only)")` |
| `progress.start_spinner("Processing with Vision LLM (multi-source)...")` (line 443) | `stages.update_text("Processing with Vision LLM...")` |
| `progress.log("LLM processing complete (vision enhanced)")` (line 481) | `stages.finalize("LLM enhanced (vision)")` |
| `progress.start_spinner("Processing with LLM...")` (line 485) | delete (the `advance("llm", ...)` above already covers it) |
| `progress.log("LLM processing complete")` (line 498) | `stages.finalize("LLM enhanced")` |
| `progress.start_spinner("Processing document and images with LLM...")` (line 502) | replace the earlier plain advance for this parallel branch: `stages.advance("llm", "Enhancing with LLM (document + images)...", pin=True)` — see 3e |
| `progress.log("LLM processing complete (document + images)")` (line 545) | `stages.finalize("LLM enhanced (document + images)")` |
| `progress.start_spinner("Processing with LLM...")` (line 548) | delete |
| `progress.log("LLM processing complete")` (line 561) | `stages.finalize("LLM enhanced")` |
| `status.stop()` + `progress.clear_and_finish()` (lines 574-575) | `stages.stop()` |
| `status.stop()` in generic `except Exception` (line 763) | `stages.fail()` then `stages.stop()` |
| `status.stop()` in `finally` (line 768) | `stages.stop()` |

3e. Structural adjustment for the LLM branch: the single `stages.advance("llm", "Enhancing with LLM...")` at the top of `if cfg.llm.enabled:` (replacing old line 387) must NOT be pinned for the sequential branches, but the parallel document+images branch needs pin. Implement by moving the advance into the branches:

```python
        if cfg.llm.enabled:
            logger.info(f"[LLM] Processing URL content: {url}")
            ...
            if use_screenshot_only and screenshot_path:
                stages.advance("llm", "Extracting content from screenshot...", pin=True)
                ...
            elif has_screenshot:
                if use_vision_enhancement and screenshot_path:
                    stages.advance("llm", "Processing with Vision LLM...", pin=True)
                    ...
                else:
                    stages.advance("llm", "Enhancing with LLM...", pin=True)
                    ...
            elif should_analyze_images:
                stages.advance("llm", "Enhancing with LLM (document + images)...", pin=True)
                ...
            else:
                stages.advance("llm", "Enhancing with LLM...", pin=True)
                ...
```

All LLM stages are pinned: `logger.info(f"[LLM] ...")` fires right before (and inside processors), and unpinned stages would have their carefully-worded text overwritten by the bridge's generic "Enhancing with LLM...". Pinning also prevents the `Analyzing images` bridge entry from advancing away mid-parallel-run. (The bridge's llm/images entries still matter for the FILE path, which has no explicit calls.)

3f. In `packages/markitai/src/markitai/cli/main.py` remove the `output_manager=om,` line from the `process_url(...)` call (line 939).

- [ ] **Step 4: Delete the replaced URL test classes**

In `packages/markitai/tests/unit/cli/test_conversion_status.py` delete classes `TestProcessorGating`, `TestStdoutStaysPure` (URL test only — keep `test_file_conversion_*` methods by moving them into a temporary holding class? NO — keep the whole `TestStdoutStaysPure` class but delete only its URL method `test_url_conversion_spinner_on_stderr_only`), and delete `TestConsentPromptInterplay` entirely (its stdout-purity assertion is covered by `TestUrlStdoutStaysPure`; the prompt-interplay v1 limitation is documented in the spec). Also delete the now-unused `process_url` import and `make_fetch_result` helper if no remaining test uses them.

- [ ] **Step 5: Run tests**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_progress_integration.py tests/unit/cli/test_stage_list.py tests/unit/cli/test_conversion_status.py -x -q`
Expected: all PASS.

Run the URL-processor suite: `cd packages/markitai && uv run pytest tests/unit/test_url_processor.py -x -q`
Expected: all PASS (these tests call `process_url`; if any passes `output_manager=`, update the call to drop it).

- [ ] **Step 6: Lint, format, typecheck, commit**

Run: `cd packages/markitai && uv run ruff check src tests && uv run ruff format src tests && uv run pyright src`
Expected: clean.

```bash
git add packages/markitai/src/markitai/cli/processors/url.py packages/markitai/src/markitai/cli/main.py packages/markitai/tests/unit/cli/test_progress_integration.py packages/markitai/tests/unit/cli/test_conversion_status.py
git commit -m "feat(cli): drive single-URL conversions with StageList (stdout mode included)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Migrate the single-file path to StageList

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/file.py` (function `process_single_file`)
- Modify: `packages/markitai/src/markitai/cli/main.py:983` (drop `output_manager=om` from the `process_single_file` call)
- Modify: `packages/markitai/tests/unit/cli/test_progress_integration.py` (add file-processor tests)
- Modify: `packages/markitai/tests/unit/cli/test_conversion_status.py` (delete remaining file-path classes)

**Interfaces:**
- Consumes: `ui.StageList` (Task 1); loguru bridge auto-advances `convert → llm → images` from `convert_document_core` logs — no changes inside `workflow/`.
- Produces: `process_single_file` signature loses `output_manager: Any = None`. New keyword set: `(input_path, output_dir, cfg, dry_run, log_file_path=None, verbose=False, quiet=False)`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/markitai/tests/unit/cli/test_progress_integration.py`:

```python
class TestFileProcessorStageList:
    """process_single_file drives StageList; bridge advances stages."""

    async def test_file_mode_persists_convert_stage(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        from markitai.cli.processors.file import process_single_file

        fake_stderr = make_tty_console()
        monkeypatch.setattr(
            "markitai.cli.ui.get_stderr_console", lambda: fake_stderr
        )

        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        await process_single_file(
            input_path=input_file,
            output_dir=out_dir,
            cfg=cfg,
            dry_run=False,
            verbose=False,
            quiet=False,
        )

        stderr_text = fake_stderr.file.getvalue()  # type: ignore[union-attr]
        # Converted stage line persisted with a duration
        assert "Converted input.txt" in stderr_text
        assert (out_dir / "input.txt.md").exists()
        captured = capsys.readouterr()
        assert HIDE_CURSOR not in captured.out

    async def test_stdout_mode_is_transient_but_active(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        from markitai.cli.processors.file import process_single_file

        fake_stderr = make_tty_console()
        monkeypatch.setattr(
            "markitai.cli.ui.get_stderr_console", lambda: fake_stderr
        )

        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\n", encoding="utf-8")

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        await process_single_file(
            input_path=input_file,
            output_dir=None,  # stdout mode
            cfg=cfg,
            dry_run=False,
            verbose=False,
            quiet=False,
        )

        # Live rendered on stderr (the fix: stdout mode is no longer silent)
        assert HIDE_CURSOR in fake_stderr.file.getvalue()  # type: ignore[union-attr]
        captured = capsys.readouterr()
        assert "hello world" in captured.out
        assert HIDE_CURSOR not in captured.out

    async def test_quiet_mode_fully_silent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from markitai.cli.processors.file import process_single_file

        fake_stderr = make_tty_console()
        monkeypatch.setattr(
            "markitai.cli.ui.get_stderr_console", lambda: fake_stderr
        )

        input_file = tmp_path / "input.txt"
        input_file.write_text("hello world\n", encoding="utf-8")
        out_dir = tmp_path / "out"

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        await process_single_file(
            input_path=input_file,
            output_dir=out_dir,
            cfg=cfg,
            dry_run=False,
            verbose=False,
            quiet=True,
        )

        assert fake_stderr.file.getvalue() == ""  # type: ignore[union-attr]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_progress_integration.py::TestFileProcessorStageList -x -q`
Expected: FAIL — "Converted input.txt" never appears (old code shows only a spinner, persists nothing).

- [ ] **Step 3: Rewrite process_single_file's progress wiring**

In `packages/markitai/src/markitai/cli/processors/file.py`:

3a. Remove `output_manager: Any = None` from the signature (and docstring). Remove the `ProgressReporter` import (check the imports at the top of the file — it's imported from `markitai.utils.progress`).

3b. Replace the ProgressReporter + ConversionStatus block (current lines 256-267) with:

```python
    # Live multi-stage progress list (stderr). Enabled in BOTH file and
    # stdout modes; suppressed for --quiet and -v/verbose. The stage text
    # follows conversion logs via the loguru bridge (see cli.ui.StageList) —
    # convert_document_core needs no changes: its "[LLM]"/"Analyzing" logs
    # advance the list automatically.
    stages = ui.StageList(
        enabled=not quiet and not verbose,
        transient=stdout_mode,
    )
```

3c. Replace call sites:

| Old code | New code |
|---|---|
| `progress.start_spinner(f"Converting {input_path.name}...")` + `status.start()` (lines 274-275) | `stages.start()` + `stages.advance("convert", f"Converting {input_path.name}...")` |
| `status.stop()` + `progress.stop_spinner()` in skip-exists branch (lines 296-297) | `stages.stop()` |
| `status.stop()` + `progress.stop_spinner()` in image-only branch (lines 310-311) | `stages.stop()` |
| `status.stop()` + `progress.stop_spinner()` before output (lines 321-322) | `stages.finalize(_file_final_stage_text(stages.active_key, input_path.name))` + `stages.stop()` |
| any `status.stop()` / `progress.*` in the error/except/finally paths further down (search the rest of the function for `status.` and `progress.` — there is a `status.stop()` in the exception handler and in `finally`) | error paths: `stages.fail()` + `stages.stop()`; `finally`: `stages.stop()` |

3d. Add this module-level helper above `process_single_file` (the bridge may have advanced the active stage past "convert", so the completion text depends on which stage is active when the pipeline returns):

```python
def _file_final_stage_text(active_key: str | None, input_name: str) -> str:
    """Completion text for the last active stage of a file conversion."""
    if active_key == "llm":
        return "LLM enhanced"
    if active_key == "images":
        return "Images analyzed"
    return f"Converted {input_name}"
```

3e. In `packages/markitai/src/markitai/cli/main.py` remove the `output_manager=om,` line from the `process_single_file(...)` call (line 983).

- [ ] **Step 4: Delete replaced file-path test classes**

In `tests/unit/cli/test_conversion_status.py`: delete what remains of `TestStdoutStaysPure` (the file-conversion methods — now covered by `TestFileProcessorStageList`). After this deletion the file should only contain: `TestSpinnerChoice`, `TestStageFromLogRecord`, `TestConversionStatusBridge`, `TestElapsedSuffix`, `TestConversionStatusTicker`, `TestConversionStatusGating` (pure component tests, deleted with the component in Task 4). Remove imports that became unused (`process_single_file`, `process_url`, `MarkitaiConfig`, `AsyncMock` etc. as applicable) so ruff stays green.

- [ ] **Step 5: Run tests**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/ -x -q`
Expected: all PASS.

Run the file-processor suite: `cd packages/markitai && uv run pytest tests/unit/cli/test_file_processor.py -x -q`
Expected: all PASS (if any test passes `output_manager=`, drop that argument).

- [ ] **Step 6: Lint, format, typecheck, commit**

Run: `cd packages/markitai && uv run ruff check src tests && uv run ruff format src tests && uv run pyright src`
Expected: clean.

```bash
git add packages/markitai/src/markitai/cli/processors/file.py packages/markitai/src/markitai/cli/main.py packages/markitai/tests/unit/cli/test_progress_integration.py packages/markitai/tests/unit/cli/test_conversion_status.py
git commit -m "feat(cli): drive single-file conversions with StageList

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Migrate init.py; retire ConversionStatus

**Files:**
- Modify: `packages/markitai/src/markitai/cli/commands/init.py:85` and `:300`
- Modify: `packages/markitai/src/markitai/cli/ui.py` (delete ConversionStatus + its private tables + old `stage_from_log_record`/`_is_stage_record`; drop now-unused imports)
- Delete: `packages/markitai/tests/unit/cli/test_conversion_status.py`
- Modify: `packages/markitai/tests/unit/cli/test_stage_list.py` (absorb the spinner-is-ASCII test)

**Interfaces:**
- Consumes: `ui.StageList` single-stage usage: `advance` + context manager.
- Produces: `markitai.cli.ui` no longer exports `ConversionStatus`, `stage_from_log_record`. `elapsed_suffix`, `STATUS_SPINNER`, `ELAPSED_SUFFIX_THRESHOLD_S` remain (StageList uses them).

- [ ] **Step 1: Migrate the two init.py call sites**

At `init.py:85` replace:

```python
    with ui.ConversionStatus("Detecting LLM providers..."):
        providers = _detect_providers()
```

with:

```python
    with ui.StageList(transient=True) as stages:
        stages.advance("detect", "Detecting LLM providers...")
        providers = _detect_providers()
```

At `init.py:300` replace:

```python
    spinner = ui.ConversionStatus("Checking dependencies...")
    spinner.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            deps_future = executor.submit(_check_deps)
            providers_future = executor.submit(_detect_providers)
            deps = deps_future.result()
            spinner.update("Detecting LLM providers...")
            providers = providers_future.result()
    finally:
        spinner.stop()
```

with:

```python
    stages = ui.StageList(transient=True)
    stages.start()
    try:
        stages.advance("init", "Checking dependencies...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            deps_future = executor.submit(_check_deps)
            providers_future = executor.submit(_detect_providers)
            deps = deps_future.result()
            stages.update_text("Detecting LLM providers...")
            providers = providers_future.result()
    finally:
        stages.stop()
```

(`transient=True`: these wizard spinners were transient before — the sections render right after; a leftover checklist line would be noise.)

- [ ] **Step 2: Delete ConversionStatus and its private helpers from ui.py**

Delete from `ui.py`: class `ConversionStatus`, `_STAGE_MESSAGE_PREFIXES`, `_STAGE_MODULES`, `stage_from_log_record`, `_is_stage_record`, and the module docstring section describing them. Keep `STATUS_SPINNER`, `ELAPSED_SUFFIX_THRESHOLD_S`, `ELAPSED_TICK_INTERVAL_S` — wait: `ELAPSED_TICK_INTERVAL_S` was only used by ConversionStatus's ticker; delete it too. Keep `elapsed_suffix` (StageList uses it). Remove now-unused imports: `asyncio`, `from rich.markup import escape` (verify with ruff — `escape` may still be used elsewhere in the file), and the `Status` TYPE_CHECKING import.

- [ ] **Step 3: Move the ASCII-spinner test, delete the old test file**

Add to `test_stage_list.py` (top level, after the helpers):

```python
class TestSpinnerChoice:
    """The configured spinner frames must be pure ASCII."""

    def test_spinner_is_pure_ascii(self) -> None:
        from rich.spinner import Spinner

        frames = Spinner(ui.STATUS_SPINNER).frames
        assert frames, "spinner has no frames"
        for frame in frames:
            assert all(ord(ch) < 128 for ch in frame), (
                f"non-ASCII frame in spinner {ui.STATUS_SPINNER!r}: {frame!r}"
            )
```

Then delete `packages/markitai/tests/unit/cli/test_conversion_status.py` entirely (its remaining component tests are superseded by `test_stage_list.py`; `elapsed_suffix` unit tests move too — add this class to `test_stage_list.py`):

```python
class TestElapsedSuffix:
    """Tests for the pure "(Ns)" elapsed-time suffix helper."""

    def test_no_suffix_before_threshold(self) -> None:
        started = 100.0
        assert ui.elapsed_suffix(started, now=started + 4.9) == ""

    def test_suffix_appears_at_threshold(self) -> None:
        started = 100.0
        assert (
            ui.elapsed_suffix(started, now=started + ui.ELAPSED_SUFFIX_THRESHOLD_S)
            == " (5s)"
        )

    def test_suffix_rounds_to_whole_seconds(self) -> None:
        started = 100.0
        assert ui.elapsed_suffix(started, now=started + 72.6) == " (73s)"

    def test_no_start_time_means_no_suffix(self) -> None:
        assert ui.elapsed_suffix(None, now=1000.0) == ""
```

- [ ] **Step 4: Verify nothing references the deleted names**

Run: `cd packages/markitai && grep -rn "ConversionStatus\|stage_from_log_record\|ELAPSED_TICK_INTERVAL_S" src tests`
Expected: no output (zero references). If `grep` finds stragglers (e.g. `tests/unit/cli/test_init.py` mocking ConversionStatus), fix them to use StageList.

- [ ] **Step 5: Run tests, lint, typecheck**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/ -x -q && uv run ruff check src tests && uv run ruff format src tests && uv run pyright src`
Expected: all PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add -A packages/markitai/src/markitai/cli/ui.py packages/markitai/src/markitai/cli/commands/init.py packages/markitai/tests/unit/cli/
git commit -m "refactor(cli): retire ConversionStatus in favor of StageList

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Retire ProgressReporter and OutputManager; clean exports; CHANGELOG

**Files:**
- Delete: `packages/markitai/src/markitai/utils/progress.py`
- Delete: `packages/markitai/src/markitai/cli/output_manager.py`
- Delete: `packages/markitai/tests/unit/test_output_manager.py`
- Modify: `packages/markitai/src/markitai/utils/__init__.py` (drop ProgressReporter import/export)
- Modify: `packages/markitai/src/markitai/cli/__init__.py` (drop ProgressReporter import/export)
- Modify: `packages/markitai/src/markitai/cli/main.py` (delete the OutputManager import + `om = OutputManager(...)` creation at lines 798-802; drop `output_manager=om` from the `attempt_login` call at line 881)
- Modify: `packages/markitai/src/markitai/providers/auth.py` (remove the dead `output_manager` parameter from `attempt_login`, `_login_copilot`, `_login_claude_agent`, `_login_chatgpt` and any internal pass-through, e.g. into oauth display helpers)
- Modify: `packages/markitai/src/markitai/providers/oauth_display.py` (docstrings mention OutputManager — update wording; check for an `output_manager` parameter on display functions and remove if present)
- Modify: `packages/markitai/CHANGELOG.md` (new Unreleased entry)

**Interfaces:**
- Consumes: nothing new.
- Produces: `markitai.utils` and `markitai.cli` no longer export `ProgressReporter`; `markitai.cli.output_manager` module no longer exists; `attempt_login(provider: str) -> AuthStatus`.

- [ ] **Step 1: Verify the retirement premise**

Run: `cd packages/markitai && grep -rn "ProgressReporter\|OutputManager\|output_manager" src | grep -v "Binary"`
Expected output: matches ONLY in the files listed above (utils/progress.py itself, cli/output_manager.py itself, the two `__init__.py` exports, main.py lines 798-802/881, providers/auth.py parameters, oauth_display.py docstrings). If anything else shows up, STOP and report — the premise "no other callers" is broken and the reviewer must decide.

- [ ] **Step 2: Delete and clean**

- `git rm packages/markitai/src/markitai/utils/progress.py packages/markitai/src/markitai/cli/output_manager.py packages/markitai/tests/unit/test_output_manager.py`
- In `utils/__init__.py`: remove `from markitai.utils.progress import ProgressReporter` and the `"ProgressReporter",` line in `__all__`.
- In `cli/__init__.py`: remove `from markitai.utils.progress import ProgressReporter` and `"ProgressReporter",` in `__all__`.
- In `main.py`: delete lines 798-802 (comment, import, `om = OutputManager(...)`); change `attempt_login(status.provider, output_manager=om)` to `attempt_login(status.provider)`.
- In `providers/auth.py`: remove the `output_manager: Any = None` parameter (and docstring text) from `attempt_login`, `_login_copilot`, `_login_claude_agent`, `_login_chatgpt`; remove any pass-through arguments. If `_login_chatgpt` forwards `output_manager` into `oauth_display` helpers, remove those forwarding arguments and the corresponding (unused) parameters in `oauth_display.py`.
- In `oauth_display.py`: update the two docstrings that mention OutputManager (lines 87, 113) to describe the actual behavior (writes to stderr console directly).
- In `tests/unit/test_oauth_display.py`: update any docstring/mock references to OutputManager.

- [ ] **Step 3: CHANGELOG entry**

Read the top of `packages/markitai/CHANGELOG.md` and mirror the existing entry format. Content to convey under Unreleased (adapt to the file's structure):

```markdown
### Added
- Multi-stage live progress checklist (StageList) for single-URL and single-file
  conversions: completed stages persist as `✓ Fetched via fxtwitter (2.1s)` lines,
  the active stage shows a spinner with an elapsed-time suffix. Stdout-mode
  conversions (no `-o`) finally show progress — they were previously fully silent
  through fetch + LLM enhancement.

### Removed
- Internal `ConversionStatus`, `ProgressReporter`, and `OutputManager` progress
  facilities, replaced by StageList. `markitai.utils`/`markitai.cli` no longer
  export `ProgressReporter`; `attempt_login()` lost its unused `output_manager`
  parameter.
```

- [ ] **Step 4: Full unit-test run, lint, typecheck**

Run: `cd packages/markitai && uv run pytest tests/unit -x -q -p no:cacheprovider`
Expected: all PASS. Auth/oauth/preflight suites (`test_auth_cli.py`, `test_provider_auth.py`, `test_oauth_display.py`, `test_preflight_auth.py`) prove the parameter removal broke nothing.

Run: `cd packages/markitai && uv run ruff check src tests && uv run ruff format src tests && uv run pyright src`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add -A packages/markitai/src packages/markitai/tests packages/markitai/CHANGELOG.md
git commit -m "refactor(cli): retire ProgressReporter and OutputManager

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(Verify with `git status` first that only feature files are staged — the repo has unrelated dirty files under `website/`.)

---

### Task 6: Full-suite verification + end-to-end evidence

**Files:** none created (verification only; fix regressions in place if found).

- [ ] **Step 1: Full test suite**

Run: `cd packages/markitai && uv run pytest -q -m "not network and not slow" -n auto`
Expected: all PASS (pytest-xdist parallel; deselect network/slow like CI does).

- [ ] **Step 2: Static gates**

Run: `cd packages/markitai && uv run ruff check src tests && uv run ruff format --check src tests && uv run pyright src`
Expected: clean, no diffs.

- [ ] **Step 3: Scripted TTY smoke test (stdout mode)**

Non-interactive TTY simulation via `script` (macOS syntax). This exercises the real CLI with a fast local file in stdout mode and asserts stdout purity:

```bash
cd packages/markitai
printf 'hello stagelist\n' > /tmp/stagelist-smoke.txt
script -q /tmp/stagelist-tty.log uv run markitai /tmp/stagelist-smoke.txt > /tmp/stagelist-stdout.txt
grep -c $'\x1b\[?25l' /tmp/stagelist-tty.log   # hide-cursor: Live ran → expect >= 1
grep -c $'\x1b' /tmp/stagelist-stdout.txt || echo "stdout clean"
cat /tmp/stagelist-stdout.txt                   # expect the markdown content
```

Note: with `script`, stdout+stderr both land in the combined TTY log, while the `>` redirect captures what the process wrote to fd 1. Expected: hide-cursor marker present in the TTY log; `stdout clean` (no ESC bytes in the redirected stdout); markdown content intact.

- [ ] **Step 4: Report for manual verification**

The finishing report must list the exact commands for the human to run in a real terminal (the agent cannot observe interactive rendering):

```bash
uv run markitai https://x.com/taresky/status/2075041205352861759 -p standard --no-cache
uv run markitai https://x.com/taresky/status/2075041205352861759 -p standard --no-cache -o /tmp/mkai-e2e/
# during the run: watch stages advance; after: -o mode keeps the ✓ list
# also try Ctrl-C mid-LLM: terminal must be left clean (cursor visible, no stray frames)
```

---

## Self-Review Notes

- Spec coverage: §1 rendering → Task 1; §2 stage model/bridge/pin → Task 1 (component) + Task 2 3e (pin usage); §3 migration table → Tasks 2-5; §4 edge behavior → Task 1 tests (TestGatingAndDegradation, TestStopBehavior) + Task 2 error-path test; §5 text spec → finalize texts in Tasks 2-3; §6 testing → every task + Task 6.
- Deviation from spec §2: no asyncio ticker — Live auto-refresh recomputes the elapsed suffix in `__rich_console__` (simpler, and works in sync contexts like init.py). Spec's "沿用 ticker 机制" is about the user-visible behavior (suffix past 5s), which is preserved via the same `elapsed_suffix()` helper and threshold.
- Deviation from spec §2: `advance` takes `pin` per-call; ALL explicit URL-path LLM advances pin (not just the parallel branch) because `[LLM]` logs fire in every branch and would clobber branch-specific texts. File path relies on unpinned bridge advancement as specced.
- Type consistency check: `StageList(enabled=, transient=, console=)` used identically in Tasks 2, 3, 4; `finalize(text, annotation=)` matches Task 1 signature; `_file_final_stage_text(active_key, input_name)` defined and used only in Task 3.
