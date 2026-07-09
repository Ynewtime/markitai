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
            (
                "JS required: string pattern matched 'x'",
                ("fetch", "Rendering (playwright)"),
            ),
            (
                "[Fetch] Enriching via FxTwitter/oEmbed: https://x.com/a",
                ("fetch", "Fetching (fxtwitter)"),
            ),
            (
                "[Defuddle] Fetching: https://example.com",
                ("fetch", "Fetching (defuddle)"),
            ),
            (
                "Fetching URL with Jina Reader (JSON mode): https://example.com",
                ("fetch", "Fetching (jina)"),
            ),
            (
                "Fetching URL with CF Browser Rendering: https://example.com",
                ("fetch", "Rendering (cloudflare)"),
            ),
            (
                "[LLM] doc.md: Starting standard LLM processing",
                ("llm", "Enhancing with LLM"),
            ),
            ("Analyzing image 1/3", ("images", "Analyzing images")),
        ],
    )
    def test_known_message_prefixes(
        self, message: str, expected: tuple[str, str]
    ) -> None:
        record = {"message": message, "module": "fetch"}
        assert ui.stage_entry_from_log_record(record) == expected

    def test_playwright_module_maps_to_fetch_rendering(self) -> None:
        record = {
            "message": "Created new cached Playwright context for: x",
            "module": "fetch_playwright",
        }
        assert ui.stage_entry_from_log_record(record) == (
            "fetch",
            "Rendering (playwright)",
        )

    def test_unknown_record_returns_none(self) -> None:
        assert (
            ui.stage_entry_from_log_record({"message": "hi", "module": "config"})
            is None
        )

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

    def test_double_stop_does_not_duplicate_failure_list(self) -> None:
        console = make_pipe_console()
        stages = ui.StageList(console=console, transient=True)
        stages.start()
        stages.advance("fetch", "Fetching...")
        stages.fail()
        stages.stop()
        stages.stop()  # e.g. explicit stop in error path + finally safety net
        out = console.file.getvalue()  # type: ignore[union-attr]
        assert out.count("Fetching") == 1


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
            segments = list(stages.__rich_console__(console, console.options))
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
