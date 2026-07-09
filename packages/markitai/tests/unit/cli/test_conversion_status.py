"""Tests for the live conversion status spinner (cli.ui.ConversionStatus).

Covers:
- the loguru log -> spinner stage-text bridge (stage_from_log_record)
- TTY / quiet / verbose gating
"""

from __future__ import annotations

import asyncio
import io
import time
from unittest.mock import patch

import pytest
from loguru import logger
from rich.console import Console

from markitai.cli import ui

# ANSI hide-cursor sequence emitted by rich Status/Live on a terminal;
# a reliable marker that the spinner rendered on a given stream.
HIDE_CURSOR = "\x1b[?25l"


def make_tty_console() -> Console:
    """Console backed by a StringIO that claims to be a terminal."""
    return Console(file=io.StringIO(), force_terminal=True, width=80)


class TestSpinnerChoice:
    """Tests for the spinner constant."""

    def test_spinner_is_pure_ascii(self) -> None:
        """The configured spinner frames must be pure ASCII."""
        from rich.spinner import Spinner

        frames = Spinner(ui.STATUS_SPINNER).frames
        assert frames, "spinner has no frames"
        for frame in frames:
            assert all(ord(ch) < 128 for ch in frame), (
                f"non-ASCII frame in spinner {ui.STATUS_SPINNER!r}: {frame!r}"
            )


class TestStageFromLogRecord:
    """Tests for the log-record -> stage-label mapping."""

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                "Fetching URL with static strategy: https://example.com",
                "Fetching (static)",
            ),
            (
                "Fetching URL with static httpx strategy: https://example.com",
                "Fetching (static)",
            ),
            ("JS required: string pattern matched 'x'", "Rendering (playwright)"),
            (
                "[Fetch] Enriching via FxTwitter/oEmbed: https://x.com/a",
                "Fetching (fxtwitter)",
            ),
            ("[Defuddle] Fetching: https://example.com", "Fetching (defuddle)"),
            (
                "Fetching URL with Jina Reader (JSON mode): https://example.com",
                "Fetching (jina)",
            ),
            (
                "Fetching URL with CF Browser Rendering: https://example.com",
                "Rendering (cloudflare)",
            ),
            ("[LLM] doc.md: Starting standard LLM processing", "Enhancing with LLM"),
            ("Analyzing image 1/3", "Analyzing images"),
        ],
    )
    def test_known_message_prefixes(self, message: str, expected: str) -> None:
        record = {"message": message, "module": "fetch"}
        assert ui.stage_from_log_record(record) == expected

    def test_playwright_module_maps_to_rendering(self) -> None:
        record = {
            "message": "Created new cached Playwright context for: x",
            "module": "fetch_playwright",
        }
        assert ui.stage_from_log_record(record) == "Rendering (playwright)"

    def test_unknown_record_returns_none(self) -> None:
        record = {"message": "some unrelated log", "module": "config"}
        assert ui.stage_from_log_record(record) is None

    def test_missing_keys_return_none(self) -> None:
        assert ui.stage_from_log_record({}) is None


class TestConversionStatusBridge:
    """Tests for the loguru sink -> spinner text bridge."""

    def test_fetch_log_updates_stage_text(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Fetching example.com...", console=console)
        with status:
            assert status.active
            logger.debug("Fetching URL with static strategy: https://example.com")
            assert status.stage_text == "Fetching (static)..."
            logger.debug("JS required: content too short (10 chars)")
            assert status.stage_text == "Rendering (playwright)..."
        assert not status.active

    def test_unrelated_log_does_not_change_text(self) -> None:
        console = make_tty_console()
        with ui.ConversionStatus("Initial...", console=console) as status:
            logger.debug("totally unrelated debug message")
            assert status.stage_text == "Initial..."

    def test_sink_removed_after_stop(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Initial...", console=console)
        status.start()
        status.stop()
        logger.debug("[Defuddle] Fetching: https://example.com")
        assert status.stage_text == "Initial..."

    def test_stop_is_idempotent(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Initial...", console=console)
        status.start()
        status.stop()
        status.stop()  # must not raise
        assert not status.active

    def test_update_before_start_is_safe(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Initial...", console=console)
        status.update("Later...")
        assert status.stage_text == "Later..."
        assert not status.active


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


class TestConversionStatusTicker:
    """Tests for the elapsed-time ticker task lifecycle."""

    async def test_ticker_started_in_async_context(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Enhancing with LLM...", console=console)
        status.start()
        try:
            assert status._ticker_task is not None
            assert not status._ticker_task.done()
        finally:
            status.stop()

    async def test_ticker_cancelled_on_stop(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Enhancing with LLM...", console=console)
        status.start()
        task = status._ticker_task
        status.stop()
        assert status._ticker_task is None
        assert task is not None
        # Let the cancellation actually propagate before asserting.
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    async def test_long_running_stage_gets_elapsed_suffix_in_render(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Enhancing with LLM...", console=console)
        status.start()
        try:
            assert status._status is not None
            with patch.object(status._status, "update") as mock_update:
                status._stage_started_at = time.monotonic() - 10.0
                status._render()
            rendered = mock_update.call_args.args[0]
            assert "(10s)" in rendered
        finally:
            status.stop()

    def test_start_without_running_loop_disables_ticker(self) -> None:
        """Sync-driven usage (no event loop) must not raise."""
        console = make_tty_console()
        status = ui.ConversionStatus("Fetching...", console=console)
        status.start()  # this test is sync: no running loop
        assert status._ticker_task is None
        assert status.active
        status.stop()


class TestConversionStatusGating:
    """Tests for TTY / enabled gating."""

    def test_non_tty_console_disables_status(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=False)
        status = ui.ConversionStatus("Fetching...", console=console)
        assert not status.enabled
        status.start()
        assert not status.active
        assert console.file.getvalue() == ""  # type: ignore[union-attr]

    def test_enabled_false_disables_status_even_on_tty(self) -> None:
        console = make_tty_console()
        status = ui.ConversionStatus("Fetching...", console=console, enabled=False)
        status.start()
        assert not status.active
        assert console.file.getvalue() == ""  # type: ignore[union-attr]

    def test_enabled_true_on_tty_renders_spinner(self) -> None:
        console = make_tty_console()
        with ui.ConversionStatus("Fetching...", console=console) as status:
            assert status.active
        assert HIDE_CURSOR in console.file.getvalue()  # type: ignore[union-attr]
