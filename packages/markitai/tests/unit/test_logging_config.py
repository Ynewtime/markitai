"""Tests for CLI logging configuration."""

from __future__ import annotations

from markitai.cli.logging_config import _should_show_log


def _record(message: str, *, level: str = "INFO", module: str = "") -> dict:
    """Build a minimal loguru-like record for filter tests."""
    logger_name = f"markitai.{module}" if module else ""
    return {
        "message": message,
        "module": module,
        "name": logger_name,
        "level": type("Level", (), {"name": level})(),
        "extra": {
            "module": module,
            "name": logger_name,
        },
    }


class TestShouldShowLog:
    """Tests for console log filtering."""

    def test_non_verbose_hides_info(self) -> None:
        """INFO logs should stay out of console by default."""
        assert not _should_show_log(_record("Written output: out.md"), verbose=False)

    def test_verbose_keeps_user_facing_info(self) -> None:
        """Useful verbose messages should remain visible."""
        assert _should_show_log(
            _record("[Config] Using MODEL env var: gpt-4o-mini", module="main"),
            verbose=True,
        )

    def test_verbose_hides_router_internal_info(self) -> None:
        """Router telemetry should go to the log file, not the terminal."""
        assert not _should_show_log(
            _record(
                "[Router] Creating with strategy=simple-shuffle", module="processor"
            ),
            verbose=True,
        )

    def test_verbose_hides_llm_write_info(self) -> None:
        """LLM output write notifications should stay off the terminal."""
        assert not _should_show_log(
            _record("Written LLM version: output/test.llm.md", module="llm"),
            verbose=True,
        )

    def test_verbose_hides_url_fetch_telemetry(self) -> None:
        """Per-URL fetch telemetry should stay in the log file."""
        assert not _should_show_log(
            _record(
                "Fetched via playwright: https://example.com/post",
                module="url",
            ),
            verbose=True,
        )

    def test_verbose_shows_llm_call_timing(self) -> None:
        """Per-call timing summaries survive the noise filter under -v.

        These are the only signal a -v user has of how long an LLM stage
        is actually taking (the spinner collapses it to one static line);
        module="document" would otherwise be blanket-filtered as noise.
        """
        assert _should_show_log(
            _record(
                "[LLM:https://x.com/a/status/1] document_process: "
                "claude-agent/haiku tokens=1574+6729 time=73558ms cost=$0.046399",
                module="document",
            ),
            verbose=True,
        )

    def test_non_verbose_still_hides_llm_call_timing(self) -> None:
        """Timing summaries are still gated behind -v, not shown by default."""
        assert not _should_show_log(
            _record(
                "[LLM:test] document_process: default tokens=1+2 time=100ms cost=$0.0",
                module="document",
            ),
            verbose=False,
        )

    def test_warning_still_shows(self) -> None:
        """Warnings must always reach the console."""
        assert _should_show_log(
            _record(
                "No content extracted from URL: https://example.com", level="WARNING"
            ),
            verbose=False,
        )


class TestQuietModeErrorHandler:
    """Tests that quiet mode still surfaces ERROR+ to stderr."""

    def test_quiet_mode_returns_console_handler(self) -> None:
        """setup_logging(quiet=True) should still return a console handler ID."""
        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        # Save and restore loguru state
        handler_id, _ = setup_logging(verbose=False, quiet=True, log_dir=None)
        assert handler_id is not None, (
            "quiet mode should add an ERROR-level console handler"
        )
        # Clean up
        logger.remove(handler_id)

    def test_quiet_mode_error_reaches_stderr(self, capsys) -> None:
        """In quiet mode, logger.error() should still appear on stderr."""
        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        handler_id, _ = setup_logging(verbose=False, quiet=True, log_dir=None)
        try:
            logger.error("LLM processing failed: test error")
            captured = capsys.readouterr()
            assert "LLM processing failed" in captured.err
        finally:
            logger.remove(handler_id)

    def test_quiet_mode_warning_hidden_from_stderr(self, capsys) -> None:
        """In quiet mode, logger.warning() should NOT appear on stderr."""
        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        handler_id, _ = setup_logging(verbose=False, quiet=True, log_dir=None)
        try:
            logger.warning("This should be hidden")
            captured = capsys.readouterr()
            assert "This should be hidden" not in captured.err
        finally:
            logger.remove(handler_id)


class TestJsonLogFormat:
    """Tests for the JSON file log format."""

    def test_json_log_lines_are_valid_json(self, tmp_path, monkeypatch) -> None:
        """Messages with quotes/newlines/backslashes must stay valid JSON."""
        import json

        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        monkeypatch.delenv("MARKITAI_LOG_DIR", raising=False)
        monkeypatch.delenv("MARKITAI_LOG_FORMAT", raising=False)

        handler_id, log_file = setup_logging(
            verbose=False,
            log_dir=str(tmp_path),
            log_format="json",
            quiet=True,
        )
        message = 'nasty "quoted" message with backslash \\ and\nnewline'
        try:
            logger.info(message)
        finally:
            logger.remove()  # flush and close all handlers

        assert log_file is not None
        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        assert lines, "expected at least one log line"

        entries = [json.loads(ln) for ln in lines]  # must not raise
        matching = [e for e in entries if e["msg"] == message]
        assert matching, "logged message should round-trip through JSON intact"
        assert matching[0]["lvl"] == "INFO"
        assert "ts" in matching[0] and "src" in matching[0]
