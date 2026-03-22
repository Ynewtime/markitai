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
