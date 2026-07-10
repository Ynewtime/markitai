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

    def test_quiet_mode_filters_instructor_retry_noise(self, capsys) -> None:
        """Instructor's retry-loop ERROR noise must not leak in quiet mode.

        markitai's own [LLM:...] failure summaries already report the same
        failure; the raw instructor lines are duplication (see
        _is_third_party_retry_noise). The non-quiet sink filters them, and
        the quiet sink must too — this is exactly the sink active in
        single-URL stdout mode where the leak was observed.
        """
        import logging

        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        handler_id, _ = setup_logging(verbose=False, quiet=True, log_dir=None)
        try:
            stdlib_logger = logging.getLogger("instructor.v2.retry")
            stdlib_logger.error(
                "API call failed on attempt 1: ChatGPT connection error:"
            )
            stdlib_logger.error(
                "Max retries exceeded. Total attempts: 1, "
                "Last error: ChatGPT connection error:"
            )
            captured = capsys.readouterr()
            assert "API call failed" not in captured.err
            assert "Max retries exceeded" not in captured.err
        finally:
            logger.remove(handler_id)

    def test_quiet_mode_keeps_other_instructor_errors(self, capsys) -> None:
        """Only the known retry-noise prefixes are filtered, not all errors."""
        import logging

        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        handler_id, _ = setup_logging(verbose=False, quiet=True, log_dir=None)
        try:
            logging.getLogger("instructor.v2.retry").error(
                "Unexpected code path in retry_sync_v2"
            )
            captured = capsys.readouterr()
            assert "Unexpected code path" in captured.err
        finally:
            logger.remove(handler_id)


class TestConsoleSinkRichRouting:
    """Console log lines must go through the shared rich stderr Console.

    Writing straight to sys.stderr tears through an active rich Live
    display (StageList): the error lands on the spinner line and Live's
    next refresh repaints stale frames, which reads as the whole pipeline
    re-running. Routing through the same Console lets rich print log
    lines above the Live region instead.
    """

    def test_quiet_sink_routes_through_rich_stderr_console(self, monkeypatch) -> None:
        import io

        from loguru import logger
        from rich.console import Console

        from markitai.cli import console as console_mod
        from markitai.cli.logging_config import setup_logging

        buf = io.StringIO()
        monkeypatch.setattr(
            console_mod, "_stderr_console", Console(file=buf, width=200)
        )

        handler_id, _ = setup_logging(verbose=False, quiet=True, log_dir=None)
        try:
            logger.error("routed through rich console (quiet)")
            assert "routed through rich console (quiet)" in buf.getvalue()
        finally:
            logger.remove(handler_id)

    def test_normal_sink_routes_through_rich_stderr_console(self, monkeypatch) -> None:
        import io

        from loguru import logger
        from rich.console import Console

        from markitai.cli import console as console_mod
        from markitai.cli.logging_config import setup_logging

        buf = io.StringIO()
        monkeypatch.setattr(
            console_mod, "_stderr_console", Console(file=buf, width=200)
        )

        handler_id, _ = setup_logging(verbose=False, quiet=False, log_dir=None)
        try:
            logger.error("routed through rich console (normal)")
            assert "routed through rich console (normal)" in buf.getvalue()
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


class TestLogUrlRedaction:
    """Every configured log sink must receive credential-safe URLs."""

    def test_file_log_redacts_userinfo_query_and_fragment(
        self, tmp_path, monkeypatch
    ) -> None:
        from loguru import logger

        from markitai.cli.logging_config import setup_logging

        monkeypatch.delenv("MARKITAI_LOG_DIR", raising=False)
        raw_url = (
            "https://alice:password@example.com/private/report"
            "?token=query-secret#access_token=fragment-secret"
        )

        _handler_id, log_file = setup_logging(
            verbose=False,
            quiet=True,
            log_dir=str(tmp_path),
        )
        try:
            logger.error(f"Failed to fetch {raw_url}")
        finally:
            logger.remove()

        assert log_file is not None
        content = log_file.read_text()
        assert "https://example.com/private/report" in content
        assert "alice" not in content
        assert "password" not in content
        assert "query-secret" not in content
        assert "fragment-secret" not in content
