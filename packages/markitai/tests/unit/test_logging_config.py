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
