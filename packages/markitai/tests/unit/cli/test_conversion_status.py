"""Tests for the live conversion status spinner (cli.ui.ConversionStatus).

Covers:
- the loguru log -> spinner stage-text bridge (stage_from_log_record)
- TTY / quiet / verbose gating
- stdout purity: spinner frames only ever go to stderr, never stdout
- the remote-consent prompt interplay (mocked prompt mid-fetch)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from rich.console import Console

from markitai.cli import ui
from markitai.cli.processors.file import process_single_file
from markitai.cli.processors.url import process_url
from markitai.config import MarkitaiConfig

# ANSI hide-cursor sequence emitted by rich Status/Live on a terminal;
# a reliable marker that the spinner rendered on a given stream.
HIDE_CURSOR = "\x1b[?25l"


def make_tty_console() -> Console:
    """Console backed by a StringIO that claims to be a terminal."""
    return Console(file=io.StringIO(), force_terminal=True, width=80)


def make_fetch_result(content: str = "# Test Page\n\nSome content.") -> MagicMock:
    """Mock FetchResult matching what process_url consumes."""
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


class TestProcessorGating:
    """Processors must gate the status on quiet/verbose/stdout-mode."""

    async def _run_process_url(self, tmp_path: Path, **kwargs) -> MagicMock:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=make_fetch_result(),
            ),
            patch("markitai.cli.ui.ConversionStatus") as status_cls,
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=kwargs.get("verbose", False),
                quiet=kwargs.get("quiet", False),
            )
        return status_cls

    async def test_default_file_mode_enables_status(self, tmp_path: Path) -> None:
        status_cls = await self._run_process_url(tmp_path)
        assert status_cls.call_args.kwargs["enabled"] is True

    async def test_verbose_disables_status(self, tmp_path: Path) -> None:
        status_cls = await self._run_process_url(tmp_path, verbose=True)
        assert status_cls.call_args.kwargs["enabled"] is False

    async def test_quiet_disables_status(self, tmp_path: Path) -> None:
        status_cls = await self._run_process_url(tmp_path, quiet=True)
        assert status_cls.call_args.kwargs["enabled"] is False

    async def test_stdout_mode_disables_status(self, capsys) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=make_fetch_result(),
            ),
            patch("markitai.cli.ui.ConversionStatus") as status_cls,
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=None,  # stdout mode
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )
        capsys.readouterr()  # discard content output
        assert status_cls.call_args.kwargs["enabled"] is False


class TestStdoutStaysPure:
    """Integration: spinner frames never leak into stdout."""

    async def test_url_conversion_spinner_on_stderr_only(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        async def slow_fetch(*args, **kwargs):
            logger.debug("Fetching URL with static strategy: https://example.com")
            return make_fetch_result()

        with patch("markitai.fetch.fetch_url", side_effect=slow_fetch):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        captured = capsys.readouterr()
        # Spinner rendered on (fake) stderr...
        assert HIDE_CURSOR in fake_stderr.file.getvalue()  # type: ignore[union-attr]
        # ...but stdout carries no spinner control codes or frames
        assert HIDE_CURSOR not in captured.out
        assert "Fetching (static)" not in captured.out
        # Output file exists and final path was reported
        assert (tmp_path / "test.md").exists()

    async def test_file_conversion_spinner_on_stderr_only(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

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

        captured = capsys.readouterr()
        assert HIDE_CURSOR in fake_stderr.file.getvalue()  # type: ignore[union-attr]
        assert HIDE_CURSOR not in captured.out
        assert (out_dir / "input.md").exists()

    async def test_file_conversion_quiet_shows_no_spinner(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

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


class TestConsentPromptInterplay:
    """The lazy remote-consent prompt fires inside the wrapped fetch coroutine.

    v1 behavior: the status stays live around the prompt (pausing would
    require changes to fetch.py, which owns the prompt). These tests pin
    down that the flow completes, the spinner is stopped before final
    output, and stdout stays pure.
    """

    async def test_mocked_prompt_mid_fetch_keeps_stdout_pure(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        prompt_calls: list[str] = []

        async def fetch_with_prompt(*args, **kwargs):
            # Simulate resolve_remote_consent(): click.confirm(err=True)
            # writes the prompt to stderr while the spinner is live.
            with patch("click.confirm", return_value=False) as confirm:
                import click

                answer = click.confirm(
                    "Allow sending URLs to remote extraction services?",
                    default=False,
                    err=True,
                )
                prompt_calls.append(f"answered={answer}")
                assert confirm.called
            sys.stderr.write("Allow sending URLs ...? [y/N]: n\n")
            return make_fetch_result()

        with patch("markitai.fetch.fetch_url", side_effect=fetch_with_prompt):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        captured = capsys.readouterr()
        assert prompt_calls == ["answered=False"]
        # Conversion completed despite the mid-fetch prompt
        assert (tmp_path / "test.md").exists()
        # stdout never received spinner frames or prompt text
        assert HIDE_CURSOR not in captured.out
        assert "Allow sending" not in captured.out
        # Spinner was cleaned up: rich shows the cursor again on stop
        stderr_output = fake_stderr.file.getvalue()  # type: ignore[union-attr]
        assert HIDE_CURSOR in stderr_output
        assert "\x1b[?25h" in stderr_output  # show-cursor on stop
