"""Tests for CLI main module interactive mode."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from markitai.cli.main import app


class TestInteractiveFlag:
    """Tests for -I/--interactive flag."""

    def test_interactive_short_flag(self) -> None:
        """Should recognize -I flag."""
        runner = CliRunner()
        with patch("markitai.cli.main.run_interactive_mode") as mock_run:
            mock_run.return_value = None
            runner.invoke(app, ["-I"])
            # Should call interactive mode, not show help
            mock_run.assert_called_once()

    def test_interactive_long_flag(self) -> None:
        """Should recognize --interactive flag."""
        runner = CliRunner()
        with patch("markitai.cli.main.run_interactive_mode") as mock_run:
            mock_run.return_value = None
            runner.invoke(app, ["--interactive"])
            mock_run.assert_called_once()

    def test_no_args_shows_help(self) -> None:
        """Should show help when no arguments provided."""
        runner = CliRunner()
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output or "usage:" in result.output.lower()
