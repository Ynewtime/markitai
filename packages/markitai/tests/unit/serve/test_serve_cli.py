"""Tests for the `markitai serve` CLI command and availability probe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


class TestServeAvailability:
    """is_serve_available() probes optional deps without importing them."""

    def test_available_in_dev_env(self) -> None:
        from markitai.serve import is_serve_available

        assert is_serve_available() is True

    def test_unavailable_when_fastapi_missing(self) -> None:
        import markitai.serve as serve_mod

        def fake_find_spec(name: str):
            return None if name == "fastapi" else MagicMock()

        with patch.object(serve_mod, "find_spec", side_effect=fake_find_spec):
            assert serve_mod.is_serve_available() is False


class TestServeCommand:
    """The serve click command."""

    def test_help_shows_options(self, cli_runner: CliRunner) -> None:
        from markitai.cli.commands.serve import serve

        result = cli_runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--no-open" in result.output

    def test_missing_extra_prints_hint_and_exits_nonzero(
        self, cli_runner: CliRunner
    ) -> None:
        from markitai.cli.commands.serve import serve

        with patch("markitai.serve.is_serve_available", return_value=False):
            result = cli_runner.invoke(serve, ["--no-open"])
        assert result.exit_code != 0
        assert "markitai[serve]" in result.output

    def test_runs_uvicorn_with_host_and_port(self, cli_runner: CliRunner) -> None:
        import pytest

        uvicorn = pytest.importorskip("uvicorn")

        from markitai.cli.commands.serve import serve

        sentinel_app = object()
        with (
            patch.object(uvicorn, "run") as mock_run,
            patch("markitai.serve.create_app", return_value=sentinel_app),
        ):
            result = cli_runner.invoke(
                serve, ["--host", "0.0.0.0", "--port", "3611", "--no-open"]
            )
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] is sentinel_app
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 3611

    def test_registered_as_lazy_subcommand(self) -> None:
        from markitai.cli.commands import _LAZY_MAP
        from markitai.cli.framework import _LAZY_COMMANDS

        assert "serve" in _LAZY_COMMANDS
        assert _LAZY_COMMANDS["serve"][0] == "markitai.cli.commands.serve"
        assert "serve" in _LAZY_MAP
