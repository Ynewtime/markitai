"""Tests for the `markitai serve` CLI command and availability probe."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
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

    def test_allowed_host_option_threads_through_to_create_app(
        self, cli_runner: CliRunner
    ) -> None:
        import pytest

        uvicorn = pytest.importorskip("uvicorn")

        from markitai.cli.commands.serve import serve

        sentinel_app = object()
        with (
            patch.object(uvicorn, "run"),
            patch(
                "markitai.serve.create_app", return_value=sentinel_app
            ) as mock_create,
        ):
            result = cli_runner.invoke(
                serve,
                [
                    "--no-open",
                    "--allowed-host",
                    "proxy.lan",
                    "--allowed-host",
                    "box.local",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_create.assert_called_once_with(allowed_hosts=("proxy.lan", "box.local"))

    def test_browser_opens_only_after_server_is_ready(self) -> None:
        import importlib

        serve_mod = importlib.import_module("markitai.cli.commands.serve")
        stop = threading.Event()
        with (
            patch.object(
                serve_mod, "_server_is_ready", side_effect=[False, True]
            ) as probe,
            patch.object(serve_mod.webbrowser, "open") as browser_open,
        ):
            serve_mod._open_browser_when_ready(
                "http://127.0.0.1:3600",
                "127.0.0.1",
                3600,
                stop,
                timeout=1.0,
                interval=0,
            )

        assert probe.call_count == 2
        browser_open.assert_called_once_with("http://127.0.0.1:3600")

    def test_browser_address_handles_wildcard_and_ipv6_hosts(self) -> None:
        from markitai.cli.commands.serve import _browser_address

        assert _browser_address("0.0.0.0") == ("127.0.0.1", "127.0.0.1")
        assert _browser_address("::") == ("::1", "[::1]")
        assert _browser_address("::1") == ("::1", "[::1]")
        assert _browser_address("[::1]") == ("::1", "[::1]")

    def test_registered_as_lazy_subcommand(self) -> None:
        from markitai.cli.commands import _LAZY_MAP
        from markitai.cli.framework import _LAZY_COMMANDS

        assert "serve" in _LAZY_COMMANDS
        assert _LAZY_COMMANDS["serve"][0] == "markitai.cli.commands.serve"
        assert "serve" in _LAZY_MAP


def _squeeze(text: str) -> str:
    """Collapse Rich's line wrapping so assertions can match whole phrases."""
    return " ".join(text.split())


class TestNonLoopbackBindWarning:
    """Binding beyond loopback opens an unauthenticated API to the network.

    The API has no authentication at all (only the settings routes are
    loopback-gated), so ``--host 0.0.0.0`` hands every reachable machine the
    job, history, download and delete endpoints. That has to be said out loud
    at startup — and only then: the default loopback path must stay silent.
    """

    def _invoke(self, cli_runner: CliRunner, args: list[str]):
        import pytest

        uvicorn = pytest.importorskip("uvicorn")

        from markitai.cli.commands.serve import serve

        with (
            patch.object(uvicorn, "run"),
            patch("markitai.serve.create_app", return_value=object()),
        ):
            return cli_runner.invoke(serve, ["--no-open", *args])

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.50", "[::]"])
    def test_exposed_bind_warns_on_stderr(
        self, cli_runner: CliRunner, host: str
    ) -> None:
        result = self._invoke(cli_runner, ["--host", host])
        assert result.exit_code == 0, result.output
        warning = _squeeze(result.stderr)
        assert host in warning
        # Why it matters ...
        assert "no authentication" in warning
        assert "conversion history" in warning
        # ... and what to do instead.
        assert "127.0.0.1" in warning

    @pytest.mark.parametrize(
        "args",
        [[], ["--host", "127.0.0.1"], ["--host", "localhost"], ["--host", "::1"]],
    )
    def test_loopback_bind_stays_silent(
        self, cli_runner: CliRunner, args: list[str]
    ) -> None:
        """The default path must not nag: UX floor for a local-first tool."""
        result = self._invoke(cli_runner, args)
        assert result.exit_code == 0, result.output
        assert "no authentication" not in _squeeze(result.stderr)
        assert result.stderr.strip() == ""

    def test_binds_beyond_loopback_classifier(self) -> None:
        from markitai.cli.commands.serve import _binds_beyond_loopback

        assert _binds_beyond_loopback("0.0.0.0") is True
        assert _binds_beyond_loopback("::") is True
        assert _binds_beyond_loopback("") is True
        assert _binds_beyond_loopback("box.lan") is True
        assert _binds_beyond_loopback("10.0.0.2") is True
        assert _binds_beyond_loopback("127.0.0.1") is False
        assert _binds_beyond_loopback("127.0.0.53") is False
        assert _binds_beyond_loopback("::1") is False
        assert _binds_beyond_loopback("[::1]") is False
        assert _binds_beyond_loopback("LocalHost") is False

    def test_host_help_states_the_exposure(self, cli_runner: CliRunner) -> None:
        from markitai.cli.commands.serve import serve

        help_text = _squeeze(cli_runner.invoke(serve, ["--help"]).output)
        assert "no authentication" in help_text

    def test_allowed_host_help_is_not_only_about_dns_rebinding(
        self, cli_runner: CliRunner
    ) -> None:
        """--allowed-host reads like a security control; say what it is not."""
        from markitai.cli.commands.serve import serve

        help_text = _squeeze(cli_runner.invoke(serve, ["--help"]).output)
        assert "not authentication" in help_text
