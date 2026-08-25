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
        mock_create.assert_called_once()
        assert mock_create.call_args.kwargs["allowed_hosts"] == (
            "proxy.lan",
            "box.local",
        )

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


def _invoke_serve(
    cli_runner: CliRunner, args: list[str], monkeypatch: pytest.MonkeyPatch
):
    """Invoke serve with uvicorn and create_app mocked, env token cleared."""
    import pytest

    uvicorn = pytest.importorskip("uvicorn")

    from markitai.cli.commands.serve import serve

    monkeypatch.delenv("MARKITAI_SERVE_TOKEN", raising=False)
    with (
        patch.object(uvicorn, "run"),
        patch("markitai.serve.create_app", return_value=object()) as mock_create,
    ):
        result = cli_runner.invoke(serve, ["--no-open", *args])
    return result, mock_create


class TestServeToken:
    """Token generation, env pinning, --no-auth, and the startup URLs."""

    def test_generated_token_reaches_create_app_and_banner(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, mock_create = _invoke_serve(cli_runner, [], monkeypatch)
        assert result.exit_code == 0, result.output
        token = mock_create.call_args.kwargs["token"]
        assert isinstance(token, str) and token.startswith("mk_")
        assert len(token) > 20
        # The banner prints a ready-to-open URL carrying the token.
        assert f"http://127.0.0.1:3600/?token={token}" in _squeeze(result.stderr)

    def test_env_var_pins_the_token(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pytest

        uvicorn = pytest.importorskip("uvicorn")

        from markitai.cli.commands.serve import serve

        monkeypatch.setenv("MARKITAI_SERVE_TOKEN", "pinned-secret")
        with (
            patch.object(uvicorn, "run"),
            patch("markitai.serve.create_app", return_value=object()) as mock_create,
        ):
            result = cli_runner.invoke(serve, ["--no-open"])
        assert result.exit_code == 0, result.output
        assert mock_create.call_args.kwargs["token"] == "pinned-secret"
        assert "?token=pinned-secret" in result.stderr

    def test_no_auth_disables_the_token(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, mock_create = _invoke_serve(cli_runner, ["--no-auth"], monkeypatch)
        assert result.exit_code == 0, result.output
        assert mock_create.call_args.kwargs["token"] is None
        assert "?token=" not in result.stderr

    def test_ready_probe_authenticates_itself(self) -> None:
        """--host <LAN-IP> makes the probe a non-loopback peer of its own
        server; without the bearer header the browser would never open."""
        import importlib
        from unittest.mock import MagicMock

        serve_mod = importlib.import_module("markitai.cli.commands.serve")
        connection = MagicMock()
        response = MagicMock()
        response.status = 200
        response.read.return_value = b'{"version": "1", "presets": []}'
        connection.getresponse.return_value = response
        with patch.object(
            serve_mod.http.client, "HTTPConnection", return_value=connection
        ):
            assert serve_mod._server_is_ready("192.168.1.50", 3600, "mk_t") is True
        connection.request.assert_called_once_with(
            "GET",
            "/api/capabilities",
            headers={"Authorization": "Bearer mk_t"},
        )

    def test_resolve_token_three_states(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from markitai.cli.commands.serve import _resolve_token

        assert _resolve_token(True) is None
        monkeypatch.setenv("MARKITAI_SERVE_TOKEN", "  fixed  ")
        assert _resolve_token(False) == "fixed"
        monkeypatch.delenv("MARKITAI_SERVE_TOKEN")
        generated = _resolve_token(False)
        assert generated is not None and generated.startswith("mk_")
        assert generated != _resolve_token(False)  # fresh secret per call


class TestNonLoopbackBindWarning:
    """Binding beyond loopback publishes the API to the network.

    With token auth (the default) the startup warning must say the token is
    now the gate; with ``--no-auth`` the pre-token "no authentication at all"
    warning must come back. The loopback path prints the token URL but never
    a warning, and ``--no-auth`` on loopback stays fully silent.
    """

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.50", "[::]"])
    def test_exposed_bind_warns_about_the_token_gate(
        self, cli_runner: CliRunner, host: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = _invoke_serve(cli_runner, ["--host", host], monkeypatch)
        assert result.exit_code == 0, result.output
        warning = _squeeze(result.stderr)
        assert host in warning
        assert "access token" in warning
        assert "no authentication" not in warning

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.50"])
    def test_no_auth_exposed_bind_keeps_the_hard_warning(
        self, cli_runner: CliRunner, host: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, _ = _invoke_serve(
            cli_runner, ["--host", host, "--no-auth"], monkeypatch
        )
        assert result.exit_code == 0, result.output
        warning = _squeeze(result.stderr)
        assert host in warning
        # Why it matters ...
        assert "no authentication" in warning
        assert "conversion history" in warning
        # ... and what to do instead.
        assert "--no-auth" in warning
        assert "127.0.0.1" in warning

    @pytest.mark.parametrize(
        "args",
        [[], ["--host", "127.0.0.1"], ["--host", "localhost"], ["--host", "::1"]],
    )
    def test_loopback_bind_prints_token_url_but_no_warning(
        self, cli_runner: CliRunner, args: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same banner everywhere (one mental model), nagging nowhere."""
        result, _ = _invoke_serve(cli_runner, args, monkeypatch)
        assert result.exit_code == 0, result.output
        stderr = _squeeze(result.stderr)
        assert "?token=mk_" in stderr
        assert "Warning" not in stderr

    def test_no_auth_loopback_stays_silent(
        self, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--no-auth restores the pre-token UX floor: nothing on stderr."""
        result, _ = _invoke_serve(cli_runner, ["--no-auth"], monkeypatch)
        assert result.exit_code == 0, result.output
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

    def test_host_help_states_the_exposure_and_the_gate(
        self, cli_runner: CliRunner
    ) -> None:
        from markitai.cli.commands.serve import serve

        help_text = _squeeze(cli_runner.invoke(serve, ["--help"]).output)
        assert "access token" in help_text
        assert "--no-auth" in help_text

    def test_allowed_host_help_is_not_only_about_dns_rebinding(
        self, cli_runner: CliRunner
    ) -> None:
        """--allowed-host reads like a security control; say what it is not."""
        from markitai.cli.commands.serve import serve

        help_text = _squeeze(cli_runner.invoke(serve, ["--help"]).output)
        assert "not authentication" in help_text
