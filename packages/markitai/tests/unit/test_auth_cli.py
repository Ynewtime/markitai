"""Tests for auth CLI commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli.main import app
from markitai.providers.auth import AuthStatus


class TestAuthCLI:
    """Tests for auth command registration and subcommands."""

    def test_auth_command_lists_all_providers(self) -> None:
        """Main CLI should expose all three provider groups."""
        runner = CliRunner()
        result = runner.invoke(app, ["auth", "--help"])

        assert result.exit_code == 0
        output = result.output.lower()
        assert "copilot" in output
        assert "claude" in output
        assert "chatgpt" in output


class TestCopilotAuthCLI:
    """Tests for markitai auth copilot subcommands."""

    def test_copilot_status_authenticated(self, tmp_path: Path) -> None:
        """auth copilot status shows user when authenticated."""
        runner = CliRunner()

        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"logged_in_users": [{"login": "ghuser"}]})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status"])

        assert result.exit_code == 0
        assert "ghuser" in result.output
        assert "logged in" in result.output.lower()

    def test_copilot_status_not_authenticated(self, tmp_path: Path) -> None:
        """auth copilot status exits 1 when not authenticated."""
        runner = CliRunner()

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status"])

        assert result.exit_code == 1

    def test_copilot_status_current_cli_config_format(self, tmp_path: Path) -> None:
        """auth copilot status parses the JSONC + camelCase config format."""
        runner = CliRunner()

        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            "// This file is managed automatically.\n"
            '{"loggedInUsers": [{"host": "https://github.com", "login": "ghuser"}]}'
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status"])

        assert result.exit_code == 0
        assert "ghuser" in result.output

    def test_copilot_status_unparseable_config_shows_unknown(
        self, tmp_path: Path
    ) -> None:
        """An unreadable config renders as unknown state, not "Not logged in"."""
        runner = CliRunner()

        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("not json at all")

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status"])

        assert result.exit_code == 1
        assert "Login state unknown" in result.output
        assert "Not logged in" not in result.output

    def test_copilot_status_json(self, tmp_path: Path) -> None:
        """auth copilot status --json outputs valid JSON."""
        runner = CliRunner()

        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"logged_in_users": [{"login": "ghuser"}]})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["authenticated"] is True
        assert payload["user"] == "ghuser"

    def test_copilot_login_invokes_attempt_login(self) -> None:
        """auth copilot login calls attempt_login."""
        runner = CliRunner()

        with patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=AuthStatus(
                provider="copilot",
                authenticated=True,
                user="ghuser",
                expires_at=None,
                error=None,
            ),
        ) as mock_login:
            result = runner.invoke(app, ["auth", "copilot", "login"])

        assert result.exit_code == 0
        mock_login.assert_called_once_with("copilot")
        assert "ghuser" in result.output


class TestClaudeAuthCLI:
    """Tests for markitai auth claude subcommands."""

    def test_claude_status_authenticated(self, tmp_path: Path) -> None:
        """auth claude status shows subscription plan when CLI unavailable."""
        runner = CliRunner()

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        expires_at = int((datetime.now().timestamp() + 3600) * 1000)
        (claude_dir / ".credentials.json").write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "tok",
                        "subscriptionType": "max",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("markitai.providers.auth._claude_cli_email", return_value=None),
        ):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 0
        assert "max plan" in result.output

    def test_claude_status_shows_email_from_cli(self, tmp_path: Path) -> None:
        """auth claude status shows email when CLI is available."""
        runner = CliRunner()

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        expires_at = int((datetime.now().timestamp() + 3600) * 1000)
        (claude_dir / ".credentials.json").write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "tok",
                        "subscriptionType": "max",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_email",
                return_value="user@example.com",
            ),
        ):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 0
        assert "user@example.com" in result.output

    def test_claude_status_not_authenticated(self, tmp_path: Path) -> None:
        """auth claude status exits 1 when not authenticated."""
        runner = CliRunner()

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("markitai.providers.auth._claude_cli_auth_status", return_value=None),
        ):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 1

    def test_claude_status_via_cli_keychain_fallback(self, tmp_path: Path) -> None:
        """auth claude status works without a credentials file (macOS Keychain)."""
        runner = CliRunner()

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_auth_status",
                return_value={
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "email": "user@example.com",
                    "subscriptionType": "max",
                },
            ),
        ):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 0
        assert "user@example.com (max plan)" in result.output

    def test_claude_status_shows_subscription_usage_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auth claude status explains how to use the Claude subscription."""
        from markitai.config import ConfigManager

        runner = CliRunner()
        # Isolate config discovery so the hint suggests `markitai init`
        # regardless of any real config on the developer machine.
        monkeypatch.delenv("MARKITAI_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ConfigManager, "DEFAULT_USER_CONFIG_DIR", tmp_path / ".markitai"
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_auth_status",
                return_value={
                    "loggedIn": True,
                    "email": "user@example.com",
                    "subscriptionType": "max",
                },
            ),
        ):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 0
        assert "CLI:" in result.output
        assert "SDK:" in result.output
        assert "claude-agent/" in result.output
        assert "subscription" in result.output.lower()
        assert "markitai init" in result.output

    def test_claude_status_json_includes_cli_and_sdk_info(self, tmp_path: Path) -> None:
        """auth claude status --json includes cli_path and sdk_installed."""
        runner = CliRunner()

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_auth_status",
                return_value={
                    "loggedIn": True,
                    "email": "user@example.com",
                    "subscriptionType": "max",
                },
            ),
        ):
            result = runner.invoke(app, ["auth", "claude", "status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["authenticated"] is True
        assert "cli_path" in payload
        assert "sdk_installed" in payload
        assert payload["details"]["source"] == "cli"

    def test_claude_login_invokes_attempt_login(self) -> None:
        """auth claude login calls attempt_login."""
        runner = CliRunner()

        with patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=AuthStatus(
                provider="claude-agent",
                authenticated=True,
                user="subscription: max",
                expires_at=None,
                error=None,
            ),
        ) as mock_login:
            result = runner.invoke(app, ["auth", "claude", "login"])

        assert result.exit_code == 0
        mock_login.assert_called_once_with("claude-agent")
        assert "max plan" in result.output


class TestChatGPTAuthCLI:
    """Tests for markitai auth chatgpt subcommands."""

    def test_chatgpt_status_authenticated(self, tmp_path: Path) -> None:
        """auth chatgpt status shows authenticated when token exists."""
        runner = CliRunner()

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text(json.dumps({"access_token": "test-token"}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "chatgpt", "status"])

        assert result.exit_code == 0
        assert "authenticated" in result.output.lower()

    def test_chatgpt_status_not_authenticated(self, tmp_path: Path) -> None:
        """auth chatgpt status exits 1 when no token."""
        runner = CliRunner()

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "chatgpt", "status"])

        assert result.exit_code == 1

    def test_chatgpt_status_json(self, tmp_path: Path) -> None:
        """auth chatgpt status --json outputs valid JSON."""
        runner = CliRunner()

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text(json.dumps({"access_token": "test-token"}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "chatgpt", "status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["authenticated"] is True

    def test_chatgpt_login_invokes_attempt_login(self) -> None:
        """auth chatgpt login triggers the device-code flow via attempt_login."""
        runner = CliRunner()

        with patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=AuthStatus(
                provider="chatgpt",
                authenticated=True,
                user="user@example.com",
                expires_at=None,
                error=None,
            ),
        ) as mock_login:
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 0
        mock_login.assert_called_once_with("chatgpt")
        assert "login successful" in result.output.lower()
        assert "user@example.com" in result.output

    def test_chatgpt_login_failure_exits_nonzero(self) -> None:
        """auth chatgpt login exits 1 and shows the error on failure."""
        runner = CliRunner()

        with patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=AuthStatus(
                provider="chatgpt",
                authenticated=False,
                user=None,
                expires_at=None,
                error="Device code expired",
            ),
        ):
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 1
        assert "login failed" in result.output.lower()
        assert "Device code expired" in result.output


class TestAuthOverview:
    """Tests for the bare `markitai auth` all-providers overview."""

    def test_bare_auth_shows_all_providers_overview(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`markitai auth` renders a glyph overview of all four providers."""
        runner = CliRunner()
        for var in (
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        ):
            monkeypatch.delenv(var, raising=False)

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_auth_status",
                return_value=None,
            ),
        ):
            result = runner.invoke(app, ["auth"])

        assert result.exit_code == 0
        for name in ("claude", "chatgpt", "copilot"):
            assert name in result.output
        assert "not logged in" in result.output
        assert "✗" in result.output  # cross glyph for unauthenticated
        assert "markitai auth <provider> login" in result.output


class TestStatusCardUnified:
    """Tests for the unified status card rendering."""

    def test_chatgpt_status_failure_points_at_markitai_login(
        self, tmp_path: Path
    ) -> None:
        """Failure card must reference the markitai login command, not litellm."""
        runner = CliRunner()

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "chatgpt", "status"])

        assert result.exit_code == 1
        assert "markitai auth chatgpt login" in result.output
        assert "pip install litellm" not in result.output
        assert "✗" in result.output

    def test_claude_status_success_card_has_glyphs(self, tmp_path: Path) -> None:
        """Success card shows title and check glyphs."""
        runner = CliRunner()

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_auth_status",
                return_value={
                    "loggedIn": True,
                    "email": "user@example.com",
                    "subscriptionType": "max",
                },
            ),
        ):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 0
        assert "Claude Authentication" in result.output
        assert "✓" in result.output  # checkmark glyph
        assert "Logged in: user@example.com (max plan)" in result.output

    def test_copilot_status_failure_points_at_markitai_login(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Copilot failure card ends with the markitai login command."""
        runner = CliRunner()
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status"])

        assert result.exit_code == 1
        assert "markitai auth copilot login" in result.output


class TestConfigAwareNextHint:
    """Login/status success hints adapt to the existing config state."""

    @pytest.fixture(autouse=True)
    def _isolate_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point config discovery at an isolated, initially-empty location."""
        from markitai.config import ConfigManager

        monkeypatch.delenv("MARKITAI_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ConfigManager, "DEFAULT_USER_CONFIG_DIR", tmp_path / ".markitai"
        )
        self.user_config = tmp_path / ".markitai" / "config.json"

    def _login_chatgpt_ok(self):
        return patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=AuthStatus(
                provider="chatgpt",
                authenticated=True,
                user="user@example.com",
                expires_at=None,
                error=None,
            ),
        )

    def _write_user_config(self, model: str) -> None:
        self.user_config.parent.mkdir(parents=True, exist_ok=True)
        self.user_config.write_text(
            json.dumps(
                {
                    "llm": {
                        "enabled": False,
                        "model_list": [
                            {
                                "model_name": "default",
                                "litellm_params": {"model": model},
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_no_config_suggests_init_autodetect(self) -> None:
        runner = CliRunner()
        with self._login_chatgpt_ok():
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 0
        assert "markitai init" in result.output
        assert "auto-detects and enables" in result.output

    def test_config_without_provider_suggests_init_update(self) -> None:
        self._write_user_config("claude-agent/sonnet")
        runner = CliRunner()
        with self._login_chatgpt_ok():
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 0
        assert "markitai init" in result.output
        assert "adds it to your existing config" in result.output

    def test_config_with_provider_reports_already_enabled(self) -> None:
        self._write_user_config("chatgpt/gpt-5.4")
        runner = CliRunner()
        with self._login_chatgpt_ok():
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 0
        assert "Already enabled in" in result.output
        assert "markitai init" not in result.output


class TestLoginFailureGuidance:
    """Login failure output is a status card with context-aware hints."""

    def _fail_status(self, provider: str, error: str) -> AuthStatus:
        return AuthStatus(
            provider=provider,
            authenticated=False,
            user=None,
            expires_at=None,
            error=error,
        )

    def test_copilot_login_failure_cli_missing_leads_with_install(self) -> None:
        """CLI-missing failure leads with install cmd; never suggests itself."""
        from markitai.providers.auth import _get_cli_install_cmd

        runner = CliRunner()
        with (
            patch(
                "markitai.cli.commands.auth.attempt_login",
                new_callable=AsyncMock,
                return_value=self._fail_status(
                    "copilot", "Copilot CLI not found in PATH"
                ),
            ),
            patch("markitai.providers.auth._resolve_cli_path", return_value=None),
        ):
            result = runner.invoke(app, ["auth", "copilot", "login"])

        assert result.exit_code == 1
        assert "✗" in result.output
        assert "Copilot login failed" in result.output
        # Never suggest the command that just failed
        assert "markitai auth copilot login" not in result.output
        # Install command leads, env-var alternative follows
        install_token = _get_cli_install_cmd("copilot").split()[0]
        assert install_token in result.output
        assert "GH_TOKEN" in result.output
        assert result.output.index("Next:") < result.output.index("GH_TOKEN")

    def test_copilot_login_failure_cli_present_suggests_direct_cli(self) -> None:
        runner = CliRunner()
        with (
            patch(
                "markitai.cli.commands.auth.attempt_login",
                new_callable=AsyncMock,
                return_value=self._fail_status("copilot", "Login failed (exit code 1)"),
            ),
            patch(
                "markitai.providers.auth._resolve_cli_path",
                return_value="/usr/local/bin/copilot",
            ),
        ):
            result = runner.invoke(app, ["auth", "copilot", "login"])

        assert result.exit_code == 1
        assert "markitai auth copilot login" not in result.output
        assert "copilot login" in result.output  # run the CLI directly
        assert "GH_TOKEN" in result.output

    def test_claude_login_failure_cli_missing_leads_with_install(self) -> None:
        from markitai.providers.auth import _get_cli_install_cmd

        runner = CliRunner()
        with (
            patch(
                "markitai.cli.commands.auth.attempt_login",
                new_callable=AsyncMock,
                return_value=self._fail_status(
                    "claude-agent", "Claude CLI not found in PATH"
                ),
            ),
            patch("markitai.providers.auth._resolve_cli_path", return_value=None),
        ):
            result = runner.invoke(app, ["auth", "claude", "login"])

        assert result.exit_code == 1
        assert "markitai auth claude login" not in result.output
        install_token = _get_cli_install_cmd("claude").split()[0]
        assert install_token in result.output
        assert "CLAUDE_CODE_USE_BEDROCK" in result.output

    def test_chatgpt_login_failure_has_retry_hint(self) -> None:
        runner = CliRunner()
        with patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=self._fail_status("chatgpt", "Device code expired"),
        ):
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 1
        assert "markitai auth chatgpt login" not in result.output
