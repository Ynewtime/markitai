"""Tests for auth CLI commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from markitai.cli.main import app
from markitai.providers.auth import AuthStatus


class TestAuthCLI:
    """Tests for auth command registration and subcommands."""

    def test_auth_command_lists_all_providers(self) -> None:
        """Main CLI should expose all four provider groups."""
        runner = CliRunner()
        result = runner.invoke(app, ["auth", "--help"])

        assert result.exit_code == 0
        output = result.output.lower()
        assert "gemini" in output
        assert "copilot" in output
        assert "claude" in output
        assert "chatgpt" in output


class TestGeminiAuthCLI:
    """Tests for markitai auth gemini subcommands."""

    def test_gemini_status_reports_managed_profile_json(self, tmp_path: Path) -> None:
        """Status should expose the active managed Gemini profile."""
        runner = CliRunner()
        auth_dir = tmp_path / ".markitai" / "auth"
        auth_dir.mkdir(parents=True)
        profile_path = auth_dir / "gemini-profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "access_token": "managed-token",
                    "refresh_token": "managed-refresh",
                    "email": "gemini@example.com",
                    "project_id": "demo-project",
                    "auth_mode": "google-one",
                    "source": "markitai",
                }
            ),
            encoding="utf-8",
        )
        (auth_dir / "gemini-current.json").write_text(
            json.dumps({"credential_path": str(profile_path)}),
            encoding="utf-8",
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "gemini", "status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["authenticated"] is True
        assert payload["user"] == "gemini@example.com"
        assert payload["details"]["project_id"] == "demo-project"
        assert payload["details"]["source"] == "markitai"

    def test_gemini_status_shared_creds_display(self, tmp_path: Path) -> None:
        """Shared credentials show clean one-liner without Source/Profile noise."""
        runner = CliRunner()
        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "oauth_creds.json").write_text(
            json.dumps({"access_token": "ya29.xxx"})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "gemini", "status"])

        assert result.exit_code == 0
        # Shared creds should be clean one-liner, no Source/Profile lines
        assert "Source:" not in result.output
        assert "Profile:" not in result.output

    def test_gemini_login_invokes_provider_login(self, tmp_path: Path) -> None:
        """Login command should call Gemini provider login and print the result."""
        runner = CliRunner()

        with patch("pathlib.Path.home", return_value=tmp_path):
            from markitai.providers.gemini_cli import GeminiCredentialRecord

            record = GeminiCredentialRecord(
                path=tmp_path / ".markitai" / "auth" / "gemini-profile.json",
                source="markitai",
                email="gemini@example.com",
                project_id="demo-project",
                auth_mode="google-one",
            )

            with patch(
                "markitai.providers.gemini_cli.GeminiCLIProvider"
            ) as MockProvider:
                MockProvider.return_value.alogin = AsyncMock(return_value=record)

                result = runner.invoke(
                    app,
                    ["auth", "gemini", "login", "--mode", "google-one"],
                )

        assert result.exit_code == 0
        assert "gemini@example.com" in result.output
        assert "demo-project" in result.output


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
        assert "authenticated" in result.output.lower()

    def test_copilot_status_not_authenticated(self, tmp_path: Path) -> None:
        """auth copilot status exits 1 when not authenticated."""
        runner = CliRunner()

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "copilot", "status"])

        assert result.exit_code == 1

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

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = runner.invoke(app, ["auth", "claude", "status"])

        assert result.exit_code == 1

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

    def test_chatgpt_login_shows_auto_login_message(self) -> None:
        """auth chatgpt login shows auto-login info."""
        runner = CliRunner()

        with patch(
            "markitai.cli.commands.auth.attempt_login",
            new_callable=AsyncMock,
            return_value=AuthStatus(
                provider="chatgpt",
                authenticated=False,
                user=None,
                expires_at=None,
                error=None,
                details={"auto_login": True},
            ),
        ):
            result = runner.invoke(app, ["auth", "chatgpt", "login"])

        assert result.exit_code == 0
        assert "device code flow" in result.output.lower()
