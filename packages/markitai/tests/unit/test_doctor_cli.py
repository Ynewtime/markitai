"""Unit tests for doctor CLI command."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner


class TestDoctorCommand:
    """Tests for doctor CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config with minimal setup."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_doctor_command_exists(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test doctor command exists and runs without errors."""
        from markitai.cli.commands.doctor import doctor

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor)

            assert result.exit_code == 0
            assert "Dependency Status" in result.output

    def test_check_deps_alias_works(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test check-deps alias works and maps to doctor command."""
        from markitai.cli.commands.doctor import check_deps

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(check_deps)

            assert result.exit_code == 0
            assert "Dependency Status" in result.output

    def test_doctor_json_output_valid(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test doctor --json outputs valid JSON."""
        from markitai.cli.commands.doctor import doctor

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            # Should have standard dependency keys
            assert "playwright" in data
            assert "libreoffice" in data
            assert "rapidocr" in data
            assert "llm-api" in data


class TestAuthenticationChecks:
    """Tests for authentication status checking in doctor command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config_with_copilot(self) -> MagicMock:
        """Create a mock config with Copilot model configured."""
        config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "copilot/claude-sonnet-4.5"
        mock_model.model_info = None
        config.llm.model_list = [mock_model]
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    @pytest.fixture
    def mock_config_with_claude_agent(self) -> MagicMock:
        """Create a mock config with Claude Agent model configured."""
        config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "claude-agent/sonnet"
        mock_model.model_info = None
        config.llm.model_list = [mock_model]
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_copilot_auth_check_authenticated(
        self, runner: CliRunner, mock_config_with_copilot: MagicMock
    ) -> None:
        """Test Copilot authentication check when authenticated."""
        from markitai.cli.commands.doctor import doctor
        from markitai.providers.auth import AuthStatus

        mock_auth_status = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="test@example.com",
            expires_at=None,
            error=None,
        )

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch("markitai.llm.get_model_info_cached") as mock_info,
            patch("markitai.providers.is_local_provider_model") as mock_local,
            patch("importlib.util.find_spec") as mock_find_spec,
            patch("markitai.cli.commands.doctor.AuthManager") as MockAuthManager,
        ):
            MockConfigManager.return_value.load.return_value = mock_config_with_copilot
            mock_pw.return_value = False
            mock_browser.return_value = False

            def which_side_effect(cmd: str) -> str | None:
                if cmd == "copilot":
                    return "/usr/bin/copilot"
                return None

            mock_which.side_effect = which_side_effect
            mock_info.return_value = {"supports_vision": True}
            mock_local.return_value = True
            mock_find_spec.return_value = MagicMock()

            mock_manager = MockAuthManager.return_value
            mock_manager.check_auth = AsyncMock(return_value=mock_auth_status)

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "copilot-auth" in data
            assert data["copilot-auth"]["status"] == "ok"

    def test_copilot_auth_check_not_authenticated(
        self, runner: CliRunner, mock_config_with_copilot: MagicMock
    ) -> None:
        """Test Copilot authentication check when not authenticated."""
        from markitai.cli.commands.doctor import doctor
        from markitai.providers.auth import AuthStatus

        mock_auth_status = AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Not authenticated",
        )

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch("markitai.llm.get_model_info_cached") as mock_info,
            patch("markitai.providers.is_local_provider_model") as mock_local,
            patch("importlib.util.find_spec") as mock_find_spec,
            patch("markitai.cli.commands.doctor.AuthManager") as MockAuthManager,
        ):
            MockConfigManager.return_value.load.return_value = mock_config_with_copilot
            mock_pw.return_value = False
            mock_browser.return_value = False

            def which_side_effect(cmd: str) -> str | None:
                if cmd == "copilot":
                    return "/usr/bin/copilot"
                return None

            mock_which.side_effect = which_side_effect
            mock_info.return_value = {"supports_vision": True}
            mock_local.return_value = True
            mock_find_spec.return_value = MagicMock()

            mock_manager = MockAuthManager.return_value
            mock_manager.check_auth = AsyncMock(return_value=mock_auth_status)

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "copilot-auth" in data
            assert data["copilot-auth"]["status"] == "error"

    def test_claude_agent_auth_check_authenticated(
        self, runner: CliRunner, mock_config_with_claude_agent: MagicMock
    ) -> None:
        """Test Claude Agent authentication check when authenticated."""
        from markitai.cli.commands.doctor import doctor
        from markitai.providers.auth import AuthStatus

        mock_auth_status = AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user=None,
            expires_at=None,
            error=None,
        )

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch("markitai.llm.get_model_info_cached") as mock_info,
            patch("markitai.providers.is_local_provider_model") as mock_local,
            patch("importlib.util.find_spec") as mock_find_spec,
            patch("markitai.cli.commands.doctor.AuthManager") as MockAuthManager,
        ):
            MockConfigManager.return_value.load.return_value = (
                mock_config_with_claude_agent
            )
            mock_pw.return_value = False
            mock_browser.return_value = False

            def which_side_effect(cmd: str) -> str | None:
                if cmd == "claude":
                    return "/usr/bin/claude"
                return None

            mock_which.side_effect = which_side_effect
            mock_info.return_value = {"supports_vision": True}
            mock_local.return_value = True
            mock_find_spec.return_value = MagicMock()

            mock_manager = MockAuthManager.return_value
            mock_manager.check_auth = AsyncMock(return_value=mock_auth_status)

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "claude-agent-auth" in data
            assert data["claude-agent-auth"]["status"] == "ok"


class TestDoctorFromMainCLI:
    """Tests for doctor command access from main CLI."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config with minimal setup."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_doctor_registered_in_main_cli(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test doctor command is registered in main CLI."""
        from markitai.cli.main import app

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(app, ["doctor"])

            assert result.exit_code == 0
            assert "Dependency Status" in result.output

    def test_check_deps_alias_in_main_cli(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test check-deps alias works in main CLI."""
        from markitai.cli.main import app

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(app, ["check-deps"])

            assert result.exit_code == 0
            assert "Dependency Status" in result.output
