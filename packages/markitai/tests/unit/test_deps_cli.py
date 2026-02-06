"""Unit tests for doctor CLI command - dependency checking."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli.commands.doctor import doctor


class TestDoctorDepsCommand:
    """Tests for doctor CLI command - dependency checking."""

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

    def test_check_deps_basic(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test check-deps runs without errors."""
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
            # Support both English and Chinese output (i18n) and unified UI format
            assert (
                "Dependency Status" in result.output
                or "System Check" in result.output
                or "系统检查" in result.output
            )

    def test_check_deps_json_format(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test check-deps --json outputs valid JSON."""
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


class TestPlaywrightDependency:
    """Tests for Playwright dependency checking."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_playwright_available_with_browser(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test Playwright status when fully installed."""
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
            mock_pw.return_value = True
            mock_browser.return_value = True
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["playwright"]["status"] == "ok"
            assert "Chromium" in data["playwright"]["message"]

    def test_playwright_available_no_browser(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test Playwright status when installed but browser missing."""
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
            mock_pw.return_value = True
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["playwright"]["status"] == "warning"
            assert "browser not found" in data["playwright"]["message"]

    def test_playwright_not_installed(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test Playwright status when not installed."""
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
            assert data["playwright"]["status"] == "missing"


class TestLibreOfficeDependency:
    """Tests for LibreOffice dependency checking."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_libreoffice_available(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test LibreOffice status when installed (found in PATH)."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["libreoffice"]["status"] == "ok"
            assert data["libreoffice"]["message"] == "installed"
            assert data["libreoffice"]["path"] == "/usr/bin/soffice"

    def test_libreoffice_not_installed(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test LibreOffice status when not installed."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value=None,
            ),
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["libreoffice"]["status"] == "missing"


class TestFFmpegDependency:
    """Tests for FFmpeg dependency checking."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_ffmpeg_available(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test FFmpeg status when installed."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch("markitai.cli.commands.doctor.subprocess.run") as mock_run,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False

            def which_side_effect(cmd: str) -> str | None:
                if cmd == "ffmpeg":
                    return "/usr/bin/ffmpeg"
                return None

            mock_which.side_effect = which_side_effect

            mock_run.return_value = MagicMock(
                returncode=0, stdout="ffmpeg version 6.0\n"
            )

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["ffmpeg"]["status"] == "ok"


class TestRapidOCRDependency:
    """Tests for RapidOCR dependency checking."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_rapidocr_available_supported_lang(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test RapidOCR status with supported language."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch.dict("sys.modules", {"rapidocr": MagicMock(__version__="1.3.0")}),
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["rapidocr"]["status"] == "ok"

    def test_rapidocr_unsupported_lang(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test RapidOCR status with unsupported language."""
        mock_config.ocr.lang = "unknown_lang"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch.dict("sys.modules", {"rapidocr": MagicMock(__version__="1.3.0")}),
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["rapidocr"]["status"] == "warning"


class TestLLMAPIDependency:
    """Tests for LLM API dependency checking."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_llm_api_configured(self, runner: CliRunner) -> None:
        """Test LLM API status when models are configured."""
        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "openai/gpt-4"
        mock_model.model_info = None
        mock_config.llm.model_list = [mock_model]
        mock_config.ocr = MagicMock()
        mock_config.ocr.lang = "en"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch("markitai.llm.get_model_info_cached") as mock_info,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None
            mock_info.return_value = {"supports_vision": True}

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["llm-api"]["status"] == "ok"
            assert "1 model" in data["llm-api"]["message"]

    def test_llm_api_not_configured(self, runner: CliRunner) -> None:
        """Test LLM API status when no models configured."""
        mock_config = MagicMock()
        mock_config.llm.model_list = []
        mock_config.ocr = MagicMock()
        mock_config.ocr.lang = "en"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["llm-api"]["status"] == "missing"


class TestVisionModelDependency:
    """Tests for vision model dependency checking."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_vision_model_available(self, runner: CliRunner) -> None:
        """Test vision model detection with supported model."""
        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "openai/gpt-4o"
        mock_model.model_info = None
        mock_config.llm.model_list = [mock_model]
        mock_config.ocr = MagicMock()
        mock_config.ocr.lang = "en"

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
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None
            mock_info.return_value = {"supports_vision": True}
            mock_local.return_value = False

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["vision-model"]["status"] == "ok"
            assert "gpt-4o" in data["vision-model"]["message"]

    def test_vision_model_not_available(self, runner: CliRunner) -> None:
        """Test vision model detection with non-vision model."""
        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "openai/gpt-3.5-turbo"
        mock_model.model_info = None
        mock_config.llm.model_list = [mock_model]
        mock_config.ocr = MagicMock()
        mock_config.ocr.lang = "en"

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
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None
            mock_info.return_value = {"supports_vision": False}
            mock_local.return_value = False

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["vision-model"]["status"] == "warning"


class TestLocalProviderSDKs:
    """Tests for local provider SDK checking (Claude Agent, Copilot)."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_claude_agent_sdk_configured(self, runner: CliRunner) -> None:
        """Test Claude Agent SDK detection when configured."""
        from unittest.mock import AsyncMock

        from markitai.providers.auth import AuthStatus

        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "claude-agent/sonnet"
        mock_model.model_info = None
        mock_config.llm.model_list = [mock_model]
        mock_config.ocr = MagicMock()
        mock_config.ocr.lang = "en"

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
            MockConfigManager.return_value.load.return_value = mock_config
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
            assert "claude-agent-sdk" in data
            assert data["claude-agent-sdk"]["status"] == "ok"


class TestOutputFormatting:
    """Tests for output formatting."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_table_output_has_all_columns(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test table output includes all expected columns."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor)

            assert result.exit_code == 0
            # New unified UI doesn't use Component/Status/Description columns
            # Check for section headers instead (i18n support)
            assert (
                "Required" in result.output
                or "必需依赖" in result.output
                or "Playwright" in result.output  # Common dependency name
            )

    def test_install_hints_shown_for_missing(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test install hints are shown for missing dependencies."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor)

            assert result.exit_code == 0
            # Should show hints panel for missing items
            assert "Installation Hints" in result.output or "uv" in result.output

    def test_all_ok_message_when_complete(self, runner: CliRunner) -> None:
        """Test success message when all dependencies are OK."""
        mock_config = MagicMock()
        mock_model = MagicMock()
        mock_model.litellm_params.model = "openai/gpt-4o"
        mock_model.model_info = MagicMock()
        mock_model.model_info.supports_vision = True
        mock_config.llm.model_list = [mock_model]
        mock_config.ocr = MagicMock()
        mock_config.ocr.lang = "en"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
            patch("markitai.cli.commands.doctor.subprocess.run") as mock_run,
            patch.dict("sys.modules", {"rapidocr": MagicMock(__version__="1.3.0")}),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = True
            mock_browser.return_value = True

            def which_side_effect(cmd: str) -> str | None:
                if cmd in ("soffice", "libreoffice", "ffmpeg"):
                    return f"/usr/bin/{cmd}"
                return None

            mock_which.side_effect = which_side_effect
            mock_run.return_value = MagicMock(returncode=0, stdout="version 1.0\n")

            result = runner.invoke(doctor)

            assert result.exit_code == 0
            # Support both English and Chinese success messages (i18n) and new unified UI
            assert (
                "properly configured" in result.output
                or "configured correctly" in result.output
                or "所有依赖配置正确" in result.output
            )


class TestDependencyStatusIcons:
    """Tests for dependency status icons in output."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.llm.model_list = []
        config.ocr = MagicMock()
        config.ocr.lang = "en"
        return config

    def test_json_contains_status_field(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test JSON output contains status field for each dependency."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as MockConfigManager,
            patch("markitai.fetch_playwright.is_playwright_available") as mock_pw,
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed"
            ) as mock_browser,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch("markitai.cli.commands.doctor.shutil.which") as mock_which,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_pw.return_value = False
            mock_browser.return_value = False
            mock_which.return_value = None

            result = runner.invoke(doctor, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)

            # All entries should have status field
            for key, info in data.items():
                assert "status" in info, f"{key} missing status field"
                assert info["status"] in (
                    "ok",
                    "warning",
                    "missing",
                    "error",
                ), f"{key} has invalid status"
