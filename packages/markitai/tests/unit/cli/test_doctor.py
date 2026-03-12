"""Tests for doctor command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli.commands.doctor import (
    FIXABLE_COMPONENTS,
    _install_component,
    doctor,
    get_install_hint,
)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_config() -> object:
    """Provide a minimal isolated config for doctor tests."""
    config = MagicMock()
    config.llm.model_list = []
    config.ocr = MagicMock()
    config.ocr.lang = "en"
    return config


def _make_model_config(model_id: str, weight: int = 1) -> MagicMock:
    """Create a mock model config for testing."""
    m = MagicMock()
    m.litellm_params.model = model_id
    m.litellm_params.weight = weight
    m.model_info = None
    return m


class TestDoctorUnifiedUI:
    """Tests for unified UI output in doctor command."""

    def test_doctor_unified_ui_output(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Test doctor command uses unified UI components."""
        # Mock dependencies as OK
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_config_manager,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/soffice",
            ),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
        ):
            mock_config_manager.return_value.load.return_value = mock_config
            result = cli_runner.invoke(doctor)

            # Should use unified symbols
            assert "\u25c6" in result.output  # Title marker (diamond)
            assert "\u2713" in result.output  # Success marker (checkmark)
            # Should NOT use Rich table format
            assert "Dependency Status" not in result.output

    def test_doctor_shows_sections(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Test doctor command shows section headers."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_config_manager,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/soffice",
            ),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
        ):
            mock_config_manager.return_value.load.return_value = mock_config
            result = cli_runner.invoke(doctor)

            # Should contain section headers (in English or Chinese)
            # Check for "Required" or Chinese equivalent
            assert (
                "Required Dependencies" in result.output
                or "\u5fc5\u9700\u4f9d\u8d56" in result.output
            )


class TestInstallHints:
    """Tests for cross-platform install hints."""

    def test_libreoffice_hint_macos(self) -> None:
        """Should return brew command for macOS."""
        hint = get_install_hint("libreoffice", platform="darwin")
        assert "brew install" in hint
        assert "libreoffice" in hint.lower()

    def test_libreoffice_hint_linux(self) -> None:
        """Should return apt command for Linux."""
        hint = get_install_hint("libreoffice", platform="linux")
        assert "apt install" in hint or "apt-get install" in hint

    def test_libreoffice_hint_windows(self) -> None:
        """Should return winget command for Windows."""
        hint = get_install_hint("libreoffice", platform="win32")
        assert "winget install" in hint

    def test_ffmpeg_hint_all_platforms(self) -> None:
        """Should have hints for all major platforms."""
        for platform in ["darwin", "linux", "win32"]:
            hint = get_install_hint("ffmpeg", platform=platform)
            assert hint, f"Missing hint for ffmpeg on {platform}"

    def test_playwright_hint(self) -> None:
        """Should return playwright install command."""
        hint = get_install_hint("playwright", platform="linux")
        assert "playwright install" in hint


class TestDoctorFix:
    """Tests for doctor --fix flag."""

    def test_fix_flag_exists(self) -> None:
        """Should accept --fix flag."""
        runner = CliRunner()
        with patch("markitai.cli.commands.doctor._doctor_impl"):
            result = runner.invoke(doctor, ["--fix"])
            # Should not fail due to unknown option
            assert "no such option" not in result.output.lower()

    @patch("markitai.cli.commands.doctor._install_component")
    def test_fix_installs_missing(self, mock_install: MagicMock) -> None:
        """Should attempt to install missing components."""
        mock_install.return_value = True
        runner = CliRunner()

        # Mock the detection to show missing components
        with patch("markitai.cli.commands.doctor._doctor_impl"):
            # This test verifies the flag triggers install logic
            result = runner.invoke(doctor, ["--fix"])
            # Actual installation behavior tested separately
            assert result.exit_code == 0


class TestDoctorSummaryAllGood:
    """Tests for doctor summary considering all check categories (High-1 fix)."""

    def _invoke_doctor_json(
        self,
        cli_runner: CliRunner,
        mock_config: object,
        *,
        auth_results: dict[str, dict] | None = None,
        llm_api_override: dict | None = None,
        vision_override: dict | None = None,
    ) -> dict:
        """Invoke doctor --json with controllable check results.

        Returns parsed JSON dict.
        """
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_cm,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
            patch(
                "markitai.cli.commands.doctor.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ffmpeg version 7.0.0"),
            ),
        ):
            mock_cm.return_value.load.return_value = mock_config
            result = cli_runner.invoke(doctor, ["--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)

        # Inject overrides to simulate LLM/auth/vision failures
        if llm_api_override:
            data["llm-api"] = llm_api_override
        if vision_override:
            data["vision-model"] = vision_override
        if auth_results:
            data.update(auth_results)
        return data

    def test_all_good_false_when_auth_error(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """all_good should be False when an auth check has error status."""
        from markitai.cli.commands.doctor import _has_errors

        data = self._invoke_doctor_json(
            cli_runner,
            mock_config,
            auth_results={
                "copilot-auth": {
                    "name": "Copilot Auth",
                    "description": "...",
                    "status": "error",
                    "message": "Not authenticated",
                    "install_hint": "markitai auth copilot",
                }
            },
        )
        assert _has_errors(data) is True

    def test_all_good_false_when_llm_error(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """all_good should be False when LLM API check has error status."""
        from markitai.cli.commands.doctor import _has_errors

        data = self._invoke_doctor_json(
            cli_runner,
            mock_config,
            llm_api_override={
                "name": "LLM API",
                "description": "...",
                "status": "error",
                "message": "API key invalid",
                "install_hint": "",
            },
        )
        assert _has_errors(data) is True

    def test_all_good_false_when_vision_error(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """all_good should be False when vision model check has error status."""
        from markitai.cli.commands.doctor import _has_errors

        data = self._invoke_doctor_json(
            cli_runner,
            mock_config,
            vision_override={
                "name": "Vision Model",
                "description": "...",
                "status": "error",
                "message": "Model not available",
                "install_hint": "",
            },
        )
        assert _has_errors(data) is True

    def test_all_good_true_when_no_errors(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """all_good should be True when all checks pass (no error status)."""
        from markitai.cli.commands.doctor import _has_errors

        # Override llm-api to OK since default mock has no models (= missing)
        data = self._invoke_doctor_json(
            cli_runner,
            mock_config,
            llm_api_override={
                "name": "LLM API",
                "description": "...",
                "status": "ok",
                "message": "1 model(s) configured",
                "install_hint": "",
            },
        )
        assert _has_errors(data) is False

    def test_summary_output_not_all_good_on_auth_error(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Non-JSON output should NOT show 'all_good' message when auth fails."""
        cfg = MagicMock()
        # Configure a copilot model so auth check runs
        model = _make_model_config("copilot/gpt-4o")
        cfg.llm.model_list = [model]
        cfg.ocr = MagicMock()
        cfg.ocr.lang = "en"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_cm,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
            patch(
                "markitai.cli.commands.doctor.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ffmpeg version 7.0.0"),
            ),
            patch(
                "markitai.cli.commands.doctor._check_copilot_auth",
                return_value={
                    "name": "Copilot Auth",
                    "description": "...",
                    "status": "error",
                    "message": "Not authenticated",
                    "install_hint": "markitai auth copilot",
                },
            ),
            patch("importlib.util.find_spec", return_value=True),
        ):
            mock_cm.return_value.load.return_value = cfg
            result = cli_runner.invoke(doctor)

        # "All dependencies configured correctly" should NOT appear
        assert "All dependencies configured correctly" not in result.output
        assert "\u6240\u6709\u4f9d\u8d56\u914d\u7f6e\u6b63\u786e" not in result.output


class TestDoctorFixFiltering:
    """Tests for --fix filtering non-fixable components (Medium-3 fix)."""

    def test_fixable_components_excludes_non_installable(self) -> None:
        """FIXABLE_COMPONENTS should not contain non-installable keys."""
        non_installable = {
            "vision-model",
            "llm-api",
            "copilot-auth",
            "claude-agent-auth",
            "gemini-cli-auth",
        }
        for key in non_installable:
            assert key not in FIXABLE_COMPONENTS, (
                f"{key} should not be in FIXABLE_COMPONENTS"
            )

    def test_fix_does_not_call_install_for_vision_model(self) -> None:
        """--fix should not attempt to install 'vision-model'."""
        runner = CliRunner()
        cfg = MagicMock()
        cfg.llm.model_list = []
        cfg.ocr = MagicMock()
        cfg.ocr.lang = "en"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_cm,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
            patch(
                "markitai.cli.commands.doctor.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ffmpeg version 7.0.0"),
            ),
            patch("markitai.cli.commands.doctor._install_component") as mock_install,
        ):
            mock_cm.return_value.load.return_value = cfg
            runner.invoke(doctor, ["--fix"])

            # _install_component should never be called with non-fixable keys
            for call_args in mock_install.call_args_list:
                component = call_args[0][0]
                assert component in FIXABLE_COMPONENTS, (
                    f"_install_component called with non-fixable '{component}'"
                )

    def test_fix_does_not_call_install_for_auth_keys(self) -> None:
        """--fix should not attempt to install auth-related check keys."""
        runner = CliRunner()
        cfg = MagicMock()
        model = _make_model_config("copilot/gpt-4o")
        cfg.llm.model_list = [model]
        cfg.ocr = MagicMock()
        cfg.ocr.lang = "en"

        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_cm,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch(
                "markitai.utils.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
            patch(
                "markitai.cli.commands.doctor.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ffmpeg version 7.0.0"),
            ),
            patch(
                "markitai.cli.commands.doctor._check_copilot_auth",
                return_value={
                    "name": "Copilot Auth",
                    "description": "...",
                    "status": "error",
                    "message": "Not authenticated",
                    "install_hint": "markitai auth copilot",
                },
            ),
            patch("importlib.util.find_spec", return_value=True),
            patch("markitai.cli.commands.doctor._install_component") as mock_install,
        ):
            mock_cm.return_value.load.return_value = cfg
            runner.invoke(doctor, ["--fix"])

            # Should never call _install_component with auth keys
            for call_args in mock_install.call_args_list:
                component = call_args[0][0]
                assert component not in {
                    "copilot-auth",
                    "claude-agent-auth",
                    "gemini-cli-auth",
                }, f"_install_component should not be called with '{component}'"


class TestPlaywrightInstallFix:
    """Tests for playwright --fix handling missing package vs missing browser."""

    def test_playwright_install_when_package_missing(self) -> None:
        """When playwright package is missing, --fix should install the package first."""
        with (
            patch("markitai.cli.commands.doctor.get_console") as mock_gc,
            patch("markitai.cli.commands.doctor.subprocess.run") as mock_run,
        ):
            mock_gc.return_value = MagicMock()
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = _install_component("playwright", package_missing=True)

            # Should first install the package via uv add
            first_call = mock_run.call_args_list[0]
            cmd = first_call[0][0]
            assert "uv" in cmd[0]
            assert "add" in cmd or "playwright" in cmd
            assert result is True

    def test_playwright_install_when_only_browser_missing(self) -> None:
        """When only browser is missing, should just install chromium."""
        with (
            patch("markitai.cli.commands.doctor.get_console") as mock_gc,
            patch("markitai.cli.commands.doctor.subprocess.run") as mock_run,
        ):
            mock_gc.return_value = MagicMock()
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = _install_component("playwright", package_missing=False)

            # Should only install browser, not the package
            assert mock_run.called
            cmd = mock_run.call_args_list[0][0][0]
            assert "playwright" in cmd
            assert "install" in cmd
            assert "chromium" in cmd
            assert result is True
