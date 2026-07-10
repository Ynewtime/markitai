"""Tests for doctor command."""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli.commands.doctor import (
    FIXABLE_COMPONENTS,
    _check_playwright,
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

        # The all-green summary should NOT appear.
        assert result.exit_code == 1
        assert "Health check failed" in result.output
        assert "All checks passed" not in result.output
        assert "\u6240\u6709\u68c0\u67e5\u5747\u901a\u8fc7" not in result.output


class TestDoctorFixFiltering:
    """Tests for --fix filtering non-fixable components (Medium-3 fix)."""

    def test_fixable_components_excludes_non_installable(self) -> None:
        """FIXABLE_COMPONENTS should not contain non-installable keys."""
        non_installable = {
            "vision-model",
            "llm-api",
            "copilot-auth",
            "claude-agent-auth",
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
                }, f"_install_component should not be called with '{component}'"


class TestPlaywrightInstallFix:
    """Tests for playwright --fix handling missing package vs missing browser."""

    def test_playwright_install_when_package_missing(self) -> None:
        """A missing package must never be installed into the caller project."""
        with (
            patch("markitai.cli.commands.doctor.get_console") as mock_gc,
            patch("markitai.cli.commands.doctor.subprocess.run") as mock_run,
        ):
            mock_gc.return_value = MagicMock()
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            result = _install_component("playwright", package_missing=True)

            mock_run.assert_not_called()
            assert result is False

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


class TestPlaywrightRuntimeCheck:
    """The normal doctor path verifies launchability, not just browser files."""

    def test_installed_browser_that_cannot_launch_is_not_green(self) -> None:
        with (
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
            patch(
                "markitai.cli.commands.doctor._smoke_test_playwright_browser",
                return_value=(False, "headless launch failed: missing libX11"),
            ),
        ):
            result = _check_playwright()

        assert result["status"] == "warning"
        assert "missing libX11" in result["message"]


class TestDoctorCapabilityContract:
    """Public CLI contract for core health and optional capabilities."""

    @staticmethod
    def _result(
        name: str,
        status: str,
        message: str,
        install_hint: str = "",
    ) -> dict[str, str]:
        return {
            "name": name,
            "description": "test dependency",
            "status": status,
            "message": message,
            "install_hint": install_hint,
        }

    def _invoke(
        self,
        cli_runner: CliRunner,
        mock_config: object,
        *args: str,
        playwright_status: str = "missing",
        playwright_after_fix_status: str | None = None,
        rapidocr_status: str = "ok",
        run_result: MagicMock | None = None,
        run_results: list[MagicMock] | None = None,
    ) -> tuple[object, MagicMock]:
        def playwright_result(status: str) -> dict[str, str]:
            if status == "ok":
                message = "Chromium installed"
            elif status == "warning":
                message = "Playwright installed but browser not found"
            else:
                message = "Playwright not installed"
            return self._result("Playwright", status, message, "install playwright")

        playwright_results = [playwright_result(playwright_status)]
        if playwright_after_fix_status is not None:
            playwright_results.append(playwright_result(playwright_after_fix_status))
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_cm,
            patch("markitai.fetch_playwright.clear_browser_cache"),
            patch(
                "markitai.cli.commands.doctor._check_playwright",
                side_effect=playwright_results,
            ),
            patch(
                "markitai.cli.commands.doctor._check_libreoffice",
                return_value=self._result(
                    "LibreOffice", "missing", "LibreOffice not installed"
                ),
            ),
            patch(
                "markitai.cli.commands.doctor._check_ffmpeg",
                return_value=self._result("FFmpeg", "missing", "FFmpeg not installed"),
            ),
            patch(
                "markitai.cli.commands.doctor._check_rapidocr",
                return_value=self._result(
                    "RapidOCR",
                    rapidocr_status,
                    "RapidOCR installed"
                    if rapidocr_status == "ok"
                    else "RapidOCR not installed",
                    "install RapidOCR",
                ),
            ),
            patch("markitai.cli.commands.doctor.subprocess.run") as mock_run,
        ):
            mock_cm.return_value.load.return_value = mock_config
            mock_cm.return_value.config_path = None
            if run_results is not None:
                mock_run.side_effect = run_results
            elif run_result is not None:
                mock_run.return_value = run_result
            result = cli_runner.invoke(doctor, list(args))
        return result, mock_run

    def test_missing_optional_capabilities_exit_zero(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Unavailable browser, Office, and media support are not core failures."""
        result, _ = self._invoke(cli_runner, mock_config)

        assert result.exit_code == 0, result.output

    def test_missing_optional_capabilities_render_as_warnings(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Unavailable optional tools are visually distinct from core failures."""
        result, _ = self._invoke(cli_runner, mock_config)

        assert "Optional Capabilities" in result.output
        assert "! Playwright: Playwright not installed" in result.output
        assert "✗ Playwright: Playwright not installed" not in result.output

    def test_unconfigured_optional_llm_is_not_a_red_failure(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """No model list is a yellow optional state, consistent with exit 0."""
        result, _ = self._invoke(cli_runner, mock_config, playwright_status="ok")

        assert result.exit_code == 0
        assert "! LLM API: No models configured" in result.output
        assert "✗ LLM API: No models configured" not in result.output

    @pytest.mark.parametrize("configured_by", ["strategy", "screenshot"])
    def test_configured_playwright_capability_is_blocking(
        self,
        cli_runner: CliRunner,
        mock_config: object,
        configured_by: str,
    ) -> None:
        """An enabled browser-dependent workflow must not exit healthy."""
        if configured_by == "strategy":
            mock_config.fetch.strategy = "playwright"  # type: ignore[attr-defined]
            mock_config.screenshot.enabled = False  # type: ignore[attr-defined]
        else:
            mock_config.fetch.strategy = "auto"  # type: ignore[attr-defined]
            mock_config.screenshot.enabled = True  # type: ignore[attr-defined]

        result, _ = self._invoke(cli_runner, mock_config)

        assert result.exit_code == 1
        assert "✗ Playwright: Playwright not installed" in result.output

    def test_missing_env_for_active_api_model_is_blocking(
        self,
        cli_runner: CliRunner,
        mock_config: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Doctor catches an env reference that the router would skip at runtime."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        model = _make_model_config("openai/gpt-4o-mini")
        model.model_info = MagicMock(supports_vision=False)
        model.litellm_params.api_key = "env:OPENAI_API_KEY"
        model.litellm_params.api_base = None
        mock_config.llm.model_list = [model]  # type: ignore[attr-defined]
        mock_config.fetch.strategy = "auto"  # type: ignore[attr-defined]
        mock_config.screenshot.enabled = False  # type: ignore[attr-defined]

        result, _ = self._invoke(
            cli_runner,
            mock_config,
            playwright_status="ok",
        )

        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    def test_nonfatal_rapidocr_language_warning_does_not_fail_core(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """An installed core with an unknown language stays a warning, not red."""
        result, _ = self._invoke(
            cli_runner,
            mock_config,
            playwright_status="ok",
            rapidocr_status="warning",
        )

        assert result.exit_code == 0
        assert "Health check failed" not in result.output

    def test_unavailable_optional_checks_use_warning_summary(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """A healthy core with degraded optional checks must not end in green."""
        result, _ = self._invoke(cli_runner, mock_config)

        summary_lines = [
            line for line in result.output.splitlines() if "Core check passed" in line
        ]
        assert summary_lines, result.output
        assert all("!" in line and "✓" not in line for line in summary_lines)

    def test_missing_rapidocr_exits_nonzero_even_with_fix(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """The core OCR dependency stays red when --fix cannot install it."""
        result, mock_run = self._invoke(
            cli_runner,
            mock_config,
            "--fix",
            playwright_status="ok",
            rapidocr_status="missing",
        )

        assert result.exit_code == 1
        mock_run.assert_not_called()

    def test_fix_installs_browser_with_markitai_python_from_isolated_cwd(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Browser repair cannot resolve tools or mutate metadata from the caller cwd."""
        result, mock_run = self._invoke(
            cli_runner,
            mock_config,
            "--fix",
            playwright_status="warning",
            playwright_after_fix_status="ok",
            run_result=MagicMock(returncode=0, stderr=""),
        )

        assert result.exit_code == 0, result.output
        assert mock_run.call_count == 2
        install_call, smoke_call = mock_run.call_args_list
        assert install_call.args[0] == [
            sys.executable,
            "-m",
            "playwright",
            "install",
            "chromium",
        ]
        assert os.path.isabs(install_call.kwargs["cwd"])
        assert install_call.kwargs["cwd"] != os.getcwd()
        assert smoke_call.args[0][:2] == [sys.executable, "-c"]
        assert "chromium.launch" in smoke_call.args[0][2]
        assert os.path.isabs(smoke_call.kwargs["cwd"])
        assert smoke_call.kwargs["cwd"] != os.getcwd()
        assert result.output.count("Playwright capability verified") == 1
        assert result.output.rfind("Core check passed") > result.output.rfind(
            "Playwright capability verified"
        )

    def test_failed_browser_repair_exits_nonzero(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """An attempted repair that fails must not report a successful exit."""
        result, _ = self._invoke(
            cli_runner,
            mock_config,
            "--fix",
            playwright_status="warning",
            run_result=MagicMock(returncode=1, stderr="download failed"),
        )

        assert result.exit_code == 1

    def test_successful_installer_exit_is_rechecked_before_success(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """A zero installer exit is not enough when Chromium remains unavailable."""
        result, _ = self._invoke(
            cli_runner,
            mock_config,
            "--fix",
            playwright_status="warning",
            playwright_after_fix_status="warning",
            run_result=MagicMock(returncode=0, stderr=""),
        )

        assert result.exit_code == 1
        assert "verification failed" in result.output.lower()

    def test_browser_marker_success_still_requires_headless_launch(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """A browser path is not healthy when Chromium cannot actually launch."""
        result, mock_run = self._invoke(
            cli_runner,
            mock_config,
            "--fix",
            playwright_status="warning",
            playwright_after_fix_status="ok",
            run_results=[
                MagicMock(returncode=0, stdout="", stderr=""),
                MagicMock(returncode=1, stdout="", stderr="missing libX11"),
            ],
        )

        assert result.exit_code == 1
        assert mock_run.call_count == 2
        assert "headless launch failed" in result.output.lower()
        assert "missing libX11" in result.output

    def test_fix_with_missing_playwright_package_is_safe_failure(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Package repair is manual and never modifies the caller's project."""
        result, mock_run = self._invoke(
            cli_runner, mock_config, "--fix", playwright_status="missing"
        )

        assert result.exit_code == 1
        mock_run.assert_not_called()
        assert "uv tool install" in result.output
        assert "pipx install" in result.output

    def test_json_and_fix_are_explicitly_mutually_exclusive(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Machine-readable output must not silently ignore a requested repair."""
        result, mock_run = self._invoke(
            cli_runner,
            mock_config,
            "--json",
            "--fix",
            playwright_status="warning",
        )

        assert result.exit_code == 2
        assert "cannot be used together" in result.output
        mock_run.assert_not_called()


class TestDoctorOutputFormat:
    """Tests for unified doctor output formatting (glyphs, layout, spacing)."""

    def _invoke_doctor_failing(
        self, cli_runner: CliRunner, mock_config: object, *, config_path: object = None
    ):
        """Invoke doctor with RapidOCR missing (required dep failure)."""
        with (
            patch("markitai.cli.commands.doctor.ConfigManager") as mock_cm,
            patch(
                "markitai.cli.commands.doctor.shutil.which",
                return_value="/usr/bin/ffmpeg",
            ),
            patch("markitai.utils.office.find_libreoffice", return_value=None),
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
                "markitai.cli.commands.doctor._check_rapidocr",
                return_value={
                    "name": "RapidOCR",
                    "description": "OCR for scanned documents",
                    "status": "missing",
                    "message": "RapidOCR not installed",
                    "install_hint": "install RapidOCR",
                },
            ),
            patch(
                "markitai.cli.commands.doctor.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="ffmpeg version 7.0.0"),
            ),
        ):
            mock_cm.return_value.load.return_value = mock_config
            mock_cm.return_value.config_path = config_path
            return cli_runner.invoke(doctor)

    def test_failure_summary_uses_cross_glyph(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Failed check summary must use a red cross, never a checkmark."""
        result = self._invoke_doctor_failing(cli_runner, mock_config)

        assert result.exit_code == 1
        summary_lines = [
            line
            for line in result.output.splitlines()
            if "Health check failed" in line or "健康检查未通过" in line
        ]
        assert summary_lines, result.output
        for line in summary_lines:
            assert "✗" in line  # cross
            assert "✓" not in line  # no checkmark on a failure line

    def test_failing_items_render_inline(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Failing dependency items render 'Name: detail' on a single line."""
        result = self._invoke_doctor_failing(cli_runner, mock_config)

        assert "! LibreOffice: soffice/libreoffice command not found" in result.output
        # No continuation-line format in dependency sections
        assert "│ soffice" not in result.output

    def test_single_blank_line_between_sections(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Output never contains two consecutive blank lines."""
        result = self._invoke_doctor_failing(cli_runner, mock_config)

        assert "\n\n\n" not in result.output

    def test_config_source_shown_with_path(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Doctor shows the loaded config file path near the top."""
        from pathlib import Path

        config_path = Path("/home/user/markitai.json")
        result = self._invoke_doctor_failing(
            cli_runner, mock_config, config_path=config_path
        )

        # str(Path) differs across platforms (backslashes on Windows)
        assert str(config_path) in result.output

    def test_config_source_defaults_when_no_file(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Doctor points to 'markitai init' when no config file was loaded."""
        result = self._invoke_doctor_failing(cli_runner, mock_config, config_path=None)

        assert "markitai init" in result.output

    def test_llm_hint_names_loaded_config_path(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """llm.model_list fix hint names the actually-loaded config file."""
        from pathlib import Path

        config_path = Path("/cfg/config.json")
        result = self._invoke_doctor_failing(
            cli_runner, mock_config, config_path=config_path
        )

        flat = " ".join(result.output.split())
        # str(Path) differs across platforms (backslashes on Windows)
        assert f"Configure llm.model_list in {config_path}" in flat
        assert "Configure llm.model_list in markitai.json" not in flat

    def test_llm_hint_suggests_init_when_no_config(
        self, cli_runner: CliRunner, mock_config: object
    ) -> None:
        """Without a loaded config, llm hint points at 'markitai init'."""
        result = self._invoke_doctor_failing(cli_runner, mock_config, config_path=None)

        flat = " ".join(result.output.split())
        assert "Configure llm.model_list (no config file found" in flat


class TestDoctorPerformance:
    """Regression tests for doctor startup cost."""

    def test_doctor_does_not_import_litellm_without_models(self, tmp_path) -> None:
        """With no models configured, doctor must not import litellm (~0.5s)."""
        import subprocess
        import sys

        config_file = tmp_path / "markitai.json"
        config_file.write_text("{}", encoding="utf-8")
        code = (
            "import sys\n"
            "from click.testing import CliRunner\n"
            "from markitai.cli.commands.doctor import doctor\n"
            "CliRunner().invoke(doctor, ['--json'])\n"
            "assert 'litellm' not in sys.modules, 'litellm eagerly imported'\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "MARKITAI_CONFIG": str(config_file)},
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
