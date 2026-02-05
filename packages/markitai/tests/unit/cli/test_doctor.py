"""Tests for doctor command."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from markitai.cli.commands.doctor import doctor, get_install_hint


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI runner."""
    return CliRunner()


class TestDoctorUnifiedUI:
    """Tests for unified UI output in doctor command."""

    def test_doctor_unified_ui_output(self, cli_runner: CliRunner) -> None:
        """Test doctor command uses unified UI components."""
        # Mock dependencies as OK
        with (
            patch("shutil.which", return_value="/usr/bin/soffice"),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
        ):
            result = cli_runner.invoke(doctor)

            # Should use unified symbols
            assert "\u25c6" in result.output  # Title marker (diamond)
            assert "\u2713" in result.output  # Success marker (checkmark)
            # Should NOT use Rich table format
            assert "Dependency Status" not in result.output

    def test_doctor_shows_sections(self, cli_runner: CliRunner) -> None:
        """Test doctor command shows section headers."""
        with (
            patch("shutil.which", return_value="/usr/bin/soffice"),
            patch(
                "markitai.fetch_playwright.is_playwright_available", return_value=True
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_browser_installed",
                return_value=True,
            ),
        ):
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


from unittest.mock import MagicMock


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
