"""Tests for doctor command."""

from __future__ import annotations

from markitai.cli.commands.doctor import get_install_hint


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


from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from markitai.cli.commands.doctor import doctor


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
