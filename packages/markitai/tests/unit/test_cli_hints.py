"""Unit tests for CLI hints module."""

from __future__ import annotations

import sys
from unittest.mock import patch


class TestGetEnvSetCommand:
    """Tests for get_env_set_command function."""

    def test_linux_macos_returns_export(self) -> None:
        """Test that Linux/macOS returns export command."""
        from markitai.cli.hints import get_env_set_command

        with patch.object(sys, "platform", "linux"):
            result = get_env_set_command("API_KEY", "secret")
            assert result == "export API_KEY=secret"

    def test_windows_powershell_returns_env_syntax(self) -> None:
        """Test that Windows PowerShell returns $env: syntax."""
        from markitai.cli.hints import get_env_set_command

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict("os.environ", {"PSModulePath": "C:\\path"}),
        ):
            result = get_env_set_command("API_KEY", "secret")
            assert result == '$env:API_KEY="secret"'

    def test_windows_cmd_returns_set_syntax(self) -> None:
        """Test that Windows CMD returns set syntax."""
        from markitai.cli.hints import get_env_set_command

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = get_env_set_command("API_KEY", "secret")
            assert result == "set API_KEY=secret"
