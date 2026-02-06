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


class TestGetLlmNotConfiguredHint:
    """Tests for get_llm_not_configured_hint function."""

    def test_returns_multiline_hint(self) -> None:
        """Test that hint contains expected content."""
        from markitai.cli.hints import get_llm_not_configured_hint

        hint = get_llm_not_configured_hint()
        assert "LLM not configured" in hint
        assert "claude login" in hint
        assert "copilot auth login" in hint
        assert "markitai init" in hint
        assert "markitai doctor" in hint
