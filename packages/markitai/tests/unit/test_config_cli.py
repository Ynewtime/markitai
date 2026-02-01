"""Unit tests for config CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli.commands.config import (
    config_get,
    config_init,
    config_list,
    config_path_cmd,
    config_set,
    config_validate,
)


class TestConfigListCommand:
    """Tests for config list CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock config object."""
        config = MagicMock()
        config.model_dump.return_value = {
            "llm": {"default_model": "gpt-4"},
            "cache": {"enabled": True},
        }
        return config

    def test_list_json_format(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test config list with JSON format (default)."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            MockManager.return_value.load.return_value = mock_config

            result = runner.invoke(config_list)

            assert result.exit_code == 0
            # Output should contain JSON-like content
            assert "llm" in result.output or "gpt-4" in result.output

    def test_list_table_format(self, runner: CliRunner, mock_config: MagicMock) -> None:
        """Test config list with table format."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            MockManager.return_value.load.return_value = mock_config

            result = runner.invoke(config_list, ["--format", "table"])

            assert result.exit_code == 0
            # Table output should have structure
            assert "Configuration" in result.output or "llm" in result.output


class TestConfigPathCommand:
    """Tests for config path CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_path_shows_search_order(self, runner: CliRunner) -> None:
        """Test that path command shows search order."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.config_path = None
            mock_manager.DEFAULT_USER_CONFIG_DIR = Path.home() / ".markitai"
            MockManager.return_value = mock_manager

            result = runner.invoke(config_path_cmd)

            assert result.exit_code == 0
            assert (
                "search order" in result.output.lower()
                or "markitai.json" in result.output
            )

    def test_path_shows_current_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that path command shows currently used config."""
        config_file = tmp_path / "config.json"

        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.config_path = config_file
            mock_manager.DEFAULT_USER_CONFIG_DIR = Path.home() / ".markitai"
            MockManager.return_value = mock_manager

            result = runner.invoke(config_path_cmd)

            assert result.exit_code == 0
            assert (
                str(config_file) in result.output or "Currently using" in result.output
            )


class TestConfigInitCommand:
    """Tests for config init CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_init_creates_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that init creates a config file."""
        output_file = tmp_path / "new_config.json"

        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.save.return_value = output_file
            mock_manager.DEFAULT_USER_CONFIG_DIR = tmp_path
            MockManager.return_value = mock_manager

            result = runner.invoke(config_init, ["--output", str(output_file)])

            assert result.exit_code == 0
            assert (
                "created" in result.output.lower() or str(output_file) in result.output
            )

    def test_init_prompts_overwrite(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that init prompts before overwriting existing file."""
        existing_file = tmp_path / "existing.json"
        existing_file.write_text("{}")

        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.DEFAULT_USER_CONFIG_DIR = tmp_path
            MockManager.return_value = mock_manager

            # User says no to overwrite
            result = runner.invoke(
                config_init, ["--output", str(existing_file)], input="n\n"
            )

            assert result.exit_code == 1  # Aborted


class TestConfigValidateCommand:
    """Tests for config validate CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_validate_valid_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test validating a valid config file."""
        config_file = tmp_path / "valid.json"
        config_file.write_text('{"llm": {}}')

        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.config_path = config_file
            MockManager.return_value = mock_manager

            result = runner.invoke(config_validate, [str(config_file)])

            assert result.exit_code == 0
            assert "valid" in result.output.lower()

    def test_validate_invalid_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test validating an invalid config file."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text('{"invalid": true}')

        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.load.side_effect = ValueError("Invalid config")
            MockManager.return_value = mock_manager

            result = runner.invoke(config_validate, [str(config_file)])

            assert result.exit_code == 2
            assert "error" in result.output.lower()


class TestConfigGetCommand:
    """Tests for config get CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_get_existing_key(self, runner: CliRunner) -> None:
        """Test getting an existing config key."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.get.return_value = "gpt-4"
            MockManager.return_value = mock_manager

            result = runner.invoke(config_get, ["llm.default_model"])

            assert result.exit_code == 0
            assert "gpt-4" in result.output

    def test_get_dict_value(self, runner: CliRunner) -> None:
        """Test getting a dict value (outputs as JSON)."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.get.return_value = {"enabled": True, "max_size": 100}
            MockManager.return_value = mock_manager

            result = runner.invoke(config_get, ["cache"])

            assert result.exit_code == 0
            # Should output as JSON
            assert "enabled" in result.output

    def test_get_nonexistent_key(self, runner: CliRunner) -> None:
        """Test getting a nonexistent key."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.get.return_value = None
            MockManager.return_value = mock_manager

            result = runner.invoke(config_get, ["nonexistent.key"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()


class TestConfigSetCommand:
    """Tests for config set CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_set_string_value(self, runner: CliRunner) -> None:
        """Test setting a string value."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.config = MagicMock()
            mock_manager.config.model_dump.return_value = {}
            MockManager.return_value = mock_manager

            result = runner.invoke(config_set, ["llm.default_model", "gpt-4"])

            assert result.exit_code == 0
            assert "Set" in result.output or "gpt-4" in result.output

    def test_set_boolean_value(self, runner: CliRunner) -> None:
        """Test setting a boolean value."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.config = MagicMock()
            mock_manager.config.model_dump.return_value = {}
            MockManager.return_value = mock_manager

            result = runner.invoke(config_set, ["cache.enabled", "true"])

            assert result.exit_code == 0
            # Should parse "true" as boolean True
            mock_manager.set.assert_called_once()
            call_args = mock_manager.set.call_args
            assert call_args[0][1] is True  # Second arg should be True

    def test_set_integer_value(self, runner: CliRunner) -> None:
        """Test setting an integer value."""
        with patch("markitai.cli.commands.config.ConfigManager") as MockManager:
            mock_manager = MagicMock()
            mock_manager.config = MagicMock()
            mock_manager.config.model_dump.return_value = {}
            MockManager.return_value = mock_manager

            result = runner.invoke(config_set, ["llm.max_tokens", "4096"])

            assert result.exit_code == 0
            mock_manager.set.assert_called_once()
            call_args = mock_manager.set.call_args
            assert call_args[0][1] == 4096  # Should be int, not string
