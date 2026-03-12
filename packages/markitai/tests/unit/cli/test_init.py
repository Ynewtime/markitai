from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


class TestInitAtomicWrites:
    """Tests that init command uses atomic writes for file operations."""

    def test_write_config_uses_atomic_write_json(self, tmp_path: Path) -> None:
        """_write_config should use atomic_write_json for crash safety."""
        from markitai.cli.commands.init import _write_config

        config_data = {"output": {"dir": "./output"}}

        with patch("markitai.cli.commands.init.atomic_write_json") as mock_atomic:
            _write_config(tmp_path / "config.json", config_data)
            mock_atomic.assert_called_once()
            call_args = mock_atomic.call_args
            assert call_args[0][0] == tmp_path / "config.json"
            assert call_args[0][1] == config_data

    def test_ensure_env_template_uses_atomic_write_text(self, tmp_path: Path) -> None:
        """_ensure_env_template should use atomic_write_text for crash safety."""
        from markitai.cli.commands.init import _ensure_env_template

        with (
            patch("markitai.cli.commands.init.ConfigManager") as mock_cm,
            patch("markitai.cli.commands.init.atomic_write_text") as mock_atomic,
        ):
            mock_cm.DEFAULT_USER_CONFIG_DIR = tmp_path
            env_path = tmp_path / ".env"
            # File doesn't exist, so it should be created
            _ensure_env_template()
            mock_atomic.assert_called_once()
            call_args = mock_atomic.call_args
            assert call_args[0][0] == env_path

    def test_write_config_produces_valid_json(self, tmp_path: Path) -> None:
        """_write_config should produce valid JSON content."""
        from markitai.cli.commands.init import _write_config

        config_data = {
            "output": {"dir": "./output"},
            "llm": {"enabled": True},
        }
        target = tmp_path / "config.json"
        _write_config(target, config_data)

        # Verify the file exists and contains valid JSON
        assert target.exists()
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == config_data
