from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from markitai.cli.main import app

# Detected providers used across merge tests; _build_config() maps these to
# claude-agent/sonnet and chatgpt/gpt-5.4-mini model entries.
_DETECTED = [("Claude CLI", True), ("ChatGPT", True)]


def test_limited_preview_models_are_not_selected_by_onboarding() -> None:
    """Automatic setup must use models available to ordinary accounts."""
    from markitai.cli.commands.init import _build_config

    config = _build_config([("ChatGPT", True), ("OpenAI API", True)])
    models = [entry["litellm_params"]["model"] for entry in config["llm"]["model_list"]]

    assert models == ["chatgpt/gpt-5.4-mini", "openai/gpt-5.4-nano"]
    assert all("gpt-5.6" not in model for model in models)


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


class TestMergeNewModels:
    """Tests for the non-destructive config merge helper."""

    def test_appends_missing_models_and_preserves_settings(self) -> None:
        """New providers are appended; all other settings stay untouched."""
        from markitai.cli.commands.init import _build_config, _merge_new_models

        existing = {
            "output": {"dir": "./custom"},
            "llm": {
                "enabled": True,
                "model_list": [
                    {
                        "model_name": "primary",
                        "litellm_params": {"model": "claude-agent/sonnet", "weight": 2},
                    }
                ],
            },
        }

        added = _merge_new_models(existing, _build_config(_DETECTED))

        assert added == ["chatgpt/gpt-5.4-mini"]
        assert existing["output"] == {"dir": "./custom"}
        assert existing["llm"]["enabled"] is True
        # Existing entry is honored (custom name and weight preserved)
        assert existing["llm"]["model_list"][0]["model_name"] == "primary"
        assert existing["llm"]["model_list"][0]["litellm_params"]["weight"] == 2
        models = [e["litellm_params"]["model"] for e in existing["llm"]["model_list"]]
        assert models == ["claude-agent/sonnet", "chatgpt/gpt-5.4-mini"]

    def test_already_up_to_date_returns_empty_and_keeps_config(self) -> None:
        """No additions when every detected provider is already configured."""
        import copy

        from markitai.cli.commands.init import _build_config, _merge_new_models

        existing = {
            "llm": {
                "enabled": False,
                "model_list": [
                    {
                        "model_name": "default",
                        "litellm_params": {"model": "claude-agent/sonnet"},
                    },
                    {
                        "model_name": "default",
                        "litellm_params": {"model": "chatgpt/gpt-5.4-mini"},
                    },
                ],
            }
        }
        snapshot = copy.deepcopy(existing)

        added = _merge_new_models(existing, _build_config(_DETECTED))

        assert added == []
        assert existing == snapshot

    def test_creates_llm_section_when_missing(self) -> None:
        """A config without an llm section gains one (LLM stays opt-in)."""
        from markitai.cli.commands.init import _build_config, _merge_new_models

        existing: dict = {"output": {"dir": "./output"}}

        added = _merge_new_models(existing, _build_config(_DETECTED))

        assert added == ["claude-agent/sonnet", "chatgpt/gpt-5.4-mini"]
        assert existing["llm"]["enabled"] is False


class TestInitYesWithExistingConfig:
    """`init -y` against an existing config applies a non-destructive update."""

    def _invoke(self, target: Path):
        runner = CliRunner()
        with (
            patch(
                "markitai.cli.commands.init._detect_providers",
                return_value=_DETECTED,
            ),
            patch(
                "markitai.cli.commands.init._ensure_env_template",
                return_value=None,
            ),
        ):
            return runner.invoke(app, ["init", "-y", "-o", str(target)])

    def _write_existing(self, target: Path) -> None:
        target.write_text(
            json.dumps(
                {
                    "output": {"dir": "./custom"},
                    "llm": {
                        "enabled": True,
                        "model_list": [
                            {
                                "model_name": "default",
                                "litellm_params": {"model": "claude-agent/sonnet"},
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_yes_appends_new_providers_and_says_so(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        self._write_existing(target)

        result = self._invoke(target)

        assert result.exit_code == 0
        assert "updated" in result.output.lower()
        assert "chatgpt/gpt-5.4-mini" in result.output
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["output"] == {"dir": "./custom"}
        assert data["llm"]["enabled"] is True
        models = [e["litellm_params"]["model"] for e in data["llm"]["model_list"]]
        assert models == ["claude-agent/sonnet", "chatgpt/gpt-5.4-mini"]

    def test_yes_is_idempotent_and_reports_up_to_date(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        self._write_existing(target)

        first = self._invoke(target)
        assert first.exit_code == 0
        content_after_first = target.read_text(encoding="utf-8")

        second = self._invoke(target)
        assert second.exit_code == 0
        assert "up to date" in second.output.lower()
        assert target.read_text(encoding="utf-8") == content_after_first

    def test_yes_unparseable_config_left_untouched(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_text("{not json", encoding="utf-8")

        result = self._invoke(target)

        assert result.exit_code == 0
        assert "could not be parsed" in result.output
        assert target.read_text(encoding="utf-8") == "{not json"


class TestWizardExistingConfig:
    """Interactive wizard offers Update / Overwrite / Keep for existing configs."""

    def _invoke_wizard(self, target: Path, input_text: str):
        runner = CliRunner()
        with (
            patch("markitai.cli.commands.init._check_deps", return_value=[]),
            patch(
                "markitai.cli.commands.init._detect_providers",
                return_value=_DETECTED,
            ),
        ):
            return runner.invoke(app, ["init", "-o", str(target)], input=input_text)

    def _write_existing(self, target: Path) -> None:
        target.write_text(
            json.dumps(
                {
                    "output": {"dir": "./custom"},
                    "llm": {
                        "enabled": True,
                        "model_list": [
                            {
                                "model_name": "default",
                                "litellm_params": {"model": "claude-agent/sonnet"},
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_update_choice_merges(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        self._write_existing(target)

        result = self._invoke_wizard(target, "1\n")

        assert result.exit_code == 0
        assert "Update" in result.output
        assert "Overwrite" in result.output
        assert "Keep" in result.output
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["output"] == {"dir": "./custom"}
        models = [e["litellm_params"]["model"] for e in data["llm"]["model_list"]]
        assert models == ["claude-agent/sonnet", "chatgpt/gpt-5.4-mini"]

    def test_keep_is_default_choice(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        self._write_existing(target)
        original = target.read_text(encoding="utf-8")

        result = self._invoke_wizard(target, "\n")  # accept default (Keep)

        assert result.exit_code == 0
        assert "Kept existing config" in result.output
        assert target.read_text(encoding="utf-8") == original

    def test_keep_leaves_file_unchanged(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        self._write_existing(target)
        original = target.read_text(encoding="utf-8")

        result = self._invoke_wizard(target, "3\n")

        assert result.exit_code == 0
        assert "Kept existing config" in result.output
        assert target.read_text(encoding="utf-8") == original

    def test_overwrite_replaces_with_fresh_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        self._write_existing(target)

        result = self._invoke_wizard(target, "2\n")

        assert result.exit_code == 0
        data = json.loads(target.read_text(encoding="utf-8"))
        # Fresh config: default output dir and llm disabled again
        assert data["output"] == {"dir": "./output"}
        assert data["llm"]["enabled"] is False
        models = [e["litellm_params"]["model"] for e in data["llm"]["model_list"]]
        assert models == ["claude-agent/sonnet", "chatgpt/gpt-5.4-mini"]
