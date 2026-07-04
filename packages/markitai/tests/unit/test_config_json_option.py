"""Tests for the --config-json global option (inline config overrides).

Precedence contract: config file < --config-json < explicit CLI flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from markitai.cli.main import app
from markitai.config import ConfigFileError, ConfigManager


def _write_config(tmp_path: Path, data: dict) -> Path:
    config_file = tmp_path / "markitai.json"
    config_file.write_text(json.dumps(data), encoding="utf-8")
    return config_file


class TestConfigManagerOverrides:
    """ConfigManager.load(overrides=...) deep-merge semantics."""

    def test_overrides_win_over_file(self, tmp_path: Path) -> None:
        config_file = _write_config(tmp_path, {"llm": {"enabled": True}})
        cfg = ConfigManager().load(
            config_path=config_file, overrides={"llm": {"enabled": False}}
        )
        assert cfg.llm.enabled is False

    def test_deep_merge_preserves_sibling_file_keys(self, tmp_path: Path) -> None:
        config_file = _write_config(
            tmp_path,
            {"llm": {"enabled": True, "concurrency": 7}, "ocr": {"enabled": True}},
        )
        cfg = ConfigManager().load(
            config_path=config_file, overrides={"llm": {"enabled": False}}
        )
        # Overridden leaf changed; untouched siblings survive the merge.
        assert cfg.llm.enabled is False
        assert cfg.llm.concurrency == 7
        assert cfg.ocr.enabled is True

    def test_overrides_apply_without_a_config_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.json"
        cfg = ConfigManager().load(
            config_path=missing, overrides={"llm": {"enabled": True}}
        )
        assert cfg.llm.enabled is True

    def test_invalid_override_value_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        config_file = _write_config(tmp_path, {})
        with pytest.raises(ConfigFileError) as excinfo:
            ConfigManager().load(
                config_path=config_file,
                overrides={"llm": {"concurrency": "not-a-number"}},
            )
        message = str(excinfo.value)
        assert "llm.concurrency" in message
        assert "--config-json" in message


class TestConfigJsonCliOption:
    """End-to-end precedence and error handling through the CLI."""

    @pytest.fixture
    def note(self, tmp_path: Path) -> Path:
        note = tmp_path / "note.txt"
        note.write_text("hello")
        return note

    def test_json_overrides_file(
        self, note: Path, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        config_file = _write_config(tmp_path, {"llm": {"enabled": True}})
        result = cli_runner.invoke(
            app,
            [
                str(note),
                "-c",
                str(config_file),
                "--config-json",
                '{"llm": {"enabled": false}}',
                "-o",
                str(tmp_path / "out"),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Features: none" in result.output

    def test_json_enables_over_file_default(
        self, note: Path, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        config_file = _write_config(tmp_path, {"llm": {"enabled": False}})
        result = cli_runner.invoke(
            app,
            [
                str(note),
                "-c",
                str(config_file),
                "--config-json",
                '{"llm": {"enabled": true}}',
                "-o",
                str(tmp_path / "out"),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "LLM" in result.output

    def test_explicit_flag_wins_over_json(
        self, note: Path, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        config_file = _write_config(tmp_path, {})
        result = cli_runner.invoke(
            app,
            [
                str(note),
                "-c",
                str(config_file),
                "--config-json",
                '{"llm": {"enabled": true}}',
                "--no-llm",
                "-o",
                str(tmp_path / "out"),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "Features: none" in result.output

    def test_bad_json_reports_position(
        self, note: Path, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        result = cli_runner.invoke(
            app,
            [str(note), "--config-json", '{"llm": {', "--dry-run"],
        )
        assert result.exit_code == 2
        assert "--config-json" in result.output
        assert "invalid JSON" in result.output
        assert "line 1 column" in result.output

    def test_non_object_json_is_rejected(
        self, note: Path, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        result = cli_runner.invoke(
            app,
            [str(note), "--config-json", "[1, 2]", "--dry-run"],
        )
        assert result.exit_code == 2
        assert "--config-json" in result.output
        assert "expected a JSON object" in result.output
        assert "list" in result.output

    def test_invalid_override_value_is_actionable(
        self, note: Path, tmp_path: Path, cli_runner: CliRunner
    ) -> None:
        config_file = _write_config(tmp_path, {})
        result = cli_runner.invoke(
            app,
            [
                str(note),
                "-c",
                str(config_file),
                "--config-json",
                '{"llm": {"concurrency": "lots"}}',
                "--dry-run",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid configuration" in result.output
        assert "llm.concurrency" in result.output
