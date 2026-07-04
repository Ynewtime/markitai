"""Tests for CLI framework option consistency."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from markitai.cli.framework import MarkitaiGroup
from markitai.cli.main import app


class TestOptionsWithValues:
    """Verify _OPTIONS_WITH_VALUES stays in sync with actual CLI options."""

    def test_all_value_options_are_registered(self) -> None:
        """Every option that takes a value must appear in _OPTIONS_WITH_VALUES."""
        value_options: set[str] = set()
        for param in app.params:
            if isinstance(param, click.Option) and not param.is_flag:
                for opt in param.opts + param.secondary_opts:
                    value_options.add(opt)

        missing = value_options - MarkitaiGroup._OPTIONS_WITH_VALUES
        assert not missing, (
            f"Options taking values but missing from _OPTIONS_WITH_VALUES: {missing}. "
            f"Add them to MarkitaiGroup._OPTIONS_WITH_VALUES in framework.py."
        )

    def test_no_stale_entries_in_options_with_values(self) -> None:
        """_OPTIONS_WITH_VALUES should not contain entries that don't exist or are flags."""
        value_options: set[str] = set()
        for param in app.params:
            if isinstance(param, click.Option) and not param.is_flag:
                for opt in param.opts + param.secondary_opts:
                    value_options.add(opt)

        extra = MarkitaiGroup._OPTIONS_WITH_VALUES - value_options
        assert not extra, (
            f"Stale entries in _OPTIONS_WITH_VALUES (not value-taking options): {extra}. "
            f"Remove them from MarkitaiGroup._OPTIONS_WITH_VALUES in framework.py."
        )


class TestInputSubcommandAmbiguity:
    """INPUT and a subcommand in the same invocation must fail loudly."""

    def test_input_plus_subcommand_is_usage_error(self, tmp_path: Path) -> None:
        """`markitai note.txt config list` must not silently drop note.txt."""
        note = tmp_path / "note.txt"
        note.write_text("hello")

        runner = CliRunner()
        result = runner.invoke(app, [str(note), "config", "list"])

        assert result.exit_code == 2
        assert "Cannot mix INPUT" in result.output

    def test_input_plus_subcommand_with_options(self, tmp_path: Path) -> None:
        """Ambiguity is detected even with options between the tokens."""
        note = tmp_path / "note.txt"
        note.write_text("hello")

        runner = CliRunner()
        result = runner.invoke(app, [str(note), "--no-llm", "doctor"])

        assert result.exit_code == 2
        assert "Cannot mix INPUT" in result.output

    def test_option_value_matching_command_name_is_not_ambiguous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A value of a value-taking option must not count as a subcommand."""
        note = tmp_path / "note.txt"
        note.write_text("hello")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        # `config` here is the value of -o, not the config subcommand
        result = runner.invoke(app, [str(note), "-o", "config", "--dry-run"])

        assert result.exit_code == 0
        assert "Cannot mix INPUT" not in result.output

    def test_subcommand_name_collision_prints_stderr_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file named like a subcommand: subcommand wins, hint on stderr."""
        (tmp_path / "config").write_text("some content")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "a file named 'config' exists" in result.stderr
        assert "./config" in result.stderr

    def test_no_hint_without_colliding_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No stderr hint when no same-named file exists in cwd."""
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "a file named" not in result.stderr

    def test_path_like_command_name_resolves_as_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`./config` is path-like, so it is INPUT, not the subcommand."""
        (tmp_path / "config").write_text("some content")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(app, ["./config"])

        # Goes down the conversion path (fails on unknown format), and must
        # NOT show the config subcommand help
        assert "Configuration management commands" not in result.output
        assert result.exit_code == 1
        assert "Unsupported file format" in result.output
