"""Tests for CLI framework option consistency."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from markitai.cli.framework import MarkitaiGroup
from markitai.cli.main import app

_SYNC_HINT = (
    "MarkitaiGroup._OPTIONS_WITH_VALUES (cli/framework.py) mirrors the value-taking "
    "options of the main command by hand; INPUT detection reads it to know which "
    "token is an option value and which is a path. Out of sync, an option value is "
    "silently swallowed as INPUT."
)


def declared_value_option_names(params: Iterable[click.Parameter]) -> set[str]:
    """Derive every option name that consumes a following token as its value.

    Flags (``--x/--no-x``, ``is_flag``) and counters (``count=True``) carry no
    value, so they are excluded — everything else does.

    Args:
        params: Click parameters to inspect (e.g. ``app.params``).

    Returns:
        The set of option strings (long and short) that take a value.
    """
    names: set[str] = set()
    for param in params:
        if not isinstance(param, click.Option):
            continue
        if param.is_flag or param.count:
            continue
        names.update(param.opts)
        names.update(param.secondary_opts)
    return names


class TestOptionsWithValues:
    """Verify _OPTIONS_WITH_VALUES stays in sync with actual CLI options."""

    def test_options_with_values_mirrors_the_command(self) -> None:
        """The hand-kept mirror must equal the reflected truth — no more, no less."""
        expected = declared_value_option_names(app.params)

        assert expected == MarkitaiGroup._OPTIONS_WITH_VALUES, _SYNC_HINT

    def test_guard_detects_an_unsynced_value_option(self) -> None:
        """Adding a value option without syncing the mirror must be detectable."""
        probe = click.Option(["--probe-value"], type=str, default=None)

        derived = declared_value_option_names([*app.params, probe])

        assert "--probe-value" in derived
        assert derived - MarkitaiGroup._OPTIONS_WITH_VALUES == {"--probe-value"}

    def test_flags_and_counters_are_not_value_options(self) -> None:
        """Flags/counters must not be demanded in the mirror (they take no value)."""
        toggle = click.Option(["--probe-flag/--no-probe-flag"], default=None)
        switch = click.Option(["--probe-switch"], is_flag=True)
        counter = click.Option(["--probe-count", "-C"], count=True)

        assert declared_value_option_names([toggle, switch, counter]) == set()


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
