"""`markitai mcp` — the CLI form of the bundled MCP server."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli import app

# rich-click forces colour under GITHUB_ACTIONS and highlights option names,
# so "--config-json" arrives wrapped in ANSI codes on CI.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Drop ANSI codes and collapse whitespace, so wrapping cannot matter."""
    return " ".join(_ANSI_RE.sub("", text).split())


def test_mcp_subcommand_runs_the_stdio_server() -> None:
    pytest.importorskip("mcp")
    with patch("markitai.mcp.server.main") as main:
        result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 0, result.output
    main.assert_called_once_with()


def test_mcp_subcommand_names_the_extra_when_the_sdk_is_missing() -> None:
    # A None entry in sys.modules makes the import raise ImportError, which is
    # what a missing mcp extra looks like — without touching the real module.
    with patch.dict(sys.modules, {"markitai.mcp.server": None}):
        result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 1
    # rich wraps at the terminal width, so match the pieces, not one line
    assert "needs the mcp extra" in _plain(result.output)
    assert '"markitai[mcp]"' in _plain(result.output)


def test_mcp_is_listed_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert "mcp" in result.output and "MCP server" in result.output


def _fake_server() -> MagicMock:
    """Stand-in for markitai.mcp.server, so these run without the mcp extra."""
    return MagicMock(main=MagicMock())


def test_root_config_path_reaches_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "markitai.json"
    config.write_text("{}")
    monkeypatch.setenv("MARKITAI_CONFIG", "unset")  # restored after the test
    seen: dict[str, str | None] = {}
    server = _fake_server()
    server.main.side_effect = lambda: seen.update(
        config=os.environ.get("MARKITAI_CONFIG")
    )
    with patch.dict(sys.modules, {"markitai.mcp.server": server}):
        result = CliRunner().invoke(app, ["-c", str(config), "mcp"])
    assert result.exit_code == 0, result.output
    assert seen["config"] == str(config)


def test_root_config_json_is_rejected_not_ignored() -> None:
    server = _fake_server()
    with patch.dict(sys.modules, {"markitai.mcp.server": server}):
        result = CliRunner().invoke(
            app, ["--config-json", '{"llm": {"enabled": true}}', "mcp"]
        )
    assert result.exit_code == 2
    assert "--config-json is not supported" in _plain(result.output)
    server.main.assert_not_called()


# The conftest drops FORCE_COLOR for the whole run, so the forced-colour
# environments a user or CI may have are set here explicitly.
@pytest.mark.parametrize(
    "env",
    [{"GITHUB_ACTIONS": "true"}, {"FORCE_COLOR": "1"}, {"TTY_COMPATIBLE": "1"}],
    ids=["github-actions", "force-color", "tty-compatible"],
)
def test_usage_error_stays_on_one_line_off_a_terminal(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """Piped stderr (CI, log files) gets click's one-line error, no panel."""
    monkeypatch.setenv("COLUMNS", "40")  # a panel would wrap the message here
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with patch.dict(sys.modules, {"markitai.mcp.server": _fake_server()}):
        result = CliRunner().invoke(
            app, ["--config-json", '{"llm": {"enabled": true}}', "mcp"]
        )
    assert result.exit_code == 2
    assert "\x1b" not in result.output  # no colour, even when forced
    lines = result.output.splitlines()
    assert lines[0] == "Usage: app mcp [OPTIONS]"
    assert lines[-1] == (
        "Error: --config-json is not supported with 'markitai mcp'; "
        "put the settings in a config file and pass it with -c."
    )
    assert "╭" not in result.output


def test_usage_error_keeps_the_panel_on_a_terminal() -> None:
    """Only the off-terminal path changed: a terminal still gets the box."""
    with (
        patch("markitai.cli.framework.is_terminal", return_value=True),
        patch.dict(sys.modules, {"markitai.mcp.server": _fake_server()}),
    ):
        result = CliRunner().invoke(
            app, ["--config-json", '{"llm": {"enabled": true}}', "mcp"]
        )
    assert result.exit_code == 2
    assert "╭" in result.output
    assert "--config-json is not supported" in _plain(result.output)


def test_a_windows_pipe_keeps_the_rounded_panel() -> None:
    """Rich calls every Windows pipe a legacy console and squares the box."""
    with (
        patch("rich.console.detect_legacy_windows", return_value=True),
        patch("markitai.cli.framework.is_terminal", return_value=True),
        patch.dict(sys.modules, {"markitai.mcp.server": _fake_server()}),
    ):
        result = CliRunner().invoke(
            app, ["--config-json", '{"llm": {"enabled": true}}', "mcp"]
        )
    assert result.exit_code == 2
    assert "╭" in result.output and "┌" not in result.output
