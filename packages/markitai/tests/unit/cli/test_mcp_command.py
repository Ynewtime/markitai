"""`markitai mcp` — the CLI form of the bundled MCP server."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli import app


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
    assert "needs the mcp extra" in result.output
    assert '"markitai[mcp]"' in result.output


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
    assert "--config-json is not supported" in result.output
    server.main.assert_not_called()
