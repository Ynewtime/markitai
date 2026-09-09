"""`markitai mcp` — the CLI form of the bundled MCP server."""

from __future__ import annotations

import sys
from unittest.mock import patch

from click.testing import CliRunner

from markitai.cli import app


def test_mcp_subcommand_runs_the_stdio_server() -> None:
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
