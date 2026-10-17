"""MCP server exposing markitai conversions as agent tools."""

from __future__ import annotations


def main() -> None:
    """The ``markitai-mcp`` script.

    Without the ``mcp`` extra the server module cannot even be imported; say
    how to install it, as ``markitai mcp`` does, instead of a traceback.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("mcp") is None:
        from markitai.utils.errors import extra_install_command

        print(
            "Error: the MCP server needs the mcp extra. Install it with: "
            f"{extra_install_command('mcp')}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    from markitai.mcp.server import main as run

    run()
