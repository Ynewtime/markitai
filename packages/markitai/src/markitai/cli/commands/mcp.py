"""`markitai mcp` — run the bundled MCP server over stdio.

The same server as the ``markitai-mcp`` console script. Having it as a
subcommand lets MCP registries and clients start it through the host CLI
package (``uvx --from "markitai[mcp]" markitai mcp``) without knowing a
second executable name.
"""

from __future__ import annotations

import rich_click as click
from rich.markup import escape

from markitai.cli.ui import get_stderr_console


@click.command("mcp")
def mcp() -> None:
    """Run the Markitai MCP server (stdio) for AI agents.

    Serves convert_document, convert_url, batch_convert and job_status over
    stdio; the client (Claude Code, Claude Desktop, any MCP host) speaks the
    protocol. The same server is published as ``markitai-mcp``.

    Examples:
        markitai mcp                              # stdio server for the host
        uvx --from "markitai[mcp]" markitai mcp   # registry-style launch
    """
    try:
        from markitai.mcp.server import main
    except ImportError:
        from markitai.utils.errors import extra_install_command

        get_stderr_console().print(
            "[red]Error:[/red] the MCP server needs the mcp extra. Install it with: "
            f"{escape(extra_install_command('mcp'))}"
        )
        raise SystemExit(1)
    main()
