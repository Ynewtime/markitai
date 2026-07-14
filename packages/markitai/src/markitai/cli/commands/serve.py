"""Serve command for Markitai CLI.

Starts the local web UI server (REST + SSE) on top of the conversion core.
"""

from __future__ import annotations

import rich_click as click

from markitai.cli.console import get_stderr_console


@click.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host interface to bind.",
)
@click.option(
    "--port",
    default=3600,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open the browser after startup.",
)
def serve(host: str, port: int, no_open: bool) -> None:
    """Run the Markitai web UI server.

    Requires the serve extra (fastapi + uvicorn + python-multipart).

    Examples:
        markitai serve                    # http://127.0.0.1:3600, opens browser
        markitai serve --port 8080        # Custom port
        markitai serve --no-open          # Don't open the browser
    """
    from rich.markup import escape

    from markitai.serve import SERVE_INSTALL_HINT, is_serve_available

    if not is_serve_available():
        get_stderr_console().print(f"[red]Error:[/red] {escape(SERVE_INSTALL_HINT)}")
        raise SystemExit(1)

    import threading
    import webbrowser

    import uvicorn

    from markitai.serve import create_app

    if not no_open:
        # This only maps a wildcard bind to a browser-safe URL; it does not bind.
        browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host  # nosec B104
        url = f"http://{browser_host}:{port}"

        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass  # opening the browser is best-effort

        threading.Timer(1.0, _open_browser).start()

    uvicorn.run(create_app(), host=host, port=port, log_level="info")
