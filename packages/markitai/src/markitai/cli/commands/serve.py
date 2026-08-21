"""Serve command for Markitai CLI.

Starts the local web UI server (REST + SSE) on top of the conversion core.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import threading
import time
import webbrowser

import rich_click as click

from markitai.cli.console import get_stderr_console

_BROWSER_READY_TIMEOUT_S = 30.0
_BROWSER_POLL_INTERVAL_S = 0.05
_EXPOSED_BIND_HELP = (
    "The default 127.0.0.1 is reachable only from this machine; any other "
    "value publishes the API — which has no authentication — to every host "
    "that can reach it."
)


def _browser_address(host: str) -> tuple[str, str]:
    """Return the connect host and URL host used by the local browser."""
    if host in {"0.0.0.0", ""}:  # nosec B104 - literal comparison to normalize the browser URL, not a bind
        return "127.0.0.1", "127.0.0.1"
    if host == "::":
        return "::1", "[::1]"
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1], host
    if ":" in host:
        return host, f"[{host}]"
    return host, host


def _binds_beyond_loopback(host: str) -> bool:
    """Whether binding to *host* makes the server reachable from other machines.

    Wildcards (``0.0.0.0``, ``::``, an empty host) and every non-loopback
    address or name count as exposed; unknown names are assumed routable
    because they are resolved by the OS, not here.
    """
    candidate = host.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if not candidate:
        return True
    if candidate.lower() == "localhost":
        return False
    try:
        return not ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return True


def _warn_exposed_bind(host: str, port: int) -> None:
    """Tell the user, on stderr, what a non-loopback bind actually opens up."""
    from rich.markup import escape

    console = get_stderr_console()
    console.print(
        f"[yellow]Warning:[/yellow] binding to {escape(host)}:{port} publishes "
        "this server to your network with no authentication."
    )
    console.print(
        "         Anyone who can reach that address can convert files and "
        "read, download or delete your whole conversion history."
    )
    console.print(
        "         Use the default --host 127.0.0.1, or keep it behind an "
        "authenticating reverse proxy."
    )


def _server_is_ready(host: str, port: int) -> bool:
    """Probe a markitai-only endpoint without honoring HTTP proxy settings."""
    connection = http.client.HTTPConnection(host, port, timeout=0.5)
    try:
        connection.request("GET", "/api/capabilities")
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = json.loads(response.read())
        return (
            isinstance(payload, dict) and "version" in payload and "presets" in payload
        )
    except (
        OSError,
        UnicodeDecodeError,
        http.client.HTTPException,
        json.JSONDecodeError,
    ):
        return False
    finally:
        connection.close()


def _open_browser_when_ready(
    url: str,
    host: str,
    port: int,
    stop: threading.Event,
    *,
    timeout: float = _BROWSER_READY_TIMEOUT_S,
    interval: float = _BROWSER_POLL_INTERVAL_S,
) -> None:
    """Open only after Uvicorn has completed application startup."""
    deadline = time.monotonic() + timeout
    while not stop.is_set() and time.monotonic() < deadline:
        if _server_is_ready(host, port):
            if not stop.is_set():
                try:
                    webbrowser.open(url)
                except Exception:
                    pass  # opening the browser is best-effort
            return
        stop.wait(interval)


@click.command("serve")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help=f"Host interface to bind. {_EXPOSED_BIND_HELP}",
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
@click.option(
    "--allowed-host",
    "allowed_hosts",
    multiple=True,
    metavar="HOSTNAME",
    help=(
        "Additional hostname to accept in the Host and Origin headers "
        "(repeatable). localhost and IP addresses are always accepted; "
        "other hostnames are rejected to block DNS rebinding. This is not "
        "authentication: with a non-loopback --host the API stays open to "
        "everyone who can reach it."
    ),
)
def serve(host: str, port: int, no_open: bool, allowed_hosts: tuple[str, ...]) -> None:
    """Run the Markitai web UI server.

    Requires the serve extra (fastapi + uvicorn + python-multipart).

    Examples:
        markitai serve                    # http://127.0.0.1:3600, opens browser
        markitai serve --port 8080        # Custom port
        markitai serve --no-open          # Don't open the browser
        markitai serve --allowed-host my-box.lan   # Accept a DNS name
    """
    from rich.markup import escape

    from markitai.serve import SERVE_INSTALL_HINT, is_serve_available

    if not is_serve_available():
        get_stderr_console().print(f"[red]Error:[/red] {escape(SERVE_INSTALL_HINT)}")
        raise SystemExit(1)

    import uvicorn

    from markitai.serve import create_app

    connect_host, url_host = _browser_address(host)
    url = f"http://{url_host}:{port}"
    if _binds_beyond_loopback(host):
        _warn_exposed_bind(host, port)
    app = create_app(allowed_hosts=allowed_hosts)

    browser_stop = threading.Event()
    browser_thread: threading.Thread | None = None
    if not no_open:
        browser_thread = threading.Thread(
            target=_open_browser_when_ready,
            args=(url, connect_host, port, browser_stop),
            name="markitai-browser",
            daemon=True,
        )
        browser_thread.start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        browser_stop.set()
        if browser_thread is not None:
            browser_thread.join(timeout=1.0)
