"""The tests' local HTTP server binds without a host name lookup."""

from __future__ import annotations

import socket
import threading
import urllib.request
from http.server import SimpleHTTPRequestHandler
from unittest.mock import patch

from tests.local_http import LocalHTTPServer


def test_binds_and_serves_without_getfqdn(tmp_path) -> None:
    """socket.getfqdn took over 25 s on macOS runners (HTTPServer.server_bind)."""
    (tmp_path / "page.html").write_text("<p>served</p>", encoding="utf-8")

    def handler(*args, **kwargs):
        return SimpleHTTPRequestHandler(*args, directory=str(tmp_path), **kwargs)

    with patch.object(socket, "getfqdn", side_effect=AssertionError("getfqdn")):
        server = LocalHTTPServer(("127.0.0.1", 0), handler)
    with server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/page.html") as resp:
                body = resp.read()
        finally:
            server.shutdown()
    assert body == b"<p>served</p>"
    assert server.server_name == "127.0.0.1" and server.server_port == port
