"""A local HTTP server for tests, without http.server's host name lookup."""

from __future__ import annotations

import socketserver
from http.server import ThreadingHTTPServer


class LocalHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that binds without ``socket.getfqdn``.

    ``HTTPServer.server_bind`` looks up the bound address's host name to fill
    ``server_name``; on macOS CI runners that reverse lookup of 127.0.0.1
    took over 25 seconds (run 38106435439). Nothing here serves CGI, the only
    reader of ``server_name``, so it is just the address.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)
