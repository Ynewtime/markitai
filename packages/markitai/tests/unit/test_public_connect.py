"""A server user's own model addresses are reached at public IPs only."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import httpx
import pytest

from markitai.public_connect import REFUSED, host_of, restricted_hosts


@pytest.fixture
def loopback() -> Iterator[tuple[int, list[bytes]]]:
    """A loopback HTTP server that records every request it accepts."""
    seen: list[bytes] = []
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            with conn:
                seen.append(conn.recv(4096))
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
                )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield server.getsockname()[1], seen
    stop.set()
    thread.join(timeout=2)
    server.close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


def test_host_of_matches_what_httpcore_connects_to() -> None:
    assert host_of("http://LocalHost:11434/v1") == "localhost"
    assert host_of("http://[::1]:8080/v1") == "::1"
    assert host_of("https://bücher.example/v1") == "xn--bcher-kva.example"
    assert host_of("env:SOME_BASE") is None
    assert host_of(None) is None


def test_a_restricted_host_on_this_machine_is_never_reached(
    loopback: tuple[int, list[bytes]],
) -> None:
    port, seen = loopback
    for host in ("localhost", "127.0.0.1"):
        url = f"http://{host}:{port}/v1/models"
        with (
            restricted_hosts([host_of(url)]),
            pytest.raises(httpx.ConnectError, match=REFUSED),
        ):
            httpx.get(url, timeout=5)
    assert seen == []
    # Outside the block, and for hosts not named, nothing changes.
    assert httpx.get(f"http://localhost:{port}/v1/models", timeout=5).text == "ok"
    with restricted_hosts(["model.example"]):
        assert httpx.get(f"http://127.0.0.1:{port}/", timeout=5).text == "ok"
    assert len(seen) == 2


async def test_the_async_client_is_held_to_the_same_rule(
    loopback: tuple[int, list[bytes]],
) -> None:
    port, seen = loopback
    url = f"http://localhost:{port}/v1/chat/completions"
    with restricted_hosts([host_of(url)]):
        async with httpx.AsyncClient(timeout=5) as client:
            with pytest.raises(httpx.ConnectError, match=REFUSED):
                await client.post(url, json={})
    assert seen == []


def test_a_proxy_that_would_carry_the_request_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Start from no proxy settings at all: the shell running the tests may set
    # its own (lower-case variants win on some platforms).
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    with (
        pytest.raises(PermissionError, match="proxy"),
        restricted_hosts(["model.example"]),
    ):
        pass
    monkeypatch.setenv("NO_PROXY", "model.example")
    with restricted_hosts(["model.example"]):
        pass


async def test_aiohttp_which_litellm_sends_through_is_held_too(
    loopback: tuple[int, list[bytes]],
) -> None:
    import aiohttp

    port, seen = loopback
    url = f"http://localhost:{port}/v1/chat/completions"
    with restricted_hosts([host_of(url)]):
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.ClientConnectorError, match=REFUSED):
                await session.post(url, json={})
    assert seen == []


async def test_a_trailing_dot_names_the_same_restricted_host(
    loopback: tuple[int, list[bytes]],
) -> None:
    """``model.example.`` is ``model.example``: aiohttp resolves it without
    the dot the restriction was given, which once let it through."""
    import aiohttp
    from aiohttp.abc import AbstractResolver

    class ToThisMachine(AbstractResolver):
        """A name that resolves (or rebinds) to loopback."""

        async def resolve(
            self, host: str, port: int = 0, family: int = socket.AF_INET
        ) -> list[dict[str, object]]:
            return [
                {
                    "hostname": host,
                    "host": "127.0.0.1",
                    "port": port,
                    "family": socket.AF_INET,
                    "proto": 0,
                    "flags": 0,
                }
            ]

        async def close(self) -> None:
            pass

    port, seen = loopback
    for base, named in (
        (f"http://rebind.example.:{port}/v1", None),
        (f"http://rebind.example..:{port}/v1", None),
        (f"http://rebind.example:{port}/v1", f"http://Rebind.Example.:{port}/v1"),
    ):
        with restricted_hosts([host_of(named or base)]):
            connector = aiohttp.TCPConnector(resolver=ToThisMachine())
            async with aiohttp.ClientSession(connector=connector) as session:
                with pytest.raises(aiohttp.ClientConnectorError, match=REFUSED):
                    await session.get(base)
    assert seen == []


def test_a_trailing_dot_is_held_by_httpcore_too(
    loopback: tuple[int, list[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    port, seen = loopback
    real = socket.getaddrinfo

    def to_this_machine(host: str, *args: object, **kwargs: object) -> object:
        if host.rstrip(".") == "rebind.example":
            host = "127.0.0.1"
        return real(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "getaddrinfo", to_this_machine)
    for base, named in (
        (f"http://rebind.example.:{port}/v1", f"http://rebind.example:{port}/v1"),
        (f"http://rebind.example:{port}/v1", f"http://rebind.example.:{port}/v1"),
    ):
        with (
            restricted_hosts([host_of(named)]),
            pytest.raises(httpx.ConnectError, match=REFUSED),
        ):
            httpx.get(base, timeout=5)
    assert seen == []
