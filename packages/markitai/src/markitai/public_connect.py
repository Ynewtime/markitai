"""Connections to a server user's own model addresses reach public IPs only.

A user token's own models name an ``api_base``; the settings routes refuse
an address written as this machine, but a name can resolve (or rebind) to
loopback, the LAN or cloud metadata. Inside :func:`restricted_hosts`, every
httpcore or aiohttp connection to one of those hosts (LiteLLM's calls and
the model discovery and tests) is resolved when it is made, and connects only
to a public address it checked; with none it fails. A configured proxy would
connect on its own, so a restricted host it would carry is refused before
any request. The same rule as the Rust implementation (``public_net``).
"""

from __future__ import annotations

import errno
import ipaddress
import socket
import urllib.request
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

#: The reason a connection was refused, without its address.
REFUSED = "this model's address is not on the public internet"

_restricted: ContextVar[frozenset[str]] = ContextVar(
    "markitai_public_connect_hosts", default=frozenset()
)
_installed = False


def host_of(base: str | None) -> str | None:
    """The host httpcore connects to for *base* (IDNA ASCII, lowercase,
    without brackets or port), or None."""
    if not base or not isinstance(base, str) or base.strip().startswith("env:"):
        return None
    import httpx

    try:
        raw = httpx.URL(base.strip()).raw_host
    except Exception:
        return None
    return raw.decode("ascii").lower() or None


def _name(host: str) -> str:
    """*host* as the restriction compares it: lowercase, without the trailing
    dots of a fully qualified name. httpcore connects to ``model.example.``
    as written, while aiohttp collapses ``model.example..`` to one dot: one
    name with or without them is one host."""
    return host.lower().rstrip(".")


def refused(error: BaseException) -> bool:
    """Whether *error* (or one it was raised from) is this module's refusal."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if REFUSED in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _checked(infos: Iterable[Any]) -> list[str]:
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    kept = [address for address in addresses if _public(address)]
    if not kept:
        raise PermissionError(REFUSED)
    return kept


def _proxied(host: str) -> bool:
    """Whether an environment proxy would carry requests to *host*."""
    proxies = urllib.request.getproxies()
    if not any(proxies.get(scheme) for scheme in ("http", "https", "all")):
        return False
    return not urllib.request.proxy_bypass(host)


@contextmanager
def restricted_hosts(hosts: Iterable[str | None]) -> Iterator[None]:
    """Connections to *hosts* inside this block reach public IPs only."""
    wanted = frozenset(_name(h) for h in hosts if h and _name(h))
    if not wanted:
        yield
        return
    if any(_proxied(host) for host in wanted):
        raise PermissionError(
            f"{REFUSED} (a proxy would carry the request, so it cannot be checked)"
        )
    install()
    token = _restricted.set(_restricted.get() | wanted)
    try:
        yield
    finally:
        _restricted.reset(token)


def install() -> None:
    """Wrap httpcore's connection setup once (idempotent)."""
    global _installed
    if _installed:
        return
    import httpcore
    from httpcore._backends.auto import AutoBackend
    from httpcore._backends.sync import SyncBackend

    original_async = AutoBackend.connect_tcp
    original_sync = SyncBackend.connect_tcp

    async def connect_tcp_async(self: Any, host: str, port: int, **kwargs: Any) -> Any:
        if _name(host) not in _restricted.get():
            return await original_async(self, host, port, **kwargs)
        import asyncio

        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )
            addresses = _checked(infos)
        except (OSError, PermissionError) as error:
            raise httpcore.ConnectError(REFUSED) from error
        last: Exception | None = None
        for address in addresses:
            try:
                return await original_async(self, address, port, **kwargs)
            except httpcore.ConnectError as error:
                last = error
        raise last or httpcore.ConnectError(REFUSED)

    def connect_tcp_sync(self: Any, host: str, port: int, **kwargs: Any) -> Any:
        if _name(host) not in _restricted.get():
            return original_sync(self, host, port, **kwargs)
        try:
            addresses = _checked(
                socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            )
        except (OSError, PermissionError) as error:
            raise httpcore.ConnectError(REFUSED) from error
        last: Exception | None = None
        for address in addresses:
            try:
                return original_sync(self, address, port, **kwargs)
            except httpcore.ConnectError as error:
                last = error
        raise last or httpcore.ConnectError(REFUSED)

    AutoBackend.connect_tcp = connect_tcp_async  # type: ignore[method-assign]
    SyncBackend.connect_tcp = connect_tcp_sync  # type: ignore[method-assign]
    _install_aiohttp()
    _installed = True


def _install_aiohttp() -> None:
    """LiteLLM sends through aiohttp: its connector connects to the
    addresses ``_resolve_host`` returns, so those are filtered there."""
    try:
        from aiohttp.connector import TCPConnector
    except ImportError:  # pragma: no cover - aiohttp is a LiteLLM dependency
        return
    original = TCPConnector._resolve_host

    async def resolve_host(self: Any, host: str, port: int, traces: Any = None) -> Any:
        hosts = await original(self, host, port, traces=traces)
        if _name(host) not in _restricted.get():
            return hosts
        kept = [entry for entry in hosts if _public(str(entry["host"]))]
        if not kept:
            # aiohttp names an OSError by its strerror.
            raise OSError(errno.EACCES, REFUSED)
        return kept

    TCPConnector._resolve_host = resolve_host  # type: ignore[method-assign]
