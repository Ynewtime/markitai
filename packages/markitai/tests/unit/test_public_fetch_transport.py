"""Network authority and DNS pinning for anonymous remote fetches."""

from unittest.mock import AsyncMock

import httpx
import pytest

from markitai.fetch_http import public_http_request


async def test_public_redirects_pin_dns_and_preserve_tls_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                302, headers={"location": "https://next.example/final"}
            )
        return httpx.Response(200, text="PUBLIC CONTENT")

    monkeypatch.setattr(
        "markitai.fetch_http._public_http_client",
        lambda _proxy, _timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ),
    )
    monkeypatch.setattr(
        "markitai.fetch_policy.resolve_hostname_addresses",
        AsyncMock(return_value=["93.184.216.34"]),
    )
    response = await public_http_request(
        "https://first.example/start", headers={"Authorization": "DO-NOT-FORWARD"}
    )
    assert str(response.url) == "https://next.example/final"
    assert response.text == "PUBLIC CONTENT"
    assert all(r.url.host == "93.184.216.34" for r in requests)
    assert [r.extensions["sni_hostname"] for r in requests] == [
        "first.example",
        "next.example",
    ]
    assert [r.headers["Host"] for r in requests] == ["first.example", "next.example"]
    assert "authorization" not in requests[1].headers


async def test_dns_change_to_private_address_never_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []

    def handle(request: httpx.Request) -> httpx.Response:
        called.append(request)
        return httpx.Response(200)

    monkeypatch.setattr(
        "markitai.fetch_http._public_http_client",
        lambda _proxy, _timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ),
    )
    monkeypatch.setattr(
        "markitai.fetch_policy.resolve_hostname_addresses",
        AsyncMock(side_effect=[["93.184.216.34"], ["127.0.0.1"]]),
    )
    with pytest.raises(PermissionError, match="non-public"):
        await public_http_request("https://rebind.example/start")
    assert not called


async def test_public_transport_falls_back_to_another_checked_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited = []

    def handle(request: httpx.Request) -> httpx.Response:
        visited.append(request.url.host)
        if ":" in request.url.host:
            raise httpx.ConnectError("IPv6 unavailable", request=request)
        return httpx.Response(200, text="PUBLIC CONTENT")

    monkeypatch.setattr(
        "markitai.fetch_http._public_http_client",
        lambda _proxy, _timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ),
    )
    monkeypatch.setattr(
        "markitai.fetch_policy.resolve_hostname_addresses",
        AsyncMock(return_value=["2606:4700::1111", "93.184.216.34"]),
    )
    response = await public_http_request("https://public.example/")
    assert response.text == "PUBLIC CONTENT"
    assert visited == ["2606:4700::1111", "93.184.216.34"]


async def test_redirect_cookies_follow_host_and_head_stays_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                303, headers={"location": "/next", "set-cookie": "session=one; Path=/"}
            )
        if len(requests) == 2:
            return httpx.Response(302, headers={"location": "https://other.example/"})
        return httpx.Response(200)

    monkeypatch.setattr(
        "markitai.fetch_http._public_http_client",
        lambda _proxy, _timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ),
    )
    monkeypatch.setattr(
        "markitai.fetch_policy.resolve_hostname_addresses",
        AsyncMock(return_value=["93.184.216.34"]),
    )
    await public_http_request("https://public.example/", method="HEAD")
    assert [request.method for request in requests] == ["HEAD"] * 3
    assert requests[1].headers["cookie"] == "session=one"
    assert "cookie" not in requests[2].headers


async def test_browser_subrequests_share_public_transport_and_block_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from markitai.fetch_playwright import _guard_public_context

    context = SimpleNamespace(route=AsyncMock(), route_web_socket=AsyncMock())
    transport = AsyncMock(
        return_value=httpx.Response(
            200,
            content=b"PUBLIC BODY",
            request=httpx.Request("GET", "https://public.example/image"),
        )
    )
    monkeypatch.setattr("markitai.fetch_http.public_http_request", transport)
    await _guard_public_context(context, 30000, None)
    pattern, handler = context.route.await_args.args
    assert pattern == "**/*"
    request = SimpleNamespace(
        url="https://public.example/image",
        method="GET",
        all_headers=AsyncMock(return_value={}),
        post_data_buffer=None,
    )
    route = SimpleNamespace(request=request, fulfill=AsyncMock(), abort=AsyncMock())
    await handler(route)
    assert route.fulfill.await_args.kwargs["body"] == b"PUBLIC BODY"
    assert transport.await_args is not None
    # Redirects are followed (and each target checked) by the transport: a
    # redirect handed to the browser would be followed without the route.
    assert transport.await_args.kwargs.get("follow_redirects", True) is True
    transport.side_effect = PermissionError("non-public target")
    await handler(route)
    route.abort.assert_awaited_once_with("blockedbyclient")
    socket = SimpleNamespace(close=AsyncMock())
    await context.route_web_socket.await_args.args[1](socket)
    socket.close.assert_awaited_once()


async def test_origin_only_headers_stay_on_the_origin_through_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(302, headers={"location": "/second"})
        if len(requests) == 2:
            return httpx.Response(
                302, headers={"location": "https://elsewhere.example/third"}
            )
        return httpx.Response(200, text="DONE")

    monkeypatch.setattr(
        "markitai.fetch_http._public_http_client",
        lambda _proxy, _timeout: httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ),
    )
    monkeypatch.setattr(
        "markitai.fetch_policy.resolve_hostname_addresses",
        AsyncMock(return_value=["93.184.216.34"]),
    )
    response = await public_http_request(
        "https://first.example/start",
        headers={"X-Account": "private-token"},
        origin_only=("x-account",),
    )
    assert str(response.request.url) == "https://elsewhere.example/third"
    assert [r.headers.get("x-account") for r in requests] == [
        "private-token",
        "private-token",
        None,
    ]


async def test_a_followed_document_gets_a_base_for_its_relative_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from markitai.fetch_playwright import _guard_public_context

    context = SimpleNamespace(route=AsyncMock(), route_web_socket=AsyncMock())
    transport = AsyncMock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html><head><title>t</title></head><body></body></html>",
            request=httpx.Request("GET", "https://public.example/doc/"),
        )
    )
    monkeypatch.setattr("markitai.fetch_http.public_http_request", transport)
    await _guard_public_context(context, 30000, None)
    _pattern, handler = context.route.await_args.args
    request = SimpleNamespace(
        url="https://public.example/hop",
        method="GET",
        all_headers=AsyncMock(return_value={}),
        post_data_buffer=None,
        is_navigation_request=lambda: False,
    )
    route = SimpleNamespace(request=request, fulfill=AsyncMock(), abort=AsyncMock())
    await handler(route)
    body = route.fulfill.await_args.kwargs["body"]
    assert body.startswith(
        b'<html><head><base href="https://public.example/doc/"><title>'
    )
