"""Fetched response bodies are capped like local files, not read unbounded."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from markitai import fetch_http
from markitai.fetch_http import (
    CurlCffiClient,
    HttpxClient,
    ResponseTooLargeError,
    public_http_request,
)

LIMIT = 1024
CHUNK = 512


@pytest.fixture(autouse=True)
def small_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch_http, "MAX_RESPONSE_BYTES", LIMIT)


class _Body:
    """A streamed body that counts how many chunks were pulled from it."""

    def __init__(self, chunks: int) -> None:
        self.chunks = chunks
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.chunks):
            self.sent += 1
            yield b"x" * CHUNK


def _httpx_client(response: httpx.Response) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: response))


async def test_httpx_client_stops_reading_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _Body(chunks=64)
    client = HttpxClient()
    monkeypatch.setattr(
        client,
        "_get_or_create_client",
        lambda _t, _p: _httpx_client(httpx.Response(200, content=body)),
    )
    with pytest.raises(ResponseTooLargeError):
        await client.get("https://big.example/", headers={}, timeout_s=5.0)
    assert body.sent == LIMIT // CHUNK + 1


async def test_httpx_client_rejects_declared_length_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _Body(chunks=64)
    client = HttpxClient()
    monkeypatch.setattr(
        client,
        "_get_or_create_client",
        lambda _t, _p: _httpx_client(
            httpx.Response(200, headers={"Content-Length": "10000000000"}, content=body)
        ),
    )
    with pytest.raises(ResponseTooLargeError):
        await client.get("https://big.example/", headers={}, timeout_s=5.0)
    assert body.sent == 0


async def test_httpx_client_keeps_a_body_at_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = HttpxClient()
    monkeypatch.setattr(
        client,
        "_get_or_create_client",
        lambda _t, _p: _httpx_client(httpx.Response(200, content=_Body(chunks=2))),
    )
    response = await client.get("https://ok.example/", headers={}, timeout_s=5.0)
    assert response.content == b"x" * LIMIT


async def test_not_modified_ignores_the_representation_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 304 may repeat the full entity's Content-Length but has no body."""
    client = HttpxClient()
    monkeypatch.setattr(
        client,
        "_get_or_create_client",
        lambda _t, _p: _httpx_client(
            httpx.Response(304, headers={"Content-Length": "10000000000"})
        ),
    )
    response = await client.get("https://ok.example/", headers={}, timeout_s=5.0)
    assert response.status_code == 304


async def test_public_request_stops_reading_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _Body(chunks=64)
    monkeypatch.setattr(
        "markitai.fetch_http._public_http_client",
        lambda _proxy, _timeout: _httpx_client(httpx.Response(200, content=body)),
    )
    monkeypatch.setattr(
        "markitai.fetch_policy.resolve_hostname_addresses",
        AsyncMock(return_value=["93.184.216.34"]),
    )
    with pytest.raises(ResponseTooLargeError):
        await public_http_request("https://big.example/")
    assert body.sent == LIMIT // CHUNK + 1


async def test_curl_client_stops_reading_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _Body(chunks=64)
    response = SimpleNamespace(
        status_code=200,
        headers={},
        aiter_content=body.__aiter__,
        aclose=AsyncMock(),
    )
    requested: dict[str, Any] = {}

    async def fake_get(url: str, **kwargs: Any) -> Any:
        requested.update(kwargs)
        return response

    client = CurlCffiClient()
    monkeypatch.setattr(
        client, "_get_or_create_session", lambda _p: SimpleNamespace(get=fake_get)
    )
    with pytest.raises(ResponseTooLargeError):
        await client.get("https://big.example/", headers={}, timeout_s=5.0)
    assert requested["stream"] is True
    assert body.sent == LIMIT // CHUNK + 1
    response.aclose.assert_awaited_once()
