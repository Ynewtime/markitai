"""Tests for the serve access-token authentication.

The trust rule under test: a loopback peer is always trusted (token or not),
a non-loopback peer must present the serve token — in the ``Authorization:
Bearer`` header or the ``?token=`` query parameter — and is then trusted like
a loopback peer (private URL targets, history deletion and LLM settings all
work). ``create_app(token=None)`` (the ``--no-auth`` CLI path and the default
for embedders) keeps the pre-token behavior, covered further by
test_serve_url_policy.py.

Harness mirrors test_serve_url_policy.py: ``create_app`` over
httpx.ASGITransport, whose ``client=`` tuple is the TCP peer the app sees.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")

import httpx

from markitai.config import MarkitaiConfig
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

TOKEN = "mk_unit-test-token"
AUTH_HEADER = {"Authorization": f"Bearer {TOKEN}"}
LAN_PEER = ("192.168.1.23", 51234)
PRIVATE_URL = "http://192.168.1.4/admin"


def _make_app(tmp_path: Path, token: str | None) -> FastAPI:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
        token=token,
    )


@asynccontextmanager
async def _serve_client(
    app: FastAPI, *, peer: tuple[str, int] | None = None
) -> AsyncIterator[httpx.AsyncClient]:
    """Run the lifespan and yield a client whose TCP peer is *peer*."""
    async with app.router.lifespan_context(app):
        transport = (
            httpx.ASGITransport(app=app, client=peer)
            if peer is not None
            else httpx.ASGITransport(app=app)
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


def _file_job(name: str = "doc.txt") -> list[tuple[str, Any]]:
    return [
        ("files", (name, b"hello", "application/octet-stream")),
        ("urls", (None, "[]")),
        ("options", (None, "{}")),
    ]


def _url_job(urls: list[str]) -> list[tuple[str, Any]]:
    return [
        ("urls", (None, json.dumps(urls))),
        ("options", (None, "{}")),
    ]


def _fake_file_converter():
    """Stub converter writing '<name>.md' into the job out dir."""

    async def fake_process_file_item(
        file_path: Path, cfg: Any, out_dir: Path, shared: Any
    ):
        from markitai.batch import ProcessResult

        out = out_dir / f"{file_path.name}.md"
        out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
        return ProcessResult(success=True, output_path=str(out))

    return fake_process_file_item


def _fake_url_converter():
    """Stub URL processing so no job ever touches the network."""

    async def fake_process_url_item(
        url: str,
        cfg: Any,
        out_dir: Path,
        shared: Any,
        url_ctx: Any,
        output_name: str | None = None,
    ):
        from markitai.batch import ProcessResult

        out = out_dir / (output_name or "page.md")
        out.write_text(f"# {url}\n", encoding="utf-8")
        return ProcessResult(success=True, output_path=str(out))

    return fake_process_url_item


async def _drain(client: httpx.AsyncClient, job_id: str, **kwargs: Any) -> dict:
    """Wait for a job to finish so lifespan shutdown cannot cancel it."""
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}", **kwargs)).json()
        if data["status"] != "running":
            return data
        await asyncio.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish")


class TestLoopbackPeer:
    """The operator's machine never needs the token."""

    async def test_no_token_passes(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, TOKEN)) as client:
            capabilities = await client.get("/api/capabilities")
            history = await client.get("/api/history")
        assert capabilities.status_code == 200
        assert history.status_code == 200

    async def test_wrong_token_still_passes(self, tmp_path: Path) -> None:
        """Peer trust wins: a stale sessionStorage token must not lock out
        the local browser, and local CLI tooling stays credential-free."""
        async with _serve_client(_make_app(tmp_path, TOKEN)) as client:
            resp = await client.get(
                "/api/capabilities", headers={"Authorization": "Bearer mk_wrong"}
            )
        assert resp.status_code == 200


class TestRemotePeerWithoutToken:
    """Every /api request from another machine is 401 without the token."""

    async def test_plain_api_rejected(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            resp = await client.get("/api/history")
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        assert "token" in resp.json()["detail"]

    async def test_sse_rejected(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            resp = await client.get("/api/jobs/some-job/events")
        assert resp.status_code == 401

    async def test_download_rejected(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            resp = await client.get("/api/jobs/some-job/files/out.md")
        assert resp.status_code == 401

    async def test_public_url_job_rejected(self, tmp_path: Path) -> None:
        """The 0.24 'public URLs only' half-open surface is gone under auth."""
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            resp = await client.post(
                "/api/jobs", files=_url_job(["https://example.com/page"])
            )
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "headers,params",
        [
            ({"Authorization": "Bearer mk_wrong"}, {}),
            ({"Authorization": TOKEN}, {}),  # credential without a scheme
            ({"Authorization": f"Basic {TOKEN}"}, {}),
            ({}, {"token": "mk_wrong"}),
        ],
    )
    async def test_bad_credentials_rejected(
        self, tmp_path: Path, headers: dict[str, str], params: dict[str, str]
    ) -> None:
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            resp = await client.get("/api/history", headers=headers, params=params)
        assert resp.status_code == 401

    async def test_static_shell_stays_reachable(self, tmp_path: Path) -> None:
        """Only /api is gated: the app shell must load from the tokened URL."""
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            resp = await client.get("/")
        assert resp.status_code == 200
        assert resp.json()["markitai"]


class TestRemotePeerWithToken:
    """A token-bearing remote peer gets the loopback trust model."""

    async def test_header_token_grants_api_access(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            history = await client.get("/api/history", headers=AUTH_HEADER)
            settings = await client.get("/api/settings/llm", headers=AUTH_HEADER)
            missing = await client.delete("/api/history/nope", headers=AUTH_HEADER)
        assert history.status_code == 200
        assert settings.status_code == 200
        # 404 (not 401/403) proves the delete reached the route handler.
        assert missing.status_code == 404

    async def test_private_url_job_allowed_without_dns_assessment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The remote-URL policy is lifted entirely, like for loopback."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", _fake_url_converter()
        )
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("10.0.0.5",),
        ) as resolver:
            async with _serve_client(
                _make_app(tmp_path, TOKEN), peer=LAN_PEER
            ) as client:
                resp = await client.post(
                    "/api/jobs", files=_url_job([PRIVATE_URL]), headers=AUTH_HEADER
                )
                assert resp.status_code == 201, resp.text
                job = await _drain(client, resp.json()["job_id"], headers=AUTH_HEADER)
        assert job["status"] == "done"
        resolver.assert_not_awaited()

    async def test_query_token_covers_sse_and_downloads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EventSource and <a download> cannot send headers; ?token= can."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _fake_file_converter()
        )
        async with _serve_client(_make_app(tmp_path, TOKEN), peer=LAN_PEER) as client:
            created = await client.post(
                "/api/jobs", files=_file_job(), headers=AUTH_HEADER
            )
            assert created.status_code == 201, created.text
            job_id = created.json()["job_id"]
            job = await _drain(client, job_id, headers=AUTH_HEADER)
            output = job["items"][0]["output"]

            events = await client.get(
                f"/api/jobs/{job_id}/events", params={"token": TOKEN}
            )
            download = await client.get(
                f"/api/jobs/{job_id}/files/{output}", params={"token": TOKEN}
            )
        assert events.status_code == 200
        assert "event: snapshot" in events.text
        assert download.status_code == 200
        assert "converted" in download.text


class TestAuthDisabled:
    """``token=None`` restores the 0.24 remote-peer surface."""

    async def test_remote_reads_pass_without_token(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, None), peer=LAN_PEER) as client:
            capabilities = await client.get("/api/capabilities")
            history = await client.get("/api/history")
        assert capabilities.status_code == 200
        assert history.status_code == 200

    async def test_remote_settings_stay_loopback_only(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path, None), peer=LAN_PEER) as client:
            resp = await client.get("/api/settings/llm")
        assert resp.status_code == 403
        assert "loopback" in resp.json()["detail"]

    async def test_remote_private_url_stays_rejected(self, tmp_path: Path) -> None:
        """A token presented while auth is off grants nothing."""
        async with _serve_client(_make_app(tmp_path, None), peer=LAN_PEER) as client:
            resp = await client.post(
                "/api/jobs", files=_url_job([PRIVATE_URL]), headers=AUTH_HEADER
            )
        assert resp.status_code == 403
        assert "loopback" in resp.json()["detail"]
