"""Tests for the serve URL fetch policy (SSRF defense by request origin).

The Host/Origin guard (test_serve_security.py) only stops a malicious *web
page* from driving the browser. A LAN user talking to the API directly is not
covered by it, so ``markitai serve --host 0.0.0.0`` would otherwise let anyone
on the network make the server fetch intranet addresses. The rule under test:
a loopback peer may target private/local URLs (it is the operator's own
machine), a non-loopback peer may not.

Harness mirrors test_serve_security.py: ``create_app`` over
httpx.ASGITransport, whose ``client=`` tuple is the TCP peer the app sees.
"""

from __future__ import annotations

import asyncio
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

LAN_PEER = ("192.168.1.23", 51234)
PRIVATE_URL = "http://192.168.1.4/admin"
METADATA_URL = "http://169.254.169.254/latest/meta-data/"
PUBLIC_URL = "https://example.com/post"


def _make_app(tmp_path: Path) -> FastAPI:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
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


def _url_job(urls: list[str]) -> list[tuple[str, Any]]:
    import json

    return [
        ("urls", (None, json.dumps(urls))),
        ("options", (None, "{}")),
    ]


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


async def _drain(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    """Wait for a job to finish so lifespan shutdown cannot cancel it."""
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] != "running":
            return data
        await asyncio.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish")


class TestNonLoopbackPeerUrlPolicy:
    """A peer that is not the local machine cannot aim the fetcher inward."""

    @pytest.mark.parametrize(
        "url", [PRIVATE_URL, METADATA_URL, "http://localhost:9200"]
    )
    async def test_private_target_rejected(self, tmp_path: Path, url: str) -> None:
        async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
            resp = await client.post("/api/jobs", files=_url_job([url]))
        assert resp.status_code == 403
        detail = resp.json()["detail"]
        # The message must explain *why* and what to do instead.
        assert "loopback" in detail
        assert "markitai serve" in detail

    async def test_hostname_resolving_to_private_address_rejected(
        self, tmp_path: Path
    ) -> None:
        """DNS-level SSRF: a public name whose records point inward."""
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("10.0.0.5",),
        ):
            async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
                resp = await client.post(
                    "/api/jobs", files=_url_job(["https://rebind.example/x"])
                )
        assert resp.status_code == 403
        assert "loopback" in resp.json()["detail"]

    async def test_public_target_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", _fake_url_converter()
        )
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("93.184.216.34",),
        ):
            async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
                resp = await client.post("/api/jobs", files=_url_job([PUBLIC_URL]))
                assert resp.status_code == 201, resp.text
                await _drain(client, resp.json()["job_id"])

    async def test_one_private_url_rejects_the_whole_job(self, tmp_path: Path) -> None:
        """No partial job: the rejected URL must not be created at all."""
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("93.184.216.34",),
        ):
            async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
                resp = await client.post(
                    "/api/jobs", files=_url_job([PUBLIC_URL, PRIVATE_URL])
                )
                assert resp.status_code == 403
                history = await client.get("/api/history")
        assert history.json() == []

    async def test_retry_cannot_smuggle_a_private_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retry route re-fetches item.name and needs the same gate."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", _fake_url_converter()
        )
        app = _make_app(tmp_path)
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("93.184.216.34",),
        ):
            async with _serve_client(app, peer=LAN_PEER) as client:
                created = await client.post("/api/jobs", files=_url_job([PUBLIC_URL]))
                assert created.status_code == 201, created.text
                job_id = created.json()["job_id"]
                await _drain(client, job_id)
                job = (await client.get(f"/api/jobs/{job_id}")).json()
                item_id = job["items"][0]["item_id"]
                # Rewrite the ledger the way a tampered/rehydrated meta would.
                registered = app.state.markitai.registry.get(job_id)
                registered.items[0].name = PRIVATE_URL
                resp = await client.post(
                    f"/api/jobs/{job_id}/items/{item_id}/retry", json={}
                )
                assert resp.status_code == 403
                assert "loopback" in resp.json()["detail"]


class TestLoopbackPeerUrlPolicy:
    """The operator's own machine keeps full access to its own network."""

    @pytest.mark.parametrize("url", [PRIVATE_URL, METADATA_URL])
    async def test_private_target_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", _fake_url_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post("/api/jobs", files=_url_job([url]))
            assert resp.status_code == 201, resp.text
            job = await _drain(client, resp.json()["job_id"])
        assert job["status"] == "done"

    async def test_private_target_needs_no_dns_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The loopback fast path must not pay for (or leak) a resolution."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", _fake_url_converter()
        )
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("10.0.0.5",),
        ) as resolver:
            async with _serve_client(_make_app(tmp_path)) as client:
                resp = await client.post("/api/jobs", files=_url_job([PUBLIC_URL]))
                assert resp.status_code == 201, resp.text
                await _drain(client, resp.json()["job_id"])
        resolver.assert_not_awaited()
