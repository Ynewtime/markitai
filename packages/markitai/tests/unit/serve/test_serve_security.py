"""Tests for the serve Host allowlist and cross-site Origin protection.

Same harness as test_serve_api.py: ``create_app`` with a tmp_path-backed jobs
root and an injected config over httpx.ASGITransport. The transport's default
peer is loopback, so these tests exercise exactly the DNS-rebinding scenario:
a loopback TCP peer whose Host/Origin headers carry an attacker's DNS name.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

import httpx

from markitai.config import MarkitaiConfig
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from fastapi import FastAPI


def _make_app(tmp_path: Path, allowed_hosts: Sequence[str] | None = None) -> FastAPI:
    """Build an app with hermetic config and tmp_path-backed jobs root."""
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
        allowed_hosts=allowed_hosts,
    )


@asynccontextmanager
async def _serve_client(
    app: FastAPI, base_url: str = "http://127.0.0.1"
) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app lifespan and yield an ASGI-backed client."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
            yield client


def _multipart(files: list[tuple[str, bytes]]) -> list[tuple[str, Any]]:
    """Build httpx multipart payload for POST /api/jobs."""
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files
    ]
    parts.append(("urls", (None, "[]")))
    parts.append(("options", (None, "{}")))
    return parts


def _fake_converter():
    """Stub converter writing '<name>.md' into the job out dir."""

    async def fake_process_file_item(
        file_path: Path, cfg: Any, out_dir: Path, shared: Any
    ):
        from markitai.batch import ProcessResult

        out = out_dir / f"{file_path.name}.md"
        out.write_text(f"# converted {file_path.name}\n", encoding="utf-8")
        return ProcessResult(success=True, output_path=str(out))

    return fake_process_file_item


class TestHostAllowlist:
    """Host header validation (DNS rebinding defense)."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:3600",
            "http://127.0.0.1:3600",
            "http://[::1]:3600",
            "http://192.168.1.7:3600",  # IP literals cannot be rebound
            "http://LOCALHOST",
        ],
    )
    async def test_local_and_ip_literal_hosts_accepted(
        self, tmp_path: Path, base_url: str
    ) -> None:
        async with _serve_client(_make_app(tmp_path), base_url=base_url) as client:
            resp = await client.get("/api/capabilities")
        assert resp.status_code == 200

    async def test_dns_name_host_rejected_despite_loopback_peer(
        self, tmp_path: Path
    ) -> None:
        """A rebound DNS name arrives from a loopback peer and must still fail."""
        async with _serve_client(
            _make_app(tmp_path), base_url="http://markitai.attacker.example:3600"
        ) as client:
            capabilities = await client.get("/api/capabilities")
            settings = await client.get("/api/settings/llm")
        assert capabilities.status_code == 400
        assert "not allowed" in capabilities.json()["detail"]
        # The settings family (which can return credential material) is
        # covered by the same guard; the loopback peer check alone passes.
        assert settings.status_code == 400

    async def test_allowed_hosts_option_admits_extra_hostname(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(tmp_path, allowed_hosts=["My-Box.LAN"])
        async with _serve_client(app, base_url="http://my-box.lan:3600") as client:
            resp = await client.get("/api/capabilities")
        assert resp.status_code == 200


class TestOriginProtection:
    """Origin allowlist on state-changing methods (CSRF/SSRF defense)."""

    @pytest.mark.parametrize("origin", ["http://evil.example", "null"])
    async def test_cross_site_post_rejected(self, tmp_path: Path, origin: str) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                files=_multipart([("doc.txt", b"hello")]),
                headers={"Origin": origin},
            )
        assert resp.status_code == 403
        assert "cross-site" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "origin",
        ["http://203.0.113.7", "http://203.0.113.7:8080", "http://[2001:db8::1]"],
    )
    async def test_cross_site_ip_literal_origin_rejected(
        self, tmp_path: Path, origin: str
    ) -> None:
        """A page served from a bare IP is still cross-site.

        The Host header trusts IP literals (they cannot be DNS-rebound), but the
        attacker-controlled Origin must not inherit that trust: hosting the
        exploit page from a raw IP would otherwise re-open the /api/jobs SSRF.
        """
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/jobs",
                files=_multipart([("doc.txt", b"hello")]),
                headers={"Origin": origin},
            )
        assert resp.status_code == 403
        assert "cross-site" in resp.json()["detail"]

    async def test_same_host_ip_origin_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A LAN-bound server accepts its own IP origin (same host as Host)."""
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _serve_client(
            _make_app(tmp_path), base_url="http://192.168.1.50:3600"
        ) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart([("doc.txt", b"hello")]),
                headers={"Origin": "http://192.168.1.50:3600"},
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            # Drain the job so lifespan shutdown does not cancel mid-run.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                data = (await client.get(f"/api/jobs/{job_id}")).json()
                if data["status"] == "done":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail(f"job {job_id} did not finish")

    async def test_cross_site_delete_rejected(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            rejected = await client.delete(
                "/api/history/nope", headers={"Origin": "http://evil.example"}
            )
            no_origin = await client.delete("/api/history/nope")
        assert rejected.status_code == 403
        assert no_origin.status_code == 404  # reached the route handler

    async def test_same_origin_post_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart([("doc.txt", b"hello")]),
                headers={"Origin": "http://127.0.0.1"},
            )
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            # Drain the job so lifespan shutdown does not cancel mid-run.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                data = (await client.get(f"/api/jobs/{job_id}")).json()
                if data["status"] == "done":
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail(f"job {job_id} did not finish")

    async def test_cross_site_origin_does_not_block_reads(self, tmp_path: Path) -> None:
        """Reads stay browser-enforced (CORS); only writes are rejected."""
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get(
                "/api/capabilities", headers={"Origin": "http://evil.example"}
            )
        assert resp.status_code == 200
        assert json.loads(resp.text)["presets"]
