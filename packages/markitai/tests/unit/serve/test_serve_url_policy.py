"""Tests for the serve URL fetch policy (SSRF defense by request origin).

The Host/Origin guard (test_serve_security.py) only stops a malicious *web
page* from driving the browser. A LAN user talking to the API directly is not
covered by it, so ``markitai serve --host 0.0.0.0`` would otherwise let anyone
on the network make the server fetch for them. The rule under test (contract
§2): a loopback peer may submit any URL (it is the operator's own machine); an
untrusted non-loopback peer may submit none, public or private (403
``remote_url_forbidden``).

These apps are built without a token (``--no-auth`` / embedder semantics), the
only mode in which an unauthenticated non-loopback peer reaches the routes at
all; token-bearing peers are covered by test_serve_auth.py.

Harness mirrors test_serve_security.py: ``create_app`` over
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


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[FastAPI]:
    async with app.router.lifespan_context(app):
        yield app


def _client_for(app: FastAPI, peer: tuple[str, int] | None) -> httpx.AsyncClient:
    """A client on an app whose lifespan already runs (see ``_lifespan``)."""
    transport = (
        httpx.ASGITransport(app=app, client=peer)
        if peer is not None
        else httpx.ASGITransport(app=app)
    )
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")


def _assert_url_refused(resp: httpx.Response) -> None:
    assert resp.status_code == 403, resp.text
    body = resp.json()
    assert body["reason"] == "remote_url_forbidden"
    # The message must explain *why* and what to do instead.
    assert "--no-auth" in body["detail"]
    assert "markitai serve" in body["detail"]
    assert "loopback" in body["detail"]


class TestNonLoopbackPeerUrlPolicy:
    """A peer that is not the local machine cannot make the server fetch."""

    @pytest.mark.parametrize(
        "url", [PUBLIC_URL, PRIVATE_URL, METADATA_URL, "http://localhost:9200"]
    )
    async def test_every_url_is_refused(self, tmp_path: Path, url: str) -> None:
        with patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            new_callable=AsyncMock,
            return_value=("93.184.216.34",),
        ) as resolver:
            async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
                resp = await client.post("/api/jobs", files=_url_job([url]))
                history = await client.get("/api/history")
        _assert_url_refused(resp)
        assert history.json() == []
        # Refused before any lookup: no DNS answer decides what gets through
        resolver.assert_not_awaited()

    async def test_a_url_list_file_is_refused(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
            resp = await client.post(
                "/api/jobs",
                files=[
                    ("files", ("links.urls", f"{PUBLIC_URL}\n".encode(), "text/plain")),
                    ("options", (None, "{}")),
                ],
            )
        _assert_url_refused(resp)

    async def test_one_url_refuses_the_whole_job(self, tmp_path: Path) -> None:
        """No partial job: the uploaded file must not be stored either."""
        async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
            resp = await client.post(
                "/api/jobs",
                files=[
                    ("files", ("doc.txt", b"hello", "text/plain")),
                    ("urls", (None, json.dumps([PUBLIC_URL]))),
                    ("options", (None, "{}")),
                ],
            )
            history = await client.get("/api/history")
        _assert_url_refused(resp)
        assert history.json() == []

    async def test_files_still_convert_with_public_only_fetches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uploads stay open; images they reference are fetched public-only."""
        from markitai.fetch_policy import public_network_only

        seen: list[bool] = []

        async def fake_process_file_item(
            file_path: Path, cfg: Any, out_dir: Path, shared: Any
        ):
            from markitai.batch import ProcessResult

            seen.append(public_network_only.get())
            out = out_dir / f"{file_path.name}.md"
            out.write_text("# converted\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", fake_process_file_item
        )
        async with _serve_client(_make_app(tmp_path), peer=LAN_PEER) as client:
            resp = await client.post(
                "/api/jobs",
                files=[
                    ("files", ("doc.txt", b"hello", "text/plain")),
                    ("options", (None, "{}")),
                ],
            )
            assert resp.status_code == 201, resp.text
            job = await _drain(client, resp.json()["job_id"])
        assert job["status"] == "done"
        assert seen == [True]

    async def test_retry_of_a_url_item_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retry route re-fetches item.name and needs the same gate."""
        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", _fake_url_converter()
        )
        app = _make_app(tmp_path)
        async with _lifespan(app):
            async with _client_for(app, None) as local:
                created = await local.post("/api/jobs", files=_url_job([PUBLIC_URL]))
                assert created.status_code == 201, created.text
                job_id = created.json()["job_id"]
                job = await _drain(local, job_id)
            item_id = job["items"][0]["item_id"]
            async with _client_for(app, LAN_PEER) as remote:
                retry = await remote.post(
                    f"/api/jobs/{job_id}/items/{item_id}/retry", json={}
                )
                enhance = await remote.post(
                    f"/api/jobs/{job_id}/items/{item_id}/retry",
                    json={"operation": "enhance"},
                )
        _assert_url_refused(retry)
        _assert_url_refused(enhance)


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


class TestUserTokenUrlPolicy:
    """A user token's URLs are fetched from the public network only.

    On the server's own network the fetch would carry the server's
    authority: a tokenless loopback request is the admin, so a user's URL job
    aimed at ``http://127.0.0.1:<port>/api/history?include=items`` read every
    job of the admin and of other users.
    """

    @staticmethod
    def _tokened_app(tmp_path: Path) -> FastAPI:
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        cfg.cache.global_dir = str(tmp_path / "cache")
        return create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            config=cfg,
            configure_logging=False,
            config_path=tmp_path / "config.json",
            token="admin-token",  # nosec B106 - a test token
        )

    async def test_user_jobs_never_use_the_remote_fallbacks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner decision 2026-10-22: a user token's job, and its retry by
        anyone, never tries defuddle.md, Jina or Cloudflare, whatever
        fetch.remote_consent says; the admin's job follows remote_consent."""
        from markitai.fetch_policy import remote_fallbacks_refused

        seen: list[tuple[str, bool]] = []
        fake = _fake_url_converter()

        async def recording(url: str, *args: Any, **kwargs: Any):
            seen.append((url, remote_fallbacks_refused.get()))
            return await fake(url, *args, **kwargs)

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", recording)
        admin = {"Authorization": "Bearer admin-token"}
        app = self._tokened_app(tmp_path)
        async with _lifespan(app), _client_for(app, None) as client:
            created = await client.post(
                "/api/access-tokens", json={"label": "guest"}, headers=admin
            )
            assert created.status_code == 201, created.text
            user = {"Authorization": f"Bearer {created.json()['token']}"}
            own = await client.post(
                "/api/jobs", files=_url_job([PUBLIC_URL]), headers=admin
            )
            theirs = await client.post(
                "/api/jobs", files=_url_job([PUBLIC_URL + "?user"]), headers=user
            )
            assert own.status_code == theirs.status_code == 201, theirs.text
            await _drain(client, own.json()["job_id"])
            job_id = theirs.json()["job_id"]
            job = await _drain(client, job_id)
            retried = await client.post(
                f"/api/jobs/{job_id}/items/{job['items'][0]['item_id']}/retry",
                json={},
                headers=admin,
            )
            assert retried.status_code == 202, retried.text
            await _drain(client, job_id)
        assert sorted(seen) == sorted(
            [
                (PUBLIC_URL, False),
                (PUBLIC_URL + "?user", True),
                (PUBLIC_URL + "?user", True),
            ]
        )

    async def test_user_jobs_and_their_retries_are_public_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.fetch_policy import public_network_only

        seen: list[tuple[str, bool]] = []
        fake = _fake_url_converter()

        async def recording(url: str, *args: Any, **kwargs: Any):
            seen.append((url, public_network_only.get()))
            return await fake(url, *args, **kwargs)

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", recording)
        admin = {"Authorization": "Bearer admin-token"}
        loopback_api = "http://127.0.0.1:3600/api/history?include=items"
        app = self._tokened_app(tmp_path)
        async with _lifespan(app), _client_for(app, None) as client:
            created = await client.post(
                "/api/access-tokens", json={"label": "guest"}, headers=admin
            )
            assert created.status_code == 201, created.text
            user = {"Authorization": f"Bearer {created.json()['token']}"}
            own = await client.post(
                "/api/jobs", files=_url_job([PRIVATE_URL]), headers=admin
            )
            theirs = await client.post(
                "/api/jobs", files=_url_job([loopback_api]), headers=user
            )
            assert own.status_code == theirs.status_code == 201, theirs.text
            await _drain(client, own.json()["job_id"])
            job_id = theirs.json()["job_id"]
            job = await _drain(client, job_id)
            retried = await client.post(
                f"/api/jobs/{job_id}/items/{job['items'][0]['item_id']}/retry",
                json={},
                headers=admin,
            )
            assert retried.status_code == 202, retried.text
            await _drain(client, job_id)
        assert sorted(seen) == sorted(
            [(PRIVATE_URL, False), (loopback_api, True), (loopback_api, True)]
        )
