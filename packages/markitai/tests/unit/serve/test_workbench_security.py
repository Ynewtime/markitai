"""Security properties of the API the shared workbench uses.

Download tickets admit exactly one GET of exactly one download path within a
minute and are kept only as digests; the LLM settings need the token from
every client when token auth is on; a state-changing request's Origin must
match the request's host and port; untrusted callers can neither aim the
fetcher inward nor bill the server's Cloudflare account; a stored original is
never rendered by a direct visit.
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
from tests.unit.serve.test_workbench_contract import check_error

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

TOKEN = "mk_security-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
LAN_PEER = ("192.168.1.23", 51234)


def _make_app(tmp_path: Path, token: str | None = TOKEN) -> FastAPI:
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
async def _client(
    app: FastAPI, *, peer: tuple[str, int] | None = None
) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = (
            httpx.ASGITransport(app=app, client=peer)
            if peer is not None
            else httpx.ASGITransport(app=app)
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:3600"
        ) as client:
            yield client


def _fake_converter() -> Any:
    async def fake(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
        from markitai.batch import ProcessResult

        out = out_dir / f"{file_path.name}.md"
        out.write_text("# done\n", encoding="utf-8")
        return ProcessResult(success=True, output_path=str(out))

    return fake


async def _finished_job(
    client: httpx.AsyncClient, name: str = "a.txt", data: bytes = b"a"
) -> str:
    created = await client.post(
        "/api/jobs",
        files=[
            ("files", (name, data, "application/octet-stream")),
            ("urls", (None, "[]")),
            ("options", (None, "{}")),
        ],
        headers=AUTH,
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = (await client.get(f"/api/jobs/{job_id}", headers=AUTH)).json()
        if status["status"] != "running":
            return job_id
        await asyncio.sleep(0.02)
    pytest.fail("job did not finish")


async def _ticket(client: httpx.AsyncClient, path: str) -> str:
    issued = await client.post(
        "/api/download-tickets", json={"path": path}, headers=AUTH
    )
    assert issued.status_code == 201, issued.text
    return issued.json()["ticket"]


class TestDownloadTickets:
    async def test_a_ticket_stands_in_for_the_token_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _client(_make_app(tmp_path), peer=LAN_PEER) as client:
            job_id = await _finished_job(client)
            path = f"/api/jobs/{job_id}/archive"
            without = await client.get(path)
            ticket = await _ticket(client, path)
            first = await client.get(f"{path}?ticket={ticket}")
            spent = await client.get(f"{path}?ticket={ticket}")
        check_error(without, 401, "token_required")
        assert first.status_code == 200
        assert first.headers["content-type"] == "application/zip"
        check_error(spent, 401, "ticket_invalid")

    async def test_a_ticket_opens_only_its_own_path_and_is_spent_trying(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _client(_make_app(tmp_path), peer=LAN_PEER) as client:
            mine = await _finished_job(client, "mine.txt")
            other = await _finished_job(client, "other.txt")
            ticket = await _ticket(client, f"/api/jobs/{mine}/archive")
            wrong = await client.get(f"/api/jobs/{other}/archive?ticket={ticket}")
            right_after = await client.get(f"/api/jobs/{mine}/archive?ticket={ticket}")
        check_error(wrong, 401, "ticket_invalid")
        check_error(right_after, 401, "ticket_invalid")

    async def test_a_ticket_admits_only_get(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _client(_make_app(tmp_path), peer=LAN_PEER) as client:
            job_id = await _finished_job(client)
            path = f"/api/history/{job_id}"
            ticket = await _ticket(client, "/api/history/archive")
            delete = await client.delete(f"{path}?ticket={ticket}")
            still_there = await client.get(f"/api/jobs/{job_id}", headers=AUTH)
        check_error(delete, 401, "ticket_invalid")
        assert still_there.status_code == 200

    async def test_a_ticket_expires_after_a_minute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        app = _make_app(tmp_path)
        async with _client(app, peer=LAN_PEER) as client:
            await _finished_job(client)
            ticket = await _ticket(client, "/api/history/archive")
            tickets = app.state.download_tickets
            later = time.monotonic() + 61
            tickets._clock = lambda: later
            expired = await client.get(f"/api/history/archive?ticket={ticket}")
        check_error(expired, 401, "ticket_invalid")

    async def test_settings_can_never_be_reached_with_a_ticket(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            refused = await client.post(
                "/api/download-tickets",
                json={"path": "/api/settings/llm/providers/x/credentials"},
                headers=AUTH,
            )
            guessed = await client.get("/api/settings/llm?ticket=" + "0" * 64)
        check_error(refused, 422, "invalid_ticket_path")
        check_error(guessed, 401, "ticket_invalid")

    async def test_only_digests_are_kept(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path)
        async with _client(app) as client:
            ticket = await _ticket(client, "/api/history/archive")
        stored = app.state.download_tickets._tickets
        assert len(stored) == 1
        (digest, (path, _expires, user)), *_ = stored.items()
        assert user is None  # the admin's
        assert ticket.encode() not in digest and ticket not in path
        assert len(digest) == 32

    async def test_an_encoded_traversal_ticket_reaches_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
        async with _client(_make_app(tmp_path), peer=LAN_PEER) as client:
            job_id = await _finished_job(client)
            path = f"/api/jobs/{job_id}/files/..%2F..%2F..%2Fsecret.txt"
            ticket = await _ticket(client, path)
            response = await client.get(f"{path}?ticket={ticket}")
        check_error(response, 404, "file_not_found")
        assert "top secret" not in response.text

    async def test_tickets_are_masked_in_access_logs(self) -> None:
        from markitai.utils.url_redaction import redact_urls_in_text

        line = '127.0.0.1:1 - "GET /api/history/archive?ticket=abcdef0123 HTTP/1.1" 200'
        assert "abcdef0123" not in redact_urls_in_text(line)


class TestSettingsNeedTheToken:
    async def test_loopback_needs_the_token_for_settings(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            refused = await client.get("/api/settings/llm")
            query_token = await client.get(f"/api/settings/llm?token={TOKEN}")
            allowed = await client.get("/api/settings/llm", headers=AUTH)
            jobs_open = await client.get("/api/history")
        check_error(refused, 401, "token_required")
        assert refused.headers["www-authenticate"] == "Bearer"
        assert refused.headers["cache-control"] == "no-store"
        assert query_token.status_code == 200
        assert allowed.status_code == 200
        assert allowed.headers["cache-control"] == "no-store"
        assert jobs_open.status_code == 200  # loopback stays open elsewhere

    async def test_without_a_token_settings_stay_on_loopback(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path, token=None), peer=LAN_PEER) as client:
            remote = await client.get("/api/settings/llm")
        check_error(remote, 403, "settings_forbidden")

    async def test_env_references_cannot_be_aimed_at_another_host(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            probe = await client.post(
                "/api/settings/llm/test",
                json={
                    "model": "openai/gpt-4o-mini",
                    "api_key": "env:OPENAI_API_KEY",
                    "api_base": "https://attacker.example/v1",
                },
                headers=AUTH,
            )
        check_error(probe, 422, "invalid_settings_request")


class TestOrigin:
    @pytest.mark.parametrize(
        "origin",
        [
            "null",
            "garbage",
            "http://127.0.0.1:3601",
            "http://127.0.0.1",
            "ws://127.0.0.1:3600",
            "http://127.0.0.1:3600/app",
            "http://user@127.0.0.1:3600",
            "http://localhost:3600",
            "https://evil.example",
        ],
    )
    @pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
    async def test_cross_origin_writes_are_refused(
        self, tmp_path: Path, origin: str, method: str
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.request(
                method, "/api/download-tickets", headers={"Origin": origin, **AUTH}
            )
        check_error(response, 403, "origin_not_allowed")

    async def test_same_origin_writes_pass(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/download-tickets",
                json={"path": "/api/history/archive"},
                headers={"Origin": "http://127.0.0.1:3600", **AUTH},
            )
        assert response.status_code == 201

    async def test_unknown_host_names_are_refused(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            response = await client.get(
                "/api/capabilities", headers={"Host": "rebind.example:3600"}
            )
        check_error(response, 400, "host_not_allowed")


class TestUntrustedCallers:
    @pytest.mark.parametrize(
        "url", ["http://192.168.1.4/admin", "https://example.com/post"]
    )
    async def test_urls_are_refused(self, tmp_path: Path, url: str) -> None:
        async with _client(_make_app(tmp_path, token=None), peer=LAN_PEER) as client:
            response = await client.post(
                "/api/jobs", files=[("urls", (None, json.dumps([url])))]
            )
        check_error(response, 403, "remote_url_forbidden")

    async def test_this_machine_still_submits_urls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_process_url_item(url, cfg, out_dir, shared, url_ctx, **_):
            from markitai.batch import ProcessResult

            out = out_dir / "page.md"
            out.write_text("# page\n", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr(
            "markitai.serve.jobs.process_url_item", fake_process_url_item
        )
        async with _client(_make_app(tmp_path, token=None)) as client:
            response = await client.post(
                "/api/jobs",
                files=[("urls", (None, json.dumps(["http://192.168.1.4/admin"])))],
            )
            assert response.status_code == 201, response.text
            job_id = response.json()["job_id"]
            for _ in range(200):
                snapshot = (await client.get(f"/api/jobs/{job_id}")).json()
                if snapshot["status"] != "running":
                    break
                await asyncio.sleep(0.02)
        assert snapshot["status"] == "done"


class TestOriginalsAreNeverRendered:
    @pytest.mark.parametrize(
        ("name", "data"),
        [
            ("page.html", b"<script>alert(1)</script>"),
            ("image.svg", b"<svg xmlns='http://www.w3.org/2000/svg'><script/></svg>"),
            ("page.xhtml", b"<html xmlns='http://www.w3.org/1999/xhtml'/>"),
        ],
    )
    async def test_markup_is_sent_as_text_in_a_sandbox(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, data: bytes
    ) -> None:
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", _fake_converter())
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _finished_job(client, name, data)
            response = await client.get(
                f"/api/jobs/{job_id}/items/i1/original/file", headers=AUTH
            )
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.headers["content-security-policy"] == (
            "sandbox; default-src 'none'"
        )
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["content-disposition"].startswith("attachment;")
