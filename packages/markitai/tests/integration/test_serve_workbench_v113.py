"""Contract v1.13 end to end on the real conversion pipeline (no fakes).

An admin and a user token share one server: real Markdown, text and web-page
conversions (the page fetched from a local site with the static fetcher),
each caller's history, batch deletion across both, the framed page's policy,
an Edit config round trip whose masked key stays on disk, and revocation.
The local site is on loopback: the admin's job fetches it, while a user
token's job, which reaches public addresses only, is refused.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

pytest.importorskip("fastapi")

from markitai.config import MarkitaiConfig
from markitai.serve import create_app
from markitai.serve.originals import VIEW_POLICY
from tests.local_http import LocalHTTPServer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from fastapi import FastAPI

ADMIN = {"Authorization": "Bearer admin-token"}
REMOTE = ("192.168.1.30", 40000)
PAGE = (
    "<!doctype html><html><head><title>Field notes</title>"
    "<script>document.title='ran'</script></head><body><article>"
    '<h1 onclick="alert(1)">Field notes</h1><p>Rain gauges were read at dawn on '
    "each of the three sites, and the readings agree within a millimetre.</p>"
    '<a href="javascript:alert(2)">details</a></article></body></html>'
)
CONFIG = (
    "{\n"
    '  "llm": {"model_list": [\n'
    '    {"model_name": "default", "litellm_params": {"model": "openai/gpt-4o-mini",'
    ' "api_key": "sk-invented-key-abcd1234"}}\n'
    "  ]},\n"
    '  "batch": {"concurrency": 2}\n'
    "}\n"
)


class _Site(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site() -> Iterator[str]:
    server = LocalHTTPServer(("127.0.0.1", 0), _Site)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/notes.html"
    finally:
        server.shutdown()
        server.server_close()


def _app(tmp_path: Path) -> FastAPI:
    (tmp_path / "config.json").write_text(CONFIG, encoding="utf-8", newline="")
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
        token="admin-token",
    )


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=REMOTE)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:3600", timeout=60
        ) as client:
            yield client


async def _convert(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    files: Sequence[tuple[str, bytes]] = (),
    urls: Sequence[str] = (),
    status: str = "done",
) -> dict[str, Any]:
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files
    ]
    parts += [
        ("urls", (None, json.dumps(list(urls)))),
        ("options", (None, json.dumps({"strategy": "static"} if urls else {}))),
    ]
    created = await client.post("/api/jobs", files=parts, headers=headers)
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]
    for _ in range(3000):
        job = (await client.get(f"/api/jobs/{job_id}", headers=headers)).json()
        if job["status"] != "running":
            assert all(item["status"] == status for item in job["items"]), job
            return job
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish")


async def test_two_callers_one_server(tmp_path: Path, site: str) -> None:
    async with _client(_app(tmp_path)) as client:
        created = await client.post(
            "/api/access-tokens", json={"label": "Field laptop"}, headers=ADMIN
        )
        user = {"Authorization": f"Bearer {created.json()['token']}"}
        admins = await _convert(
            client, ADMIN, files=[("plan.md", b"# Plan\n\nSurvey the north site.\n")]
        )
        mine = await _convert(
            client,
            user,
            files=[
                ("readings.txt", b"Gauge A: 12 mm\nGauge B: 13 mm\n"),
                ("summary.md", b"# Summary\n\nAll gauges agree.\n"),
            ],
        )
        # The page is on this machine: a user token's job reaches public
        # addresses only, so the user's fetch is refused; the admin's is not
        page = await _convert(client, user, urls=[site], status="error")
        assert page["items"][0]["error_code"] == "fetch_error", page
        assert "private_or_local_host" in page["items"][0]["error"], page
        framed = await _convert(client, ADMIN, urls=[site])

        # Each caller's list
        listed = (await client.get("/api/history", headers=user)).json()
        everything = (await client.get("/api/history", headers=ADMIN)).json()
        assert {entry["job_id"] for entry in listed} == {mine["job_id"], page["job_id"]}
        assert {entry["job_id"] for entry in everything} == {
            admins["job_id"],
            mine["job_id"],
            page["job_id"],
            framed["job_id"],
        }
        hidden = await client.get(
            f"/api/jobs/{admins['job_id']}/items/i1/result", headers=user
        )
        assert (hidden.status_code, hidden.json()["reason"]) == (404, "job_not_found")
        result = await client.get(
            f"/api/jobs/{mine['job_id']}/items/i1/result", headers=user
        )
        assert "Gauge A: 12 mm" in result.json()["markdown"]

        # The fetched page, framed under the script-less same-origin policy
        view = await client.get(
            f"/api/jobs/{framed['job_id']}/items/i1/original/view", headers=ADMIN
        )
        assert view.headers["content-security-policy"] == VIEW_POLICY
        assert "<script>document.title='ran'</script>" in view.text
        assert '<base href="' in view.text

        # One request deletes across both callers: the admin's job is not the
        # user's to delete, its own item is
        deleted = await client.post(
            "/api/history/delete",
            json={
                "jobs": [admins["job_id"]],
                "items": [{"job_id": mine["job_id"], "item_id": "i2"}],
            },
            headers=user,
        )
        assert [(r["ok"], r.get("reason")) for r in deleted.json()["results"]] == [
            (False, "job_not_found"),
            (True, None),
        ]
        left = (await client.get(f"/api/jobs/{mine['job_id']}", headers=user)).json()
        assert [item["name"] for item in left["items"]] == ["readings.txt"]
        assert not list((tmp_path / "jobs" / mine["job_id"]).rglob("summary.md*"))

        # Revoked: refused at once; the admin keeps the user's jobs
        await client.delete(f"/api/access-tokens/{created.json()['id']}", headers=ADMIN)
        refused = await client.get("/api/history", headers=user)
        assert refused.status_code == 401
        kept = await client.get(f"/api/jobs/{page['job_id']}", headers=ADMIN)
        assert kept.status_code == 200


async def test_edit_config_round_trip(tmp_path: Path) -> None:
    async with _client(_app(tmp_path)) as client:
        shown = (await client.get("/api/settings/config", headers=ADMIN)).json()
        assert "sk-invented-key-abcd1234" not in shown["text"]
        assert '"api_key": "sk…1234"' in shown["text"]
        edited = shown["text"].replace('"concurrency": 2', '"concurrency": 4')
        saved = await client.put(
            "/api/settings/config",
            json={"expected_revision": shown["revision"], "text": edited},
            headers=ADMIN,
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["restart_required"] is True
        on_disk = (tmp_path / "config.json").read_text(encoding="utf-8")
        assert on_disk == CONFIG.replace('"concurrency": 2', '"concurrency": 4')
        stale = await client.put(
            "/api/settings/config",
            json={"expected_revision": shown["revision"], "text": edited},
            headers=ADMIN,
        )
        assert stale.json()["detail"] == {
            "code": "stale_revision",
            "current_revision": saved.json()["revision"],
        }
        # The CLI reads the saved file as before
        from markitai.config import ConfigManager

        loaded = ConfigManager().load(config_path=tmp_path / "config.json")
        assert loaded.batch.concurrency == 4
        assert loaded.llm.model_list[0].litellm_params.api_key == (
            "sk-invented-key-abcd1234"
        )
