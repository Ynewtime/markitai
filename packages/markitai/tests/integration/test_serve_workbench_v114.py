"""Contract v1.14 end to end: a real browser loads a local site's page.

The server converts an invented page with the local browser (Playwright's
Chromium; skipped where it is not installed) and keeps, for Compare, the
page, how it read, its compare screenshots and their block boxes; with
screenshots asked for, the item's own screenshots carry the boxes instead.
A server started again on the same folder frames the page at its URL.
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
from tests.local_http import LocalHTTPServer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from fastapi import FastAPI

PAGE = (
    "<!doctype html><html><head><title>River notes</title>"
    '<link rel="stylesheet" href="style.css"></head><body><article>'
    "<h1>River notes</h1>"
    + "".join(
        f"<h2>Day {day}</h2><p>{'The gauge at the old mill read the same at dawn. ' * 8}"
        f'</p><img src="mill-{day}.gif" width="400" height="160" alt="Mill {day}">'
        for day in range(1, 19)
    )
    + "</article></body></html>"
)
GIF = bytes.fromhex(
    "47494638396101000100800000ffffff00000021f90401000000002c00000000010001000002024401003b"
)


class _Site(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.endswith(".gif"):
            body, kind = GIF, "image/gif"
        elif self.path.endswith(".css"):
            body, kind = b"body{font:18px sans-serif}", "text/css"
        else:
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site() -> Iterator[str]:
    from markitai.fetch_playwright import (
        is_playwright_available,
        is_playwright_browser_installed,
    )

    if not is_playwright_available() or not is_playwright_browser_installed(
        use_cache=False
    ):
        pytest.skip("Playwright's Chromium is not installed")
    server = LocalHTTPServer(("127.0.0.1", 0), _Site)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/notes/river.html"
    finally:
        server.shutdown()
        server.server_close()


def _app(tmp_path: Path) -> FastAPI:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    cfg.screenshot.tile_height = 1500
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
    )


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:3600", timeout=120
        ) as client:
            yield client


async def _convert(client: httpx.AsyncClient, url: str, options: dict[str, Any]) -> str:
    created = await client.post(
        "/api/jobs",
        files=[
            ("urls", (None, json.dumps([url]))),
            ("options", (None, json.dumps(options))),
        ],
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]
    for _ in range(3000):
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["status"] != "running":
            assert job["items"][0]["status"] == "done", job
            return job_id
        await asyncio.sleep(0.05)
    pytest.fail(f"job {job_id} did not finish")


async def test_a_browser_load_keeps_compare_screenshots_and_boxes(
    tmp_path: Path, site: str
) -> None:
    async with _client(_app(tmp_path)) as client:
        job_id = await _convert(client, site, {"strategy": "playwright"})
        prefix = f"/api/jobs/{job_id}/items/i1"
        descriptor = (await client.get(f"{prefix}/original")).json()
        blocks = (await client.get(f"{prefix}/original/blocks")).json()
        first = await client.get(f"{prefix}/original/pages/1?width=1400")
        result = (await client.get(f"{prefix}/result")).json()
    assert descriptor["kind"] == "html"
    assert descriptor["capture"]["page"]["state"] == "ok"
    assert descriptor["capture"]["screenshot"]["state"] == "ok"
    assert descriptor["capture"]["screenshot"]["title"] == "River notes"
    pages = descriptor["pages"]
    assert len(pages) >= 2
    assert all(page["height"] == 1500 for page in pages[:-1])
    assert descriptor["blocks"] == f"{prefix}/original/blocks"
    assert blocks["width"] == pages[0]["width"]
    assert blocks["height"] == sum(page["height"] for page in pages)
    assert [b["text"] for b in blocks["blocks"] if b["tag"] == "h2"][:2] == [
        "Day 1",
        "Day 2",
    ]
    assert first.headers["content-type"] == "image/jpeg"
    # The tiles are kept beside the page, never as output
    shots = sorted((tmp_path / "jobs" / job_id / "snapshots" / "i1.shots").iterdir())
    assert [shot.name for shot in shots] == [
        f"{n}.jpg" for n in range(1, len(pages) + 1)
    ]
    assert not list((tmp_path / "jobs" / job_id / "out").rglob("*.jpg"))
    assert all(".shots" not in a["relpath"] for a in result["artifacts"])

    # Started again on the same folder: framed at the page's own URL
    async with _client(_app(tmp_path)) as client:
        view = await client.get(f"{prefix}/original/view")
        again = (await client.get(f"{prefix}/original")).json()
    assert f'<base href="{site}" target="_blank">' in view.text
    assert again["blocks"] == descriptor["blocks"]
    assert again["capture"] == descriptor["capture"]


async def test_asked_for_screenshots_carry_the_boxes(tmp_path: Path, site: str) -> None:
    async with _client(_app(tmp_path)) as client:
        job_id = await _convert(
            client, site, {"strategy": "static", "screenshot": True}
        )
        prefix = f"/api/jobs/{job_id}/items/i1"
        descriptor = (await client.get(f"{prefix}/original")).json()
        blocks = (await client.get(f"{prefix}/original/blocks")).json()
    pages = descriptor["pages"]
    assert descriptor["kind"] == "html"
    assert pages and descriptor["blocks"] == f"{prefix}/original/blocks"
    assert blocks["height"] == sum(page["height"] for page in pages)
    assert descriptor["capture"]["screenshot"]["state"] == "ok"
    # The page came from the static fetch; the screenshots from the browser
    assert descriptor["capture"]["page"]["state"] == "ok"
    assert not (tmp_path / "jobs" / job_id / "snapshots" / "i1.shots").exists()
