"""Workbench API contract v1.11: what Python 1.2.1 adds for the e62d82f bundle.

- One item's ZIP (section 6.7, ``features.item_archive``): its result's files
  and its kept original.
- A web page's original is the page the server fetched (section 7): kept as
  ``snapshots/<item_id>.html``, described as ``html`` with ``view``, served
  for the UI's sandboxed frame by ``…/original/view`` (v1.12: under a policy
  that loads the site's pictures, style sheets, fonts and media over HTTPS
  and runs no script); the page's own policy allows ``frame-src 'self'``.
- ``POST /api/access-token/regenerate`` (section 2,
  ``features.token_regenerate``): the old token, its tickets and its event
  streams stop working at once.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

pytest.importorskip("fastapi")

from markitai.config import MarkitaiConfig
from markitai.constants import SCREENSHOTS_REL_PATH
from markitai.serve import create_app
from markitai.serve.originals import MAX_SNAPSHOT_BYTES, framed_page
from markitai.serve.tickets import DownloadTickets, downloadable
from markitai.serve.workbench import CSP
from tests.unit.serve.test_serve_item_files import _pdf_like_converter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

URL = "https://example.test/articles/a?b=1&c=2"
PAGE = (
    "<!doctype html><html lang=en><head><title>A page</title>"
    "<style>h1{color:#c00}</style><script>document.title='ran'</script></head>"
    '<body><h1>Fetched</h1><img src="/remote.png"><a href="next">next</a></body></html>'
)
META = '<meta http-equiv="x-dns-prefetch-control" content="off">'
# Contract v1.12 section 7, exactly
VIEW_POLICY = (
    "sandbox allow-popups allow-popups-to-escape-sandbox; default-src 'none'; "
    "img-src https: data:; style-src https: 'unsafe-inline'; "
    "font-src https: data:; media-src https:; "
    "form-action 'none'; frame-ancestors 'self'"
)


def _make_app(tmp_path: Path, **kwargs: Any) -> FastAPI:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
        **kwargs,
    )


@asynccontextmanager
async def _client(
    app: FastAPI,
    *,
    peer: tuple[str, int] | None = None,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = (
            httpx.ASGITransport(app=app, client=peer)
            if peer is not None
            else httpx.ASGITransport(app=app)
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:3600", headers=headers
        ) as client:
            yield client


def _form(
    files: list[tuple[str, bytes]] | None = None, urls: list[str] | None = None
) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files or []
    ]
    parts.append(("urls", (None, json.dumps(urls or []))))
    parts.append(("options", (None, "{}")))
    return parts


async def _settled(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    """The job's snapshot once nothing in it is queued or running."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] != "running" and all(
            item["status"] in ("done", "error") for item in data["items"]
        ):
            return data
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not settle")


async def _job(client: httpx.AsyncClient, **form: Any) -> str:
    created = await client.post("/api/jobs", files=_form(**form))
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]
    await _settled(client, job_id)
    return job_id


class _FakeUrlConversion:
    """``serve.jobs.process_url_item`` stand-in: writes the item's Markdown and
    a page screenshot, and reports the page's HTML as a fetch would."""

    def __init__(self) -> None:
        self.html: str | None = PAGE
        self.fail = False
        self.screenshot = True

    async def __call__(
        self,
        url: str,
        cfg: Any,
        out_dir: Path,
        shared: Any,
        url_ctx: Any,
        output_name: str | None = None,
    ) -> Any:
        from markitai.batch import ProcessResult

        if self.fail:
            return ProcessResult(success=False, error="fetch failed")
        name = output_name or "page.md"
        out = out_dir / name
        out.write_text(f"# Fetched\n\n{url}\n", encoding="utf-8")
        if self.screenshot:
            shots = out_dir / SCREENSHOTS_REL_PATH
            shots.mkdir(parents=True, exist_ok=True)
            # A 1x1 PNG, named after the output as URL screenshots are
            (shots / f"{name.removesuffix('.md')}.full.png").write_bytes(_PNG)
            out.write_text(
                out.read_text(encoding="utf-8")
                + f"\n![](.markitai/screenshots/{name.removesuffix('.md')}.full.png)\n",
                encoding="utf-8",
            )
        return ProcessResult(success=True, output_path=str(out), page_html=self.html)


_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


@pytest.fixture
def url_conversion(monkeypatch: pytest.MonkeyPatch) -> _FakeUrlConversion:
    fake = _FakeUrlConversion()
    monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake)
    # The fake never fetches; the server's own URL policy still runs
    monkeypatch.setattr(
        "markitai.serve.app._guard_fetch_targets", _allow_every_target, raising=True
    )
    return fake


async def _allow_every_target(request: Any, urls: Any) -> None:
    return None


@pytest.fixture
def pdf_like(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("markitai.serve.jobs.process_file_item", _pdf_like_converter())


# --------------------------------------------------------------------- 6.7


class TestItemArchive:
    @pytest.mark.usefixtures("pdf_like")
    async def test_it_holds_the_result_files_and_the_kept_original(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(
                client, files=[("a.pdf", b"%PDF-a"), ("b b.pdf", b"%PDF-b")]
            )
            result = (await client.get(f"/api/jobs/{job_id}/items/i2/result")).json()
            response = await client.get(f"/api/jobs/{job_id}/items/i2/archive")
            features = (await client.get("/api/capabilities")).json()["features"]
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert response.headers["content-disposition"] == (
            "attachment; filename*=UTF-8''b%20b.pdf.zip"
        )
        assert response.headers["x-content-type-options"] == "nosniff"
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            assert archive.read("original/b b.pdf") == b"%PDF-b"
        # Exactly the result's artifacts, at their relpaths, then the original
        assert names == [a["relpath"] for a in result["artifacts"]] + [
            "original/b b.pdf"
        ]
        assert not any("a.pdf" in name for name in names)
        assert features["item_archive"] is True
        # Nothing is left beside the job once it was sent
        assert not list((tmp_path / "jobs" / job_id).glob(".*archive*"))

    @pytest.mark.usefixtures("pdf_like")
    async def test_its_path_is_a_ticket_path(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path, token="the-token")
        bearer = {"Authorization": "Bearer the-token"}
        async with _client(app, headers=bearer) as client:
            job_id = await _job(client, files=[("a.pdf", b"%PDF")])
            path = f"/api/jobs/{job_id}/items/i1/archive"
            ticket = (
                await client.post("/api/download-tickets", json={"path": path})
            ).json()
        async with _client(app, peer=("192.168.1.30", 40000)) as remote:
            used = await remote.get(ticket["url"])
            again = await remote.get(ticket["url"])
        assert ticket["url"].startswith(f"{path}?ticket=")
        assert used.status_code == 200
        assert used.headers["content-type"] == "application/zip"
        assert again.status_code == 401
        assert again.json()["reason"] == "ticket_invalid"

    @pytest.mark.usefixtures("pdf_like")
    async def test_refusals(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, files=[("a.pdf", b"%PDF")])
            state = client._transport.app.state.markitai  # type: ignore[attr-defined]
            item = state.registry.get(job_id).items[0]
            item.status = "running"
            busy = await client.get(f"/api/jobs/{job_id}/items/i1/archive")
            item.status = "error"
            item.output = None
            no_result = await client.get(f"/api/jobs/{job_id}/items/i1/archive")
            no_item = await client.get(f"/api/jobs/{job_id}/items/i9/archive")
            no_job = await client.get("/api/jobs/nope/items/i1/archive")
        assert (busy.status_code, busy.json()["reason"]) == (409, "item_busy")
        assert (no_result.status_code, no_result.json()["reason"]) == (
            404,
            "result_unavailable",
        )
        assert (no_item.status_code, no_item.json()["reason"]) == (
            404,
            "item_not_found",
        )
        assert (no_job.status_code, no_job.json()["reason"]) == (404, "job_not_found")

    async def test_a_url_item_holds_the_page_it_was_converted_from(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            snapshot = await client.get(f"/api/jobs/{job_id}")
            output = snapshot.json()["items"][0]["output"]
            response = await client.get(f"/api/jobs/{job_id}/items/i1/archive")
        base = output.removesuffix(".md")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.read(f"original/{base}.html").decode() == PAGE
            assert output in archive.namelist()


def test_tickets_name_the_new_download_paths() -> None:
    for path in (
        "/api/jobs/j1/items/i1/archive",
        "/api/jobs/j1/items/i1/original/view",
        "/api/jobs/j1/items/i1/original/file",
    ):
        assert downloadable(path), path
    for path in (
        "/api/jobs/j1/items/../archive",
        "/api/jobs/j1/items/i1/archive/x",
        "/api/jobs/j1/items/i1/original",
        "/api/jobs/j1/items/i1/original/pages/1",
        "/api/jobs/j1/items//archive",
    ):
        assert not downloadable(path), path


def test_tickets_can_all_be_revoked() -> None:
    tickets = DownloadTickets()
    ticket = tickets.issue("/api/history/archive")
    assert ticket is not None
    tickets.clear()
    assert not tickets.redeem(ticket, "/api/history/archive")


# ----------------------------------------------------------------------- 7


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        (
            "<!doctype html><html lang=en><head><title>T</title></head><body>x</body></html>",
            "<!doctype html><html lang=en><head>{}<title>T</title></head><body>x</body></html>",
        ),
        (
            "<HTML><HEAD\n data-x=1><TITLE>T</TITLE></HEAD></HTML>",
            "<HTML><HEAD\n data-x=1>{}<TITLE>T</TITLE></HEAD></HTML>",
        ),
        ("<head/><p>x", "<head/>{}<p>x"),
        (
            "<html><header>h</header><p>x</p></html>",
            "<html>{}<header>h</header><p>x</p></html>",
        ),
        ("<header>h</header><heading><head>", "<header>h</header><heading><head>{}"),
        ("<htmlx><HTML\tdir=ltr><p>x", "<htmlx><HTML\tdir=ltr>{}<p>x"),
        ("  <!DOCTYPE html>\n<p>x</p>", "  <!DOCTYPE html>{}\n<p>x</p>"),
        ("<p>x</p>", "{}<p>x</p>"),
        ("﻿<p>x</p>", "﻿{}<p>x</p>"),
        ("", "{}"),
        ("<head", "{}<head"),
    ],
)
def test_a_framed_page_gets_one_insertion(page: str, expected: str) -> None:
    """The same table as Rust's ``original.rs`` test."""
    upload = f'{META}<base target="_blank">'
    assert framed_page(page, None) == expected.format(upload)


def test_a_url_items_base_is_its_escaped_address() -> None:
    assert framed_page(
        "<head><title>T</title>", "https://example.test/a?b=1&c=\"<'>"
    ) == (
        f'<head>{META}<base href="https://example.test/a?b=1&amp;c=&quot;&lt;&#39;&gt;"'
        ' target="_blank"><title>T</title>'
    )


def test_the_page_policy_lets_the_workbench_frame_its_own_pages() -> None:
    assert "frame-src 'self'" in CSP
    assert "frame-src 'none'" not in CSP


class TestKeptWebPage:
    async def test_a_url_item_is_its_fetched_page(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            item = (await client.get(f"/api/jobs/{job_id}")).json()["items"][0]
            prefix = f"/api/jobs/{job_id}/items/{item['item_id']}"
            descriptor = (await client.get(f"{prefix}/original")).json()
            file = await client.get(f"{prefix}/original/file")
            view = await client.get(f"{prefix}/original/view")
            page = await client.get(f"{prefix}/original/pages/1?width=200")
        base = item["output"].removesuffix(".md")
        assert descriptor["kind"] == "html"
        assert descriptor["name"] == URL
        assert descriptor["media_type"] == "text/html"
        assert descriptor["size"] == len(PAGE.encode())
        assert descriptor["download"] == f"{prefix}/original/file"
        assert descriptor["view"] == f"{prefix}/original/view"
        # The page's screenshot is its page
        assert descriptor["page_count"] == 1
        assert descriptor["pages"] == [{"width": 1.0, "height": 1.0}]
        assert page.status_code == 200
        assert page.headers["content-type"] == "image/png"
        # The bytes, never rendered by a direct visit
        assert file.content.decode() == PAGE
        assert file.headers["content-type"] == "text/plain; charset=utf-8"
        assert file.headers["content-disposition"] == (
            f"attachment; filename*=UTF-8''{base}.html"
        )
        # The framed page: one insertion, and exactly these headers
        assert view.status_code == 200
        assert view.text == PAGE.replace(
            "<head>",
            f'<head>{META}<base href="{URL.replace("&", "&amp;")}" target="_blank">',
            1,
        )
        assert view.headers["content-type"] == "text/html; charset=utf-8"
        assert view.headers["content-security-policy"] == VIEW_POLICY
        assert view.headers["x-content-type-options"] == "nosniff"
        assert view.headers["referrer-policy"] == "no-referrer"
        assert view.headers["x-dns-prefetch-control"] == "off"
        assert view.headers["cache-control"] == "no-store"
        assert view.headers["content-disposition"] == "inline"
        assert "x-frame-options" not in view.headers
        snapshots = list((tmp_path / "jobs" / job_id / "snapshots").iterdir())
        assert [path.name for path in snapshots] == [f"{item['item_id']}.html"]

    def test_the_frame_loads_the_sites_files_and_runs_no_script(self) -> None:
        from markitai.serve.originals import VIEW_POLICY as SERVED

        assert SERVED == VIEW_POLICY
        directives = {
            name: values
            for name, *values in (part.split() for part in SERVED.split(";"))
        }
        # No script: none is allowed, and the sandbox grants neither scripts,
        # forms nor the frame an origin of its own
        assert directives["default-src"] == ["'none'"]
        assert not {"script-src", "script-src-elem", "connect-src"} & set(directives)
        assert directives["sandbox"] == [
            "allow-popups",
            "allow-popups-to-escape-sandbox",
        ]
        assert directives["form-action"] == ["'none'"]
        assert directives["frame-ancestors"] == ["'self'"]
        # The page's pictures, style sheets, fonts and media, over HTTPS only
        assert directives["img-src"] == ["https:", "data:"]
        assert directives["style-src"] == ["https:", "'unsafe-inline'"]
        assert directives["font-src"] == ["https:", "data:"]
        assert directives["media-src"] == ["https:"]
        assert "http:" not in SERVED and "*" not in SERVED

    async def test_an_uploaded_page_is_framed_with_a_bare_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _pdf_like_converter()
        )
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, files=[("p.html", b"<p>hi</p>")])
            prefix = f"/api/jobs/{job_id}/items/i1"
            descriptor = (await client.get(f"{prefix}/original")).json()
            view = await client.get(f"{prefix}/original/view")
        assert descriptor["kind"] == "html"
        assert descriptor["view"] == f"{prefix}/original/view"
        assert view.text == f'{META}<base target="_blank"><p>hi</p>'

    @pytest.mark.usefixtures("pdf_like")
    async def test_other_originals_have_no_view(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, files=[("a.pdf", b"%PDF")])
            descriptor = await client.get(f"/api/jobs/{job_id}/items/i1/original")
            view = await client.get(f"/api/jobs/{job_id}/items/i1/original/view")
        assert descriptor.json()["view"] is None
        assert (view.status_code, view.json()["reason"]) == (
            404,
            "original_unavailable",
        )

    async def test_a_page_without_html_keeps_nothing(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        # A remote service's Markdown (Jina, Defuddle) or a fetch-cache hit
        url_conversion.html = None
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = await client.get(f"/api/jobs/{job_id}/items/i1/original")
            view = await client.get(f"/api/jobs/{job_id}/items/i1/original/view")
        assert descriptor.json()["kind"] == "screenshots"
        assert descriptor.json()["view"] is None
        assert view.status_code == 404

    async def test_a_page_over_10_mib_is_not_kept(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        url_conversion.html = "<p>" + "x" * MAX_SNAPSHOT_BYTES
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = await client.get(f"/api/jobs/{job_id}/items/i1/original")
        assert descriptor.json()["kind"] == "screenshots"

    async def test_a_kept_page_grown_over_10_mib_is_not_framed(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            snapshot = tmp_path / "jobs" / job_id / "snapshots" / "i1.html"
            snapshot.write_bytes(b"<p>" + b"x" * MAX_SNAPSHOT_BYTES)
            # A rerun would clear the descriptor cache; drop it here
            client._transport.app.state.original_cache.clear()  # type: ignore[attr-defined]
            view = await client.get(f"/api/jobs/{job_id}/items/i1/original/view")
        assert (view.status_code, view.json()["reason"]) == (413, "result_too_large")

    async def test_a_rerun_replaces_it_and_a_failed_one_keeps_it(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            snapshot = tmp_path / "jobs" / job_id / "snapshots" / "i1.html"
            assert snapshot.read_text(encoding="utf-8") == PAGE

            url_conversion.html = "<p>second fetch</p>"
            await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            await _settled(client, job_id)
            assert snapshot.read_text(encoding="utf-8") == "<p>second fetch</p>"

            url_conversion.fail = True
            url_conversion.html = "<p>never kept</p>"
            await client.post(f"/api/jobs/{job_id}/items/i1/retry")
            data = await _settled(client, job_id)
        assert data["items"][0]["status"] == "done"
        assert snapshot.read_text(encoding="utf-8") == "<p>second fetch</p>"

    async def test_deleting_the_item_removes_it(
        self, tmp_path: Path, url_conversion: _FakeUrlConversion
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL, "https://example.test/other"])
            snapshots = tmp_path / "jobs" / job_id / "snapshots"
            assert sorted(p.name for p in snapshots.iterdir()) == ["i1.html", "i2.html"]
            deleted = await client.delete(f"/api/jobs/{job_id}/items/i1")
        assert deleted.status_code == 204
        assert [p.name for p in snapshots.iterdir()] == ["i2.html"]


async def test_a_url_conversion_reports_the_page_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``process_url_item`` hands the cascade's ``html`` on as ``page_html``."""
    from markitai.serve.jobs import UrlJobContext, process_url_item
    from markitai.workflow.url import UrlCascadeResult

    out = tmp_path / "page.md"
    out.write_text("# x\n", encoding="utf-8")

    async def cascade(*_args: Any, **_kwargs: Any) -> UrlCascadeResult:
        return UrlCascadeResult(
            markdown="# x", output_path=out, llm_output_path=None, html=PAGE
        )

    monkeypatch.setattr("markitai.workflow.url.convert_url_cascade", cascade)
    cfg = MarkitaiConfig()
    cfg.screenshot.enabled = False
    result = await process_url_item(
        URL,
        cfg,
        tmp_path,
        None,
        UrlJobContext(strategy=None, cache=None, screenshot_dir=None),
    )
    assert result.success and result.page_html == PAGE


# ----------------------------------------------------------------------- 2


class TestRegenerateToken:
    async def test_the_old_token_stops_working_at_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from loguru import logger

        handler = logger.add(caplog.handler, level="INFO", format="{message}")
        app = _make_app(tmp_path, token="old-token")
        try:
            async with _client(app, peer=("192.168.1.30", 40000)) as client:
                old = {"Authorization": "Bearer old-token"}
                ticket = (
                    await client.post(
                        "/api/download-tickets",
                        json={"path": "/api/history/archive"},
                        headers=old,
                    )
                ).json()
                answer = await client.post("/api/access-token/regenerate", headers=old)
                body = answer.json()
                new = {"Authorization": f"Bearer {body['token']}"}
                with_old = await client.get("/api/capabilities", headers=old)
                with_old_query = await client.get("/api/capabilities?token=old-token")
                with_new = await client.get("/api/capabilities", headers=new)
                spent = await client.get(ticket["url"])
                again = await client.post("/api/access-token/regenerate", headers=new)
        finally:
            logger.remove(handler)
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-store"
        assert set(body) == {"token", "network_url", "persistent", "startup_variable"}
        assert body["token"].startswith("mk_") and len(body["token"]) >= 40
        assert body["token"] != "old-token"
        assert body["persistent"] is False
        assert body["startup_variable"] is None
        assert body["network_url"] is None
        assert (with_old.status_code, with_old.json()["reason"]) == (
            401,
            "token_required",
        )
        assert with_old_query.status_code == 401
        assert with_new.status_code == 200
        assert with_new.json()["features"]["token_regenerate"] is True
        assert (spent.status_code, spent.json()["reason"]) == (401, "ticket_invalid")
        assert again.status_code == 200 and again.json()["token"] != body["token"]
        logged = [r.getMessage() for r in caplog.records]
        assert (
            "[Serve] Access token regenerated from the web workbench; other tabs "
            "and devices must sign in again."
        ) in logged
        assert not any(body["token"] in line for line in logged)

    async def test_it_names_the_sign_in_link_and_the_startup_variable(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(
            tmp_path,
            token="pinned",
            network_origin=lambda: "http://192.168.1.5:3600",
            token_variable="MARKITAI_SERVE_TOKEN",
        )
        async with _client(app, headers={"Authorization": "Bearer pinned"}) as client:
            body = (await client.post("/api/access-token/regenerate", json={})).json()
        assert body["network_url"] == f"http://192.168.1.5:3600/#token={body['token']}"
        assert body["startup_variable"] == "MARKITAI_SERVE_TOKEN"
        assert body["persistent"] is False

    async def test_open_event_streams_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = asyncio.Event()

        async def slow(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
            await release.wait()
            return await _pdf_like_converter()(file_path, cfg, out_dir, shared)

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", slow)
        app = _make_app(tmp_path, token="old-token")
        old = {"Authorization": "Bearer old-token"}
        async with _client(app, headers=old) as client:
            created = await client.post(
                "/api/jobs", files=_form(files=[("a.pdf", b"%")])
            )
            job_id = created.json()["job_id"]
            stream = asyncio.create_task(client.get(f"/api/jobs/{job_id}/events"))
            await asyncio.sleep(0.2)
            assert not stream.done()
            regenerated = await client.post("/api/access-token/regenerate")
            ended = await asyncio.wait_for(stream, timeout=5)
            reconnect = await client.get(f"/api/jobs/{job_id}/events")
            release.set()
            new = {"Authorization": f"Bearer {regenerated.json()['token']}"}
            await asyncio.sleep(0)
            done = await client.get(f"/api/jobs/{job_id}", headers=new)
        assert ended.status_code == 200
        assert "event: snapshot" in ended.text
        assert (reconnect.status_code, reconnect.json()["reason"]) == (
            401,
            "token_required",
        )
        assert done.status_code == 200

    async def test_trust_as_for_settings(self, tmp_path: Path) -> None:
        tokened = _make_app(tmp_path / "a", token="the-token")
        async with _client(tokened) as local:
            # Loopback needs the token too, as for the settings
            without = await local.post("/api/access-token/regenerate")
        no_auth = _make_app(tmp_path / "b")
        async with _client(no_auth) as local:
            disabled = await local.post("/api/access-token/regenerate")
            features = (await local.get("/api/capabilities")).json()["features"]
        async with _client(no_auth, peer=("192.168.1.30", 40000)) as remote:
            untrusted = await remote.post("/api/access-token/regenerate")
        async with _client(
            tokened, headers={"Authorization": "Bearer the-token"}
        ) as local:
            foreign = await local.post(
                "/api/access-token/regenerate",
                headers={"Origin": "http://evil.example"},
            )
        assert (without.status_code, without.json()["reason"]) == (
            401,
            "token_required",
        )
        assert (disabled.status_code, disabled.json()["reason"]) == (
            409,
            "token_disabled",
        )
        assert features["token_regenerate"] is False
        assert (untrusted.status_code, untrusted.json()["reason"]) == (
            403,
            "settings_forbidden",
        )
        assert (foreign.status_code, foreign.json()["reason"]) == (
            403,
            "origin_not_allowed",
        )

    @pytest.mark.parametrize(
        ("content", "status", "reason"),
        [
            (b"[]", 422, "invalid_json"),
            (b'{"token": "mine"}', 422, "invalid_json"),
            (b"not json", 422, "invalid_json"),
            (b"{" + b" " * 1100 + b"}", 413, "request_too_large"),
        ],
        ids=["array", "an-object", "not-json", "over-1-kib"],
    )
    async def test_a_body_other_than_nothing_or_an_empty_object_is_refused(
        self, tmp_path: Path, content: bytes, status: int, reason: str
    ) -> None:
        app = _make_app(tmp_path, token="the-token")
        async with _client(
            app, headers={"Authorization": "Bearer the-token"}
        ) as client:
            refused = await client.post(
                "/api/access-token/regenerate",
                content=content,
                headers={"Content-Type": "application/json"},
            )
            still = await client.get("/api/capabilities")
        assert (refused.status_code, refused.json()["reason"]) == (status, reason)
        assert still.status_code == 200
