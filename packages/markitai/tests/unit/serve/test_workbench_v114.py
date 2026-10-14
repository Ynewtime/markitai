"""Workbench API contract v1.14: a web page's capture, kept for Compare.

- ``capture`` on a URL item's original descriptor, recorded as
  ``snapshots/<item_id>.capture.json`` (an older job's kept page is judged
  when asked);
- the screenshots' block boxes, ``snapshots/<item_id>.blocks.json``, served
  by ``…/original/blocks`` and named by the descriptor's ``blocks``;
- compare screenshots ``snapshots/<item_id>.shots/<n>.jpg`` when a browser
  loaded the page and no screenshots were asked for: the item's pages, never
  output, archived or listed in a result;
- a retry replaces them, deleting removes them;
- the framed page's ``<base href>`` is the item's URL also for a job read
  back from disk (it was ``None``).

The conversion is faked; it fills the capture sink the way the browser fetch
does (``tests/unit/test_fetch_page_capture.py`` covers the browser side).
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from markitai.constants import SCREENSHOTS_REL_PATH
from markitai.page_capture import CAPTURE, ShotView, page_quality, sized_blocks
from tests.unit.serve.test_workbench_contract import check_error, check_shape
from tests.unit.serve.test_workbench_v111 import (
    META,
    _allow_every_target,
    _client,
    _job,
    _make_app,
    _settled,
)

URL = "https://example.test/notes/river?day=3&x=<1>"
PAGE = (
    "<!doctype html><html><head><title>River notes</title></head><body><article>"
    "<h1>River notes</h1><p>"
    + "The gauge at the old mill read the same at dawn and at dusk. " * 3
    + '</p><img src="pictures/mill.jpg" alt="The mill"></article></body></html>'
)
OK = {"state": "ok", "text_length": 200, "title": "River notes", "reason": None}


def _jpeg(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 150)).save(buffer, "JPEG")
    return buffer.getvalue()


def _blocks(width: int, height: int) -> bytes:
    value = {
        "version": 1,
        "width": 0,
        "height": 0,
        "blocks": [
            {"tag": "h1", "text": "River notes", "bbox": [8, 20, 1272, 60]},
            {
                "tag": "img",
                "text": "The mill",
                "src": "https://example.test/notes/pictures/mill.jpg",
                "bbox": [8, 2100, 408, 2220],
            },
        ],
    }
    data = sized_blocks(value, width, height)
    assert data is not None
    return data


class _FakeCapture:
    """``serve.jobs.process_url_item`` stand-in that loads "the page" as the
    browser fetch would: it writes the Markdown, maybe screenshots of its
    own, and fills the capture sink."""

    def __init__(self) -> None:
        self.html: str | None = PAGE
        self.fail = False
        #: Tile heights of the conversion's own screenshots (none: compare)
        self.own: list[int] = []
        #: Tile heights of compare screenshots (none: the browser loaded nothing)
        self.compare: list[int] = [2000, 2000, 500]
        self.blocks = True
        self.quality: dict[str, Any] = dict(OK)
        self.refused: tuple[str, dict[str, Any]] | None = None
        self.sinks: list[Any] = []

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

        sink = CAPTURE.get()
        self.sinks.append(sink)
        if self.fail:
            return ProcessResult(success=False, error="fetch failed")
        name = output_name or "page.md"
        out = out_dir / name
        out.write_text(f"# River notes\n\n{url}\n", encoding="utf-8")
        assert sink is not None
        sink.view, sink.refused = None, self.refused
        if self.own:
            shots = out_dir / SCREENSHOTS_REL_PATH
            shots.mkdir(parents=True, exist_ok=True)
            stem = name.removesuffix(".md")
            paths = [
                shots / (f"{stem}.full.jpg" if i == 0 else f"{stem}.full--{i}.jpg")
                for i in range(len(self.own))
            ]
            for path, height in zip(paths, self.own, strict=True):
                path.write_bytes(_jpeg(1920, height))
            out.write_text(
                out.read_text(encoding="utf-8")
                + "".join(f"\n![](.markitai/screenshots/{p.name})\n" for p in paths),
                encoding="utf-8",
            )
            sink.view = ShotView(
                quality=self.quality,
                blocks=_blocks(1920, sum(self.own)) if self.blocks else None,
                screenshots=paths,
            )
        elif self.compare:
            sink.view = ShotView(
                quality=self.quality,
                blocks=_blocks(1280, sum(self.compare)) if self.blocks else None,
                compare=[_jpeg(1280, height) for height in self.compare],
            )
        return ProcessResult(
            success=True,
            output_path=str(out),
            page_html=self.html,
            screenshots=len(self.own),
        )


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> _FakeCapture:
    fake = _FakeCapture()
    monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake)
    monkeypatch.setattr(
        "markitai.serve.app._guard_fetch_targets", _allow_every_target, raising=True
    )
    return fake


def _prefix(job_id: str, item_id: str = "i1") -> str:
    return f"/api/jobs/{job_id}/items/{item_id}"


def _snapshots(tmp_path: Path, job_id: str) -> list[str]:
    folder = tmp_path / "jobs" / job_id / "snapshots"
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


class TestCompareScreenshots:
    async def test_a_page_loaded_without_screenshots_keeps_compare_ones(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            prefix = _prefix(job_id)
            descriptor = (await client.get(f"{prefix}/original")).json()
            blocks = await client.get(f"{prefix}/original/blocks")
            second = await client.get(f"{prefix}/original/pages/2?width=1400")
            result = (await client.get(f"{prefix}/result")).json()
            item_zip = await client.get(f"{prefix}/archive")
            job_zip = await client.get(f"/api/jobs/{job_id}/archive")
        # The sink was there, with the configured tile height
        assert capture.sinks[0].tile_height == 2000
        check_shape("OriginalDescriptor", descriptor)
        assert descriptor["kind"] == "html"
        assert descriptor["pages"] == [
            {"width": 1280.0, "height": 2000.0},
            {"width": 1280.0, "height": 2000.0},
            {"width": 1280.0, "height": 500.0},
        ]
        assert descriptor["page_count"] == 3
        assert descriptor["blocks"] == f"{prefix}/original/blocks"
        assert descriptor["capture"] == {
            "page": page_quality(URL, PAGE),
            "screenshot": OK,
        }
        assert descriptor["capture"]["page"]["state"] == "ok"
        # The block boxes as recorded, never cached
        assert blocks.status_code == 200
        assert blocks.headers["content-type"] == "application/json"
        assert blocks.headers["cache-control"] == "no-store"
        check_shape("ShotBlocks", blocks.json())
        assert (blocks.json()["width"], blocks.json()["height"]) == (1280, 4500)
        assert second.status_code == 200
        assert second.headers["content-type"] == "image/jpeg"
        # Not output: not in out/, the result or an archive
        assert _snapshots(tmp_path, job_id) == [
            "i1.blocks.json",
            "i1.capture.json",
            "i1.html",
            "i1.shots",
            "i1.shots/1.jpg",
            "i1.shots/2.jpg",
            "i1.shots/3.jpg",
        ]
        assert not list((tmp_path / "jobs" / job_id / "out").rglob("*.jpg"))
        assert not any(".shots" in a["relpath"] for a in result["artifacts"])
        for archive in (item_zip, job_zip):
            with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
                assert not [n for n in zf.namelist() if n.endswith(".jpg")]
                assert not [n for n in zf.namelist() if "capture" in n]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
    async def test_kept_files_are_private(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
        folder = tmp_path / "jobs" / job_id / "snapshots"
        for path in folder.rglob("*"):
            mode = os.stat(path).st_mode & 0o777
            assert mode & 0o077 == 0, (path.name, oct(mode))

    async def test_own_screenshots_carry_the_boxes_and_no_copy(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        capture.own = [2000, 700]
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["pages"] == [
            {"width": 1920.0, "height": 2000.0},
            {"width": 1920.0, "height": 700.0},
        ]
        assert descriptor["blocks"] == f"{_prefix(job_id)}/original/blocks"
        assert descriptor["capture"]["screenshot"] == OK
        assert _snapshots(tmp_path, job_id) == [
            "i1.blocks.json",
            "i1.capture.json",
            "i1.html",
        ]

    async def test_boxes_that_do_not_cover_the_pages_are_not_offered(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        capture.own = [2000, 700]
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            boxes = tmp_path / "jobs" / job_id / "snapshots" / "i1.blocks.json"
            boxes.write_bytes(_blocks(1920, 5000))
            client._transport.app.state.original_cache.clear()  # type: ignore[attr-defined]
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["blocks"] is None

    async def test_without_boxes_the_route_is_404(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        capture.blocks = False
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
            missing = await client.get(f"{_prefix(job_id)}/original/blocks")
        assert descriptor["blocks"] is None
        assert descriptor["page_count"] == 3
        check_error(missing, 404, "original_unavailable")

    async def test_a_page_with_no_browser_load_has_no_screenshot(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        capture.compare = []
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["kind"] == "html"
        assert descriptor["pages"] == []
        assert descriptor["capture"]["screenshot"] is None
        assert descriptor["blocks"] is None
        assert _snapshots(tmp_path, job_id) == ["i1.capture.json", "i1.html"]

    async def test_an_upload_has_no_capture(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, files=[("a.md", b"# A\n")])
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["capture"] is None
        assert descriptor["blocks"] is None


class TestCaptureQuality:
    async def test_a_verification_page_is_kept_aside_to_say_why(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        """The text came from elsewhere; the browser showed a captcha."""
        captcha = '<html><head><title>Video</title></head><body><div id="risk-captcha-app"></div></body></html>'
        verdict = page_quality("https://www.bilibili.com/video/BV0", captcha)
        capture.html = None
        capture.compare = []
        capture.refused = (captcha, verdict)
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["kind"] == "html"
        assert descriptor["capture"] == {
            "page": {
                "state": "challenge",
                "text_length": 0,
                "title": "Video",
                "reason": "verification",
            },
            "screenshot": None,
        }

    async def test_a_screenshot_verdict_is_recorded_as_measured(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        capture.quality = {
            "state": "empty",
            "text_length": 12,
            "title": None,
            "reason": "blank",
        }
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        record = json.loads(
            (tmp_path / "jobs" / job_id / "snapshots" / "i1.capture.json").read_text()
        )
        assert record == {
            "version": 1,
            "page": page_quality(URL, PAGE),
            "screenshot": capture.quality,
        }
        assert descriptor["capture"]["screenshot"] == capture.quality

    @pytest.mark.parametrize(
        "record",
        [None, '{"version": 2}', '{"version": 1, "page": {"state": "odd"}}', "{"],
    )
    async def test_an_older_job_has_its_kept_page_judged(
        self, tmp_path: Path, capture: _FakeCapture, record: str | None
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            path = tmp_path / "jobs" / job_id / "snapshots" / "i1.capture.json"
            if record is None:
                path.unlink()
            else:
                path.write_text(record, encoding="utf-8")
            client._transport.app.state.original_cache.clear()  # type: ignore[attr-defined]
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["capture"] == {
            "page": page_quality(URL, PAGE),
            "screenshot": None,
        }

    async def test_a_record_written_by_the_other_backend_is_read(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            path = tmp_path / "jobs" / job_id / "snapshots" / "i1.capture.json"
            challenge = {
                "state": "challenge",
                "text_length": 41,
                "title": "Just a moment...",
                "reason": "challenge",
            }
            path.write_text(
                json.dumps({"version": 1, "page": challenge, "screenshot": None}),
                encoding="utf-8",
            )
            client._transport.app.state.original_cache.clear()  # type: ignore[attr-defined]
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert descriptor["capture"] == {"page": challenge, "screenshot": None}


class TestRetryAndDelete:
    async def test_a_retry_replaces_everything_it_captured_again(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            capture.compare = [2000, 300]
            capture.blocks = False
            capture.html = "<html><body><p>second load</p></body></html>"
            await client.post(f"{_prefix(job_id)}/retry")
            await _settled(client, job_id)
            replaced = _snapshots(tmp_path, job_id)
            html = (tmp_path / "jobs" / job_id / "snapshots" / "i1.html").read_text()
            # A failed retry keeps them all
            capture.fail = True
            await client.post(f"{_prefix(job_id)}/retry")
            await _settled(client, job_id)
            after_failure = _snapshots(tmp_path, job_id)
            # A retry that loaded no page (a cache hit, a remote reader)
            # leaves the kept ones
            capture.fail = False
            capture.html = None
            capture.compare = []
            await client.post(f"{_prefix(job_id)}/retry")
            await _settled(client, job_id)
            unchanged = _snapshots(tmp_path, job_id)
            # One with its own screenshots drops the compare copies
            capture.html = PAGE
            capture.own = [900]
            capture.blocks = True
            await client.post(f"{_prefix(job_id)}/retry")
            await _settled(client, job_id)
            own = _snapshots(tmp_path, job_id)
        assert replaced == [
            "i1.capture.json",
            "i1.html",
            "i1.shots",
            "i1.shots/1.jpg",
            "i1.shots/2.jpg",
        ]
        assert "second load" in html
        assert after_failure == replaced
        assert unchanged == replaced
        assert own == ["i1.blocks.json", "i1.capture.json", "i1.html"]

    async def test_a_retry_shown_a_check_page_keeps_the_earlier_page(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        """The text came from the cache and the screenshot browser met a
        verification page: the good page and its screenshots stay."""
        captcha = '<html><head><title>Video</title></head><body><div id="risk-captcha-app"></div></body></html>'
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            before = _snapshots(tmp_path, job_id)
            folder = tmp_path / "jobs" / job_id / "snapshots"
            record = (folder / "i1.capture.json").read_bytes()
            capture.html = None
            capture.compare = []
            capture.refused = (
                captcha,
                page_quality("https://www.bilibili.com/video/BV0", captcha),
            )
            await client.post(f"{_prefix(job_id)}/retry")
            await _settled(client, job_id)
            after = _snapshots(tmp_path, job_id)
            html = (folder / "i1.html").read_text(encoding="utf-8")
            client._transport.app.state.original_cache.clear()  # type: ignore[attr-defined]
            descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        assert after == before
        assert html == PAGE
        assert (folder / "i1.capture.json").read_bytes() == record
        assert descriptor["capture"]["page"]["state"] == "ok"

    async def test_deleting_an_item_removes_its_kept_files(
        self, tmp_path: Path, capture: _FakeCapture
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(
                client, urls=[URL, "https://example.test/b", "https://example.test/c"]
            )
            deleted = await client.delete(f"{_prefix(job_id, 'i1')}")
            batch = await client.post(
                "/api/history/delete",
                json={"jobs": [], "items": [{"job_id": job_id, "item_id": "i2"}]},
            )
        assert deleted.status_code == 204
        assert batch.json()["results"][0]["ok"] is True
        assert _snapshots(tmp_path, job_id) == [
            "i3.blocks.json",
            "i3.capture.json",
            "i3.html",
            "i3.shots",
            "i3.shots/1.jpg",
            "i3.shots/2.jpg",
            "i3.shots/3.jpg",
        ]


async def test_a_job_read_back_from_disk_frames_its_page_at_its_url(
    tmp_path: Path, capture: _FakeCapture
) -> None:
    """Round 13: a rehydrated item has no source; the base was ``None``."""
    async with _client(_make_app(tmp_path)) as client:
        job_id = await _job(client, urls=[URL])
        live = await client.get(f"{_prefix(job_id)}/original/view")
    async with _client(_make_app(tmp_path)) as client:
        view = await client.get(f"{_prefix(job_id)}/original/view")
        descriptor = (await client.get(f"{_prefix(job_id)}/original")).json()
        blocks = await client.get(f"{_prefix(job_id)}/original/blocks")
    escaped = URL.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    base = f'{META}<base href="{escaped}" target="_blank">'
    assert view.status_code == 200
    assert view.text == live.text
    assert base in view.text
    assert 'href="None"' not in view.text
    assert descriptor["capture"]["screenshot"] == OK
    assert descriptor["blocks"] is not None and blocks.status_code == 200
