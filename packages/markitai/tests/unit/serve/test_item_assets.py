"""GET /api/jobs/{job_id}/items/{item_id}/assets/{relpath} (contract v1.6, 6.2).

The workbench draws a conversion's own extracted images through this route:
only a raster image the item's ``…/result`` lists, inline and sandboxed,
from inside the job's out dir. Everything else is 404 ``file_not_found``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from markitai.constants import ASSETS_REL_PATH, SCREENSHOTS_REL_PATH
from tests.unit.serve.test_serve_item_files import (
    _make_app,
    _multipart,
    _pdf_like_converter,
    _serve_client,
    _wait_job_done,
)

IMAGE = f"{ASSETS_REL_PATH}/a.pdf-0001-01.png"


async def _job(client: Any, files: list[tuple[str, bytes]] | None = None) -> str:
    created = await client.post(
        "/api/jobs",
        files=_multipart(files=files or [("a.pdf", b"%PDF"), ("b.pdf", b"%PDF")]),
    )
    job_id = created.json()["job_id"]
    await _wait_job_done(client, job_id)
    return job_id


@pytest.fixture
def pdf_like(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("markitai.serve.jobs.process_file_item", _pdf_like_converter())


def _out_dir(tmp_path: Path, job_id: str) -> Path:
    return tmp_path / "jobs" / job_id / "out"


@pytest.mark.usefixtures("pdf_like")
async def test_an_item_image_is_served_inline_and_sandboxed(tmp_path: Path) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client)
        result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        listed = {a["relpath"] for a in result.json()["artifacts"]}
        response = await client.get(f"/api/jobs/{job_id}/items/i1/assets/{IMAGE}")
        capabilities = (await client.get("/api/capabilities")).json()
    assert IMAGE in listed
    assert response.status_code == 200
    assert response.content == b"img"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"] == (
        "inline; filename*=UTF-8''a.pdf-0001-01.png"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.headers["etag"].startswith('W/"')
    assert capabilities["features"]["item_assets"] is True


@pytest.mark.usefixtures("pdf_like")
async def test_every_listed_image_is_served_and_page_renders_too(
    tmp_path: Path,
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client)
        result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        images = [
            a["relpath"]
            for a in result.json()["artifacts"]
            if not a["relpath"].endswith(".md")
        ]
        answers = [
            await client.get(f"/api/jobs/{job_id}/items/i1/assets/{relpath}")
            for relpath in images
        ]
    assert f"{SCREENSHOTS_REL_PATH}/a.pdf.page0001.jpg" in images
    assert [a.status_code for a in answers] == [200] * len(images)
    assert {a.headers["content-type"] for a in answers} == {"image/png", "image/jpeg"}


@pytest.mark.usefixtures("pdf_like")
@pytest.mark.parametrize(
    "header", ["{etag}", "W/{bare}", '"other", {etag}', "*"], ids=str
)
async def test_a_matching_validator_is_304_without_a_body(
    tmp_path: Path, header: str
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client)
        url = f"/api/jobs/{job_id}/items/i1/assets/{IMAGE}"
        etag = (await client.get(url)).headers["etag"]
        sent = header.format(etag=etag, bare=etag.removeprefix("W/"))
        again = await client.get(url, headers={"If-None-Match": sent})
        changed = await client.get(url, headers={"If-None-Match": '"stale"'})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["etag"] == etag
    assert changed.status_code == 200


@pytest.mark.usefixtures("pdf_like")
@pytest.mark.parametrize(
    "relpath",
    [
        # Markdown, not an image
        "a.pdf.md",
        # another item's image
        f"{ASSETS_REL_PATH}/b.pdf-0001-01.png",
        # not on disk
        f"{ASSETS_REL_PATH}/a.pdf-0009-01.png",
        # escapes and odd segments
        f"{ASSETS_REL_PATH}/../a.pdf.md",
        f"{ASSETS_REL_PATH}/%2e%2e/a.pdf-0001-01.png",
        f"{ASSETS_REL_PATH}//a.pdf-0001-01.png",
        f"{ASSETS_REL_PATH}\\a.pdf-0001-01.png",
        "%2Fetc%2Fhostname.png",
    ],
)
async def test_anything_but_the_items_own_raster_image_is_not_found(
    tmp_path: Path, relpath: str
) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client)
        response = await client.get(f"/api/jobs/{job_id}/items/i1/assets/{relpath}")
    assert response.status_code == 404
    assert response.json()["reason"] == "file_not_found"


async def test_svg_and_other_files_are_never_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
        from markitai.batch import ProcessResult

        assets = out_dir / ASSETS_REL_PATH
        assets.mkdir(parents=True, exist_ok=True)
        for name in ("a.pdf.0001.svg", "a.pdf.0002.bmp", "a.pdf.0003.png"):
            (assets / name).write_bytes(b"<svg onload=alert(1)/>")
        out = out_dir / "a.pdf.md"
        out.write_text(
            "![](.markitai/assets/a.pdf.0001.svg)\n![](.markitai/assets/a.pdf.0002.bmp)\n",
            encoding="utf-8",
        )
        return ProcessResult(success=True, output_path=str(out))

    monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client, [("a.pdf", b"%PDF")])
        base = f"/api/jobs/{job_id}/items/i1/assets/{ASSETS_REL_PATH}"
        svg = await client.get(f"{base}/a.pdf.0001.svg")
        bmp = await client.get(f"{base}/a.pdf.0002.bmp")
        png = await client.get(f"{base}/a.pdf.0003.png")
    assert (svg.status_code, bmp.status_code) == (404, 404)
    # Only the bytes are served, as an image the page cannot run
    assert png.status_code == 200
    assert png.headers["content-security-policy"] == "default-src 'none'; sandbox"


@pytest.mark.usefixtures("pdf_like")
@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
async def test_a_symlink_named_like_the_items_image_is_not_followed(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"outside")
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client)
        image = _out_dir(tmp_path, job_id) / IMAGE
        image.unlink()
        image.symlink_to(secret)
        response = await client.get(f"/api/jobs/{job_id}/items/i1/assets/{IMAGE}")
    assert response.status_code == 404


async def test_an_item_without_a_result_serves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
        from markitai.batch import ProcessResult

        assets = out_dir / ASSETS_REL_PATH
        assets.mkdir(parents=True, exist_ok=True)
        (assets / "a.pdf-0001-01.png").write_bytes(b"img")
        return ProcessResult(success=False, error="broken PDF")

    monkeypatch.setattr("markitai.serve.jobs.process_file_item", fail)
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client, [("a.pdf", b"%PDF")])
        failed = await client.get(f"/api/jobs/{job_id}/items/i1/assets/{IMAGE}")
        no_item = await client.get(f"/api/jobs/{job_id}/items/i9/assets/{IMAGE}")
        no_job = await client.get(f"/api/jobs/000000000000/items/i1/assets/{IMAGE}")
    assert (failed.status_code, failed.json()["reason"]) == (404, "file_not_found")
    assert (no_item.status_code, no_item.json()["reason"]) == (404, "item_not_found")
    assert (no_job.status_code, no_job.json()["reason"]) == (404, "job_not_found")


@pytest.mark.usefixtures("pdf_like")
async def test_no_download_ticket_names_it(tmp_path: Path) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id = await _job(client)
        response = await client.post(
            "/api/download-tickets",
            json={"path": f"/api/jobs/{job_id}/items/i1/assets/{IMAGE}"},
        )
    assert response.status_code == 422
    assert response.json()["reason"] == "invalid_ticket_path"
