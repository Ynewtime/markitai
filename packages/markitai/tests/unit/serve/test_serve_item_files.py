"""Which files belong to an item: result artifacts, item deletion, names.

Covers the shared ownership rule in ``markitai.serve.artifacts`` (PDF image
names, page screenshots, URL screenshot references), that deleting an item
removes its upload and assets but never a sibling's, and that per-job
output/upload names are unique case-insensitively.
"""

from __future__ import annotations

import asyncio
import json
import time
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

import httpx

from markitai.config import MarkitaiConfig
from markitai.constants import ASSETS_REL_PATH, SCREENSHOTS_REL_PATH
from markitai.serve import create_app
from markitai.serve.artifacts import claims_asset_name, referenced_assets

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


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
async def _serve_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


def _multipart(
    files: list[tuple[str, bytes]] | None = None, urls: list[str] | None = None
) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = [
        ("files", (name, content, "application/octet-stream"))
        for name, content in files or []
    ]
    parts.append(("urls", (None, json.dumps(urls or []))))
    parts.append(("options", (None, "{}")))
    return parts


async def _wait_job_done(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        if data["status"] == "done":
            return data
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish")


def _pdf_like_converter():
    """Write what a PDF conversion leaves behind, prefixed by the output base.

    ``<base>-0001-01.png`` (pymupdf4llm images adopted under the output
    prefix), ``<base>.page0001.jpg`` (page render) and ``<base>.0001.png``
    (a numbered asset); the markdown references the extracted image.
    """

    async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
        from markitai.batch import ProcessResult

        base = file_path.name
        assets = out_dir / ASSETS_REL_PATH
        shots = out_dir / SCREENSHOTS_REL_PATH
        assets.mkdir(parents=True, exist_ok=True)
        shots.mkdir(parents=True, exist_ok=True)
        (assets / f"{base}-0001-01.png").write_bytes(b"img")
        (assets / f"{base}.0001.png").write_bytes(b"img")
        (shots / f"{base}.page0001.jpg").write_bytes(b"shot")
        out = out_dir / f"{base}.md"
        out.write_text(
            f"# {base}\n\n![](.markitai/assets/{base.replace(' ', '%20')}-0001-01.png)\n",
            encoding="utf-8",
        )
        return ProcessResult(success=True, output_path=str(out))

    return convert


class TestOwnershipRule:
    """serve.artifacts.claims_asset_name / referenced_assets."""

    @pytest.mark.parametrize(
        "name",
        [
            "report.pdf-0001-10.jpg",
            "report.pdf-0001.png",
            "report.pdf.0001.png",
            "report.pdf.page0001.jpg",
            "report.pdf.slide0003.png",
            "report.pdf.full.jpg",
            "report.pdf.full--2.jpg",
        ],
    )
    def test_claims_names_the_converters_write(self, name: str) -> None:
        assert claims_asset_name("report.pdf", name)

    @pytest.mark.parametrize(
        ("base", "name"),
        [
            # a renamed re-run's assets belong to the renamed output
            ("report.pdf", "report.pdf.v2-0001-01.jpg"),
            ("report.pdf", "report.pdf.v2.0001.png"),
            # sibling "a.txt" of item "a"
            ("a", "a.txt.0001.png"),
            ("a", "a.txt-0001-01.png"),
            # sibling "a.pdf-1" (its images are "a.pdf-1-0001-01.png")
            ("a.pdf", "a.pdf-1-0001-01.png"),
            ("a.pdf", "a.pdf.md"),
            ("a.pdf", "a.pdf-notes.png"),
        ],
    )
    def test_never_claims_a_sibling(self, base: str, name: str) -> None:
        assert not claims_asset_name(base, name)

    def test_renamed_output_claims_its_own_assets(self) -> None:
        assert claims_asset_name("report.pdf.v2", "report.pdf.v2-0001-01.jpg")

    def test_referenced_assets_forms(self) -> None:
        markdown = (
            "![a](.markitai/assets/a%20b.pdf-0001-01.png)\n"
            "<!-- ![Screenshot](.markitai/screenshots/example.com_x.full.jpg) -->\n"
            "![[assets/c d.png]]\n"
            "![up](.markitai/assets/../../meta.json)\n"
        )
        assert referenced_assets(markdown) == {
            (ASSETS_REL_PATH, "a b.pdf-0001-01.png"),
            (SCREENSHOTS_REL_PATH, "example.com_x.full.jpg"),
            ("assets", "c d.png"),
        }


class TestResultArtifacts:
    """GET /items/{id}/result lists every file the item owns."""

    async def test_lists_pdf_images_and_page_screenshots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _pdf_like_converter()
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("a.pdf", b"%PDF"), ("a.pdf-1.pdf", b"%PDF")]),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        assert result.status_code == 200
        relpaths = {a["relpath"] for a in result.json()["artifacts"]}
        assert relpaths == {
            "a.pdf.md",
            f"{ASSETS_REL_PATH}/a.pdf-0001-01.png",
            f"{ASSETS_REL_PATH}/a.pdf.0001.png",
            f"{SCREENSHOTS_REL_PATH}/a.pdf.page0001.jpg",
        }

    async def test_lists_url_screenshot_named_after_the_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_url(
            url: str,
            cfg: Any,
            out_dir: Path,
            shared: Any,
            url_ctx: Any,
            output_name: str | None = None,
        ):
            from markitai.batch import ProcessResult

            shots = out_dir / SCREENSHOTS_REL_PATH
            shots.mkdir(parents=True, exist_ok=True)
            (shots / "example.com_docs_guide.full.jpg").write_bytes(b"shot")
            out = out_dir / (output_name or "guide.md")
            out.write_text(
                "# guide\n\n<!-- ![Screenshot]"
                "(.markitai/screenshots/example.com_docs_guide.full.jpg) -->\n",
                encoding="utf-8",
            )
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake_url)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs", files=_multipart(urls=["https://example.com/docs/guide"])
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            result = await client.get(f"/api/jobs/{job_id}/items/i1/result")
        relpaths = {a["relpath"] for a in result.json()["artifacts"]}
        assert relpaths == {
            "guide.md",
            f"{SCREENSHOTS_REL_PATH}/example.com_docs_guide.full.jpg",
        }


class TestDeleteItemFiles:
    """DELETE /items/{id} removes the upload, outputs and owned assets."""

    async def test_removes_upload_and_assets_but_keeps_siblings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.serve.jobs.process_file_item", _pdf_like_converter()
        )
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("a.pdf", b"%PDF"), ("a.pdf-1.pdf", b"%PDF"), ("a", b"x")]
                ),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            before = await client.get(f"/api/jobs/{job_id}/archive")
            assert before.status_code == 200

            deleted = await client.delete(f"/api/jobs/{job_id}/items/i1")
            assert deleted.status_code == 204

            remaining = {
                p.relative_to(job.job_dir).as_posix()
                for p in job.job_dir.rglob("*")
                if p.is_file() and p.name != "meta.json"
            }
            assert remaining == {
                "uploads/a.pdf-1.pdf",
                "uploads/a",
                "out/a.pdf-1.pdf.md",
                f"out/{ASSETS_REL_PATH}/a.pdf-1.pdf-0001-01.png",
                f"out/{ASSETS_REL_PATH}/a.pdf-1.pdf.0001.png",
                f"out/{SCREENSHOTS_REL_PATH}/a.pdf-1.pdf.page0001.jpg",
                "out/a.md",
                f"out/{ASSETS_REL_PATH}/a-0001-01.png",
                f"out/{ASSETS_REL_PATH}/a.0001.png",
                f"out/{SCREENSHOTS_REL_PATH}/a.page0001.jpg",
            }

            archive = await client.get(f"/api/jobs/{job_id}/archive")
            with zipfile.ZipFile(BytesIO(archive.content)) as zf:
                names = set(zf.namelist())
        assert not any(name.startswith("a.pdf.") for name in names)
        assert not any("/a.pdf-0001" in name or "/a.pdf." in name for name in names)
        assert "a.pdf-1.pdf.md" in names

    async def test_keeps_a_screenshot_a_sibling_still_references(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two URL items of one page share its URL-named screenshot."""

        async def fake_url(
            url: str,
            cfg: Any,
            out_dir: Path,
            shared: Any,
            url_ctx: Any,
            output_name: str | None = None,
        ):
            from markitai.batch import ProcessResult

            shots = out_dir / SCREENSHOTS_REL_PATH
            shots.mkdir(parents=True, exist_ok=True)
            (shots / "example.com_page.full.jpg").write_bytes(b"shot")
            out = out_dir / (output_name or "page.md")
            out.write_text(
                "<!-- ![Screenshot](.markitai/screenshots/example.com_page.full.jpg) -->",
                encoding="utf-8",
            )
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake_url)
        app = _make_app(tmp_path)
        url = "https://example.com/page"
        async with _serve_client(app) as client:
            created = await client.post("/api/jobs", files=_multipart(urls=[url, url]))
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            shot = job.out_dir / SCREENSHOTS_REL_PATH / "example.com_page.full.jpg"

            assert (
                await client.delete(f"/api/jobs/{job_id}/items/i1")
            ).status_code == 204
            assert not (job.out_dir / "page.md").exists()
            assert shot.is_file()  # still referenced by i2
            assert (job.out_dir / "page (2).md").is_file()

            # With no sibling left to reference it, the last delete drops the
            # whole job directory.
            assert (
                await client.delete(f"/api/jobs/{job_id}/items/i2")
            ).status_code == 204
        assert not job.job_dir.exists()

    async def test_failed_item_delete_removes_upload_and_partial_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            if file_path.name == "bad.pdf":
                assets = out_dir / ASSETS_REL_PATH
                assets.mkdir(parents=True, exist_ok=True)
                (assets / "bad.pdf-0001-01.png").write_bytes(b"img")
                return ProcessResult(success=False, error="boom")
            out = out_dir / f"{file_path.name}.md"
            out.write_text("ok", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        app = _make_app(tmp_path)
        async with _serve_client(app) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(files=[("bad.pdf", b"x"), ("good.txt", b"y")]),
            )
            job_id = created.json()["job_id"]
            await _wait_job_done(client, job_id)
            job = app.state.markitai.registry.get(job_id)
            assert job is not None
            assert (
                await client.delete(f"/api/jobs/{job_id}/items/i1")
            ).status_code == 204
            assert not (job.uploads_dir / "bad.pdf").exists()
            assert not (job.out_dir / ASSETS_REL_PATH / "bad.pdf-0001-01.png").exists()
            assert (job.uploads_dir / "good.txt").is_file()
            assert (job.out_dir / "good.txt.md").is_file()


class TestCaseInsensitiveNames:
    """Per-job output and upload names never differ only by case."""

    async def test_url_outputs_differing_only_by_case_are_deconflicted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names: list[str | None] = []

        async def fake_url(
            url: str,
            cfg: Any,
            out_dir: Path,
            shared: Any,
            url_ctx: Any,
            output_name: str | None = None,
        ):
            from markitai.batch import ProcessResult

            names.append(output_name)
            out = out_dir / (output_name or "x.md")
            out.write_text(url, encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            out = out_dir / f"{file_path.name}.md"
            out.write_text("upload", encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake_url)
        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("PAGE.html", b"<p>x</p>")],
                    urls=[
                        "https://example.com/Page.html",
                        "https://example.org/page.html",
                    ],
                ),
            )
            job_id = created.json()["job_id"]
            data = await _wait_job_done(client, job_id)
        outputs = [item["output_name"] for item in data["items"][1:]]
        # Neither URL may share the upload's "PAGE.html.md" (nor each other's
        # name) on a case-insensitive file system.
        assert outputs == ["Page.html (2).md", "page.html (3).md"]
        assert len({name.casefold() for name in outputs if name}) == 2
        assert sorted(n for n in names if n) == sorted(outputs)

    async def test_uploads_differing_only_by_case_are_deconflicted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def convert(file_path: Path, cfg: Any, out_dir: Path, shared: Any):
            from markitai.batch import ProcessResult

            out = out_dir / f"{file_path.name}.md"
            out.write_text(file_path.read_text(), encoding="utf-8")
            return ProcessResult(success=True, output_path=str(out))

        monkeypatch.setattr("markitai.serve.jobs.process_file_item", convert)
        async with _serve_client(_make_app(tmp_path)) as client:
            created = await client.post(
                "/api/jobs",
                files=_multipart(
                    files=[("Report.txt", b"one"), ("report.txt", b"two")]
                ),
            )
            body = created.json()
            data = await _wait_job_done(client, body["job_id"])
        assert [item["name"] for item in body["items"]] == [
            "Report.txt",
            "report (2).txt",
        ]
        assert [item["output"] for item in data["items"]] == [
            "Report.txt.md",
            "report (2).txt.md",
        ]
