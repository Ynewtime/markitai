"""Document JSON in ``markitai serve`` (contract v1.19/v1.20).

Every finished item keeps its document (``GET …/document``, what the
workbench's Download menu offers as ``.json``); a job with ``format: "json"``
also gets ``<base>.json`` beside the item's Markdown, one of its files. The
job options ``ocr_lang`` and ``ocr_all_pages`` and the matching convert
defaults. All inputs are invented.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

import httpx

from markitai.config import MarkitaiConfig
from markitai.serve.schemas import JobOptions
from tests.unit.serve.test_serve_item_files import _make_app, _serve_client


def _form(name: str, content: bytes, options: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("files", (name, content, "application/octet-stream")),
        ("urls", (None, "[]")),
        ("options", (None, json.dumps(options))),
    ]


async def _job(client: httpx.AsyncClient, options: dict[str, Any]) -> tuple[str, str]:
    created = await client.post(
        "/api/jobs", files=_form("notes.txt", b"# Notes\n\nPlain words.\n", options)
    )
    assert created.status_code in (200, 201, 202), created.text
    job_id = created.json()["job_id"]
    return job_id, await _finished_item(client, job_id)


async def _finished_item(client: httpx.AsyncClient, job_id: str) -> str:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        data = (await client.get(f"/api/jobs/{job_id}")).json()
        items = data["items"]
        # finished_at comes with the item's event, after its document is kept
        if items and all(item["finished_at"] for item in items):
            return items[0]["item_id"]
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish")


async def test_every_finished_item_keeps_its_document(tmp_path: Path) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id, item_id = await _job(client, {})
        got = await client.get(f"/api/jobs/{job_id}/items/{item_id}/document")
        result = await client.get(f"/api/jobs/{job_id}/items/{item_id}/result")
        missing = await client.get(f"/api/jobs/{job_id}/items/nope/document")
    assert got.status_code == 200, got.text
    assert got.headers["content-type"].startswith("application/json")
    document = got.json()
    assert document["schema_version"] == "1.0"
    # The upload name, never the server's path
    assert document["source"]["path"] == "notes.txt"
    assert document["source"]["size"] == 22
    assert document["status"] == "completed"
    assert document["text"] == "Notes\n\nPlain words."
    assert document["outputs"]["markdown"] == "notes.txt.md"
    # format markdown: no <base>.json among the item's files
    relpaths = [a["relpath"] for a in result.json()["artifacts"]]
    assert relpaths == ["notes.txt.md"]
    assert missing.status_code == 404


async def test_a_failed_item_has_no_document(tmp_path: Path) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        created = await client.post(
            "/api/jobs", files=_form("odd.unknownext", b"?", {})
        )
        job_id = created.json()["job_id"]
        item_id = await _finished_item(client, job_id)
        got = await client.get(f"/api/jobs/{job_id}/items/{item_id}/document")
    assert got.status_code == 404
    assert got.json()["reason"] == "result_unavailable"


async def test_format_json_writes_base_json_that_belongs_to_the_item(
    tmp_path: Path,
) -> None:
    jobs_root = tmp_path / "jobs"
    async with _serve_client(_make_app(tmp_path)) as client:
        job_id, item_id = await _job(client, {"format": "json"})
        result = await client.get(f"/api/jobs/{job_id}/items/{item_id}/result")
        relpaths = [a["relpath"] for a in result.json()["artifacts"]]
        download = await client.get(f"/api/jobs/{job_id}/files/notes.txt.json")
        out = jobs_root / job_id / "out"
        kept = jobs_root / job_id / "documents" / f"{item_id}.json"
        assert (out / "notes.txt.json").is_file() and kept.is_file()
        # A rerun publishing Markdown only takes the old JSON away
        retry = await client.post(
            f"/api/jobs/{job_id}/items/{item_id}/retry",
            json={"options": {"format": "markdown"}},
        )
        assert retry.status_code in (200, 202), retry.text
        deadline = time.monotonic() + 30
        while (out / "notes.txt.json").exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert not (out / "notes.txt.json").exists()
        deleted = await client.delete(f"/api/jobs/{job_id}/items/{item_id}")
        assert deleted.status_code in (200, 204), deleted.text
    assert relpaths[:2] == ["notes.txt.md", "notes.txt.json"]
    assert download.status_code == 200
    assert download.json()["outputs"]["markdown"] == "notes.txt.md"
    assert not kept.exists()


async def test_an_unknown_ocr_language_is_refused(tmp_path: Path) -> None:
    async with _serve_client(_make_app(tmp_path)) as client:
        capabilities = (await client.get("/api/capabilities")).json()
        refused = await client.post(
            "/api/jobs", files=_form("notes.txt", b"x", {"ocr_lang": "klingon"})
        )
        wrong = await client.post(
            "/api/jobs", files=_form("notes.txt", b"x", {"format": "yaml"})
        )
    assert capabilities["ocr_languages"] == [
        "en", "zh", "zh_tw", "ja", "ko", "ru", "ar", "th", "hi", "el",
    ]  # fmt: skip
    assert capabilities["ocr_lang"] == "en"
    assert refused.status_code == 422
    assert refused.json()["reason"] == "invalid_options"
    assert refused.json()["detail"] == (
        "option 'ocr_lang' must be one of: en, zh, zh_tw, ja, ko, ru, ar, th, hi, el"
    )
    assert (wrong.status_code, wrong.json()["reason"]) == (422, "invalid_options")


def test_every_offered_ocr_language_is_one_rapidocr_reads() -> None:
    from markitai.ocr import resolve_ocr_language
    from markitai.serve.convert_defaults import OCR_LANGUAGES

    for code in OCR_LANGUAGES:
        resolve_ocr_language(code)
    assert resolve_ocr_language("ru") == ("eslav", True)
    assert resolve_ocr_language("hi") == ("devanagari", True)


def test_the_options_map_onto_the_configuration() -> None:
    from markitai.serve.app import _build_job_config

    cfg = _build_job_config(
        MarkitaiConfig(),
        JobOptions(format="json", ocr_lang="ko", ocr_all_pages=True),
    )
    assert (cfg.output.format, cfg.ocr.lang, cfg.ocr.per_page_routing) == (
        "json",
        "ko",
        False,
    )
    unchanged = _build_job_config(MarkitaiConfig(), JobOptions())
    assert (unchanged.output.format, unchanged.ocr.per_page_routing) == (
        "markdown",
        True,
    )


def test_a_configured_language_outside_the_list_is_offered_too() -> None:
    from markitai.serve.convert_defaults import ocr_languages

    assert ocr_languages("fr")[-1] == "fr"
    assert ocr_languages("ko").count("ko") == 1
