"""Serve items say where each PDF page's text came from (``page_sources``).

Beyond the workbench contract (the UI ignores it): listed in
``test_contract_sync.PYTHON_EXTRAS``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from markitai.batch import ProcessResult
from markitai.converter.pdf_text_trust import NO_TEXT_LAYER, TRUSTED
from markitai.serve.jobs import (
    JobItem,
    RetryWork,
    _item_from_payload,
    _restore_prior_result,
)
from markitai.serve.schemas import ItemPayload
from tests import text_layer_pdfs as fixtures
from tests.unit.serve.test_serve_api import (
    _make_app,
    _multipart,
    _serve_client,
    _wait_job_done,
)

SOURCES = [
    {"page": 1, "source": "text", "reason": TRUSTED},
    {"page": 2, "source": "text", "reason": NO_TEXT_LAYER},
]


async def test_a_pdf_item_lists_its_pages(tmp_path: Path) -> None:
    pdf = fixtures.mapped_helvetica_pdf(
        tmp_path / "doc.pdf", [fixtures.LATIN_LINES, []], {}
    )
    async with _serve_client(_make_app(tmp_path)) as client:
        resp = await client.post(
            "/api/jobs",
            files=_multipart(
                files=[("doc.pdf", pdf.read_bytes()), ("a.txt", b"hello")]
            ),
        )
        assert resp.status_code == 201, resp.text
        snapshot = await _wait_job_done(client, resp.json()["job_id"])
    items = {item["name"]: item for item in snapshot["items"]}
    assert items["doc.pdf"]["status"] == "done"
    assert items["doc.pdf"]["page_sources"] == SOURCES
    assert items["a.txt"]["page_sources"] == []
    for item in snapshot["items"]:
        ItemPayload.model_validate(item)


def test_saved_payload_round_trips(tmp_path: Path) -> None:
    item = JobItem(
        item_id="i1", name="doc.pdf", kind="file", source=None, status="done"
    )
    item.page_sources = [dict(entry) for entry in SOURCES]
    payload = item.to_payload()
    assert payload["page_sources"] == SOURCES
    assert _item_from_payload(payload, 1).page_sources == SOURCES
    # A meta.json from before this field, or a damaged entry
    del payload["page_sources"]
    assert _item_from_payload(payload, 1).page_sources == []
    payload["page_sources"] = [{"page": 1, "source": "scan", "reason": "x"}, 3]
    assert _item_from_payload(payload, 1).page_sources == []


def test_a_failed_rerun_restores_them(tmp_path: Path) -> None:
    from markitai.config import MarkitaiConfig

    item = JobItem(item_id="i1", name="doc.pdf", kind="file", source=None)
    work = RetryWork(
        item_id="i1",
        cfg=MarkitaiConfig(),
        prior_output="doc.pdf.md",
        prior_page_sources=[dict(entry) for entry in SOURCES],
    )
    _restore_prior_result(None, item, work)  # type: ignore[arg-type]
    assert item.page_sources == SOURCES


def test_result_sets_them() -> None:
    from markitai.serve.jobs import _apply_result

    item = JobItem(item_id="i1", name="doc.pdf", kind="file", source=None)
    job = type("J", (), {"out_dir": Path("/nonexistent")})()
    _apply_result(job, item, ProcessResult(success=True, page_sources=SOURCES))  # type: ignore[arg-type]
    assert item.page_sources == SOURCES
    _apply_result(job, item, ProcessResult(success=False, error="boom"))  # type: ignore[arg-type]
    assert item.page_sources == []
