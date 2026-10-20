"""Where each PDF page's text came from, from the converter to every output.

The converter records ``metadata["page_sources"]``; the Python API, the
batch ProcessResult, ``--json`` items and recorded history carry it on.
Synthetic PDFs only (``tests.text_layer_pdfs``).
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from markitai.api import convert
from markitai.config import MarkitaiConfig
from markitai.converter.base import ConvertResult
from markitai.converter.pdf_text_trust import NO_TEXT_LAYER, TRUSTED
from markitai.runs import json_output
from markitai.runs.types import Outcome
from markitai.workflow.core import ConversionContext, ConversionStepResult
from markitai.workflow.results import document_process_result
from tests import text_layer_pdfs as fixtures

SOURCES = [
    {"page": 1, "source": "text", "reason": TRUSTED},
    {"page": 2, "source": "text", "reason": NO_TEXT_LAYER},
]


def _pdf(tmp_path: Path) -> Path:
    """Page 1 a correct text layer, page 2 none at all."""
    return fixtures.mapped_helvetica_pdf(
        tmp_path / "doc.pdf", [fixtures.LATIN_LINES, []], {}
    )


def test_api_output_lists_each_page(tmp_path: Path) -> None:
    out = convert(_pdf(tmp_path), config=MarkitaiConfig())
    assert out.page_sources == SOURCES


def test_api_output_is_empty_for_other_formats(tmp_path: Path) -> None:
    text = tmp_path / "a.txt"
    text.write_text("hello", encoding="utf-8")
    assert convert(text, config=MarkitaiConfig()).page_sources == []


def test_process_result_carries_them(tmp_path: Path) -> None:
    ctx = ConversionContext(
        input_path=tmp_path / "doc.pdf", output_dir=tmp_path, config=MarkitaiConfig()
    )
    ctx.conversion_result = ConvertResult(
        markdown="# x", metadata={"page_sources": SOURCES}
    )
    ctx.output_file = tmp_path / "doc.pdf.md"
    ctx.output_file.write_text("# x", encoding="utf-8")
    result = document_process_result(ctx, ConversionStepResult(success=True))
    assert result.page_sources == SOURCES


def test_json_item_renders_them() -> None:
    outcome = Outcome(
        kind="file", source="a.pdf", status="completed", page_sources=SOURCES
    )
    envelope = json_output.build_envelope([outcome])
    assert envelope["items"][0]["page_sources"] == SOURCES
    plain = Outcome(kind="file", source="a.txt", status="completed")
    assert json_output.build_envelope([plain])["items"][0]["page_sources"] == []


def test_cli_json_single_file(tmp_path: Path) -> None:
    from markitai.cli.main import app

    result = CliRunner().invoke(
        app, [str(_pdf(tmp_path)), "-o", str(tmp_path / "out"), "--json"]
    )
    assert result.exit_code == 0, result.output
    (item,) = json.loads(result.stdout)["items"]
    assert item["page_sources"] == SOURCES


def test_cli_json_batch(tmp_path: Path) -> None:
    from markitai.cli.main import app

    source_dir = tmp_path / "in"
    source_dir.mkdir()
    _pdf(source_dir)
    (source_dir / "a.txt").write_text("hello", encoding="utf-8")
    result = CliRunner().invoke(
        app, [str(source_dir), "-o", str(tmp_path / "out"), "--json"]
    )
    assert result.exit_code == 0, result.output
    items = {item["source"]: item for item in json.loads(result.stdout)["items"]}
    assert items["doc.pdf"]["page_sources"] == SOURCES
    assert items["a.txt"]["page_sources"] == []


def test_history_records_them(tmp_path: Path) -> None:
    from markitai.runs.history import record_cli_job

    output = tmp_path / "doc.pdf.md"
    output.write_text("# x", encoding="utf-8")
    job_dir = record_cli_job(
        [
            Outcome(
                kind="file",
                source="doc.pdf",
                status="completed",
                output_path=output,
                page_sources=SOURCES,
            )
        ],
        options={},
        jobs_root=tmp_path / "jobs",
    )
    assert job_dir is not None
    meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["items"][0]["page_sources"] == SOURCES
