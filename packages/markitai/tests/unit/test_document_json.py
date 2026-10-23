"""The document JSON (``-f json``) against markitai-rust's.

The fixtures in ``tests/fixtures/document_json`` are markitai-rust's own: its
frozen field set of schema 1.0, its published JSON Schemas and the snapshot
its tests pin byte for byte. The same inputs give the same bytes here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from markitai.document_json import (
    DocumentParts,
    build_document,
    classify_notice,
    document_url,
    plain_text,
    render,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "document_json"
OUT = Path("/out")

PAGED = (
    "<!-- Page number: 1 -->\n\n# Report\n\nIntro with a [link](https://example.test)"
    " and `code`.\n\n![A chart](.markitai/assets/chart.png)\n\n<!-- ![Page 1]"
    "(.markitai/screenshots/report.pdf.page0001.jpg) -->\n\n<!-- Page number: 2 -->"
    "\n\n| Name | Value |\n|---|---|\n| a | 1 |\n| b | 2 |\n\n```text\n"
    "<!-- Page number: 9 -->\n```\n\n<!-- Page images for reference -->\n"
    "<!-- ![Page 2](.markitai/screenshots/report.pdf.page0002.jpg) -->\n"
)


def _paged() -> DocumentParts:
    """markitai-rust's snapshot input: a scanned two-page PDF read by a
    model, with page markers, screenshots, an analysed picture, a table,
    notices and model usage."""
    llm = PAGED.replace("<!-- Page number: 1 -->\n\n", "").replace(
        "\n\n<!-- Page number: 2 -->\n", "\n"
    )
    return DocumentParts(
        source="https://user:secret@example.test/report.pdf?token=abc",
        source_name="example_test_report.pdf",
        base="# Report\n\nBase text.\n",
        llm=llm,
        paged=PAGED,
        frontmatter={
            "tags": ["report"],
            "title": "Report",
            "source": "report.pdf",
            "description": "A short report.",
            "markitai_processed": "2026-10-17T12:00:00.000+08:00",
        },
        llm_output_path=OUT / "report.pdf.llm.md",
        assets=[OUT / ".markitai/assets/chart.png"],
        screenshots=[
            OUT / ".markitai/screenshots/report.pdf.page0001.jpg",
            OUT / ".markitai/screenshots/report.pdf.page0002.jpg",
        ],
        analysis=[
            {
                "asset": "/out/.markitai/assets/chart.png",
                "alt": "A chart",
                "desc": "Bars rising to the right.",
                "text": "",
            }
        ],
        llm_usage={
            "openai/gpt-test": {
                "requests": 2,
                "input_tokens": 100,
                "output_tokens": 40,
                "cost_usd": 0.0012,
            }
        },
        duration_s=1.23456,
        warnings=[
            "PDF page 2: native text was not recovered (no text layer); OCR is "
            "required for this page.",
            "Something nobody classified yet.",
        ],
    )


def _snapshot_text(parts: DocumentParts) -> str:
    document = build_document(parts, OUT)
    document["tool"] = {"name": "markitai", "version": "VERSION", "commit": "COMMIT"}
    return render(document)


def test_a_paged_document_matches_markitai_rusts_snapshot_byte_for_byte() -> None:
    expected = (FIXTURES / "snapshot-paged.json").read_text(encoding="utf-8")
    # The snapshot's LLM body is the paged one without its markers
    assert json.loads(expected)["markdown"]["llm"] == _paged().llm
    assert _snapshot_text(_paged()) == expected


def _fields(value: Any, path: str, out: set[str]) -> None:
    kind = {
        type(None): "null",
        bool: "boolean",
        int: "integer",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }[type(value)]
    if path:
        out.add(f"{path}: {kind}")
    if isinstance(value, dict) and path != "properties":
        for key, item in value.items():
            _fields(item, f"{path}.{key}" if path else key, out)
    elif isinstance(value, list):
        for item in value:
            _fields(item, f"{path}[]", out)


def test_the_field_set_is_markitai_rusts_frozen_schema_1_0(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-1.4")
    local = DocumentParts(
        source=str(source),
        base="![x](.markitai/assets/a.png)\n\n|h|\n|-|\n|v|\n\n<!-- Page number: 1 -->"
        "\n\nText\n",
        output_path=OUT / "scan.pdf.md",
        analysis=[
            {"asset": "/out/.markitai/assets/a.png", "alt": "x", "text": "Words"}
        ],
    )
    skipped = DocumentParts(
        source="notes.txt",
        base="Plain.\n",
        skip_reason="image_only",
        frontmatter={"title": "notes"},
    )
    found: set[str] = set()
    for parts in (_paged(), skipped, local):
        _fields(json.loads(render(build_document(parts, OUT))), "", found)
    frozen = {
        line
        for line in (FIXTURES / "fields-v1.0.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert found == frozen


def _validate(schema: dict, value: Any, path: str, root: dict) -> None:
    """The subset of JSON Schema 2020-12 the document schemas use; strict:
    a key the schema does not list fails, except in the front matter."""
    if "$ref" in schema:
        name = schema["$ref"].removeprefix("#/$defs/")
        return _validate(root["$defs"][name], value, path, root)
    kinds = schema.get("type")
    if kinds is not None:
        allowed = [kinds] if isinstance(kinds, str) else kinds
        kind = {
            type(None): "null",
            bool: "boolean",
            int: "integer",
            float: "number",
            str: "string",
            list: "array",
            dict: "object",
        }[type(value)]
        assert kind in allowed or (kind == "integer" and "number" in allowed), path
    if "enum" in schema:
        assert value in schema["enum"], path
    if "const" in schema:
        assert value == schema["const"], path
    if "pattern" in schema and isinstance(value, str):
        assert re.search(schema["pattern"], value), path
    if isinstance(value, dict):
        for key in schema.get("required", []):
            assert key in value, f"{path}: missing {key}"
        properties = schema.get("properties")
        if properties is not None:
            for key, item in value.items():
                assert key in properties, f"{path}: {key} is not in the schema"
                _validate(properties[key], item, f"{path}.{key}", root)
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate(schema["items"], item, f"{path}[{index}]", root)
    return None


def test_the_published_schema_describes_every_key(tmp_path: Path) -> None:
    from markitai.workflow.document import index_item, write_index

    schema = json.loads((FIXTURES / "document-v1.schema.json").read_text())
    for parts in (_paged(), DocumentParts(source="notes.txt", base="Plain.\n")):
        _validate(schema, json.loads(render(build_document(parts, OUT))), "", schema)
    (tmp_path / "a.txt.md").write_text("A\n")
    (tmp_path / "a.txt.json").write_text("{}")
    write_index(
        tmp_path,
        [
            index_item("a.txt", "completed", tmp_path, output=tmp_path / "a.txt.md"),
            index_item("b.png", "skipped", tmp_path, skip_reason="image_only"),
        ],
    )
    index = json.loads((tmp_path / "markitai.index.json").read_text())
    index_schema = json.loads((FIXTURES / "index-v1.schema.json").read_text())
    _validate(index_schema, index, "", index_schema)
    assert index["items"][0]["document"] == "a.txt.json"


def test_pages_tables_and_pictures_follow_the_page_markers() -> None:
    document = build_document(_paged(), OUT)
    pages = document["pages"]
    assert len(pages) == 2, "a marker inside a code block is text"
    assert pages[0]["images"] == [".markitai/assets/chart.png"]
    assert pages[1]["screenshot"] == ".markitai/screenshots/report.pdf.page0002.jpg"
    assert "Page images" not in pages[1]["markdown"]
    assert document["tables"] == [
        {"page": 2, "header": ["Name", "Value"], "rows": [["a", "1"], ["b", "2"]]}
    ]
    image = document["images"][0]
    assert (image["description"], image["text"], image["page"]) == (
        "Bars rising to the right.",
        None,
        1,
    )
    assert document["source"]["url"] == "https://example.test/report.pdf?token=REDACTED"
    assert document["outputs"]["llm_markdown"] == "report.pdf.llm.md"
    assert document["timings"]["duration_s"] == 1.235


def test_the_rag_profiles_markers_divide_pages_too() -> None:
    body = "<!-- page: 1 -->\n\nOne.\n\n<!-- page: 2 -->\n\n![p](assets/p.png)\n"
    document = build_document(DocumentParts(source="a.pdf", base=body))
    assert [page["number"] for page in document["pages"]] == [1, 2]
    assert document["images"][0]["page"] == 2


def test_page_screenshots_are_found_by_name_when_the_markdown_does_not_link_them() -> (
    None
):
    """Python's Markdown does not link its page renders (markitai-rust's does)."""
    parts = DocumentParts(
        source="deck.pptx",
        base="<!-- Slide number: 1 -->\n\nA\n\n<!-- Slide number: 2 -->\n\nB\n",
        screenshots=[
            OUT / ".markitai/screenshots/deck.pptx.slide0001.jpg",
            OUT / ".markitai/screenshots/deck.pptx.slide0002.jpg",
        ],
    )
    pages = build_document(parts, OUT)["pages"]
    assert [page["screenshot"] for page in pages] == [
        ".markitai/screenshots/deck.pptx.slide0001.jpg",
        ".markitai/screenshots/deck.pptx.slide0002.jpg",
    ]


@pytest.mark.parametrize(
    ("markdown", "text"),
    [
        (
            "# Title\n\nA **bold** [link](x) and ![pic](p.png) `code`.\n\n<!-- note -->"
            "\n\n- one\n- two\n\n|a|b|\n|-|-|\n|1|2|\n",
            "Title\n\nA bold link and  code.\n\none\ntwo\n\na\tb\n1\t2",
        ),
        ("", ""),
        # A task marker is no text; a loose list's items are blocks
        ("- [ ] open\n- [x] done\n", "open\ndone"),
        ("1. a\n\n2. b\n", "a\n\nb"),
        # CommonMark strips spaces and tabs only: a Chinese paragraph keeps
        # the ideographic spaces that indent it
        ("　　段落。\n", "　　段落。"),
        ("| 表 | 二 |\n|---|---|\n|\u3000值\u3000| x |\n", "表\t二\n\u3000值\u3000\tx"),
        # An HTML block line by line, comments left out
        ("<div>\n<b>bold</b> block\n</div>\n", "bold block"),
        ("Inline <span>html</span> &amp; ok.\n", "Inline html & ok."),
    ],
    ids=["markup", "empty", "tasks", "loose", "u3000", "u3000-cell", "html", "inline"],
)
def test_plain_text_reads_markdown_as_pulldown_cmark_does(
    markdown: str, text: str
) -> None:
    assert plain_text(markdown) == text


def test_a_table_row_is_padded_and_cut_to_its_header() -> None:
    body = "| h1 | h2 |\n| --- | --- |\n| a | b | extra |\n| c |\n"
    tables = build_document(DocumentParts(source="t.md", base=body))["tables"]
    assert tables == [
        {"page": None, "header": ["h1", "h2"], "rows": [["a", "b"], ["c", ""]]}
    ]


def test_picture_targets_stay_as_written() -> None:
    """No percent-encoding and every scheme: stdout links ``file://`` URIs."""
    body = "![中文](.markitai/assets/图.png)\n\n![](<file:///tmp/a b.png>)\n"
    images = build_document(DocumentParts(source="p.md", base=body))["images"]
    assert [image["path"] for image in images] == [
        ".markitai/assets/图.png",
        "file:///tmp/a b.png",
    ]


@pytest.mark.parametrize(
    ("message", "code", "level", "items"),
    [
        (
            "PDF page 3: native text was not recovered (no text layer); OCR is "
            "required for this page.",
            "pdf.page_without_text",
            "warning",
            ["3"],
        ),
        (
            "PDF page 2: read as plain text because the layout is ambiguous.",
            "pdf.page_plain_text",
            "info",
            ["2"],
        ),
        (
            'Hidden worksheet "Hidden notes" is omitted by the native reader.',
            "spreadsheet.hidden_sheet",
            "info",
            ["Hidden notes"],
        ),
        (
            "Image analysis failed for asset-1.emf: Cannot decode; existing alt "
            "text was retained.",
            "image.analysis_failed",
            "warning",
            ["asset-1.emf"],
        ),
        (
            "Office pages 2–3 (2 of 3): OCR found no text.",
            "office.page",
            "warning",
            ["2", "3"],
        ),
        (
            "No model succeeded for this document; only the base Markdown was kept.",
            "llm.no_model_succeeded",
            "warning",
            [],
        ),
        (
            "The model's tool-call answer was not used (the answer changed a "
            "marker); the request was sent again in JSON-schema mode and a later "
            "answer was used",
            "llm.answer_retried",
            "info",
            [],
        ),
        # Python's own sentences for the same families
        (
            "[PDF] scan.pdf: 3 of 3 pages look scanned/garbled (pages 1\u20133); "
            "consider re-running with --ocr",
            "pdf.page_without_text",
            "warning",
            ["1", "2", "3"],
        ),
        (
            "No price is known for LLM model fake-model: its cost is reported as "
            "unknown, not $0",
            "llm.cost_incomplete",
            "warning",
            [],
        ),
        (
            "[PPTX] Cannot render slides of deck.pptx: LibreOffice not found.",
            "office.renderer_missing",
            "warning",
            [],
        ),
        ("Something new happened.", "other", "warning", []),
    ],
    ids=lambda value: value if isinstance(value, str) and "." in value[:20] else None,
)
def test_notices_get_markitai_rusts_codes(
    message: str, code: str, level: str, items: list[str]
) -> None:
    found = classify_notice(message)
    assert (found.code, found.level, found.items) == (code, level, items)


def test_a_url_loses_its_credentials_and_secret_values() -> None:
    assert document_url("https://u:p@example.test/a?token=x&page=2#k") == (
        "https://example.test/a?token=REDACTED&page=2#k"
    )
    assert (
        document_url("https://example.test/plain?q=1")
        == "https://example.test/plain?q=1"
    )


def test_a_local_file_has_its_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "note.TXT"
    path.write_bytes(b"hello")
    source = build_document(DocumentParts(source=str(path)))["source"]
    assert (source["size"], source["type"], source["name"]) == (5, "txt", "note.TXT")
    assert source["sha256"] == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    missing = build_document(DocumentParts(source="gone.txt"))["source"]
    assert (missing["size"], missing["sha256"]) == (None, None)
