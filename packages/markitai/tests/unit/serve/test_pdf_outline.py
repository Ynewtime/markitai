"""A PDF's own outline (contract v1.17): ``GET …/original/outline``.

Bookmarks of an invented PDF: nested entries, a GoTo-style entry the
document cannot place, a UTF-16 title (written by PyMuPDF), control
characters and an entry without a page. Nothing leaves the machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pymupdf = pytest.importorskip("pymupdf")

from markitai.serve.originals import MAX_OUTLINE_ENTRIES, pdf_outline
from tests.unit.serve.test_workbench_contract import (
    _client,
    _make_app,
    check_error,
    check_shape,
)


def _pdf(path: Path, toc: list[list[Any]] | None, pages: int = 4) -> Path:
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 60), f"Heading {index + 1}", fontsize=18)
    if toc is not None:
        document.set_toc(toc)
    document.save(path)
    document.close()
    return path


def _toc_is(monkeypatch: pytest.MonkeyPatch, toc: list[list[Any]]) -> None:
    def get_toc(_document: Any, simple: bool = True) -> list[list[Any]]:
        assert simple is True
        return toc

    monkeypatch.setattr(pymupdf.Document, "get_toc", get_toc)


def test_nested_bookmarks_in_outline_order(tmp_path: Path) -> None:
    path = _pdf(
        tmp_path / "a.pdf",
        [
            [1, "Garden handbook", 1],
            [2, "Soil\tand\n  compost", 2],
            [3, "Café — 堆肥", 3],
            [1, "Index", 4],
        ],
    )
    got = pdf_outline(path)
    assert got["truncated"] is False
    assert got["items"] == [
        {"title": "Garden handbook", "page": 1, "level": 1},
        {"title": "Soil and compost", "page": 2, "level": 2},
        {"title": "Café — 堆肥", "page": 3, "level": 3},
        {"title": "Index", "page": 4, "level": 1},
    ]


def test_an_entry_without_a_page_takes_the_next_ones_and_the_last_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "b.pdf", [[1, "Part", 1], [2, "Chapter", 2]])
    toc: list[list[Any]] = [
        [1, "Part", -1],
        [2, "Chapter", 3],
        [2, "Elsewhere", 99],
        [1, "Appendix", 4],
        [1, "Dangling", 0],
    ]
    # PyMuPDF refuses a page it cannot place; read the walk directly
    _toc_is(monkeypatch, toc)
    got = pdf_outline(path)
    assert got["items"] == [
        {"title": "Part", "page": 3, "level": 1},
        {"title": "Chapter", "page": 3, "level": 2},
        {"title": "Elsewhere", "page": 4, "level": 2},
        {"title": "Appendix", "page": 4, "level": 1},
    ]


def test_titles_are_one_line_and_at_most_300_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "c.pdf", None)
    _toc_is(monkeypatch, [[1, "x" * 400, 1], [1, "  a\x07b\x00\n c  ", 2]])
    items = pdf_outline(path)["items"]
    assert items[0]["title"] == "x" * 300
    assert items[1]["title"] == "a b c"


def test_the_list_is_cut_at_2000_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pdf(tmp_path / "d.pdf", None, pages=2)
    _toc_is(monkeypatch, [[1, f"Entry {n}", 1] for n in range(MAX_OUTLINE_ENTRIES + 5)])
    got = pdf_outline(path)
    assert len(got["items"]) == MAX_OUTLINE_ENTRIES and got["truncated"] is True


def test_no_outline_or_no_pdf_is_an_empty_list_never_an_error(tmp_path: Path) -> None:
    assert pdf_outline(_pdf(tmp_path / "e.pdf", None)) == {
        "items": [],
        "truncated": False,
    }
    broken = tmp_path / "f.pdf"
    broken.write_bytes(b"not a pdf")
    assert pdf_outline(broken) == {"items": [], "truncated": False}
    assert pdf_outline(tmp_path / "missing.pdf") == {"items": [], "truncated": False}


class TestRoute:
    async def test_pdf_has_an_outline_and_other_originals_are_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.unit.serve.test_workbench_contract import TestOriginals

        source = _pdf(tmp_path / "g.pdf", [[1, "Garden handbook", 1], [2, "Soil", 2]])
        async with _client(_make_app(tmp_path)) as client:
            holder = TestOriginals()
            pdf_job = await holder._job_of(
                client, monkeypatch, "doc.pdf", source.read_bytes()
            )
            text_job = await holder._job_of(client, monkeypatch, "note.txt", b"hello")
            ok = await client.get(f"/api/jobs/{pdf_job}/items/i1/original/outline")
            other = await client.get(f"/api/jobs/{text_job}/items/i1/original/outline")
            gone = await client.get("/api/jobs/nope/items/i1/original/outline")
        assert ok.status_code == 200, ok.text
        check_shape("PdfOutline", ok.json())
        assert ok.json()["items"] == [
            {"title": "Garden handbook", "page": 1, "level": 1},
            {"title": "Soil", "page": 2, "level": 2},
        ]
        check_error(other, 404, "original_unavailable")
        check_error(gone, 404, "job_not_found")
