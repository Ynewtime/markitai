"""PDF page extraction across worker processes (markitai.converter.pdf_parallel).

The parallel path must produce exactly what one ``pymupdf4llm.to_markdown``
call does: the pages are parsed in runs, then joined and given heading
levels from the whole document before rendering.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

pymupdf = pytest.importorskip("pymupdf")
pymupdf4llm = pytest.importorskip("pymupdf4llm")

from markitai.converter import pdf_parallel  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pdf_parallel.layout_engine_active(),
    reason="PyMuPDF-Layout is not installed",
)


def _make_pdf(path: Path, pages: int = 16) -> Path:
    """Headings shrink page by page, so a run of pages sees other sizes."""
    doc = pymupdf.open()
    sizes = [30, 22, 16, 12]
    for n in range(pages):
        page = doc.new_page()
        size = sizes[min(n * len(sizes) // pages, len(sizes) - 1)]
        page.insert_text((72, 80), f"Heading probe {n}", fontsize=size)
        page.insert_textbox(
            pymupdf.Rect(72, 110, 540, 400), "Body text sentence. " * 30, fontsize=9
        )
    doc.save(path)
    return path


def _options(image_dir: Path) -> dict[str, Any]:
    return {
        "write_images": True,
        "image_path": str(image_dir),
        "image_format": "png",
        "dpi": 150,
    }


def _normalized(chunks: Any, *image_dirs: Path) -> str:
    text = json.dumps(chunks, sort_keys=True, default=repr)
    for image_dir in image_dirs:
        text = text.replace(str(image_dir.resolve()), "IMAGES")
        text = text.replace(str(image_dir), "IMAGES")
    return re.sub(r"\s+", " ", text)


def _serial(path: Path, image_dir: Path) -> Any:
    return pdf_parallel._serial(
        pymupdf4llm.to_markdown, path, None, _options(image_dir)
    )


def test_joined_page_runs_render_like_one_document(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "headings.pdf")
    serial_dir, runs_dir = tmp_path / "serial", tmp_path / "runs"
    serial_dir.mkdir()
    runs_dir.mkdir()

    parsed = [
        pdf_parallel._parse_pages(str(pdf), run, _options(runs_dir))
        for run in (list(range(8)), list(range(8, 16)))
    ]

    def header_sizes(part: Any) -> set[int]:
        return {
            box.max_fontsize
            for page in part.pages
            for box in page.boxes
            if box.boxclass in ("title", "section-header")
        }

    # The runs see different heading sizes: levels computed per run would
    # differ from the document's, so this exercises the whole-document pass
    assert header_sizes(parsed[0]) != header_sizes(parsed[1])
    joined = pdf_parallel._render(parsed, _options(runs_dir))

    assert _normalized(joined, runs_dir) == _normalized(
        _serial(pdf, serial_dir), serial_dir
    )


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch):
    """The pool enabled with two workers; stopped afterwards."""
    monkeypatch.setattr(pdf_parallel, "_enabled", True)
    monkeypatch.setenv("MARKITAI_PDF_WORKERS", "2")
    yield
    pdf_parallel.shutdown_pool()


def test_the_worker_pool_extracts_the_same_output(
    tmp_path: Path, workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _make_pdf(tmp_path / "headings.pdf")
    monkeypatch.setattr(pdf_parallel, "PARALLEL_MIN_PAGES", 4)
    serial_dir, pool_dir = tmp_path / "serial", tmp_path / "pool"
    serial_dir.mkdir()
    pool_dir.mkdir()

    chunks = pdf_parallel.to_markdown_chunks(
        pdf, extract=pymupdf4llm.to_markdown, **_options(pool_dir)
    )

    assert pdf_parallel._pool is not None  # the pool did the work
    assert _normalized(chunks, pool_dir) == _normalized(
        _serial(pdf, serial_dir), serial_dir
    )


def test_without_enable_the_pages_stay_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host program that did not opt in never spawns workers."""
    monkeypatch.setattr(pdf_parallel, "_enabled", False)
    monkeypatch.setattr(pdf_parallel, "PARALLEL_MIN_PAGES", 1)
    pdf = _make_pdf(tmp_path / "doc.pdf", pages=4)

    pdf_parallel.to_markdown_chunks(
        pdf, extract=pymupdf4llm.to_markdown, **_options(tmp_path)
    )

    assert pdf_parallel._pool is None


def test_a_substituted_extractor_is_called_as_before(
    tmp_path: Path, workers: None
) -> None:
    """A patched pymupdf4llm.to_markdown (tests, wrappers) runs in-process."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def extract(path: str, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append((path, kwargs))
        return [{"text": "page"}]

    result = pdf_parallel.to_markdown_chunks(
        tmp_path / "any.pdf", extract=extract, pages=[0, 1], **_options(tmp_path)
    )

    assert result == [{"text": "page"}]
    ((path, kwargs),) = calls
    assert path.endswith("any.pdf")
    assert kwargs["pages"] == [0, 1]
    assert kwargs["page_chunks"] is True
    assert kwargs["use_ocr"] is False
    assert pdf_parallel._pool is None


def test_worker_count_honours_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKITAI_PDF_WORKERS", "3")
    assert pdf_parallel.worker_count() == 3
    monkeypatch.setenv("MARKITAI_PDF_WORKERS", "0")
    assert pdf_parallel.worker_count() == 0
    monkeypatch.delenv("MARKITAI_PDF_WORKERS")
    assert 1 <= pdf_parallel.worker_count() <= pdf_parallel._MAX_WORKERS


def test_the_registry_maps_pdf_to_the_converter() -> None:
    """The converter class, not a helper beside it, is registered for PDF."""
    from markitai.converter import get_converter
    from markitai.converter.pdf import PdfConverter

    assert isinstance(get_converter(Path("report.pdf")), PdfConverter)


def test_a_patched_module_attribute_is_not_taken_for_the_real_one(
    tmp_path: Path, workers: None
) -> None:
    """``patch("pymupdf4llm.to_markdown")`` replaces the attribute the
    genuine-function check would compare against; the check must not be
    fooled into opening the (fake) file in the workers."""
    from unittest.mock import patch

    fake = tmp_path / "not-really.pdf"
    fake.write_bytes(b"%PDF-1.4 fake")
    with patch("pymupdf4llm.to_markdown", return_value=[{"text": "mocked"}]):
        result = pdf_parallel.to_markdown_chunks(
            fake, extract=pymupdf4llm.to_markdown, **_options(tmp_path)
        )

    assert result == [{"text": "mocked"}]
    assert pdf_parallel._pool is None
