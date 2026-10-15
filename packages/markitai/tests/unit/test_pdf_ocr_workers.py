"""Scanned PDF pages read in the extraction workers (converter.pdf_ocr).

A page of OCR costs seconds of CPU, nearly all in the recognizer; the pages
go to the worker processes one task each, each worker keeping its engine,
a bounded number at a time. The text must be what reading them in this
process gives.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

pymupdf = pytest.importorskip("pymupdf")

from markitai.converter import pdf_parallel  # noqa: E402

# Invented lines, one page each; "" is a blank scanned page
_PAGES = [
    "Harbour survey of the north quay",
    "Tide tables for the spring months",
    "",
    "Ferry timetable from the east pier",
    "Notes on the lighthouse keepers",
]


def _scanned_pdf(path: Path, lines: list[str]) -> Path:
    """An image-only PDF: each page is a picture of its line, no text layer."""
    out = pymupdf.open()
    for line in lines:
        source = pymupdf.open()
        page = source.new_page(width=420, height=300)
        if line:
            page.insert_text((30, 150), line, fontsize=18)
        picture = page.get_pixmap(dpi=150).tobytes("png")
        scanned = out.new_page(width=420, height=300)
        scanned.insert_image(scanned.rect, stream=picture)
    out.save(path)
    return path


@pytest.fixture
def workers(monkeypatch: pytest.MonkeyPatch):
    """The pool enabled with two workers; stopped afterwards."""
    monkeypatch.setattr(pdf_parallel, "_enabled", True)
    monkeypatch.setenv("MARKITAI_PDF_WORKERS", "2")
    yield
    pdf_parallel.shutdown_pool()


def test_scanned_pages_read_in_the_workers_match_this_process(
    tmp_path: Path, workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("rapidocr")
    from markitai.config import MarkitaiConfig, OCRConfig, ScreenshotConfig
    from markitai.converter.pdf import PdfConverter
    from markitai.notices import capture_task_notices
    from markitai.ocr import OCRProcessor

    pdf = _scanned_pdf(tmp_path / "harbour.pdf", _PAGES)
    config = MarkitaiConfig(
        ocr=OCRConfig(enabled=True), screenshot=ScreenshotConfig(enabled=True)
    )
    monkeypatch.setattr(OCRProcessor, "_global_engine", None)

    with capture_task_notices() as notices:
        pooled = PdfConverter(config).convert(pdf, tmp_path / "pooled")

    # Read in the workers: this process never built an engine
    assert OCRProcessor._global_engine is None
    assert pdf_parallel._pool is not None
    # The workers saved every page's screenshot
    shots = pooled.metadata["page_images"]
    assert [shot["page"] for shot in shots] == [1, 2, 3, 4, 5]
    assert all(Path(shot["path"]).is_file() for shot in shots)
    assert notices == [
        "[OCR] harbour.pdf: 1 of 5 pages has no text (page 3); "
        "it is left blank in the output"
    ]

    pdf_parallel.shutdown_pool()
    monkeypatch.setattr(pdf_parallel, "_enabled", False)
    here = PdfConverter(config).convert(pdf, tmp_path / "here")

    assert pooled.markdown == here.markdown
    for line in _PAGES:
        if line:
            assert line.split()[0] in pooled.markdown


class _WindowPool:
    """A pool whose tasks finish later, on a timer: counts how many run."""

    def __init__(self, *, fail: BaseException | None = None) -> None:
        self._broken = False
        self._shutdown_thread = False
        self.running = 0
        self.most = 0
        self.calls: list[tuple[Any, ...]] = []
        self._fail = fail
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.calls.append(args)

        def finish() -> None:
            with self._lock:
                self.running -= 1
            if not future.set_running_or_notify_cancel():
                return  # cancelled while queued
            if self._fail is not None:
                future.set_exception(self._fail)
            else:
                try:
                    future.set_result(fn(*args))
                except Exception as e:
                    future.set_exception(e)

        threading.Timer(0.01, finish).start()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._shutdown_thread = True


def _page_text(path: str, page: int, threads: int, prefix: str) -> str:
    if page == 13:
        raise ValueError("unreadable page 13")
    return f"{prefix}{page}"


def _collect() -> tuple[dict[int, Any], Callable[[int, Future[Any]], None]]:
    seen: dict[int, Any] = {}

    def on_done(page: int, future: Future[Any]) -> None:
        try:
            seen[page] = future.result()
        except ValueError as e:
            seen[page] = e

    return seen, on_done


def test_pages_go_to_the_workers_a_bounded_number_at_a_time(
    workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _WindowPool()
    monkeypatch.setattr(pdf_parallel, "_pool", pool)
    monkeypatch.setattr(pdf_parallel, "ocr_slots", lambda _workers: 2)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    seen, on_done = _collect()

    left = pdf_parallel.run_page_tasks(
        Path("book.pdf"), _page_text, list(range(20)), ("p",), on_done
    )

    assert left == []
    assert pool.most == 2  # never more pages at once than the slots
    # each page gets its share of the cores: 8 cores over 2 pages
    assert {call[2] for call in pool.calls} == {4}
    assert {page: text for page, text in seen.items() if page != 13} == {
        page: f"p{page}" for page in range(20) if page != 13
    }
    # A page's own error is that page's result; the pool stays
    assert isinstance(seen[13], ValueError)
    assert pdf_parallel._pool is pool and not pool._shutdown_thread


def test_a_short_document_is_read_in_process_unless_the_workers_run(
    workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seen, on_done = _collect()
    few = list(range(pdf_parallel.OCR_PARALLEL_MIN_PAGES - 1))

    assert (
        pdf_parallel.run_page_tasks(Path("a.pdf"), _page_text, few, ("p",), on_done)
        is None
    )
    assert pdf_parallel._pool is None  # nothing started for it

    monkeypatch.setattr(pdf_parallel, "_pool", _WindowPool())
    assert (
        pdf_parallel.run_page_tasks(Path("a.pdf"), _page_text, few, ("p",), on_done)
        == []
    )


def test_without_enable_every_page_is_read_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pdf_parallel, "_enabled", False)
    _seen, on_done = _collect()

    assert (
        pdf_parallel.run_page_tasks(
            Path("a.pdf"), _page_text, list(range(40)), ("p",), on_done
        )
        is None
    )


def test_pages_a_failed_pool_left_are_returned_to_read_here(
    workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _WindowPool(fail=BrokenProcessPool("a worker died"))
    monkeypatch.setattr(pdf_parallel, "_pool", pool)
    monkeypatch.setattr(pdf_parallel, "ocr_slots", lambda _workers: 3)
    seen, on_done = _collect()

    left = pdf_parallel.run_page_tasks(
        Path("a.pdf"), _page_text, list(range(10)), ("p",), on_done
    )

    assert left == list(range(10))
    assert seen == {}
    assert pdf_parallel._pool is None and pool._shutdown_thread


def test_a_pool_that_refuses_pages_leaves_them_all_here(
    workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing(_WindowPool):
        def submit(self, fn: Callable[..., Any], /, *args: Any) -> Future[Any]:
            raise BrokenProcessPool("broken, not flagged yet")

    pool = Refusing()
    monkeypatch.setattr(pdf_parallel, "_pool", pool)
    _seen, on_done = _collect()

    left = pdf_parallel.run_page_tasks(
        Path("a.pdf"), _page_text, list(range(6)), ("p",), on_done
    )

    assert left == list(range(6))
    assert pdf_parallel._pool is None


def test_a_stop_ends_the_reading_instead_of_redoing_it(
    workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _WindowPool(fail=BrokenProcessPool("the workers were stopped"))
    monkeypatch.setattr(pdf_parallel, "_pool", pool)
    monkeypatch.setattr(pdf_parallel, "_stopping", True)
    _seen, on_done = _collect()

    with pytest.raises(BrokenProcessPool):
        pdf_parallel._get_pool(2)  # a stop refuses new work outright
    monkeypatch.setattr(pdf_parallel, "_get_pool", lambda _workers: pool)
    with pytest.raises(BrokenProcessPool):
        pdf_parallel.run_page_tasks(
            Path("a.pdf"), _page_text, list(range(6)), ("p",), on_done
        )


def test_an_error_the_caller_raises_cancels_the_pages_not_started(
    workers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing OCR backend fails the whole document: no page after it."""
    pool = _WindowPool()
    monkeypatch.setattr(pdf_parallel, "_pool", pool)
    monkeypatch.setattr(pdf_parallel, "ocr_slots", lambda _workers: 2)

    def on_done(page: int, future: Future[Any]) -> None:
        raise ImportError("no OCR backend")

    with pytest.raises(ImportError):
        pdf_parallel.run_page_tasks(
            Path("a.pdf"), _page_text, list(range(30)), ("p",), on_done
        )
    assert len(pool.calls) <= 3


def test_worker_sessions_keep_a_thread_count_set_on_purpose() -> None:
    ort = pytest.importorskip("onnxruntime")
    unset, chosen = ort.SessionOptions(), ort.SessionOptions()
    chosen.intra_op_num_threads = 3

    pdf_parallel._one_thread_unless_set(unset)
    pdf_parallel._one_thread_unless_set(chosen)

    assert (unset.intra_op_num_threads, unset.inter_op_num_threads) == (1, 1)
    assert (chosen.intra_op_num_threads, chosen.inter_op_num_threads) == (3, 1)


def test_the_in_process_readers_follow_the_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from markitai.converter.pdf import PdfConverter

    monkeypatch.setattr(pdf_parallel, "ocr_slots", lambda workers: workers)
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    assert PdfConverter._ocr_thread_count(100) == 5
    assert PdfConverter._ocr_thread_count(2) == 2
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert PdfConverter._ocr_thread_count(100) == 8
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert PdfConverter._ocr_thread_count(100) == 1
    monkeypatch.setattr(pdf_parallel, "ocr_slots", lambda _workers: 1)
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    assert PdfConverter._ocr_thread_count(100) == 1  # memory allows one
