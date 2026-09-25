"""PDF page extraction across worker processes.

pymupdf4llm's layout engine (PyMuPDF-Layout) runs two ONNX models per page
and dominates PDF conversion time. In one process onnxruntime spreads each
page over every core and spends about three CPU seconds for each second it
saves; concurrent conversions (a batch) then oversubscribe the machine. Worker
processes that each run onnxruntime on a single thread do the same pages for a
third of the CPU, and in parallel.

The output is identical to one ``pymupdf4llm.to_markdown(..., page_chunks=True)``
call: each worker parses a contiguous run of pages with the same
``parse_document`` arguments the library uses, and the parent joins the pages
in order, then assigns heading levels from the font sizes of the whole
document (``update_header_tags``) — the one step that looks across pages —
before rendering Markdown.

Small single conversions stay in-process: starting the workers costs more
than a few pages take. The pool starts for a long document, or as soon as
two PDF conversions run at once, and then serves every later conversion of
the process.

Only processes markitai owns turn the pool on (:func:`enable`: the CLI,
``markitai serve`` and the MCP server). Worker processes are spawned, and a
spawned child imports its parent's ``__main__``: in a host script without an
``if __name__ == "__main__"`` guard that would run the host's own code again.
"""

from __future__ import annotations

import atexit
import importlib
import inspect
import math
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    # Imported when the pool starts: the CLI imports this module at startup
    from collections.abc import Callable
    from concurrent.futures import Future, ProcessPoolExecutor

#: A document with at least this many pages is split across the workers.
PARALLEL_MIN_PAGES = 12
#: Pages per worker task at least: fewer make the per-task overhead show.
MIN_PAGES_PER_TASK = 4
#: Memory one worker holds (the layout models plus a page's buffers).
_WORKER_RAM_BYTES = 450 * 1024 * 1024
#: The workers together use at most this share of the available memory.
_RAM_SHARE = 0.25
_MAX_WORKERS = 12

_pool: ProcessPoolExecutor | None = None
_lock = threading.Lock()
_active = 0
_enabled = False


def enable() -> None:
    """Allow worker processes in this process (entry points markitai owns)."""
    global _enabled
    _enabled = True


def layout_engine_active() -> bool:
    """Whether pymupdf4llm renders through PyMuPDF-Layout (its default)."""
    import pymupdf4llm

    return bool(getattr(pymupdf4llm, "_use_layout", False))


def worker_count() -> int:
    """Workers for this machine: cores and memory bound it, 12 at most.

    Two cores are left to the parent and the rest of the system, and the
    workers take at most a quarter of the available memory (about 450 MB
    each).

    ``MARKITAI_PDF_WORKERS`` overrides it; ``0`` or ``1`` disables the pool.
    """
    override = os.environ.get("MARKITAI_PDF_WORKERS")
    if override is not None:
        try:
            return max(0, int(override))
        except ValueError:
            logger.warning("Ignoring MARKITAI_PDF_WORKERS={!r}", override)
    cpus = os.cpu_count() or 1
    workers = min(_MAX_WORKERS, max(1, cpus - 2))
    try:
        import psutil

        available = psutil.virtual_memory().available * _RAM_SHARE
        workers = min(workers, max(1, int(available // _WORKER_RAM_BYTES)))
    except Exception:  # psutil missing or unsupported: cores decide
        pass
    return workers


def to_markdown_chunks(
    path: Path,
    *,
    extract: Callable[..., Any],
    pages: list[int] | None = None,
    write_images: bool,
    image_path: str,
    image_format: str,
    dpi: int,
) -> Any:
    """``pymupdf4llm.to_markdown(..., page_chunks=True)``, in parallel when it pays.

    Args:
        path: The PDF.
        extract: The caller's ``pymupdf4llm.to_markdown`` (looked up through
            its own module, so a patched one is honored); the workers run
            only for the genuine function.
        pages: 0-based pages to extract (all when None).
        write_images: Write the page images into *image_path*.
        image_path: Directory the images go to.
        image_format: Image file format (``png``, ``jpg`` ...).
        dpi: Image resolution.

    Returns:
        One chunk dict per page, as pymupdf4llm returns them.
    """
    global _active
    options: dict[str, Any] = {
        "write_images": write_images,
        "image_path": image_path,
        "image_format": image_format,
        "dpi": dpi,
    }
    with _lock:
        _active += 1
        concurrent = _active > 1
    try:
        if not _enabled or not _is_genuine(extract) or not layout_engine_active():
            return _serial(extract, path, pages, options)
        page_list = _page_list(path, pages)
        workers = worker_count()
        use_pool = workers >= 2 and (
            len(page_list) >= PARALLEL_MIN_PAGES or concurrent or _pool is not None
        )
        if not use_pool or not page_list:
            return _serial(extract, path, pages, options)
        try:
            return _parallel(path, page_list, options, workers)
        except Exception as e:  # a broken pool must not fail the conversion
            logger.warning(
                "[PDF] Parallel extraction failed ({}); extracting in-process", e
            )
            _discard_pool()
            return _serial(extract, path, pages, options)
    finally:
        with _lock:
            _active -= 1


def prestart() -> None:
    """Start the workers ahead of a batch that will convert several PDFs.

    Each worker loads the layout models when it starts; spawned now, they
    are ready by the time the batch reaches its PDFs instead of stalling
    the first of them. No-op where the pool would not be used.
    """
    if not _enabled or not layout_engine_active():
        return
    workers = worker_count()
    if workers < 2:
        return
    pool = _get_pool(workers)
    for _ in range(workers):  # one task per worker spawns them all
        pool.submit(_warm)


def _warm() -> None:
    """Worker task that only makes the worker start (see ``prestart``)."""


def _page_list(path: Path, pages: list[int] | None) -> list[int]:
    if pages is not None:
        return sorted(set(pages))
    import pymupdf

    with pymupdf.open(path) as doc:
        return list(range(doc.page_count))


def _is_genuine(extract: Callable[..., Any]) -> bool:
    """Whether *extract* is pymupdf4llm's own ``to_markdown``.

    Checked by what the function is, not by identity with the module
    attribute: a patch on ``pymupdf4llm.to_markdown`` replaces that too.
    """
    return (
        inspect.isfunction(extract)
        and extract.__module__ == "pymupdf4llm"
        and extract.__name__ == "to_markdown"
    )


def _serial(
    extract: Callable[..., Any],
    path: Path,
    pages: list[int] | None,
    options: dict[str, Any],
) -> Any:
    if pages is not None:
        options = {**options, "pages": pages}
    return extract(
        str(path),
        force_text=True,
        page_chunks=True,
        use_ocr=False,  # Markitai handles OCR separately; suppress Tesseract probing
        **options,
    )


def map_page_runs(
    path: Path, scan: Callable[[str, list[int]], Any], page_count: int
) -> list[Future[Any]] | None:
    """Run ``scan(path, pages)`` over runs of pages in the workers.

    For per-page checks that would otherwise run in this process while the
    workers extract the same document (holding the GIL an event loop needs):
    under the rules that send the extraction to the workers, the checks go
    there too. *scan* must be a module-level function (it is pickled by
    reference).

    Returns:
        One future per run of pages, in page order; None when the pool
        would not serve this document (run the check in-process instead).
    """
    if not _enabled or page_count < 1 or not layout_engine_active():
        return None
    workers = worker_count()
    if workers < 2 or not (
        page_count >= PARALLEL_MIN_PAGES or _active > 0 or _pool is not None
    ):
        return None
    pool = _get_pool(workers)
    return [
        pool.submit(scan, str(path), run)
        for run in _page_runs(list(range(page_count)), workers)
    ]


def _page_runs(page_list: list[int], workers: int) -> list[list[int]]:
    """Contiguous runs of pages, one per task, at least MIN_PAGES_PER_TASK."""
    tasks = max(1, min(workers, math.ceil(len(page_list) / MIN_PAGES_PER_TASK)))
    size = math.ceil(len(page_list) / tasks)
    return [page_list[i : i + size] for i in range(0, len(page_list), size)]


def _parallel(
    path: Path, page_list: list[int], options: dict[str, Any], workers: int
) -> list[dict[str, Any]]:
    pool = _get_pool(workers)
    runs = _page_runs(page_list, workers)
    futures = [pool.submit(_parse_pages, str(path), run, options) for run in runs]
    parsed = [future.result() for future in futures]
    logger.debug(
        "[PDF] {} page(s) of {} parsed in {} worker task(s)",
        len(page_list),
        path.name,
        len(runs),
    )
    return _render(parsed, options)


def _render(parsed: list[Any], options: dict[str, Any]) -> list[dict[str, Any]]:
    """Join the parsed page runs and render them like one document."""
    from pymupdf4llm.helpers.document_layout import update_header_tags

    document = parsed[0]
    document.pages = [page for part in parsed for page in part.pages]
    header_fontsizes = {
        box.max_fontsize
        for page in document.pages
        for box in page.boxes
        if box.boxclass in ("title", "section-header")
    }
    if header_fontsizes:
        update_header_tags(document.pages, header_fontsizes)
    # The arguments pymupdf4llm's own layout path renders with
    return document.to_markdown(
        header=True,
        footer=True,
        write_images=options["write_images"],
        embed_images=False,
        ignore_code=False,
        show_progress=False,
        page_separators=False,
        page_chunks=True,
    )


def _parse_pages(path: str, pages: list[int], options: dict[str, Any]) -> Any:
    """Worker task: parse *pages* exactly as pymupdf4llm's layout path does."""
    from pymupdf4llm.helpers.document_layout import parse_document

    return parse_document(
        path,
        filename="",
        image_dpi=options["dpi"],
        image_format=options["image_format"],
        image_path=options["image_path"],
        pages=pages,
        ocr_dpi=150,
        write_images=options["write_images"],
        embed_images=False,
        show_progress=False,
        force_text=True,
        use_ocr=False,
        force_ocr=False,
        ocr_language="eng",
        ocr_function=None,
        render_html_tables=None,
        edge_threshold=None,
    )


def _init_worker() -> None:
    """Run onnxruntime on one thread and keep stdout free for the parent."""
    import sys

    import onnxruntime as ort

    # parse_document prints its messages; the parent may be writing Markdown
    # to stdout. The parent owns logging: a worker's loguru would print its
    # debug lines to the terminal.
    sys.stdout = sys.stderr
    from loguru import logger as worker_logger

    worker_logger.remove()
    original = ort.InferenceSession

    class _SingleThreadSession(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            options = kwargs.get("sess_options")
            if options is None and len(args) > 1:
                options = args[1]
            if options is None:
                options = ort.SessionOptions()
                kwargs["sess_options"] = options
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            super().__init__(*args, **kwargs)

    ort.InferenceSession = _SingleThreadSession  # type: ignore[misc]
    # Load the layout engine once per worker, not on its first task
    importlib.import_module("pymupdf4llm")


def _get_pool(workers: int) -> ProcessPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            # spawn: a forked child would inherit the parent's threads and
            # onnxruntime state
            _pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
            )
            logger.debug("[PDF] Started {} extraction worker(s)", workers)
        return _pool


def _discard_pool() -> None:
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def shutdown_pool() -> None:
    """Stop the workers (idempotent). Called on exit; the CLI's ``os._exit``
    skips ``atexit``, so ``finalize_process`` calls it too."""
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is None:
        return
    processes = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            process.terminate()
        except Exception:
            pass


atexit.register(shutdown_pool)
