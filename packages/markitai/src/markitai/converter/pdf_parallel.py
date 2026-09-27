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
import functools
import importlib
import inspect
import math
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
#: ProcessPoolExecutor refuses more workers than this on Windows.
_MAX_WINDOWS_WORKERS = 61
_WINDOWS = sys.platform == "win32"

_R = TypeVar("_R")

_pool: ProcessPoolExecutor | None = None
_lock = threading.Lock()
_active = 0
_enabled = False
# Set by shutdown_pool: an extraction the stop interrupted must end, not
# start over in-process, and no new pool starts until enable()
_stopping = False


def enable() -> None:
    """Allow worker processes in this process (entry points markitai owns).

    Also ends a stop (:func:`shutdown_pool`): the workers start again when
    a conversion needs them.
    """
    global _enabled, _stopping
    with _lock:
        _enabled = True
        _stopping = False


def layout_engine_active() -> bool:
    """Whether pymupdf4llm renders through PyMuPDF-Layout (its default)."""
    import pymupdf4llm

    return bool(getattr(pymupdf4llm, "_use_layout", False))


def worker_count() -> int:
    """Workers for this machine: cores and memory bound it, 12 at most.

    Two cores are left to the parent and the rest of the system, and the
    workers take at most a quarter of the available memory (about 450 MB
    each).

    ``MARKITAI_PDF_WORKERS`` overrides it, up to the same maximum; ``1`` or
    less disables the pool.
    """
    maximum = _max_workers()
    override = os.environ.get("MARKITAI_PDF_WORKERS")
    if override is not None:
        workers = _workers_override(override, maximum)
        if workers is not None:
            return workers
    cpus = os.cpu_count() or 1
    workers = min(maximum, max(1, cpus - 2))
    try:
        import psutil

        available = psutil.virtual_memory().available * _RAM_SHARE
        workers = min(workers, max(1, int(available // _WORKER_RAM_BYTES)))
    except Exception:  # psutil missing or unsupported: cores decide
        pass
    return workers


def _max_workers() -> int:
    if _WINDOWS:
        return min(_MAX_WORKERS, _MAX_WINDOWS_WORKERS)
    return _MAX_WORKERS


@functools.cache
def _workers_override(value: str, maximum: int) -> int | None:
    """``MARKITAI_PDF_WORKERS`` as a worker count; None when it is no number.

    Cached: worker_count runs up to three times per PDF, and a bad value is
    reported once per process, not on every call.
    """
    try:
        workers = int(value)
    except ValueError:
        logger.warning("Ignoring MARKITAI_PDF_WORKERS={!r} (not a number)", value)
        return None
    if workers > maximum:
        logger.warning(
            "MARKITAI_PDF_WORKERS={} is above the maximum; using {}", workers, maximum
        )
        return maximum
    return max(0, workers)


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
        pool: ProcessPoolExecutor | None = None
        try:
            pool = _get_pool(workers)
            return _parallel(pool, path, page_list, options, workers)
        except Exception as e:  # a failed worker must not fail the conversion
            if _stopping:
                # shutdown_pool stopped the workers (Ctrl-C, server stop),
                # before or during this extraction: end now, do not redo it
                raise
            if pool is None or _pool_failed(pool, e):
                logger.warning(
                    "[PDF] The extraction workers failed ({}); extracting {} "
                    "in-process, new workers start with the next document",
                    e,
                    path.name,
                )
                if pool is not None:
                    _discard_pool(pool)
            else:
                # The document's own error (encrypted, a page out of range):
                # the pool, and other documents' work in it, are fine
                logger.warning(
                    "[PDF] A worker could not extract {} ({}); retrying in-process",
                    path.name,
                    e,
                )
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
    if not _enabled or _stopping or not layout_engine_active():
        return
    workers = worker_count()
    if workers < 2:
        return
    pool: ProcessPoolExecutor | None = None
    try:
        pool = _get_pool(workers)
        for _ in range(workers):  # one task per worker spawns them all
            pool.submit(_warm)
    except Exception as e:  # stopped, or the pool broke: the conversions decide
        if pool is not None and not _stopping:
            _discard_pool(pool)
        logger.debug("[PDF] Extraction workers not started ahead: {}", e)


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
    path: Path, scan: Callable[[str, list[int]], _R], page_count: int
) -> list[PageRun[_R]] | None:
    """Run ``scan(path, pages)`` over runs of pages in the workers.

    For per-page checks that would otherwise run in this process while the
    workers extract the same document (holding the GIL an event loop needs):
    under the rules that send the extraction to the workers, the checks go
    there too. *scan* must be a module-level function (it is pickled by
    reference).

    Returns:
        One pending result per run of pages, in page order (a run the
        workers do not finish is checked in this process instead); None
        when the pool would not serve this document, or cannot (run the
        check in-process instead).

    Raises:
        BrokenProcessPool: :func:`shutdown_pool` stopped the workers.
    """
    if not _enabled or page_count < 1 or not layout_engine_active():
        return None
    workers = worker_count()
    if workers < 2 or not (
        page_count >= PARALLEL_MIN_PAGES or _active > 0 or _pool is not None
    ):
        return None
    runs = _page_runs(list(range(page_count)), workers)
    pool: ProcessPoolExecutor | None = None
    futures: list[Future[_R]] = []
    try:
        pool = _get_pool(workers)
        for run in runs:
            futures.append(pool.submit(scan, str(path), run))
    except Exception as e:  # the pool broke, or was shut down, meanwhile
        if _stopping:
            raise  # no new work after a stop, in the workers or here
        for future in futures:
            future.cancel()
        if pool is not None:
            _discard_pool(pool)
        logger.debug(
            "[PDF] Extraction workers unavailable ({}); checking {} in-process",
            e,
            path.name,
        )
        return None
    return [
        PageRun(future, scan, str(path), run)
        for future, run in zip(futures, runs, strict=True)
    ]


class PageRun(Generic[_R]):
    """A check of one run of pages, running in a worker (see map_page_runs).

    A run the workers do not finish (a worker died, or a discarded pool
    cancelled it) is checked in this process instead, so a check always
    covers every page: the hidden-text scan is a security check.
    """

    def __init__(
        self,
        future: Future[_R],
        scan: Callable[[str, list[int]], _R],
        path: str,
        pages: list[int],
    ) -> None:
        self._future = future
        self._scan = scan
        self._path = path
        self._pages = pages

    def result(self) -> _R:
        """The check's result for these pages, from the worker or from here.

        Raises:
            BrokenProcessPool: :func:`shutdown_pool` stopped the workers.
        """
        try:
            return self._future.result()
        except Exception as e:
            if _stopping:
                raise  # the stop ends the conversion; nothing is redone here
            logger.debug(
                "[PDF] Page check in a worker failed ({}); checking pages {}-{} "
                "of {} in-process",
                e,
                self._pages[0] + 1,
                self._pages[-1] + 1,
                Path(self._path).name,
            )
            return self._scan(self._path, self._pages)


def _page_runs(page_list: list[int], workers: int) -> list[list[int]]:
    """Contiguous runs of pages, one per task, at least MIN_PAGES_PER_TASK."""
    tasks = max(1, min(workers, math.ceil(len(page_list) / MIN_PAGES_PER_TASK)))
    size = math.ceil(len(page_list) / tasks)
    return [page_list[i : i + size] for i in range(0, len(page_list), size)]


def _parallel(
    pool: ProcessPoolExecutor,
    path: Path,
    page_list: list[int],
    options: dict[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    runs = _page_runs(page_list, workers)
    futures: list[Future[Any]] = []
    try:
        for run in runs:
            futures.append(pool.submit(_parse_pages, str(path), run, options))
        parsed = [future.result() for future in futures]
    except BaseException:
        for future in futures:  # this document's runs still queued: drop them
            future.cancel()
        raise
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
    """Ignore Ctrl-C, run onnxruntime on one thread, keep stdout for the parent."""
    import signal

    # A terminal's Ctrl-C reaches every process in its foreground group. The
    # parent stops the workers itself (shutdown_pool, or the watch below when
    # it dies); a worker's own KeyboardInterrupt would print a traceback
    # when idle, and come back as a busy task's exception, which escapes the
    # parent's error handling. SIGINT and SIG_IGN exist on Windows too.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _exit_with_parent()
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


def _exit_with_parent() -> None:
    """End this worker as soon as the process that spawned it is gone.

    The pool's queues hand every worker both ends of their pipes, so a
    worker whose parent was killed (a signal, a crash, ``os._exit`` without
    ``shutdown_pool``) never sees EOF and would wait forever. The parent's
    sentinel does close with it.
    """
    import multiprocessing

    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        os._exit(0)

    threading.Thread(target=watch, name="markitai-parent-watch", daemon=True).start()


def _get_pool(workers: int) -> ProcessPoolExecutor:
    """The shared pool, started on first use; a broken one is replaced.

    Raises:
        BrokenProcessPool: :func:`shutdown_pool` stopped the workers (what
            an extraction the stop interrupts gets from them too); none
            start again until :func:`enable`.
    """
    global _pool
    stale: ProcessPoolExecutor | None = None
    with _lock:
        if _stopping:
            from concurrent.futures.process import BrokenProcessPool

            raise BrokenProcessPool("the PDF extraction workers were stopped")
        if _pool is not None and _unusable(_pool):
            # A worker died while the pool was idle (the OOM killer, a kill):
            # the pool refuses every task from then on
            stale, _pool = _pool, None
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
        pool = _pool
    if stale is not None:
        logger.warning(
            "[PDF] The extraction workers failed ({}); starting new ones",
            getattr(stale, "_broken", None) or "shut down",
        )
        stale.shutdown(wait=False, cancel_futures=True)
    return pool


def _unusable(pool: ProcessPoolExecutor) -> bool:
    """Whether *pool* refuses tasks: broken (a worker died) or shut down."""
    return bool(
        getattr(pool, "_broken", False) or getattr(pool, "_shutdown_thread", False)
    )


def _pool_failed(pool: ProcessPoolExecutor, error: BaseException) -> bool:
    """Whether *error* is the pool's failure rather than the task's own.

    A worker that died breaks the whole pool, and a pool that was shut down
    cancels what it still held; an exception the task raised itself (an
    encrypted PDF, a page out of range) leaves the pool usable.
    """
    from concurrent.futures import BrokenExecutor, CancelledError

    return isinstance(error, (BrokenExecutor, CancelledError)) or _unusable(pool)


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    """Drop *pool*, which failed; the next conversion starts a new one.

    Only that pool: one another thread has already put in its place stays,
    with the other documents' work in it.
    """
    global _pool
    with _lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


def shutdown_pool() -> None:
    """Stop the workers now, cancelling what they have not finished.

    Idempotent. Called on exit (the CLI's ``os._exit`` skips ``atexit``, so
    ``finalize_process`` calls it too), on Ctrl-C (an extraction waiting on
    the workers then fails instead of finishing the document first), and
    by ``markitai serve`` before uvicorn re-raises its stop signal.

    The stop lasts until :func:`enable`: meanwhile a conversion that needs
    the workers fails at once, instead of starting new ones or extracting
    the document in-process while the process exits.
    """
    global _pool, _stopping
    with _lock:
        pool, _pool = _pool, None
        _stopping = True
    if pool is None:
        return
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            process.terminate()
        except Exception:
            pass
    for process in processes:
        try:
            process.join(timeout=1)
        except Exception:
            pass
    # With its workers gone the pool's manager thread ends and releases the
    # queues; their semaphores would otherwise be reported leaked at exit.
    # Bounded: a stop must never hang on it.
    closer = threading.Thread(target=pool.shutdown, kwargs={"wait": True}, daemon=True)
    closer.start()
    closer.join(timeout=2)


atexit.register(shutdown_pool)
