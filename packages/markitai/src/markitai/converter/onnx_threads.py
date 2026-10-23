"""onnxruntime threads for the models markitai runs in its own process.

An onnxruntime session's intra-op pool defaults to every core, its threads
spinning while they wait. The layout models pymupdf4llm loads and the
RapidOCR engine run in this process for short documents, and a job converts
several files at once: each session spread over every core, they spent
four times the CPU for the same wall time (a 5-page PDF 4.6 CPU seconds in
0.45 s, a 3-page scan 15 CPU seconds a page), and once there are more
threads than free cores (a laptop's efficiency cores, which onnxruntime
counts like the others) the pages wait for descheduled threads: 1.4 to 1.9
times slower on four cores. Four threads are as fast on these models.

The PDF extraction workers (``pdf_parallel``) run their sessions on one
thread each and are not concerned here.
"""

from __future__ import annotations

import threading
from types import ModuleType
from typing import Any

#: Threads one in-process session uses at most.
LOCAL_SESSION_THREADS = 4

_lock = threading.Lock()
_patched = 0
_original: Any = None


def local_threads() -> int:
    """Intra-op threads for a session in this process: four, or the cores
    the process may run on when fewer."""
    from markitai.converter.pdf_parallel import usable_cores

    return max(1, min(LOCAL_SESSION_THREADS, usable_cores()))


def _is_layout_model(model: Any) -> bool:
    """A model file of PyMuPDF-Layout (``pymupdf/layout/resources/...``)."""
    path = str(model).replace("\\", "/")
    return "/pymupdf/layout/" in path


def _cap(options: Any) -> None:
    if not options.intra_op_num_threads:
        options.intra_op_num_threads = local_threads()
    if not options.inter_op_num_threads:
        options.inter_op_num_threads = 1


def import_pymupdf4llm() -> ModuleType:
    """``import pymupdf4llm``, its layout models capped at :func:`local_threads`.

    pymupdf4llm loads its three layout sessions while it is imported. For
    that import only, sessions built from PyMuPDF-Layout's own model files
    get the cap; any other session (a host program's, built on another
    thread meanwhile) is left as its owner made it. Already imported (by a
    host program, or a worker that set its own threads), it is returned as
    it is.
    """
    import importlib
    import sys

    loaded = sys.modules.get("pymupdf4llm")
    if loaded is not None and not getattr(loaded.__spec__, "_initializing", False):
        return loaded
    # Being imported on another thread (serve's warm-up): the import below
    # waits for it instead of returning the half-loaded module
    try:
        import onnxruntime as ort
    except ImportError:  # no layout engine without it: nothing to cap
        return importlib.import_module("pymupdf4llm")

    global _patched, _original
    with _lock:
        if _patched == 0:
            _original = ort.InferenceSession
            original = _original

            class _LocalSession(original):  # type: ignore[misc,valid-type]
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    model = args[0] if args else kwargs.get("path_or_bytes")
                    if _is_layout_model(model):
                        options = kwargs.get("sess_options")
                        if options is None and len(args) > 1:
                            options = args[1]
                        if options is None:
                            options = ort.SessionOptions()
                            kwargs["sess_options"] = options
                        _cap(options)
                    super().__init__(*args, **kwargs)

            ort.InferenceSession = _LocalSession  # type: ignore[misc]
        _patched += 1
    try:
        return importlib.import_module("pymupdf4llm")
    finally:
        with _lock:
            _patched -= 1
            if _patched == 0:
                ort.InferenceSession = _original  # type: ignore[misc]
                _original = None


#: Libraries the office and image converters load on first use.
_CONVERTER_LIBRARIES = ("numpy", "PIL.Image", "lxml.etree", "pptx")
_warm_started = False


def warm_up_in_background() -> None:
    """Load the PDF layout models and converter libraries in a daemon thread.

    For ``markitai serve``: its first job otherwise paid them, and every
    file of that job waited (pymupdf4llm loads three ONNX models as it is
    imported). Once per process. As for serve's LiteLLM prewarm, the thread
    imports only third-party modules: the markitai module the cap reads its
    core count from is imported here first.
    """
    global _warm_started
    import importlib

    with _lock:
        if _warm_started:
            return
        _warm_started = True
    importlib.import_module("markitai.converter.pdf_parallel")

    def run() -> None:
        try:
            import_pymupdf4llm()
        except Exception:  # the conversion reports it where it matters
            pass
        for name in _CONVERTER_LIBRARIES:
            try:
                importlib.import_module(name)
            except Exception:
                pass

    threading.Thread(target=run, name="markitai-converter-prewarm", daemon=True).start()
