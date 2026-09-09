"""Native parser-noise suppression shared by every entry point.

PyMuPDF and ONNX Runtime emit diagnostics from C/C++ code that bypasses
Python logging entirely. These helpers historically lived in
``cli.logging_config`` and only ran once the CLI had initialized logging,
so a plain library call (``markitai.convert``) still leaked parser noise
into the process streams. They live here — in the foundation layer, free
of any CLI dependency — so the CLI, the serve app, and the public API all
share one idempotent implementation.
"""

from __future__ import annotations

import os

_APPLIED = False


def _suppress_onnx_runtime_logs() -> None:
    """Suppress ONNX Runtime C++ logs via environment variables.

    ONNX Runtime logs directly to stderr in C++, bypassing Python logging.
    Must be called before any ONNX Runtime imports.
    """
    # Suppress ONNX Runtime session logging
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # WARNING level
    os.environ.setdefault("ORT_CPP_LOG_SEVERITY_LEVEL", "3")


def _suppress_mupdf_logs() -> None:
    """Suppress MuPDF C-level logs that bypass Python logging.

    MuPDF (via PyMuPDF) logs directly to stderr, which can clutter output
    with format warnings (e.g., "No common ancestor in structure tree").

    Imports ``pymupdf``, never the legacy ``fitz`` alias: since PyMuPDF
    1.28.2 the alias prints its deprecation notice on **stdout**, which lands
    inside piped markdown (`markitai doc.pdf | ...`) — i.e. a noise-
    suppression helper that emitted noise of its own.
    """
    try:
        # PyMuPDF might not be installed in all environments
        import pymupdf

        if hasattr(pymupdf, "TOOLS") and hasattr(pymupdf.TOOLS, "mupdf_display_errors"):
            pymupdf.TOOLS.mupdf_display_errors(False)
    except ImportError:
        pass


def suppress_parser_noise() -> None:
    """Apply all native parser-noise suppressions, once per process.

    Idempotent: safe to call from every entry point (CLI startup, each
    ``markitai.convert()`` call) — repeat calls return immediately.
    """
    global _APPLIED
    if _APPLIED:
        return
    _suppress_onnx_runtime_logs()
    _suppress_mupdf_logs()
    _APPLIED = True
