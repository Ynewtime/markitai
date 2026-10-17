"""Import markitdown completely, once, however many threads ask.

``markitdown`` and ``markitdown.converters`` import each other. Two threads
that first touch them through different doors (``from markitdown import
MarkItDown`` vs ``from markitdown.converters import ...``) each take one
module's import lock and wait for the other's: Python reports it as
``_DeadlockError`` and the file fails. Batch workers convert in threads, so
every first import goes through :func:`import_markitdown`.

A foundation module (like :mod:`markitai.utils.litellm_import`), so the
converters and the pure extraction domain can both use it.
"""

from __future__ import annotations

import importlib
import threading

_markitdown_import_lock = threading.Lock()


def import_markitdown() -> None:
    """Import markitdown completely, once, however many threads ask.

    Call before the first ``from markitdown... import``. Later calls find the
    package in ``sys.modules`` and return at once. Raises ``ImportError``
    when markitdown is not installed.
    """
    with _markitdown_import_lock:
        importlib.import_module("markitdown")
        importlib.import_module("markitdown.converters")
