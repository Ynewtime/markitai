"""The command line's standard streams, set up before anything prints.

``markitai.cli`` calls this before importing its UI: ``markitai.utils.term``
picks the status marks from the streams' encoding when it is imported.
"""

from __future__ import annotations

import sys


def configure_standard_streams() -> None:
    """UTF-8 on Windows; elsewhere, replace what a non-UTF-8 locale cannot encode.

    Windows: consoles, pipes and files all get UTF-8 (a pipe would otherwise
    use the ANSI code page, which has no status marks), as in markitai-rust.
    A non-UTF-8 locale elsewhere (``LANG=en_US.ISO-8859-1``,
    ``PYTHONIOENCODING=ascii``): a character the stream cannot encode (a
    dash, a CJK file name) is replaced instead of crashing the run, and the
    status marks switch to ASCII (``markitai.utils.term``).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        if sys.platform == "win32":
            reconfigure(encoding="utf-8", errors="replace")
            continue
        encoding = (getattr(stream, "encoding", None) or "utf-8").lower()
        if not encoding.replace("-", "").startswith("utf"):
            reconfigure(errors="replace")
