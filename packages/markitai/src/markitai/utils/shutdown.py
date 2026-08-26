"""Deterministic process exit for short-lived markitai processes.

Every markitai process loads onnxruntime, whether or not it will ever run a
model: markitdown's ``_stream_info`` imports Magika for content sniffing, and
Magika imports onnxruntime eagerly. Converting a plain ``.txt`` file pays for
it too.

onnxruntime's C++ statics are torn down after Python is finished, and under
load that teardown can abort the process:

    libc++abi: terminating due to uncaught exception of type
    std::__1::system_error: recursive_mutex lock failed: Invalid argument

The output has already been written by then, so the visible damage is exit
code 134 on a conversion that in fact succeeded — a false failure for any
script or CI job that checks the status. Nothing on the Python side can make
a third-party library's static destructors safe; what markitai can do is stop
reaching them once its own work is done.

Confirmed by keeping onnxruntime out of the import graph: the abort, which
appeared 3-12 times per full parallel test run, disappeared entirely across
three runs.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn


def finalize_process(code: int = 0) -> NoReturn:
    """Release what markitai owns, then leave without unwinding the process.

    Runs the cleanup that has user-visible consequences (tracked temporary
    directories) and flushes both streams, then exits through ``os._exit``,
    which skips interpreter shutdown and the native static destructors that
    can abort there.

    Only for a process markitai owns end to end — the CLI. A host embedding
    markitai as a library decides its own exit; this would take its process
    down with it.

    Args:
        code: Exit status to report.
    """
    from markitai.utils.paths import cleanup_tracked_temp_dirs

    try:
        cleanup_tracked_temp_dirs()
    except Exception:  # noqa: BLE001 - cleanup must not change the exit status
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 - a closed pipe is not a failure here
            pass
    os._exit(code)
