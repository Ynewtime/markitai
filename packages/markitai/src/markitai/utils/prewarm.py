"""Import LiteLLM in the background while a run does its local work.

``import litellm`` takes most of a second, and an LLM run pays it when the
first document reaches enhancement, after its conversion or fetch has
finished. Started as soon as the run knows it will use LLM, the import
overlaps that work instead.

The thread imports only ``litellm``, never a markitai module: when the
conversion later imports ``markitai.llm`` it waits for LiteLLM's module
lock, and the prewarm thread never waits for a lock the main thread holds,
so the two cannot deadlock on imports. Callbacks queued with ``after`` run
on that thread once the import is done, under the same rule (third-party
calls only; report through thread-safe logging).
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Callable

from loguru import logger

_lock = threading.Lock()
_thread: threading.Thread | None = None
_done = False
_callbacks: list[Callable[[], None]] = []


def prewarm_litellm(after: Callable[[], None] | None = None) -> None:
    """Start importing LiteLLM in a daemon thread (at most once per process).

    Args:
        after: Run on the prewarm thread once LiteLLM is imported (right
            away on a new thread when it already is).
    """
    global _thread
    with _lock:
        if after is not None:
            _callbacks.append(after)
        if _thread is not None and not _done:
            return  # the running thread picks the callback up
        if _thread is None and "litellm" not in sys.modules:
            # Same default as the CLI: no model cost map fetched over the network
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        elif not _callbacks:
            return
        _thread = threading.Thread(
            target=_run, name="markitai-litellm-prewarm", daemon=True
        )
        _thread.start()


def _run() -> None:
    global _done
    try:
        importlib.import_module("litellm")
    except Exception:  # the real import reports any failure where it matters
        pass
    while True:
        with _lock:
            if not _callbacks:
                _done = True
                return
            callback = _callbacks.pop(0)
        try:
            callback()
        except Exception as e:
            logger.debug("[Prewarm] Callback failed: {}", e)
