"""Prepare the process for ``import litellm``, whichever thread runs it.

LiteLLM 1.104.2 attaches secret-redaction filters to third-party loggers
(``uvicorn.access``, ``uvicorn.error``, ``httpx``, ``asyncio``, ``backoff``,
``apscheduler``) early in its ``__init__``, and every one of those filters
lazily imports ``litellm.rust_bridge.catalog``. That import only works once
``litellm/__init__`` has finished: ``catalog`` imports ``litellm.types``, and
``litellm/__init__`` itself reaches ``catalog`` later through ``utils``. A
record on one of those loggers from another thread while the import is still
running enters the cycle from the wrong end: the logging thread fails with
"cannot import name 'RouteContext' from partially initialized module
'litellm.rust_bridge.catalog'", and the importing thread can fail with a
``_DeadlockError``. markitai imports LiteLLM on a prewarm thread while the
server answers requests (every request writes an ``uvicorn.access`` record)
and while conversions run, so ``serve``'s first requests could be dropped and
LiteLLM left half imported.

A second trap is Python's own import locks. Importing a submodule
(``from litellm.x.y import z``) takes ``litellm.x.y``'s lock *before* it
imports the ``litellm`` package, and ``litellm/__init__`` imports some of those
submodules itself. With the prewarm thread inside ``litellm/__init__``,
another thread that starts with a submodule holds the lock the prewarm thread
needs next while it waits for the package: Python breaks the deadlock by
failing one import, and LiteLLM stays half imported for the whole process
(every model call then fails with "partially initialized module 'litellm'").
:func:`import_litellm` imports the package first; call it before importing
any LiteLLM submodule on a path that can run while the prewarm thread works.

``prepare_litellm_import`` puts a guard first on those loggers. While
LiteLLM is being imported, the guard hands each record straight to the
logger's handlers and ends the filter chain, so LiteLLM's filter never runs
mid-import; afterwards it lets every record through untouched. markitai's own
log redaction (``redact_log_record``) still masks keys and the ``serve``
token on the records it passes.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from types import ModuleType

# The loggers LiteLLM 1.104.2 attaches import-dependent filters to
# (litellm/_logging.py: _REDACTED_THIRD_PARTY_LOGGERS, _REDACTED_ACCESS_LOGGERS).
_LITELLM_FILTERED_LOGGERS = (
    "apscheduler.executors.default",
    "apscheduler.scheduler",
    "asyncio",
    "backoff",
    "httpx",
    "uvicorn.access",
    "uvicorn.error",
)


def litellm_is_importing() -> bool:
    """Whether some thread is executing ``litellm/__init__`` right now."""
    module = sys.modules.get("litellm")
    return bool(getattr(getattr(module, "__spec__", None), "_initializing", False))


class _ImportGuard(logging.Filter):
    """Keep one logger's later filters from running while LiteLLM imports."""

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self._logger = logger

    def filter(self, record: logging.LogRecord) -> bool:
        if not litellm_is_importing():
            return True
        self._logger.callHandlers(record)
        return False


def prepare_litellm_import() -> None:
    """Call before markitai can first import LiteLLM. Idempotent and cheap.

    Also keeps LiteLLM from fetching its model cost map over the network on
    import (an explicit ``LITELLM_LOCAL_MODEL_COST_MAP`` wins).
    """
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in _LITELLM_FILTERED_LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(f, _ImportGuard) for f in logger.filters):
            # First, so it runs before any filter LiteLLM appends
            logger.filters.insert(0, _ImportGuard(logger))


def import_litellm() -> ModuleType:
    """Import the ``litellm`` package (waiting for another thread that is
    importing it), so that importing its submodules afterwards takes no lock
    ``litellm/__init__`` needs. Raises what the import raises."""
    prepare_litellm_import()
    return importlib.import_module("litellm")
