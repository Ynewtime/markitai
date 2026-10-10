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

``prepare_litellm_import`` puts a guard first on those loggers. While
LiteLLM is being imported, the guard hands each record straight to the
logger's handlers and ends the filter chain, so LiteLLM's filter never runs
mid-import; afterwards it lets every record through untouched. markitai's own
log redaction (``redact_log_record``) still masks keys and the ``serve``
token on the records it passes.
"""

from __future__ import annotations

import logging
import os
import sys

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
