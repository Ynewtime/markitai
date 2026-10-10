"""Logging on another thread while LiteLLM is being imported.

LiteLLM 1.104.2 attaches filters to ``uvicorn.access`` and other third-party
loggers early in its ``__init__``; each filter lazily imports
``litellm.rust_bridge.catalog``, which only works once the import is done.
``serve`` imports LiteLLM on its prewarm thread while the event loop writes
an access record for every request, so the first requests failed with
"cannot import name 'RouteContext' from partially initialized module" (seen
on Windows as an aborted connection) and the import itself could die with a
``_DeadlockError``, leaving LiteLLM half imported.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
import types

import pytest

from markitai.utils import litellm_import

# Run in a fresh interpreter: the race needs LiteLLM's first import. Only
# public pieces markitai had before the fix, so it fails on the old code.
_RACE = textwrap.dedent(
    """
    import logging, sys, time

    access = logging.getLogger("uvicorn.access")
    access.setLevel(logging.INFO)
    access.addHandler(logging.NullHandler())
    access.propagate = False

    from markitai.utils import prewarm

    prewarm.prewarm_litellm()  # imports LiteLLM on its own thread, as serve does
    deadline = time.monotonic() + 120
    records = 0
    while not prewarm._done and time.monotonic() < deadline:
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:1", "GET", "/api/x", "1.1", 200)
        records += 1
        time.sleep(0.0002)
    litellm = sys.modules.get("litellm")
    assert prewarm._done, "prewarm never finished"
    assert litellm is not None and not litellm.__spec__._initializing, "half imported"
    assert callable(getattr(litellm, "completion", None)), "LiteLLM import failed"
    print(f"ok records={records}")
    """
)


@pytest.mark.parametrize("attempt", range(3))
def test_access_records_during_the_prewarm_import_do_not_break_it(
    attempt: int,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _RACE],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.startswith("ok records="), result.stdout


class _Raising(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        raise ImportError("partially initialized module (simulated)")


@pytest.fixture
def guarded_logger(monkeypatch: pytest.MonkeyPatch):
    """A guarded logger whose next filter fails like LiteLLM's mid-import."""
    logger = logging.getLogger("uvicorn.access")
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Keep()
    monkeypatch.setattr(logger, "filters", [])
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "disabled", False)
    logger.setLevel(logging.INFO)
    litellm_import.prepare_litellm_import()
    logger.addFilter(_Raising())  # appended after the guard, as LiteLLM does
    return logger, records


def test_mid_import_records_reach_the_handlers_without_litellm_filters(
    guarded_logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger, records = guarded_logger
    importing = types.ModuleType("litellm")
    importing.__spec__ = types.SimpleNamespace(_initializing=True)  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "litellm", importing)

    logger.info("GET /api/capabilities 200")

    assert [r.getMessage() for r in records] == ["GET /api/capabilities 200"]


def test_after_the_import_every_filter_runs_again(
    guarded_logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger, records = guarded_logger
    imported = types.ModuleType("litellm")
    imported.__spec__ = types.SimpleNamespace(_initializing=False)  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "litellm", imported)

    with pytest.raises(ImportError):
        logger.info("GET /api/capabilities 200")  # LiteLLM's filter is reached
    assert records == []


def test_the_guard_is_installed_once_and_first() -> None:
    litellm_import.prepare_litellm_import()
    litellm_import.prepare_litellm_import()
    for name in ("uvicorn.access", "uvicorn.error", "httpx", "asyncio"):
        filters = logging.getLogger(name).filters
        guards = [f for f in filters if isinstance(f, litellm_import._ImportGuard)]
        assert len(guards) == 1
        assert filters[0] is guards[0]
