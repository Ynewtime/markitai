"""User-facing notices: persistent first-use notices and actionable warnings.

Persistent notices (:func:`notify_once`) are informational records; they
never grant consent.

Actionable warnings (:func:`user_notice`) are the handful of WARNING lines a
user can do something about -- "these pages look scanned, re-run with
--ocr", "slides could not be rendered, install LibreOffice", "OCR found no
text". They are ordinary loguru warnings tagged with :data:`USER_NOTICE`,
which lets the presentation layer show them where it shows nothing else:
the quiet single-file console (which otherwise only lets errors through) and
the batch summary (the console log handler is detached while the progress
bar is live). ``--quiet`` still hides them. Everything else stays a plain
warning, so routine diagnostics do not leak into the default console.

Hosts without a console (``markitai serve``, the Python API, the MCP server)
hand the notices to the caller instead: :func:`capture_task_notices` collects
the notices raised by the current task only, through a context variable, so
concurrent conversions never see each other's notices.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger

#: loguru ``extra`` key marking a record as an actionable user-facing notice.
USER_NOTICE = "user_notice"


def notify_once(key: str, message: str, notify: Callable[[str], None]) -> bool:
    """Show a notice once per OS user, across runs and concurrent processes.

    A marker stores only the notice identifier, never URLs or credentials.
    If the state directory is unwritable, delivery still succeeds; the
    caller's process-local guard prevents repetition within that run.
    """
    marker = Path.home() / ".markitai" / "notices" / key
    recorded = False
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        recorded = True
    except FileExistsError:
        return False
    except OSError:
        logger.debug("Cannot persist first-use notice {}", key)
    try:
        notify(message)
    except Exception:
        if recorded:
            marker.unlink(missing_ok=True)
        raise
    return True


class _TaskNotices:
    """Notices captured for one task, forwarded to an enclosing capture."""

    def __init__(self, parent: _TaskNotices | None) -> None:
        self.notices: list[str] = []
        self._parent = parent
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        with self._lock:
            if text not in self.notices:
                self.notices.append(text)
        if self._parent is not None:
            self._parent.add(text)


#: The capture of the running task, if any. A context variable, so each
#: asyncio task (and the converter threads it runs work in, which get a copy
#: of its context) reports to its own capture.
_TASK_NOTICES: ContextVar[_TaskNotices | None] = ContextVar(
    "markitai_task_notices", default=None
)


def user_notice(message: str, *args: Any) -> None:
    """Log an actionable warning the user should see even in default mode.

    Same call shape as ``logger.warning`` (``{}`` placeholders filled from
    ``args``). Name the file in the message: in a batch the notice is
    listed in the summary, away from the item that raised it. Inside
    :func:`capture_task_notices` the formatted text is also recorded there.
    """
    capture = _TASK_NOTICES.get()
    if capture is not None:
        # loguru formats the same way: str.format only when args are given
        capture.add(message.format(*args) if args else message)
    logger.bind(**{USER_NOTICE: True}).opt(depth=1).warning(message, *args)


@contextmanager
def capture_task_notices() -> Iterator[list[str]]:
    """Collect the user notices raised by the current task.

    Unlike :class:`UserNoticeCollector` (a global loguru sink, right for the
    one CLI run of a process), the capture is scoped to the current context:
    a notice lands here only when it is raised by this task, by work it
    awaits, or by a converter thread running on a copy of its context.
    Concurrent tasks (serve job items, MCP batch items) each get their own
    list. A nested capture also forwards its notices to the enclosing one.

    Yields:
        The notices (deduplicated, in first arrival order); the list keeps
        filling while the block runs and is complete once it exits.
    """
    capture = _TaskNotices(_TASK_NOTICES.get())
    token = _TASK_NOTICES.set(capture)
    try:
        yield capture.notices
    finally:
        _TASK_NOTICES.reset(token)


def is_user_notice(record: Any) -> bool:
    """Loguru filter: True for records logged through :func:`user_notice`."""
    extra = record.get("extra", {}) if isinstance(record, dict) else {}
    return bool(extra.get(USER_NOTICE))


class UserNoticeCollector:
    """Capture the user notices logged between :meth:`start` and :meth:`stop`.

    :attr:`notices` fills in as notices arrive (deduplicated, in first
    arrival order). Converter threads log concurrently, so appends are
    serialized. Stopping removes the capture sink and keeps the list.
    """

    def __init__(self) -> None:
        self.notices: list[str] = []
        self._lock = threading.Lock()
        self._sink_id: int | None = None

    def _sink(self, message: Any) -> None:
        text = str(message.record["message"])
        with self._lock:
            if text not in self.notices:
                self.notices.append(text)

    def start(self) -> None:
        """Attach the capture sink (idempotent)."""
        if self._sink_id is None:
            self._sink_id = logger.add(
                self._sink, level="WARNING", filter=is_user_notice, format="{message}"
            )

    def stop(self) -> None:
        """Detach the capture sink (idempotent); the notices are kept."""
        if self._sink_id is not None:
            try:
                logger.remove(self._sink_id)
            except ValueError:  # logger reconfigured underneath us
                pass
            self._sink_id = None


@contextmanager
def collect_user_notices() -> Iterator[list[str]]:
    """Context-manager form of :class:`UserNoticeCollector`."""
    collector = UserNoticeCollector()
    collector.start()
    try:
        yield collector.notices
    finally:
        collector.stop()
