"""Persistent, informational notices; these records never grant consent."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from loguru import logger


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
