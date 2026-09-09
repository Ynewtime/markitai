"""Timestamps shared by the serve layer and the run history it persists."""

from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    """Browser-portable RFC 3339 timestamp with millisecond precision."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
