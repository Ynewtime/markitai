"""Machine-readable run result for ``markitai --json``.

The CLI's four conversion paths already collect one :class:`Outcome` per work
item for ``--record-history``. This module renders that same collection as a
single JSON document on stdout, so a script or agent can read
``{source, status, output, error, cost_usd}`` without parsing Rich panels or
hunting for the batch report file.

The module holds no process state: the CLI's own ``--json`` flag decides
whether the envelope is emitted, and every function here is pure. That keeps
one source of truth for "is JSON mode on" and makes the renderers trivial to
test.

This module sits below :mod:`markitai.cli` and must never import it (the
import-layer contracts in the root ``pyproject.toml`` make that direction
forbidden); the CLI imports this module, never the other way round.
"""

from __future__ import annotations

import json
from typing import Any

from markitai.runs.types import Outcome

ENVELOPE_VERSION = "1.0"


def _item(outcome: Outcome) -> dict[str, Any]:
    """Render one Outcome as a stable, flat JSON object."""
    return {
        "kind": outcome.kind,
        "source": outcome.source,
        "status": outcome.status,
        "output": str(outcome.output_path) if outcome.output_path else None,
        "error": outcome.error,
        "skip_reason": outcome.skip_reason,
        "images": outcome.images,
        "screenshots": outcome.screenshots,
        "cost_usd": round(outcome.cost_usd, 6),
        "duration_s": (
            round(outcome.duration, 3) if outcome.duration is not None else None
        ),
        "cache_hit": outcome.cache_hit,
        "fetch_cache_hit": outcome.fetch_cache_hit,
        "llm_cache_hit": outcome.llm_cache_hit,
        "fetch_strategy": outcome.fetch_strategy,
        "source_file": outcome.source_file,
        "llm_usage": outcome.llm_usage,
    }


def build_envelope(
    outcomes: list[Outcome], *, error: str | None = None
) -> dict[str, Any]:
    """Build the top-level ``--json`` document from collected outcomes.

    Args:
        outcomes: One Outcome per work item, in completion order.
        error: A run-level failure that produced no Outcome (a missing input
            path, an unsupported format, an interrupt). Without it, a run that
            died before its first item would report ``ok: true`` with an empty
            list while exiting non-zero.

    Returns:
        ``{"version", "ok", "error", "items", "totals"}`` where ``ok`` is False
        when any item failed or a run-level error was reported (partial batch
        failure included).
    """
    items = [_item(outcome) for outcome in outcomes]
    completed = sum(1 for i in items if i["status"] == "completed")
    failed = sum(1 for i in items if i["status"] == "failed")
    skipped = sum(1 for i in items if i["status"] == "skipped")
    return {
        "version": ENVELOPE_VERSION,
        "ok": failed == 0 and error is None,
        "error": error,
        "items": items,
        "totals": {
            "total": len(items),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "cost_usd": round(sum(i["cost_usd"] for i in items), 6),
            "duration_s": round(
                sum(i["duration_s"] or 0.0 for i in items),
                3,
            ),
        },
    }


def render(outcomes: list[Outcome], *, error: str | None = None) -> str:
    """Render the envelope as pretty JSON with a trailing newline.

    ``ensure_ascii`` stays on: a CJK source name written raw raises
    ``UnicodeEncodeError`` when stdout is an ASCII stream (``LC_ALL=C``, a
    Windows pipe), and escaped JSON is still valid UTF-8 JSON for ``jq``.
    """
    return (
        json.dumps(build_envelope(outcomes, error=error), indent=2, ensure_ascii=True)
        + "\n"
    )


__all__ = [
    "ENVELOPE_VERSION",
    "build_envelope",
    "render",
]
