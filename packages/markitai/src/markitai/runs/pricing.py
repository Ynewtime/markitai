"""Request coverage and models of a conversion's LLM usage.

Shared by ``markitai serve`` (an item's ``pricing``) and the run history the
CLI records, so both say the same about a cost that is partly or wholly
unknown.
"""

from __future__ import annotations

from typing import Any


def pricing_from_usage(
    usage: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Request coverage of one item's model cost, or None when no model ran.

    A request whose price is unknown is counted in ``unpriced_requests``; a
    cost is ``complete`` only when every recorded request was priced.
    """
    requests = unpriced = 0
    for stats in (usage or {}).values():
        if not isinstance(stats, dict):
            continue
        requests += max(0, int(stats.get("requests") or 0))
        unpriced += max(0, int(stats.get("unpriced_requests") or 0))
    if requests <= 0:
        return None
    unpriced = min(unpriced, requests)
    priced = requests - unpriced
    return {
        "priced_requests": priced,
        "unpriced_requests": unpriced,
        "cost_status": cost_status(priced, unpriced),
    }


def cost_status(priced: int, unpriced: int) -> str:
    if priced == 0:
        return "unknown"
    return "partial" if unpriced else "complete"


def models_from_usage(usage: dict[str, dict[str, Any]] | None) -> list[str]:
    """Models that answered requests for an item, first use first."""
    return [
        str(name)
        for name, stats in (usage or {}).items()
        if isinstance(stats, dict) and int(stats.get("requests") or 0) > 0
    ]


#: The processor's own once-per-run notice (llm/processor.py): in a batch it
#: lands on whichever item answered first, so the serve items carry
#: :func:`unpriced_warning` instead, on every item that has an unpriced request.
PROCESSOR_UNPRICED = "No price is known for LLM model "


def unpriced_warning(usage: dict[str, dict[str, Any]] | None) -> str | None:
    """The "Cost is incomplete" notice of one item, in the words the workbench
    (and Rust) use: which models' requests have no reviewed price."""
    rows = [
        (str(model), int(stats.get("unpriced_requests") or 0))
        for model, stats in (usage or {}).items()
        if isinstance(stats, dict)
    ]
    rows = [(model, count) for model, count in rows if count > 0]
    if not rows:
        return None
    parts = [
        f"{count} request{'' if count == 1 else 's'} to {model} "
        f"{'has' if count == 1 else 'have'} no reviewed price"
        for model, count in rows[:3]
    ]
    if len(rows) > 3:
        parts.append(f"{len(rows) - 3} more models")
    return (
        f"Cost is incomplete: {'; '.join(parts)}. cost_usd is the known priced "
        "subtotal; the complete cost is unknown."
    )
