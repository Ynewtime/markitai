"""Provider picker metadata for the shared workbench (API contract v1.5, section 8).

Each provider card gains the group, brand-mark key and aliases markitai-rust's
catalog gives the same provider, so the workbench's picker groups Python's
cards as it groups Rust's. ``provider_catalog.json`` is that catalog, exported
from markitai-rust (its ``source`` names the commit); refresh it when a newly
vendored workbench changes the catalog or its brand marks.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any


@cache
def _catalog() -> dict[str, dict[str, Any]]:
    text = (
        resources.files("markitai.serve")
        .joinpath("provider_catalog.json")
        .read_text("utf-8")
    )
    return {row["provider"]: row for row in json.loads(text)["providers"]}


def catalog_entry(provider: str) -> dict[str, Any] | None:
    """The catalog row of *provider*, or of the provider a further model prefix
    (``ollama_chat``, ``vertex_ai_beta``) is routed with."""
    table = _catalog()
    name = provider.lower()
    if name in table:
        return table[name]
    for row in table.values():
        # Aliases also hold searched names ("aws", "openai"); only the
        # prefix-shaped ones stand for a card's provider
        if "_" in name and name in row["aliases"]:
            return row
    return None


def annotate_card(card: dict[str, Any]) -> dict[str, Any]:
    """Add ``group``, ``icon`` and ``aliases`` to a provider card, keeping any
    the card already has; a provider the catalog lacks gets none of them (the
    workbench then lists it under All providers with a monogram)."""
    entry = catalog_entry(str(card.get("provider", "")))
    if entry is None:
        return card
    card.setdefault("group", entry["group"])
    card.setdefault("icon", entry["icon"])
    card.setdefault("aliases", list(entry["aliases"]))
    return card
