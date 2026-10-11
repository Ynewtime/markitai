"""Provider picker metadata for the shared workbench (API contract v1.5, section 8).

Each provider card gains the group, brand-mark key and aliases markitai-rust's
catalog gives the same provider, so the workbench's picker groups Python's
cards as it groups Rust's. ``provider_catalog.json`` is that catalog, exported
from markitai-rust (its ``source`` names the commit); refresh it when a newly
vendored workbench changes the catalog or its brand marks.

``card_facts`` adds what the picker's provider page needs to connect a
provider (its default address, key variable, whether a key is optional), in
the form this build routes it (``markitai.providers.connectable``).
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


def card_facts(card: dict[str, Any]) -> dict[str, Any]:
    """Add the catalog's label and the connection facts to a provider card.

    ``default_base`` (null when an address must be entered), ``key_variable``
    and ``key_optional`` are those of ``markitai.providers.connectable``;
    a subscription runtime takes neither key nor address. Values the card
    already has stay.
    """
    from markitai.providers.connectable import connectable

    provider = str(card.get("provider", ""))
    entry = _catalog().get(provider.lower())
    if entry is not None and card.get("kind") in {"common", "configured"}:
        card["label"] = entry["label"]
    row = connectable(provider)
    if row is not None:
        card.setdefault("default_base", row.base)
        card.setdefault("key_variable", row.key_variable)
        card.setdefault("key_optional", row.key_optional)
    elif entry is not None and entry["group"] == "subscriptions":
        card.setdefault("default_base", None)
        card.setdefault("key_variable", None)
        card.setdefault("key_optional", True)
    return card
