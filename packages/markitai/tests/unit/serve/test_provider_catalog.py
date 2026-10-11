"""Provider cards carry the workbench picker's group, brand mark and aliases.

Contract v1.5 (section 8): without ``group`` the picker lists every provider
but the common eight under All providers, with no filter. Python's cards get
the values markitai-rust's catalog gives the same provider.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from markitai.config import MarkitaiConfig, ModelConfig
from markitai.serve.provider_catalog import _catalog, annotate_card, catalog_entry
from tests.unit.serve.test_serve_settings import _make_app, _model_entry, _serve_client
from tests.unit.serve.test_workbench_contract import PROVIDER_GROUP

SERVE = Path(__file__).resolve().parents[3] / "src" / "markitai" / "serve"


def test_every_brand_mark_the_catalog_names_is_in_the_vendored_workbench() -> None:
    marks = set(
        re.findall(
            r"[{,]\s*\"?([\w$-]+)\"?:\{viewBox",
            (SERVE / "workbench" / "ui" / "brands.js").read_text(encoding="utf-8"),
        )
    )
    icons = {row["icon"] for row in _catalog().values() if row["icon"] is not None}
    assert marks, "no brand marks found in brands.js"
    assert icons <= marks, f"no mark for {sorted(icons - marks)}"


def test_rows_use_the_contract_groups() -> None:
    rows = json.loads((SERVE / "provider_catalog.json").read_text(encoding="utf-8"))
    groups = {row["group"] for row in rows["providers"]}
    assert groups == PROVIDER_GROUP


@pytest.mark.parametrize(
    ("prefix", "provider"),
    [
        ("openai", "openai"),
        ("Gemini", "gemini"),
        ("ollama_chat", "ollama"),
        ("vertex_ai_beta", "vertex_ai"),
        ("openai_like", "custom"),
    ],
)
def test_a_model_prefix_finds_its_provider(prefix: str, provider: str) -> None:
    entry = catalog_entry(prefix)
    assert entry is not None and entry["provider"] == provider


@pytest.mark.parametrize("name", ["aws", "google cloud", "my-proxy", ""])
def test_searched_names_and_unknown_providers_find_none(name: str) -> None:
    assert catalog_entry(name) is None


def test_a_card_keeps_what_it_already_says() -> None:
    card = annotate_card({"provider": "anthropic", "icon": None})
    assert card == {
        "provider": "anthropic",
        "icon": None,
        "group": "common",
        "aliases": ["claude"],
    }
    assert annotate_card({"provider": "my-proxy"}) == {"provider": "my-proxy"}


async def test_cards_of_every_kind_carry_the_catalog_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from markitai.providers import discovery

    cfg = MarkitaiConfig()
    cfg.llm.model_list = [
        _model_entry("ollama_chat/llama3", api_base="http://127.0.0.1:11434"),
        _model_entry("my-proxy/model-a", api_key="sk-proxy-0000-0000", model_name="b"),
    ]

    async def fake_connections(
        _configured: list[ModelConfig], *, refresh: bool = False
    ) -> list[dict[str, Any]]:
        def card(provider: str, kind: str, **extra: Any) -> dict[str, Any]:
            return {
                "id": f"{kind}:{provider}",
                "provider": provider,
                "label": provider,
                "kind": kind,
                "status": "ready",
                "source": "built_in",
                "supports_discovery": True,
                **extra,
            }

        return [
            card("claude-agent", "local_cli"),
            card("chatgpt", "oauth"),
            card("gemini", "environment", credential="env:GEMINI_API_KEY"),
            card("openai", "common"),
            card("custom", "common"),
        ]

    monkeypatch.setattr(discovery, "detect_provider_connections", fake_connections)
    async with _serve_client(_make_app(tmp_path, cfg)) as client:
        response = await client.get("/api/settings/llm/providers")
    assert response.status_code == 200, response.text
    cards = {card["provider"]: card for card in response.json()["providers"]}
    picked = {
        provider: (card.get("group"), card.get("icon"), card.get("aliases"))
        for provider, card in cards.items()
    }
    assert picked == {
        "claude-agent": ("subscriptions", "claude", ["claude code", "anthropic"]),
        "chatgpt": ("subscriptions", "openai", ["codex", "openai"]),
        "gemini": ("common", "gemini", ["google", "google ai studio"]),
        "openai": ("common", "openai", []),
        # No mark: the picker draws a generic plug for it
        "custom": (
            "local",
            None,
            ["openai_like", "custom", "self-hosted", "llama.cpp"],
        ),
        # A saved model under a further prefix is its provider's connection
        "ollama_chat": ("common", "ollama", ["ollama_chat"]),
        # Not in the catalog: listed under All providers, with a monogram
        "my-proxy": (None, None, None),
    }
    assert "icon" in cards["custom"] and "group" not in cards["my-proxy"]
