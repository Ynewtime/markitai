"""Workbench round 18: one atomic batch adds and removes models, and every
routing weight a write sets is a whole number from 0 to 100.

A provider page that adds one model and removes another sent a DELETE, then a
batch whose ``credential_deployment_id`` named the model the provider card
stands for (the connection's first entry): after the DELETE that entry was
gone, or (hand-written entries) had a new id, so the batch answered 404
``settings_entry_missing`` after the removal had been saved. The batch now
takes ``remove`` and resolves every id against the list its revision names.
All keys and addresses are invented; nothing leaves the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.unit.serve.test_user_models import (
    ADMIN,
    USER_A_KEY,
    _add,
    _app,
    _revision,
    _server,
    _user,
    _why,
)

GARDEN_BASE = "https://garden.example/v1"
RULE = "Routing weight must be a whole number from 0 to 100"


def _garden() -> dict[str, Any]:
    """Three models of one connection written by hand (no ids)."""
    return {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": f"openai/{name}",
                        "api_key": "sk-fixture-garden-0000",
                        "api_base": GARDEN_BASE,
                    },
                }
                for name in ("garden-lite", "garden-flash", "garden-pro")
            ]
        }
    }


@pytest.mark.parametrize("removed", ["openai/garden-lite", "openai/garden-flash"])
async def test_one_batch_adds_and_removes_through_the_card_of_a_removed_model(
    tmp_path: Path, removed: str
) -> None:
    async with _server(_app(tmp_path, _garden())) as client:
        view = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        cards = (await client.get("/api/settings/llm/providers", headers=ADMIN)).json()
        card = next(c for c in cards["providers"] if c["kind"] == "configured")
        target = next(d for d in view["deployments"] if d["model"] == removed)
        saved = await client.post(
            "/api/settings/llm/deployments/batch",
            json={
                "expected_revision": view["revision"],
                "remove": [target["deployment_id"]],
                "deployments": [
                    {
                        "model_name": "default",
                        "model": "openai/garden-new",
                        "provider": "openai",
                        "credential_deployment_id": card["deployment_id"],
                    }
                ],
            },
            headers=ADMIN,
        )
    assert view["batch_remove"] is True
    assert card["deployment_id"] == view["deployments"][0]["deployment_id"]
    assert saved.status_code == 200, saved.text
    expected = ["openai/garden-lite", "openai/garden-flash", "openai/garden-pro"]
    expected.remove(removed)
    assert [d["model"] for d in saved.json()["deployments"]] == [
        *expected,
        "openai/garden-new",
    ]
    stored = json.loads((tmp_path / "config.json").read_text())
    providers = stored["llm"]["providers"]
    assert len(providers) == 1
    assert providers[0]["api_key"] == "sk-fixture-garden-0000"
    assert all(
        entry["model_info"]["provider_id"] == providers[0]["id"]
        for entry in stored["llm"]["model_list"]
    )


async def test_a_batch_is_all_or_nothing(tmp_path: Path) -> None:
    async with _server(_app(tmp_path, _garden())) as client:
        view = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        ids = [d["deployment_id"] for d in view["deployments"]]
        before = (tmp_path / "config.json").read_bytes()

        async def batch(body: dict[str, Any]) -> Any:
            return await client.post(
                "/api/settings/llm/deployments/batch",
                json={"expected_revision": view["revision"], **body},
                headers=ADMIN,
            )

        add = {"model_name": "default", "model": "openai/garden-new"}
        refused = [
            await batch({"remove": [ids[1], "no-such-id"], "deployments": [add]}),
            await batch({"remove": [ids[1], ids[1]]}),
            await batch({"remove": [ids[1]], "deployments": [{**add, "weight": 101}]}),
            await batch({"remove": [], "deployments": []}),
            await batch({"remove": [7]}),
        ]
        unchanged = (tmp_path / "config.json").read_bytes()
        removed = await batch({"remove": [ids[2], ids[0]]})
    assert [_why(r) for r in refused] == [
        (404, "settings_entry_missing"),
        (422, "invalid_settings_request"),
        (422, "invalid_weight"),
        (422, "invalid_settings_request"),
        (422, "invalid_settings_request"),
    ]
    assert unchanged == before
    assert removed.status_code == 200, removed.text
    assert [d["model"] for d in removed.json()["deployments"]] == [
        "openai/garden-flash"
    ]


async def test_every_weight_write_is_a_whole_number_from_0_to_100(
    tmp_path: Path,
) -> None:
    async with _server(_app(tmp_path, _garden())) as client:
        view = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        target = view["deployments"][1]["deployment_id"]
        refused = []
        for weight in (101, -1, 1.5, "3", True):
            refused.append(
                await client.post(
                    "/api/settings/llm/deployments/batch",
                    json={
                        "expected_revision": view["revision"],
                        "deployments": [
                            {"model_name": "x", "model": "openai/x", "weight": weight}
                        ],
                    },
                    headers=ADMIN,
                )
            )
            refused.append(
                await client.patch(
                    f"/api/settings/llm/deployments/{target}",
                    json={"expected_revision": view["revision"], "weight": weight},
                    headers=ADMIN,
                )
            )
        top = await client.patch(
            f"/api/settings/llm/deployments/{target}",
            json={"expected_revision": view["revision"], "weight": 100},
            headers=ADMIN,
        )
    assert {_why(r) for r in refused} == {(422, "invalid_weight")}
    assert {r.json()["detail"] for r in refused} == {RULE}
    assert top.status_code == 200, top.text


async def test_edit_config_refuses_a_new_weight_over_100_and_keeps_an_old_one(
    tmp_path: Path,
) -> None:
    config = _garden()
    config["llm"]["model_list"][2]["litellm_params"]["weight"] = 500
    async with _server(_app(tmp_path, config)) as client:
        loaded = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        document = (await client.get("/api/settings/config", headers=ADMIN)).json()
        value = json.loads(document["text"])

        async def put(change: Any) -> Any:
            edited = json.loads(document["text"])
            change(edited)
            return await client.put(
                "/api/settings/config",
                json={
                    "expected_revision": document["revision"],
                    "text": json.dumps(edited),
                },
                headers=ADMIN,
            )

        def weights(first: Any, third: Any) -> Any:
            def change(edited: Any) -> None:
                edited["llm"]["model_list"][0]["litellm_params"]["weight"] = first
                edited["llm"]["model_list"][2]["litellm_params"]["weight"] = third

            return change

        too_big = await put(weights(101, 500))
        fraction = await put(weights(2.5, 500))
        other_big = await put(weights(100, 600))
        kept = await put(weights(100, 500))
    assert [d["weight"] for d in loaded["deployments"]] == [1, 1, 500]
    assert value["llm"]["model_list"][2]["litellm_params"]["weight"] == 500
    assert _why(too_big) == (422, "invalid_weight")
    assert too_big.json()["detail"] == (
        f"/llm/model_list/0 (openai/garden-lite in default): {RULE}"
    )
    assert _why(fraction) == (422, "invalid_weight")
    assert other_big.json()["detail"].startswith("/llm/model_list/2 ")
    assert kept.status_code == 200, kept.text


async def test_a_users_batch_adds_and_removes_in_one_change(tmp_path: Path) -> None:
    async with _server(_app(tmp_path)) as client:
        _, a = await _user(client, "Laptop A")
        model = {"api_key": USER_A_KEY, "api_base": GARDEN_BASE, "weight": 2}
        added = await _add(
            client,
            a,
            {"model": "openai/garden-lite", **model},
            {"model": "openai/garden-flash", **model},
        )
        view = added.json()
        cards = (await client.get("/api/me/models/providers", headers=a)).json()
        card = next(c for c in cards["providers"] if c["kind"] == "configured")
        lite = view["deployments"][0]["deployment_id"]
        new = {
            "model_name": "default",
            "model": "openai/garden-new",
            "credential_provider_id": card["provider_id"],
        }
        unknown = await client.post(
            "/api/me/models/deployments/batch",
            json={"expected_revision": view["revision"], "remove": [lite, "nobody"]},
            headers=a,
        )
        heavy = await client.post(
            "/api/me/models/deployments/batch",
            json={
                "expected_revision": view["revision"],
                "remove": [lite],
                "deployments": [{**new, "weight": 150}],
            },
            headers=a,
        )
        unchanged = await _revision(client, a)
        saved = await client.post(
            "/api/me/models/deployments/batch",
            json={
                "expected_revision": view["revision"],
                "remove": [lite],
                "deployments": [{**new, "weight": 3}],
            },
            headers=a,
        )
    assert view["batch_remove"] is True
    assert _why(unknown) == (404, "settings_entry_missing")
    assert _why(heavy) == (422, "invalid_weight")
    assert unchanged == view["revision"]
    assert saved.status_code == 200, saved.text
    assert [(d["model"], d["weight"]) for d in saved.json()["deployments"]] == [
        ("openai/garden-flash", 2),
        ("openai/garden-new", 3),
    ]
