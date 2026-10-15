"""Workbench contract v1.16: users weigh and test shared models, and edit
their own models as text.

A user's own weight for a model the admin shares lives in its own file
(``shared_weights``, only while it differs from the admin's) and applies to
its pool only; a Test of a shared model runs on the server with the admin's
settings and says only whether it passed; ``/api/me/config`` shows and saves
the user's models, connections and shared weights as one masked document.
All keys and addresses are invented; nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from markitai.serve import user_models
from markitai.serve.user_models import Lists, UserModels
from tests.unit.serve.test_user_models import (
    ADMIN,
    ADMIN_BASE,
    ADMIN_KEY,
    USER_A_KEY,
    _add,
    _app,
    _base,
    _revision,
    _server,
    _user,
    _why,
    record_network,
)
from tests.unit.serve.test_workbench_contract import check_shape


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every model test or discovery that would send a request, recorded."""
    return record_network(monkeypatch)


def _config() -> dict[str, Any]:
    """An admin with two shared models (weights 3 and 1) and a private one."""
    return {
        "llm": {
            "model_list": [
                {
                    "model_name": "fast",
                    "litellm_params": {
                        "model": "openai/garden-large",
                        "api_key": ADMIN_KEY,
                        "api_base": ADMIN_BASE,
                        "weight": 3,
                    },
                    "model_info": {
                        "id": "s1",
                        "provider_id": "admin-1",
                        "share": "shared",
                    },
                },
                {
                    "model_name": "fast",
                    "litellm_params": {
                        "model": "openai/garden-small",
                        "api_key": ADMIN_KEY,
                        "api_base": ADMIN_BASE,
                    },
                    "model_info": {"id": "s2", "share": "shared"},
                },
                {
                    "model_name": "fast",
                    "litellm_params": {
                        "model": "openai/admin-private",
                        "api_key": ADMIN_KEY,
                        "api_base": ADMIN_BASE,
                    },
                    "model_info": {"id": "p1"},
                },
            ],
            "providers": [
                {
                    "id": "admin-1",
                    "provider": "custom",
                    "api_key": ADMIN_KEY,
                    "api_base": ADMIN_BASE,
                }
            ],
        }
    }


async def _weigh(
    client: Any, headers: dict[str, str], shared_id: str, weight: Any, **extra: Any
) -> Any:
    return await client.patch(
        f"/api/me/models/shared/{shared_id}",
        json={
            "expected_revision": await _revision(client, headers),
            "weight": weight,
            **extra,
        },
        headers=headers,
    )


def _shared(view: dict[str, Any]) -> list[tuple[str, int, int]]:
    return [(row["id"], row["weight"], row["default_weight"]) for row in view["shared"]]


# ---------------------------------------------------------------- weights


def test_a_users_weight_for_a_shared_model_applies_to_its_pool_only() -> None:
    from markitai.config import ModelConfig

    admin = [
        ModelConfig.model_validate(entry) for entry in _config()["llm"]["model_list"]
    ]
    weights = lambda pool: [m.litellm_params.weight for m in pool.models]  # noqa: E731
    names = lambda pool: [m.litellm_params.model for m in pool.models]  # noqa: E731
    plain = user_models.build_pool(_base(), Lists(), admin, [])
    assert names(plain) == ["openai/garden-large", "openai/garden-small"]
    assert weights(plain) == [3, 1]
    # A heavier weight for one, 0 for the other: only the first routes, with
    # the user's weight; an override of an id no longer shared changes nothing
    own = Lists(shared_weights={"s1": 7, "s2": 0, "gone": 5})
    pool = user_models.build_pool(_base(), own, admin, [])
    assert (names(pool), weights(pool)) == (["openai/garden-large"], [7])
    # Every shared model off for this user: an empty, sealed pool
    off = user_models.build_pool(
        _base(), Lists(shared_weights={"s1": 0, "s2": 0}), admin, []
    )
    assert off.models == []
    assert off.config.llm.enabled is False
    # The admin's entries are untouched
    assert [m.litellm_params.weight for m in admin] == [3, 1, 1]


def test_a_files_shared_weights_are_kept_and_count_in_its_revision(
    tmp_path: Path,
) -> None:
    from markitai.serve.app import _settings_revision, _user_revision

    directory = tmp_path / "serve" / "user-models"
    users = UserModels(directory)
    entry = {
        "model_name": "default",
        "litellm_params": {"model": "openai/own", "api_key": USER_A_KEY},
    }
    plain = users.write(
        "u_000000000001", lambda: True, lambda lists: lists.models.append(entry)
    )
    path = directory / "u_000000000001.json"
    # No overrides: v1.15's revision, and no shared_weights in the file
    assert _user_revision(plain) == _settings_revision(plain.models, plain.providers)
    assert "shared_weights" not in path.read_text()
    weighted = users.write(
        "u_000000000001",
        lambda: True,
        lambda lists: lists.shared_weights.update({"s1": 0}),
    )
    assert _user_revision(weighted) != _user_revision(plain)
    saved = json.loads(path.read_text())
    assert (saved["version"], saved["shared_weights"]) == (1, {"s1": 0})
    assert UserModels(directory).lists("u_000000000001") == weighted
    # A change that changes nothing writes nothing
    path.write_bytes(b"{not json")
    users.write("u_000000000001", lambda: True, lambda _lists: None)
    assert path.read_bytes() == b"{not json"
    # A file with a weight that is no weight is not read
    path.write_text(
        json.dumps(
            {"version": 1, "models": [], "providers": [], "shared_weights": {"s1": -1}}
        )
    )
    assert UserModels(directory).lists("u_000000000001") == Lists()


async def test_users_weigh_shared_models_for_themselves_only(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path, _config())) as client:
        a_id, a = await _user(client, "Laptop A")
        _b_id, b = await _user(client, "Laptop B")
        first = (await client.get("/api/me/models", headers=a)).json()
        off = await _weigh(client, a, "s2", 0)
        heavier = await _weigh(client, a, "s1", 5)
        capabilities = (await client.get("/api/capabilities", headers=a)).json()
        other = (await client.get("/api/me/models", headers=b)).json()
        admin = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        same = await _weigh(client, a, "s1", 5)
        back = await _weigh(client, a, "s1", None)
        to_admins = await _weigh(client, a, "s2", 1)
        refused = [
            await _weigh(client, a, "p1", 2),  # private
            await _weigh(client, a, "nobody", 2),
            await client.patch(
                "/api/me/models/shared/nobody",
                json={"expected_revision": "stale", "weight": 2},
                headers=a,
            ),
            await client.patch(
                "/api/me/models/shared/s1",
                json={"expected_revision": "stale", "weight": 2},
                headers=a,
            ),
            await _weigh(client, a, "s1", -1),
            await _weigh(client, a, "s1", True),
            await _weigh(client, a, "s1", "2"),
            await _weigh(client, a, "s1", 1.5),
            await _weigh(client, a, "s1", 101),
            await _weigh(client, a, "s1", 2, share="shared"),
            await client.patch(
                "/api/me/models/shared/s1",
                json={"weight": 2},
                headers=a,
            ),
            await client.patch(
                "/api/me/models/shared/s1",
                content=b"weight=2",
                headers={**a, "Content-Type": "text/plain"},
            ),
            await client.patch(
                "/api/me/models/shared/s1",
                content=b"{weight",
                headers={**a, "Content-Type": "application/json"},
            ),
            await client.patch(
                "/api/me/models/shared/s1",
                json={"expected_revision": "x", "weight": 2},
                headers=ADMIN,
            ),
        ]
        # Unshared, an override stays in the file; shared again, it is back
        kept = await _weigh(client, a, "s1", 4)
        revision = (await client.get("/api/settings/llm", headers=ADMIN)).json()[
            "revision"
        ]
        await client.patch(
            "/api/settings/llm/deployments/s1",
            json={"expected_revision": revision, "share": "private"},
            headers=ADMIN,
        )
        unshared = (await client.get("/api/me/models", headers=a)).json()
        revision = (await client.get("/api/settings/llm", headers=ADMIN)).json()[
            "revision"
        ]
        await client.patch(
            "/api/settings/llm/deployments/s1",
            json={"expected_revision": revision, "share": "shared"},
            headers=ADMIN,
        )
        reshared = (await client.get("/api/me/models", headers=a)).json()
    check_shape("UserModelsView", first)
    assert _shared(first) == [("s1", 3, 3), ("s2", 1, 1)]
    assert off.status_code == 200, off.text
    assert _shared(off.json()) == [("s1", 3, 3), ("s2", 0, 1)]
    assert _shared(heavier.json()) == [("s1", 5, 3), ("s2", 0, 1)]
    assert heavier.json()["revision"] != first["revision"]
    # The pool follows (s2 off), the other user and the admin do not
    assert capabilities["llm"]["models"] == ["openai/garden-large"]
    assert _shared(other) == [("s1", 3, 3), ("s2", 1, 1)]
    assert [d["weight"] for d in admin["deployments"]] == [3, 1, 1]
    assert same.json()["revision"] == heavier.json()["revision"]
    # null, or the admin's weight, follows the admin again
    assert _shared(back.json()) == [("s1", 3, 3), ("s2", 0, 1)]
    assert _shared(to_admins.json()) == [("s1", 3, 3), ("s2", 1, 1)]
    assert to_admins.json()["revision"] == first["revision"]
    saved = json.loads((tmp_path / "user-models" / f"{a_id}.json").read_text())
    assert "shared_weights" not in saved or saved["shared_weights"] == {"s1": 4}
    assert [_why(r) for r in refused] == [
        (404, "settings_entry_missing"),
        (404, "settings_entry_missing"),
        (404, "settings_entry_missing"),  # not shared: 404 before the revision
        (409, "stale_revision"),
        # Round 18: a weight is a whole number from 0 to 100
        (422, "invalid_weight"),
        (422, "invalid_weight"),
        (422, "invalid_weight"),
        (422, "invalid_weight"),
        (422, "invalid_weight"),
        (422, "invalid_settings_request"),
        (422, "invalid_settings_request"),
        (422, "json_required"),
        (422, "invalid_json"),
        (403, "user_token_required"),
    ]
    assert (
        refused[3].json()["detail"]["current_revision"] == to_admins.json()["revision"]
    )
    assert _shared(kept.json())[0] == ("s1", 4, 3)
    assert [row["id"] for row in unshared["shared"]] == ["s2"]
    assert unshared["revision"] == kept.json()["revision"]
    assert _shared(reshared)[0] == ("s1", 4, 3)
    assert no_network == []


# ------------------------------------------------------------------- test


async def test_a_user_tests_a_shared_model_without_learning_where_it_is(
    tmp_path: Path, no_network: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refused(Exception):
        status_code = 401

    async def failing(body: Any) -> str:
        no_network.append(("test", body.model, body.api_key, body.api_base))
        raise Refused(f"Incorrect API key provided at {ADMIN_BASE} for {ADMIN_KEY}")

    async with _server(_app(tmp_path, _config())) as client:
        _a_id, a = await _user(client, "Laptop A")
        passed = await client.post(
            "/api/me/models/test", json={"shared_id": "s1"}, headers=a
        )
        monkeypatch.setattr("markitai.serve.app._probe_llm", failing)
        failed = await client.post(
            "/api/me/models/test", json={"shared_id": "s2"}, headers=a
        )
        refused = [
            await client.post(
                "/api/me/models/test", json={"shared_id": "p1"}, headers=a
            ),
            await client.post(
                "/api/me/models/test", json={"shared_id": "nobody"}, headers=a
            ),
            await client.post(
                "/api/me/models/test",
                json={"shared_id": "s1", "model": "openai/x"},
                headers=a,
            ),
            await client.post("/api/me/models/test", json={"shared_id": 1}, headers=a),
            await client.post(
                "/api/me/models/test", json={"shared_id": "s1"}, headers=ADMIN
            ),
            # The admin's route takes no shared_id
            await client.post(
                "/api/settings/llm/test", json={"shared_id": "s1"}, headers=ADMIN
            ),
        ]
    assert passed.json() == {"ok": True, "detail": "Model test passed"}
    assert failed.json() == {
        "ok": False,
        "reason": "key_rejected",
        "detail": "The shared model did not pass its test; ask the admin to check it",
    }
    for answer in (passed.text, failed.text):
        for hidden in (ADMIN_KEY, ADMIN_BASE, "models.admin.example", "OPENAI"):
            assert hidden not in answer
    assert [_why(r) for r in refused] == [
        (422, "invalid_settings_request"),
        (422, "invalid_settings_request"),
        (422, "invalid_settings_request"),
        (422, "invalid_settings_request"),
        (403, "user_token_required"),
        (422, "invalid_settings_request"),
    ]
    assert refused[0].json()["detail"] == "deployment not found"
    # Sent with the admin's key to its address, once per test
    assert no_network == [
        ("test", "openai/garden-large", ADMIN_KEY, ADMIN_BASE),
        ("test", "openai/garden-small", ADMIN_KEY, ADMIN_BASE),
    ]


async def test_shared_model_tests_are_one_at_a_time_and_six_a_minute(
    tmp_path: Path, no_network: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow(body: Any) -> str:
        no_network.append(body.model)
        started.set()
        await release.wait()
        return "ok"

    async with _server(_app(tmp_path, _config())) as client:
        _a_id, a = await _user(client, "Laptop A")
        _b_id, b = await _user(client, "Laptop B")
        monkeypatch.setattr("markitai.serve.app._probe_llm", slow)
        running = asyncio.create_task(
            client.post("/api/me/models/test", json={"shared_id": "s1"}, headers=a)
        )
        await started.wait()
        busy = await client.post(
            "/api/me/models/test", json={"shared_id": "s2"}, headers=a
        )
        release.set()
        other = await client.post(
            "/api/me/models/test", json={"shared_id": "s2"}, headers=b
        )
        first = await running
        later = [
            await client.post(
                "/api/me/models/test", json={"shared_id": "s1"}, headers=a
            )
            for _ in range(6)
        ]
    assert first.json()["ok"] is True
    assert _why(busy) == (429, "provider_busy")
    assert busy.headers["Retry-After"] == "1"
    assert other.json()["ok"] is True
    assert [r.status_code for r in later] == [200] * 5 + [429]
    assert 1 <= int(later[-1].headers["Retry-After"]) <= 60
    # The refused ones sent nothing
    assert len(no_network) == 1 + 1 + 5


# ----------------------------------------------------------- edit config


def _same_id_twice(value: dict[str, Any]) -> None:
    copy = json.loads(json.dumps(value["models"][0]))
    copy["litellm_params"]["api_key"] = USER_A_KEY
    value["models"].append(copy)


async def test_a_users_edit_config_is_its_own_models_file(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path, _config())) as client:
        a_id, a = await _user(client, "Laptop A")
        _b_id, b = await _user(client, "Laptop B")
        empty = (await client.get("/api/me/config", headers=a)).json()
        added = await _add(client, a, {"model": "openai/own", "api_key": USER_A_KEY})
        assert added.status_code == 200, added.text
        shown = (await client.get("/api/me/config", headers=a)).json()
        models_revision = await _revision(client, a)

        async def put(text: str, revision: str | None = None) -> Any:
            return await client.put(
                "/api/me/config",
                json={
                    "expected_revision": revision or await _revision(client, a),
                    "text": text,
                },
                headers=a,
            )

        path = tmp_path / "user-models" / f"{a_id}.json"
        before = path.read_bytes()
        unchanged = await put(shown["text"])
        unchanged_bytes = path.read_bytes()
        document = json.loads(shown["text"])
        document["shared"][1]["weight"] = 0
        document["models"][0]["litellm_params"]["weight"] = 2
        document["providers"].append(
            {"id": "own-1", "provider": "openai", "api_key": "user-a-second-key"}
        )
        saved = await put(json.dumps(document, indent=2))
        view = (await client.get("/api/me/models", headers=a)).json()
        stale = await put(shown["text"], revision=shown["revision"])
        current = (await client.get("/api/me/config", headers=a)).json()["text"]

        def edited(change: Any) -> str:
            value = json.loads(current)
            change(value)
            return json.dumps(value)

        refused = [
            await put("{"),
            await put("[]"),
            await put('{"llm": {}}'),
            await put(edited(lambda v: v["shared"].append({"id": "p1", "weight": 1}))),
            await put(edited(lambda v: v["shared"].append({"id": "s1", "weight": 2}))),
            await put(edited(lambda v: v["shared"][0].update(weight=-1))),
            await put(edited(lambda v: v["shared"][0].update(api_key="x"))),
            await put(
                edited(
                    lambda v: (
                        v["models"][0]
                        .setdefault("model_info", {})
                        .update(provider_id="nobody")
                    )
                )
            ),
            # The masked key moved to another model
            await put(
                edited(
                    lambda v: v["models"].append(
                        {
                            "model_name": "default",
                            "litellm_params": {
                                "model": "openai/copy",
                                "api_key": v["models"][0]["litellm_params"]["api_key"],
                            },
                        }
                    )
                )
            ),
            await put(
                edited(
                    lambda v: v["models"].append(
                        {
                            "model_name": "default",
                            "litellm_params": {
                                "model": "vertex_ai/gemini-2.5-pro",
                                "api_key": USER_A_KEY,
                            },
                        }
                    )
                )
            ),
            await put(
                edited(
                    lambda v: v["models"].append(
                        {
                            "model_name": "default",
                            "litellm_params": {"model": "openai/keyless"},
                        }
                    )
                )
            ),
            await put(
                edited(
                    lambda v: v["models"][0]["litellm_params"].update(
                        api_base="http://127.0.0.1:8000/v1"
                    )
                )
            ),
            await put(
                edited(
                    lambda v: v["models"][0]["litellm_params"].update(
                        api_key="env:OPENAI_API_KEY"
                    )
                )
            ),
            await put(edited(_same_id_twice)),
            await put(
                edited(lambda v: v["models"][0]["litellm_params"].update(weight="x"))
            ),
            await client.get("/api/me/config", headers=ADMIN),
            await client.put(
                "/api/me/config",
                json={"expected_revision": "x", "text": "{}"},
                headers=ADMIN,
            ),
            await client.put(
                "/api/me/config",
                json={"expected_revision": "x", "text": "{}", "path": "/etc"},
                headers=a,
            ),
        ]
        other = (await client.get("/api/me/config", headers=b)).json()
        config_json = (tmp_path / "config.json").read_text()
        # Over the size limit
        too_large = await put(" " * (1024 * 1024 + 1))
    check_shape("ConfigDocument", empty)
    assert (empty["path"], empty["origin"], empty["exists"]) == (None, "user", False)
    assert json.loads(empty["text"]) == {
        "models": [],
        "providers": [],
        "shared": [
            {"id": "s1", "model": "openai/garden-large", "weight": 3},
            {"id": "s2", "model": "openai/garden-small", "weight": 1},
        ],
    }
    assert list(json.loads(empty["text"])) == ["models", "providers", "shared"]
    assert shown["text"].startswith('{\n  "models": [\n') and shown["text"].endswith(
        "}\n"
    )
    # The model and the connection the Models page made for it
    assert (shown["exists"], shown["masked"], shown["read_only"]) == (True, 2, False)
    assert USER_A_KEY not in shown["text"]
    assert shown["revision"] == models_revision
    for hidden in (ADMIN_KEY, ADMIN_BASE, "models.admin.example", "admin-private"):
        assert hidden not in shown["text"]
    # Unchanged: nothing written, same revision
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["revision"] == shown["revision"]
    assert unchanged.json()["restart_required"] is False
    assert unchanged_bytes == before
    assert saved.status_code == 200, saved.text
    assert saved.json()["masked"] == 3
    assert _shared(view) == [("s1", 3, 3), ("s2", 0, 1)]
    assert [d["weight"] for d in view["deployments"]] == [2]
    assert view["revision"] == saved.json()["revision"]
    stored = json.loads(path.read_text())
    # The kept mask is the stored key again; the new connection is sealed
    assert stored["models"][0]["litellm_params"]["api_key"] == USER_A_KEY
    assert stored["providers"][-1]["use_environment_credentials"] is False
    assert stored["providers"][-1]["api_key"] == "user-a-second-key"
    assert stored["shared_weights"] == {"s2": 0}
    assert _why(stale) == (409, "stale_revision")
    assert [_why(r) for r in refused] == [
        (422, "invalid_config"),
        (422, "invalid_config"),
        (422, "invalid_config"),
        (422, "invalid_config"),
        (422, "invalid_config"),
        (422, "invalid_weight"),  # shared weight -1
        (422, "invalid_config"),
        (422, "invalid_config"),
        (422, "invalid_config"),
        (422, "server_credentials"),
        (422, "key_or_address_required"),
        (422, "server_address"),
        (422, "invalid_settings_request"),
        (422, "invalid_config"),  # the same model id twice
        (422, "invalid_weight"),  # a model's weight "x"
        (403, "user_token_required"),
        (403, "user_token_required"),
        (422, "invalid_settings_request"),
    ]
    details = [r.json()["detail"] for r in refused]
    assert details[0].startswith("line 1, column 2")
    assert details[1] == "the configuration must be a JSON object"
    assert details[2] == (
        "llm is not part of your models file; it holds only models, providers "
        "and shared"
    )
    assert details[3] == "p1 is not a model the admin shares"
    assert details[4] == "shared lists s1 twice"
    assert details[8].startswith("/models/1/litellm_params/api_key: ")
    assert "used twice" in details[13]
    assert json.loads(other["text"])["models"] == []
    assert "openai/own" not in config_json
    assert _why(too_large) == (413, "config_too_large")
    assert no_network == []
