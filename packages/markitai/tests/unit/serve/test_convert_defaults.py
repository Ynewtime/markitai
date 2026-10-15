"""Convert defaults (contract v1.17): ``GET``/``PUT /api/convert-defaults``.

The admin's defaults are the server's configuration and are saved into its
config file; a user token keeps its own in its models file; an anonymous
caller of a ``--no-auth`` server only reads. All values are invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.unit.serve.test_serve_auth import _serve_client
from tests.unit.serve.test_user_models import (
    ADMIN,
    _app,
    _revision,
    _server,
    _user,
)

URL = "/api/convert-defaults"
ALL = {
    "llm": False,
    "ocr": False,
    "alt": False,
    "desc": False,
    "screenshot": False,
    "screenshot_only": False,
    "pure": False,
    "no_cache": False,
    "no_compress": False,
    "profile": None,
    "strategy": "auto",
}


async def test_the_server_defaults_read_from_its_configuration(
    tmp_path: Path,
) -> None:
    config = {
        "llm": {"enabled": True},
        "image": {"compress": False, "alt_enabled": True},
        "fetch": {"strategy": "cloudflare"},
        "output": {"profile": "rag"},
    }
    async with _server(_app(tmp_path, config)) as client:
        got = await client.get(URL, headers=ADMIN)
    assert got.status_code == 200, got.text
    assert got.headers["cache-control"] == "no-store"
    body = got.json()
    assert body["server"] == {
        **ALL,
        "llm": True,
        "no_compress": True,
        "alt": True,
        "profile": "rag",
        "strategy": "auto",
    }
    assert body["defaults"] == body["server"]
    assert (body["own"], body["origin"], body["scope"], body["can_save"]) == (
        None,
        "server",
        "server",
        True,
    )
    assert body["revision"].startswith("sha256:")


async def test_capabilities_name_the_features(tmp_path: Path) -> None:
    async with _server(_app(tmp_path)) as client:
        features = (await client.get("/api/capabilities", headers=ADMIN)).json()[
            "features"
        ]
    assert features["convert_defaults"] is True
    assert features["pdf_outline"] is True


async def test_the_admin_saves_into_the_file_and_the_running_config(
    tmp_path: Path,
) -> None:
    config = {"llm": {"enabled": False, "model_list": []}, "output": {"profile": "rag"}}
    app = _app(tmp_path, config)
    async with _server(app) as client:
        view = (await client.get(URL, headers=ADMIN)).json()
        put = await client.put(
            URL,
            json={
                "expected_revision": view["revision"],
                "defaults": {
                    "llm": True,
                    # The built-in value and the file lacks the key: not added
                    "ocr": False,
                    "no_compress": True,
                    "profile": None,
                    "strategy": "jina",
                },
            },
            headers=ADMIN,
        )
        assert put.status_code == 200, put.text
        again = (await client.get(URL, headers=ADMIN)).json()
        stale = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": {"llm": False}},
            headers=ADMIN,
        )
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved == {
        "llm": {"enabled": True, "model_list": []},
        "image": {"compress": False},
        "fetch": {"strategy": "jina"},
    }
    assert (tmp_path / "config.json").stat().st_mode & 0o077 == 0
    assert put.json() == again
    assert again["server"]["llm"] is True
    assert again["server"]["no_compress"] is True
    assert again["server"]["profile"] is None
    assert app.state.markitai.config.fetch.strategy == "jina"
    assert app.state.markitai.config.image.compress is False
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "stale_revision"
    assert stale.json()["detail"]["current_revision"] == again["revision"]


async def test_an_empty_output_section_goes_with_the_profile(tmp_path: Path) -> None:
    async with _server(_app(tmp_path, {"output": {"profile": "okf"}})) as client:
        view = (await client.get(URL, headers=ADMIN)).json()
        put = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": {"profile": None}},
            headers=ADMIN,
        )
    assert put.status_code == 200, put.text
    assert json.loads((tmp_path / "config.json").read_text()) == {}


async def test_a_file_that_does_not_exist_yet_is_created(tmp_path: Path) -> None:
    async with _server(_app(tmp_path)) as client:
        view = (await client.get(URL, headers=ADMIN)).json()
        put = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": {"ocr": True}},
            headers=ADMIN,
        )
    assert put.status_code == 200, put.text
    assert json.loads((tmp_path / "config.json").read_text()) == {
        "ocr": {"enabled": True}
    }


@pytest.mark.parametrize(
    "defaults",
    [
        {"backend": "cloudflare"},
        {"strategy": "cloudflare"},
        {"strategy": None},
        {"llm": "yes"},
        {"llm": None},
        {"profile": "html"},
        {"preset": "rich"},
        None,
        [],
    ],
)
async def test_bad_defaults_are_refused(tmp_path: Path, defaults: Any) -> None:
    async with _server(_app(tmp_path)) as client:
        view = (await client.get(URL, headers=ADMIN)).json()
        put = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": defaults},
            headers=ADMIN,
        )
    assert (put.status_code, put.json()["reason"]) == (422, "invalid_convert_defaults")
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize(
    "body",
    [
        {"defaults": {}},
        {"expected_revision": "x"},
        {"expected_revision": "x", "defaults": {}, "extra": 1},
        {"expected_revision": 3, "defaults": {}},
        [],
    ],
)
async def test_another_body_shape_is_refused(tmp_path: Path, body: Any) -> None:
    async with _server(_app(tmp_path)) as client:
        put = await client.put(URL, json=body, headers=ADMIN)
    assert (put.status_code, put.json()["reason"]) == (422, "invalid_convert_defaults")


async def test_a_user_keeps_its_own_defaults_over_the_servers(tmp_path: Path) -> None:
    async with _server(_app(tmp_path, {"llm": {"enabled": True}})) as client:
        user_id, headers = await _user(client, "Rota")
        view = (await client.get(URL, headers=headers)).json()
        assert view["scope"] == "user" and view["can_save"] is True
        assert view["own"] is None and view["origin"] == "server"
        put = await client.put(
            URL,
            json={
                "expected_revision": view["revision"],
                "defaults": {"llm": False, "profile": None, "ocr": True},
            },
            headers=headers,
        )
        assert put.status_code == 200, put.text
        mine = put.json()
        admin = (await client.get(URL, headers=ADMIN)).json()
        stale = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": {"ocr": False}},
            headers=headers,
        )
        back = await client.put(
            URL,
            json={"expected_revision": mine["revision"], "defaults": None},
            headers=headers,
        )
    assert mine["own"] == {"llm": False, "profile": None, "ocr": True}
    assert mine["origin"] == "user"
    assert mine["server"]["llm"] is True
    assert mine["defaults"]["llm"] is False and mine["defaults"]["ocr"] is True
    assert admin["server"]["llm"] is True and admin["own"] is None
    assert stale.status_code == 409
    assert stale.json()["detail"]["current_revision"] == mine["revision"]
    assert back.status_code == 200
    assert back.json()["own"] is None and back.json()["origin"] == "server"
    assert user_id


async def test_a_users_file_holds_the_defaults_and_its_models_revision_ignores_them(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    async with _server(app) as client:
        user_id, headers = await _user(client, "Rota")
        before = await _revision(client, headers)
        view = (await client.get(URL, headers=headers)).json()
        put = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": {"alt": True}},
            headers=headers,
        )
        assert put.status_code == 200, put.text
        after = await _revision(client, headers)
        models = (await client.get("/api/me/models", headers=headers)).json()
        files = list(tmp_path.rglob(f"{user_id}.json"))
    assert before == after
    assert models["configured"] is False
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["version"] == 1
    assert saved["convert_defaults"] == {"alt": True}
    assert saved["models"] == [] and saved["providers"] == []


async def test_a_revoked_user_cannot_save(tmp_path: Path) -> None:
    async with _server(_app(tmp_path)) as client:
        user_id, headers = await _user(client, "Rota")
        view = (await client.get(URL, headers=headers)).json()
        await client.delete(f"/api/access-tokens/{user_id}", headers=ADMIN)
        put = await client.put(
            URL,
            json={"expected_revision": view["revision"], "defaults": {"alt": True}},
            headers=headers,
        )
    assert put.status_code == 401


async def test_an_anonymous_caller_reads_and_cannot_save(tmp_path: Path) -> None:
    from markitai.config import MarkitaiConfig
    from markitai.serve import create_app

    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    app = create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
        token=None,
    )
    async with _serve_client(app, peer=("203.0.113.9", 4000)) as client:
        got = await client.get(URL)
        assert got.status_code == 200, got.text
        put = await client.put(
            URL,
            json={"expected_revision": got.json()["revision"], "defaults": {}},
        )
    assert got.json()["can_save"] is False and got.json()["scope"] == "server"
    assert (put.status_code, put.json()["reason"]) == (403, "settings_forbidden")
