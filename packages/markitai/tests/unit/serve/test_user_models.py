"""Workbench contract v1.15: a user token's own models and the admin's shared ones.

A user (v1.13 user token) keeps models of its own in
``serve/user-models/<token id>.json``, under rules that keep the server's
credentials and machine out of its reach, and its conversions route over one
pool: its own models, then the ones the admin shares. These tests drive the
routes with two users and an admin on one server; nothing leaves the machine
(the end-to-end routing with model servers is in
``tests/integration/test_serve_user_models.py``). All keys are invented.
"""

from __future__ import annotations

import json
import stat
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

from markitai.config import LLMProviderConfig, MarkitaiConfig, ModelConfig
from markitai.serve import user_models
from markitai.serve.app import NO_KEY
from markitai.serve.user_models import Lists, UserModels
from tests.unit.serve.test_workbench_contract import check_shape
from tests.unit.serve.test_workbench_v111 import _client

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx
    from fastapi import FastAPI

ADMIN_TOKEN = "admin-token"  # nosec B105 - a test token
ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
USER_A_KEY = "user-a-fixture-key"
USER_B_KEY = "user-b-fixture-key"
ADMIN_KEY = "admin-fixture-key"
SENTINEL_KEY = "env-sentinel-not-for-users"
ADMIN_BASE = "https://models.admin.example/v1"
USER_BASE = "https://models.user.example/v1"


# ----------------------------------------------------------------- rules


@pytest.mark.parametrize(
    "own",
    [
        "http://localhost:11434",
        "http://LOCALHOST./v1",
        "http://models.localhost:8000/v1",
        "http://127.0.0.1:1234/v1",
        "http://127.8.9.10/v1",
        "http://127.1/v1",
        "http://0x7f.0.0.1/v1",
        "http://2130706433/v1",
        "http://[::1]:8000/v1",
        "http://[::ffff:127.0.0.1]/v1",
        "http://0.0.0.0:8000",
        "http://[::]:8000",
    ],
)
def test_the_servers_own_machine_is_found_by_its_address(own: str) -> None:
    assert user_models.server_machine(own)


@pytest.mark.parametrize(
    "other",
    [
        "https://api.openai.com/v1",
        "http://192.0.2.10:8000/v1",
        "http://[2001:db8::1]/v1",
        "http://localhost.example/v1",
        "http://10.0.0.1/v1",
        "not an address",
    ],
)
def test_other_addresses_are_not_the_servers_machine(other: str) -> None:
    assert not user_models.server_machine(other)


def _reason(provider: str, key: Any, base: Any) -> str | None:
    try:
        user_models.check_connection(provider, key, base)
    except Exception as e:
        return getattr(e, "reason", type(e).__name__)
    return None


def test_the_rules_refuse_the_servers_credentials_and_machine() -> None:
    assert _reason("openai", USER_A_KEY, None) is None
    assert _reason("openai", None, USER_BASE) is None
    assert _reason("openai", NO_KEY, USER_BASE) is None
    for login in sorted(user_models.SERVER_LOGIN):
        assert _reason(login, USER_A_KEY, None) == "server_credentials", login
    assert _reason("bedrock", None, None) == "server_credentials"
    assert _reason("bedrock", NO_KEY, USER_BASE) == "server_credentials"
    assert _reason("bedrock", USER_A_KEY, None) is None
    assert _reason("openai", None, None) == "key_or_address_required"
    assert _reason("openai", NO_KEY, None) == "key_or_address_required"
    # A local server's documented address is the server's own machine
    assert _reason("ollama", None, None) == "server_address"
    assert _reason("ollama", USER_A_KEY, None) == "server_address"
    assert _reason("lm_studio", USER_A_KEY, None) == "server_address"
    assert _reason("ollama", None, "http://127.0.0.1:11434") == "server_address"
    assert _reason("ollama", None, "http://192.0.2.4:11434") is None
    assert _reason("openai", "env:OPENAI_API_KEY", None) == "invalid_settings_request"
    assert _reason("openai", USER_A_KEY, "env:OPENAI_API_BASE") == (
        "invalid_settings_request"
    )
    assert _reason("openai", USER_A_KEY, "ftp://models.example") == (
        "invalid_settings_request"
    )
    # An address httpx cannot name escapes the public-address rule (yarl,
    # which aiohttp reads it with, maps it to evil.example)
    assert _reason("openai", USER_A_KEY, "http://\u24d4vil.example/v1") == (
        "invalid_settings_request"
    )


def test_a_bare_model_name_is_judged_by_where_litellm_sends_it() -> None:
    """LiteLLM routes ``gemini-2.5-pro`` to Vertex AI (Google credentials)."""
    assert user_models.route_provider("gemini-2.5-pro") == "vertex_ai"
    with pytest.raises(Exception) as refused:
        user_models.precheck("gemini-2.5-pro")
    assert getattr(refused.value, "reason", None) == "server_credentials"
    user_models.precheck("openai/gpt-4o-mini")


# ------------------------------------------------------------------ pool


def _base() -> MarkitaiConfig:
    cfg = MarkitaiConfig.model_validate(
        {
            "llm": {
                "enabled": True,
                "model_list": [
                    {
                        "model_name": "fast",
                        "litellm_params": {
                            "model": "openai/admin-only",
                            "api_key": ADMIN_KEY,
                        },
                    }
                ],
                "router_settings": {"fallbacks": [{"fast": ["slow"]}]},
            },
            "image": {"alt_enabled": True, "desc_enabled": True},
        }
    )
    return cfg


def _admin(models: list[dict[str, Any]], providers: list[dict[str, Any]] = ()):  # type: ignore[assignment]
    return (
        [ModelConfig.model_validate(entry) for entry in models],
        [LLMProviderConfig.model_validate(entry) for entry in providers],
    )


def test_an_empty_pool_turns_model_processing_off_and_holds_nothing() -> None:
    models, providers = _admin(
        [
            {
                "model_name": "a",
                "litellm_params": {"model": "openai/private", "api_key": ADMIN_KEY},
            },
            {
                "model_name": "b",
                "litellm_params": {
                    "model": "openai/off",
                    "api_key": ADMIN_KEY,
                    "weight": 0,
                },
                "model_info": {"share": "shared"},
            },
            {
                "model_name": "c",
                "litellm_params": {"model": "openai/other", "api_key": ADMIN_KEY},
                "model_info": {"share": "public"},
            },
        ]
    )
    pool = user_models.build_pool(_base(), Lists(), models, providers)
    assert pool.models == []
    cfg = pool.config
    assert (cfg.llm.enabled, cfg.image.alt_enabled, cfg.image.desc_enabled) == (
        False,
        False,
        False,
    )
    assert cfg.llm.model_list == [] and cfg.llm.providers == []
    assert cfg.llm.router_settings.fallbacks == []
    assert user_models.shared_models(models) == []


def test_own_models_come_first_then_shared_ones_in_one_default_group() -> None:
    own = Lists(
        models=[
            {
                "model_name": "mine",
                "litellm_params": {
                    "model": "openai/own",
                    "api_key": USER_A_KEY,
                    "api_base": USER_BASE,
                    "weight": 2,
                },
                "model_info": {"id": "m1", "provider_id": "p-own"},
            },
            {
                "model_name": "mine",
                "litellm_params": {
                    "model": "openai/own-off",
                    "api_key": USER_A_KEY,
                    "weight": 0,
                },
            },
        ],
        providers=[
            {
                "id": "p-own",
                "provider": "openai",
                "api_key": USER_A_KEY,
                "api_base": USER_BASE,
            }
        ],
    )
    models, providers = _admin(
        [
            {
                "model_name": "fast",
                "litellm_params": {
                    "model": "openai/shared",
                    "weight": 3,
                    "api_base": "http://10.1.2.3:8080/v1",
                },
                "model_info": {
                    "id": "s1",
                    "share": "shared",
                    "provider_id": "p-admin",
                },
            },
            {
                "model_name": "slow",
                "litellm_params": {"model": "openai/private", "api_key": ADMIN_KEY},
                "model_info": {"provider_id": "p-private"},
            },
        ],
        [
            {"id": "p-admin", "provider": "custom", "api_key": ADMIN_KEY},
            {"id": "p-private", "provider": "openai", "api_key": "admin-other-key"},
        ],
    )
    pool = user_models.build_pool(_base(), own, models, providers)
    cfg = pool.config
    assert [m.litellm_params.model for m in cfg.llm.model_list] == [
        "openai/own",
        "openai/shared",
    ]
    assert all(m.model_name == "default" for m in cfg.llm.model_list)
    assert [m.litellm_params.weight for m in cfg.llm.model_list] == [2, 3]
    assert [p.id for p in cfg.llm.providers] == ["p-own", "p-admin"]
    assert cfg.llm.router_settings.fallbacks == []
    assert cfg.llm.enabled is True
    assert "admin-other-key" not in cfg.model_dump_json()
    assert pool.hidden_hosts == ("10.1.2.3:8080", "10.1.2.3")
    assert (
        user_models.connection_provider(models[0], providers) == "custom"
    )  # an OpenAI-compatible endpoint
    # A user entry that breaks the rules (a subscription runtime) is left out
    # even when a file holds it
    broken = Lists(
        models=[
            {
                "model_name": "x",
                "litellm_params": {"model": "claude-agent/sonnet", "api_key": "k"},
            }
        ]
    )
    assert user_models.build_pool(_base(), broken, [], []).models == []


def test_a_shared_hosts_name_is_hidden_in_what_a_user_reads() -> None:
    text = (
        "LLM enhancement failed (APIConnectionError: cannot connect to "
        "http://10.1.2.3:8080/v1/chat/completions; 10.1.2.3:8080 refused)"
    )
    hidden = user_models.hide_hosts(text, ("10.1.2.3:8080", "10.1.2.3"))
    assert "10.1.2.3" not in hidden
    assert "<shared model address>" in hidden
    assert "<shared model host>" in hidden


# --------------------------------------------------------------- storage


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_files_are_owner_only_and_revoked_or_orphaned_ones_are_deleted(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "serve" / "user-models"
    store = UserModels(folder)
    entry = {
        "model_name": "default",
        "litellm_params": {"model": "openai/m", "api_key": USER_A_KEY},
    }
    saved = store.write(
        "u_000000000001", lambda: True, lambda lists: lists.models.append(entry)
    )
    assert len(saved.models) == 1
    path = folder / "u_000000000001.json"
    assert json.loads(path.read_text())["version"] == 1
    if sys.platform != "win32":
        assert _mode(path) == 0o600
        assert _mode(folder) == 0o700
    restarted = UserModels(folder)
    assert restarted.count("u_000000000001") == 1
    assert restarted.count("u_000000000002") == 0
    # A revoked token cannot write; an invalid file holds nothing and is kept
    with pytest.raises(Exception) as refused:
        restarted.write("u_000000000001", lambda: False, lambda _l: None)
    assert getattr(refused.value, "reason", None) == "token_required"
    (folder / "u_000000000003.json").write_bytes(b"{not json")
    assert restarted.count("u_000000000003") == 0
    assert (folder / "u_000000000003.json").read_bytes() == b"{not json"
    (folder / "u_000000000004.json").write_bytes(b"{}")
    (folder / ".tmp-u_000000000005-0123").write_bytes(b"half")
    restarted.sweep(lambda token: token != "u_000000000004")
    assert not (folder / "u_000000000004.json").exists()
    assert not (folder / ".tmp-u_000000000005-0123").exists()
    assert path.exists()
    assert restarted.delete("u_000000000001") == 1
    assert not path.exists()
    assert restarted.count("u_000000000001") == 0


# ---------------------------------------------------------------- routes


def _app(tmp_path: Path, config: dict[str, Any] | None = None) -> FastAPI:
    path = tmp_path / "config.json"
    if config is not None:
        path.write_text(json.dumps(config), encoding="utf-8")
    cfg = MarkitaiConfig.model_validate(config or {})
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    from markitai.serve import create_app

    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=path,
        token=ADMIN_TOKEN,
    )


@asynccontextmanager
async def _server(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with _client(app) as client:
        yield client


async def _user(client: httpx.AsyncClient, label: str) -> tuple[str, dict[str, str]]:
    created = await client.post(
        "/api/access-tokens", json={"label": label}, headers=ADMIN
    )
    assert created.status_code == 201, created.text
    body = created.json()
    return body["id"], {"Authorization": f"Bearer {body['token']}"}


async def _revision(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    return (await client.get("/api/me/models", headers=headers)).json()["revision"]


async def _add(
    client: httpx.AsyncClient, headers: dict[str, str], *deployments: dict[str, Any]
) -> httpx.Response:
    return await client.post(
        "/api/me/models/deployments/batch",
        json={
            "expected_revision": await _revision(client, headers),
            "deployments": [{"model_name": "default", **d} for d in deployments],
        },
        headers=headers,
    )


def _why(response: httpx.Response) -> tuple[int, str]:
    return response.status_code, response.json().get("reason")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every model test or discovery that would send a request, recorded."""
    return record_network(monkeypatch)


def record_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record, instead of sending, every model test or discovery."""
    # Start from no proxy settings at all: the shell running the tests may set
    # its own, and a user's model is refused outright behind a proxy.
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    sent: list[Any] = []

    async def probe(body: Any) -> str:
        sent.append(("test", body.model, body.api_key, body.api_base))
        return f"{body.model} responded"

    async def discover(provider: str, **kwargs: Any) -> dict[str, Any]:
        sent.append(
            ("discover", provider, kwargs.get("api_key"), kwargs.get("api_base"))
        )
        return {
            "provider": provider,
            "status": "ok",
            "source": "live_api",
            "authoritative": True,
            "cached": False,
            "stale": False,
            "models": [],
        }

    monkeypatch.setattr("markitai.serve.app._probe_llm", probe)
    monkeypatch.setattr("markitai.providers.discovery.discover_models", discover)
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    return sent


async def test_only_a_user_token_reaches_its_own_models(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path)) as client:
        _id, user = await _user(client, "Laptop A")
        admin = await client.get("/api/me/models", headers=ADMIN)
        loopback = await client.get("/api/me/models")
        settings = await client.get("/api/settings/llm", headers=user)
        capabilities = (await client.get("/api/capabilities", headers=user)).json()
        empty = await client.get("/api/me/models", headers=user)
    assert _why(admin) == (403, "user_token_required")
    assert admin.headers["cache-control"] == "no-store"
    assert _why(loopback) == (403, "user_token_required")
    assert _why(settings) == (403, "settings_forbidden")
    assert capabilities["features"]["user_models"] is True
    assert capabilities["llm"] == {
        "configured": False,
        "routable": False,
        "effective": False,
        "models": [],
    }
    assert empty.headers["cache-control"] == "no-store"
    view = empty.json()
    check_shape("UserModelsView", view)
    assert view == {
        "configured": False,
        "routable": False,
        "source": "user",
        "config_path": None,
        "config_origin": "user",
        "revision": view["revision"],
        "deployments": [],
        "detected": [],
        "shared": [],
        "limit": 50,
        "batch_remove": True,
    }


async def test_users_keep_their_models_apart_masked_and_within_the_rules(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path)) as client:
        a_id, a = await _user(client, "Laptop A")
        _b_id, b = await _user(client, "Laptop B")
        added = await _add(
            client,
            a,
            {"model": "openai/own-model", "api_key": USER_A_KEY, "api_base": USER_BASE},
        )
        assert added.status_code == 200, added.text
        assert added.headers["cache-control"] == "no-store"
        view = added.json()
        assert (view["configured"], view["routable"]) == (True, True)
        row = view["deployments"][0]
        assert row["model"] == "openai/own-model"
        assert row["api_key_configured"] is True
        assert "share" not in row
        deployment = row["deployment_id"]
        cards = (await client.get("/api/me/models/providers", headers=a)).json()
        card = next(c for c in cards["providers"] if c["kind"] == "configured")
        connection = card["provider_id"]
        credentials = (
            await client.get(
                f"/api/me/models/providers/{connection}/credentials", headers=a
            )
        ).json()

        # B sees nothing of A's and can change or test none of it
        other = (await client.get("/api/me/models", headers=b)).json()
        b_rev = other["revision"]
        refusals = [
            (
                await client.patch(
                    f"/api/me/models/deployments/{deployment}",
                    json={"expected_revision": b_rev, "weight": 3},
                    headers=b,
                ),
                (404, "settings_entry_missing"),
            ),
            (
                await client.delete(
                    f"/api/me/models/deployments/{deployment}",
                    params={"expected_revision": b_rev},
                    headers=b,
                ),
                (404, "settings_entry_missing"),
            ),
            (
                await client.post(
                    "/api/me/models/test", json={"deployment_id": deployment}, headers=b
                ),
                (422, "invalid_settings_request"),
            ),
            (
                await client.get(
                    f"/api/me/models/providers/{connection}/credentials", headers=b
                ),
                (404, "settings_entry_missing"),
            ),
            (
                await client.patch(
                    f"/api/me/models/providers/{connection}",
                    json={"expected_revision": b_rev, "api_key": "replaced-key-value"},
                    headers=b,
                ),
                (404, "settings_entry_missing"),
            ),
            (
                await client.post(
                    "/api/me/models/deployments/batch",
                    json={
                        "expected_revision": b_rev,
                        "deployments": [
                            {
                                "model_name": "default",
                                "model": "openai/x",
                                "credential_deployment_id": deployment,
                            }
                        ],
                    },
                    headers=b,
                ),
                (404, "settings_entry_missing"),
            ),
            (
                await client.post(
                    "/api/me/models/model-discovery",
                    json={"provider": "openai", "provider_id": connection},
                    headers=b,
                ),
                (404, "settings_entry_missing"),
            ),
        ]
        settings = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        tokens = (await client.get("/api/access-tokens", headers=ADMIN)).json()
        answers = [
            (await client.get(path, headers=headers)).text
            for headers in (ADMIN, a, b)
            for path in (
                "/api/capabilities",
                "/api/me/models",
                "/api/me/models/providers",
                "/api/settings/llm",
                "/api/settings/llm/providers",
                "/api/settings/config",
                "/api/access-tokens",
                "/api/history?include=items",
            )
        ]
    assert credentials["api_key_masked"] == "us…-key"
    assert credentials["api_key_source"] is None
    assert other["deployments"] == []
    for response, expected in refusals:
        assert _why(response) == expected, response.text
    assert settings["deployments"] == []
    counts = {t["label"]: t["model_count"] for t in tokens["tokens"]}
    assert counts == {"Laptop A": 1, "Laptop B": 0}
    for answer in answers:
        assert USER_A_KEY not in answer and SENTINEL_KEY not in answer
    # Saved owner-only beside the history, never in the configuration file
    folder = tmp_path / "user-models"
    path = folder / f"{a_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert USER_A_KEY in path.read_text(encoding="utf-8")
    assert all(
        entry["litellm_params"]["use_environment_credentials"] is False
        for entry in saved["models"]
    )
    if sys.platform != "win32":
        assert (_mode(path), _mode(folder)) == (0o600, 0o700)
    config = tmp_path / "config.json"
    assert not config.exists() or USER_A_KEY not in config.read_text()
    assert no_network == []


@pytest.mark.parametrize(
    ("deployment", "expected"),
    [
        ({"model": "claude-agent/sonnet"}, (422, "server_credentials")),
        (
            {"model": "copilot/gpt-5", "api_key": USER_A_KEY},
            (422, "server_credentials"),
        ),
        (
            {"model": "vertex_ai/gemini-pro", "api_key": USER_A_KEY},
            (422, "server_credentials"),
        ),
        (
            {"model": "gemini-2.5-pro", "api_key": USER_A_KEY},
            (422, "server_credentials"),
        ),
        ({"model": "bedrock/anthropic.claude"}, (422, "server_credentials")),
        (
            {"model": "openai/x", "provider": "watsonx", "api_key": USER_A_KEY},
            (422, "server_credentials"),
        ),
        ({"model": "openai/no-key"}, (422, "key_or_address_required")),
        ({"model": "ollama/llama3"}, (422, "server_address")),
        (
            {"model": "ollama/llama3", "api_base": "http://127.0.0.1:11434"},
            (422, "server_address"),
        ),
        (
            {
                "model": "openai/local",
                "api_key": USER_A_KEY,
                "api_base": "http://localhost:9/v1",
            },
            (422, "server_address"),
        ),
        (
            {"model": "openai/x", "credential_provider_id": "env:openai"},
            (422, "invalid_settings_request"),
        ),
        (
            {"model": "openai/x", "api_key": "env:OPENAI_API_KEY"},
            (422, "invalid_settings_request"),
        ),
    ],
)
async def test_what_a_user_may_save(
    tmp_path: Path,
    no_network: list[Any],
    deployment: dict[str, Any],
    expected: tuple[int, str],
) -> None:
    async with _server(_app(tmp_path)) as client:
        a_id, a = await _user(client, "Laptop A")
        response = await _add(client, a, deployment)
    assert _why(response) == expected, response.text
    assert not (tmp_path / "user-models" / f"{a_id}.json").exists()


async def test_a_keyless_model_at_its_own_address_stores_the_placeholder(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path)) as client:
        a_id, a = await _user(client, "Laptop A")
        response = await _add(
            client, a, {"model": "openai/local-model", "api_base": USER_BASE}
        )
    assert response.status_code == 200, response.text
    saved = json.loads((tmp_path / "user-models" / f"{a_id}.json").read_text())
    params = saved["models"][0]["litellm_params"]
    assert (params["api_key"], params["api_base"]) == (NO_KEY, USER_BASE)
    assert response.json()["deployments"][0]["api_key_configured"] is False


async def test_revisions_limits_and_share_are_the_settings_rules(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path)) as client:
        _a_id, a = await _user(client, "Laptop A")
        first = await _add(client, a, {"model": "openai/m0", "api_key": USER_A_KEY})
        deployment = first.json()["deployments"][0]["deployment_id"]
        revision = first.json()["revision"]
        stale = await client.post(
            "/api/me/models/deployments/batch",
            json={
                "expected_revision": "stale",
                "deployments": [
                    {
                        "model_name": "default",
                        "model": "openai/y",
                        "api_key": USER_A_KEY,
                    }
                ],
            },
            headers=a,
        )
        share = await client.patch(
            f"/api/me/models/deployments/{deployment}",
            json={"expected_revision": revision, "share": "shared"},
            headers=a,
        )
        many = await _add(
            client,
            a,
            *(
                {"model": f"openai/m{i}", "api_key": USER_A_KEY, "weight": 0}
                for i in range(1, 50)
            ),
        )
        one_more = await _add(
            client, a, {"model": "openai/one-more", "api_key": USER_A_KEY}
        )
        edited = await client.patch(
            f"/api/me/models/deployments/{deployment}",
            json={"expected_revision": many.json()["revision"], "weight": 4},
            headers=a,
        )
        moved = await client.patch(
            f"/api/me/models/deployments/{deployment}",
            json={
                "expected_revision": edited.json()["revision"],
                "api_key": USER_A_KEY,
                "api_base": "http://127.0.0.1:8000/v1",
            },
            headers=a,
        )
    assert stale.status_code == 409
    assert stale.json()["detail"] == {
        "code": "stale_revision",
        "current_revision": revision,
    }
    assert _why(share) == (422, "invalid_settings_request")
    assert many.status_code == 200, many.text
    assert len(many.json()["deployments"]) == 50
    assert _why(one_more) == (409, "model_limit")
    assert edited.status_code == 200, edited.text
    assert edited.json()["deployments"][0]["weight"] == 4
    assert _why(moved) == (422, "server_address")


async def test_tests_and_discoveries_follow_the_rules_before_anything_is_sent(
    tmp_path: Path, no_network: list[Any]
) -> None:
    async with _server(_app(tmp_path)) as client:
        _a_id, a = await _user(client, "Laptop A")
        refused = [
            (
                await client.post("/api/me/models/test", json=body, headers=a),
                expected,
            )
            for body, expected in (
                ({"model": "openai/draft"}, "key_or_address_required"),
                ({"model": "copilot/gpt-5"}, "server_credentials"),
                (
                    {
                        "model": "openai/draft",
                        "api_key": USER_A_KEY,
                        "api_base": "http://[::1]:9/v1",
                    },
                    "server_address",
                ),
            )
        ] + [
            (
                await client.post(
                    "/api/me/models/model-discovery", json=body, headers=a
                ),
                expected,
            )
            for body, expected in (
                ({"provider": "ollama"}, "server_address"),
                ({"provider": "claude-agent"}, "server_credentials"),
                ({"provider": "bedrock"}, "server_credentials"),
                (
                    {"provider": "openai", "provider_id": "env:openai"},
                    "invalid_settings_request",
                ),
            )
        ]
        assert no_network == []
        # Without a key or an address, a discovery names no variable
        keyless = (
            await client.post(
                "/api/me/models/model-discovery", json={"provider": "openai"}, headers=a
            )
        ).json()
        tested = (
            await client.post(
                "/api/me/models/test",
                json={"model": "openai/draft", "api_base": USER_BASE},
                headers=a,
            )
        ).json()
        listed = (
            await client.post(
                "/api/me/models/model-discovery",
                json={"provider": "custom", "api_base": USER_BASE},
                headers=a,
            )
        ).json()
    for response, expected in refused:
        assert _why(response) == (422, expected), response.text
    assert keyless["reason"] == "no_key"
    assert "key_variable" not in keyless
    assert "OPENAI_API_KEY" not in json.dumps(keyless)
    assert tested["ok"] is True
    assert listed["status"] == "ok"
    # Keyless drafts at an address go with the placeholder, never the
    # server's key variable
    assert no_network == [
        ("test", "openai/draft", NO_KEY, USER_BASE),
        ("discover", "custom", NO_KEY, USER_BASE),
    ]


async def test_a_users_provider_cards_name_nothing_of_the_server(
    tmp_path: Path, no_network: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-env-fixture-value")
    async with _server(_app(tmp_path)) as client:
        _a_id, a = await _user(client, "Laptop A")
        cards = (await client.get("/api/me/models/providers", headers=a)).json()
        admin = (await client.get("/api/settings/llm/providers", headers=ADMIN)).json()
    providers = {card["provider"]: card for card in cards["providers"]}
    assert all(card["kind"] in ("common", "configured") for card in cards["providers"])
    assert all(card["key_variable"] is None for card in cards["providers"])
    assert all("credential" not in card for card in cards["providers"])
    for refused in ("claude-agent", "copilot", "chatgpt", "vertex_ai", "watsonx"):
        assert refused not in providers
    assert providers["ollama"]["default_base"] is None
    assert providers["openai"]["default_base"] == "https://api.openai.com/v1"
    assert providers["bedrock"]["key_optional"] is False
    assert "custom" in providers
    # The admin's list does show the environment card
    assert any(card["kind"] == "environment" for card in admin["providers"])


async def test_the_admin_shares_a_model_and_a_user_sees_only_what_it_is(
    tmp_path: Path, no_network: list[Any]
) -> None:
    config = {
        "llm": {
            "model_list": [
                {
                    "model_name": "fast",
                    "litellm_params": {
                        "model": "openai/fake-model",
                        "api_key": ADMIN_KEY,
                        "api_base": ADMIN_BASE,
                    },
                    "model_info": {"id": "shared-model-1", "provider_id": "admin-1"},
                }
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
    async with _server(_app(tmp_path, config)) as client:
        _b_id, b = await _user(client, "Laptop B")
        view = (await client.get("/api/settings/llm", headers=ADMIN)).json()

        async def share(value: Any) -> httpx.Response:
            revision = (await client.get("/api/settings/llm", headers=ADMIN)).json()[
                "revision"
            ]
            return await client.patch(
                "/api/settings/llm/deployments/shared-model-1",
                json={"expected_revision": revision, "share": value},
                headers=ADMIN,
            )

        public = await share("public")
        null = await share(None)
        shared = await share("shared")
        saved = json.loads((tmp_path / "config.json").read_text())
        mine = (await client.get("/api/me/models", headers=b)).json()
        capabilities = (await client.get("/api/capabilities", headers=b)).json()
        refused = [
            await client.post(
                "/api/me/models/test",
                json={"deployment_id": "shared-model-1"},
                headers=b,
            ),
            await client.get("/api/me/models/providers/admin-1/credentials", headers=b),
            await client.post(
                "/api/me/models/deployments/batch",
                json={
                    "expected_revision": mine["revision"],
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "openai/fake-model",
                            "credential_provider_id": "admin-1",
                        }
                    ],
                },
                headers=b,
            ),
        ]
        private = await share("private")
        after = (await client.get("/api/me/models", headers=b)).json()
        refused_job = await client.post(
            "/api/jobs",
            files=[
                ("files", ("notes.md", b"# Notes\n", "text/markdown")),
                ("options", (None, json.dumps({"llm": True}))),
            ],
            headers=b,
        )
    assert view["deployments"][0]["share"] == "private"
    assert view["detected"] == []
    assert _why(public) == (422, "invalid_settings_request")
    assert _why(null) == (422, "invalid_settings_request")
    assert shared.json()["deployments"][0]["share"] == "shared"
    assert saved["llm"]["model_list"][0]["model_info"]["share"] == "shared"
    assert mine["shared"] == [
        {
            "id": "shared-model-1",
            "model": "openai/fake-model",
            "provider": "custom",
            "weight": 1,
            "default_weight": 1,
        }
    ]
    assert (mine["routable"], mine["configured"]) == (True, False)
    assert capabilities["llm"]["routable"] is True
    assert capabilities["llm"]["models"] == ["openai/fake-model"]
    for answer in (json.dumps(mine), json.dumps(capabilities)):
        for hidden in (ADMIN_KEY, ADMIN_BASE, "models.admin.example", "routing_group"):
            assert hidden not in answer
    assert [_why(r) for r in refused] == [
        (422, "invalid_settings_request"),
        (404, "settings_entry_missing"),
        (404, "settings_entry_missing"),
    ]
    assert private.json()["deployments"][0]["share"] == "private"
    assert (after["shared"], after["routable"]) == ([], False)
    assert _why(refused_job) == (422, "llm_unavailable")
    assert no_network == []


async def test_revoking_deletes_the_users_models_and_regenerating_keeps_them(
    tmp_path: Path, no_network: list[Any]
) -> None:
    from loguru import logger

    lines: list[str] = []
    handler = logger.add(lines.append, format="{message}", level="INFO")
    try:
        async with _server(_app(tmp_path)) as client:
            a_id, a = await _user(client, "Laptop A")
            b_id, b = await _user(client, "Laptop B")
            await _add(client, a, {"model": "openai/a", "api_key": USER_A_KEY})
            await _add(client, b, {"model": "openai/b", "api_key": USER_B_KEY})
            folder = tmp_path / "user-models"
            b_bytes = (folder / f"{b_id}.json").read_bytes()
            orphan = folder / "u_0000000000ff.json"
            orphan.write_bytes((folder / f"{a_id}.json").read_bytes())
            regenerated = await client.post(
                "/api/access-token/regenerate", json={}, headers=ADMIN
            )
            admin = {"Authorization": f"Bearer {regenerated.json()['token']}"}
            revoked = await client.delete(f"/api/access-tokens/{a_id}", headers=admin)
            gone = await client.get("/api/me/models", headers=a)
            a_after = (folder / f"{a_id}.json").exists()
            b_after = (folder / f"{b_id}.json").read_bytes()
        # A restart deletes the file of a token that no longer exists
        async with _server(_app(tmp_path)) as client:
            b_view = (await client.get("/api/me/models", headers=b)).json()
    finally:
        logger.remove(handler)
    assert revoked.status_code == 204
    assert gone.status_code == 401
    assert a_after is False
    assert b_after == b_bytes
    assert any(
        'Access token "Laptop A" revoked from the web workbench; its 1 model was '
        "deleted." in line
        for line in lines
    )
    assert not orphan.exists()
    assert [row["model"] for row in b_view["deployments"]] == ["openai/b"]


# ----------------------------------------------- the server's key variables


def test_a_keyless_entry_is_sealed_with_the_placeholder() -> None:
    """Saved without a key of its own or of its connection, a user's model
    (from Edit config as much as from the routes) stores the placeholder:
    LiteLLM would otherwise send the server's OPENAI_API_KEY to its address."""
    linked = {"id": "p-own", "provider": "openai", "api_key": USER_A_KEY}
    lists = Lists(
        models=[
            {"litellm_params": {"model": "openai/a", "api_base": USER_BASE}},
            {
                "litellm_params": {"model": "openai/b"},
                "model_info": {"provider_id": "p-own"},
            },
        ],
        providers=[
            linked,
            {"id": "p-bare", "provider": "custom", "api_base": USER_BASE},
        ],
    )
    user_models.check(lists)
    assert lists.models[0]["litellm_params"]["api_key"] == NO_KEY
    assert "api_key" not in lists.models[1]["litellm_params"]
    assert lists.providers[0]["api_key"] == USER_A_KEY
    assert lists.providers[1]["api_key"] == NO_KEY


def test_the_pool_never_leaves_a_user_model_to_the_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file saved before the placeholder was stored still routes with the
    entry's own key, its connection's, or the placeholder."""
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    own = Lists(
        models=[
            {
                "model_name": "default",
                "litellm_params": {"model": "openai/a", "api_base": USER_BASE},
            },
            {
                "model_name": "default",
                "litellm_params": {"model": "openai/b"},
                "model_info": {"provider_id": "p-own"},
            },
        ],
        providers=[
            {
                "id": "p-own",
                "provider": "openai",
                "api_key": USER_A_KEY,
                "api_base": USER_BASE,
            }
        ],
    )
    pool = user_models.build_pool(_base(), own, [], [])
    sent = [(m.litellm_params.api_key, m.litellm_params.api_base) for m in pool.models]
    assert sent == [(NO_KEY, USER_BASE), (USER_A_KEY, USER_BASE)]


async def test_edit_config_cannot_save_a_model_that_borrows_the_server_key(
    tmp_path: Path,
) -> None:
    async with _server(_app(tmp_path)) as client:
        user_id, headers = await _user(client, "Laptop A")
        view = (await client.get("/api/me/config", headers=headers)).json()
        text = json.dumps(
            {
                "models": [
                    {
                        "model_name": "default",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": USER_BASE,
                        },
                    }
                ],
                "providers": [],
                "shared": [],
            }
        )
        saved = await client.put(
            "/api/me/config",
            json={"expected_revision": view["revision"], "text": text},
            headers=headers,
        )
        listed = (await client.get("/api/me/models", headers=headers)).json()
    assert saved.status_code == 200, saved.text
    stored = json.loads((tmp_path / "user-models" / f"{user_id}.json").read_text())
    assert [m["litellm_params"]["api_key"] for m in stored["models"]] == [NO_KEY]
    assert [row["api_key_configured"] for row in listed["deployments"]] == [False]


# ------------------------------------------------- Cloudflare processing


async def test_cloudflare_processing_billed_to_the_server_is_the_admins_alone(
    tmp_path: Path,
) -> None:
    config = {
        "fetch": {
            "remote_consent": "ask",
            "cloudflare": {
                "api_token": "synthetic-token",
                "account_id": "0123456789abcdef0123456789abcdef",
                "convert_enabled": True,
            },
        }
    }
    async with _server(_app(tmp_path, config)) as client:
        _, user = await _user(client, "guest")
        admin_caps = (await client.get("/api/capabilities", headers=ADMIN)).json()
        user_caps = (await client.get("/api/capabilities", headers=user)).json()
        refused = await client.post(
            "/api/jobs",
            files=[
                ("files", ("no.txt", b"local", "text/plain")),
                (
                    "options",
                    (
                        None,
                        json.dumps(
                            {"backend": "cloudflare", "remote_processing": "cloudflare"}
                        ),
                    ),
                ),
            ],
            headers=user,
        )
    assert admin_caps["remote_services"]["cloudflare"]["reason"] != "client_not_trusted"
    assert user_caps["remote_services"]["cloudflare"]["available"] is False
    assert user_caps["remote_services"]["cloudflare"]["reason"] == "client_not_trusted"
    assert _why(refused) == (403, "remote_processing_forbidden")


# ------------------------------------------- the server's machine and network


def test_the_pool_restricts_only_the_users_own_model_hosts() -> None:
    own = Lists(
        models=[
            {
                "model_name": "default",
                "litellm_params": {
                    "model": "openai/a",
                    "api_base": "http://model.test:8000/v1",
                    "api_key": USER_A_KEY,
                },
            }
        ]
    )
    pool = user_models.build_pool(_base(), own, [], [])
    assert pool.own_hosts == ("model.test",)


async def test_a_users_model_test_never_reaches_this_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket
    import threading

    seen: list[bytes] = []
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.2)
    port = listener.getsockname()[1]
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn:
                seen.append(conn.recv(4096))

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    real = socket.getaddrinfo

    def resolve(host: Any, *args: Any, **kwargs: Any) -> Any:
        # A name that (re)binds to this machine
        if host == "model.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    try:
        async with _server(_app(tmp_path)) as client:
            _, user = await _user(client, "guest")
            tested = await client.post(
                "/api/me/models/test",
                json={
                    "model": "openai/gpt-x",
                    "api_base": f"http://model.test:{port}/v1",
                    "api_key": USER_A_KEY,
                },
                headers=user,
            )
    finally:
        stop.set()
        thread.join(timeout=2)
        listener.close()
    answer = tested.json()
    assert seen == []
    assert answer["ok"] is False, answer
    assert "not on the public internet" in answer["detail"], answer


async def test_a_proxy_that_would_carry_a_users_model_fails_only_its_model_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behind a proxy a user's own model host cannot be held to public
    addresses: a run that would call it fails with that reason (it once
    crashed as an internal error), and a run without the LLM converts."""
    import asyncio

    for name in ("HTTP_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    async with _server(_app(tmp_path)) as client:
        _, user = await _user(client, "guest")
        added = await _add(
            client,
            user,
            {"model": "openai/gpt-x", "api_key": USER_A_KEY, "api_base": USER_BASE},
        )
        assert added.status_code == 200, added.text
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")

        async def finished(llm: bool) -> dict[str, Any]:
            created = await client.post(
                "/api/jobs",
                files=[
                    ("files", ("notes.md", b"# Notes\n\nText.\n", "text/markdown")),
                    ("options", (None, json.dumps({"llm": llm}))),
                ],
                headers=user,
            )
            assert created.status_code in (200, 201, 202), created.text
            job_id = created.json()["job_id"]
            for _ in range(500):
                job = (await client.get(f"/api/jobs/{job_id}", headers=user)).json()
                if job["status"] not in ("queued", "running"):
                    return job["items"][0]
                await asyncio.sleep(0.02)
            pytest.fail(f"job {job_id} did not finish")

        plain = await finished(False)
        enhanced = await finished(True)
    assert plain["status"] == "done", plain
    assert enhanced["status"] == "error", enhanced
    assert enhanced["error_code"] == "enhancement_failed", enhanced
    assert "proxy" in enhanced["error"], enhanced
