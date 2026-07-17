"""Tests for the LLM settings endpoints (``/api/settings/llm``).

Same harness as test_serve_api.py: ``create_app`` with a tmp_path-backed jobs
root, an injected config and a tmp_path config file target, over
httpx.ASGITransport — the real ``~/.markitai`` is never touched.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

import httpx

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def _model_entry(
    model: str,
    api_key: str | None = None,
    api_base: str | None = None,
    model_name: str = "default",
) -> ModelConfig:
    """Build one llm.model_list entry."""
    return ModelConfig(
        model_name=model_name,
        litellm_params=LiteLLMParams(model=model, api_key=api_key, api_base=api_base),
    )


def _make_app(
    tmp_path: Path,
    cfg: MarkitaiConfig | None = None,
    config_path: Path | None = None,
) -> FastAPI:
    """Build an app with hermetic config, jobs root and config file target."""
    if cfg is None:
        cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=(
            config_path if config_path is not None else tmp_path / "config.json"
        ),
    )


@asynccontextmanager
async def _serve_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app lifespan and yield an ASGI-backed client."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


class TestMaskApiKey:
    """Masking rules for api_key display."""

    @pytest.mark.parametrize(
        ("api_key", "masked"),
        [
            (None, None),
            ("env:GEMINI_API_KEY", "env:GEMINI_API_KEY"),
            ("sk-1234567890abcdef", "sk…cdef"),
            ("abcdef123456", "ab…3456"),  # 12 chars: 6 shown == half
            ("abcdefgh", "…gh"),  # short: last 2 only
            ("abcd", "…cd"),
            ("abc", "…"),  # tiny: nothing revealed
            ("", "…"),
        ],
    )
    def test_masking_table(self, api_key: str | None, masked: str | None) -> None:
        from markitai.serve.app import _mask_api_key

        assert _mask_api_key(api_key) == masked


class TestGetLLMSettings:
    """GET /api/settings/llm masked view and source tracking."""

    async def test_masked_view_with_env_literal_and_null_keys(
        self, tmp_path: Path
    ) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry("gemini/gemini-2.5-flash", api_key="env:GEMINI_API_KEY"),
            _model_entry(
                "openai/gpt-4o-mini",
                api_key="sk-1234567890abcdef",
                api_base="https://proxy.local/v1",
            ),
            _model_entry("claude-agent/sonnet"),
        ]
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            resp = await client.get("/api/settings/llm")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is True
        assert data["source"] == "config"
        deployments = data["deployments"]
        assert [entry["routing_group"] for entry in deployments] == [
            "default",
            "default",
            "default",
        ]
        assert [entry["model"] for entry in deployments] == [
            "gemini/gemini-2.5-flash",
            "openai/gpt-4o-mini",
            "claude-agent/sonnet",
        ]
        assert [entry["api_key_configured"] for entry in deployments] == [
            True,
            True,
            False,
        ]
        assert deployments[1]["api_base"] == "https://proxy.local"
        assert all(entry["deployment_id"] for entry in deployments)
        assert "sk-1234567890abcdef" not in resp.text
        assert "env:GEMINI_API_KEY" not in resp.text

    async def test_source_none_for_empty_config(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/settings/llm")
        data = resp.json()
        assert data["configured"] is False
        assert data["routable"] is False
        assert data["source"] == "none"
        assert data["deployments"] == []
        assert data["detected"] == []
        assert len(data["revision"]) == 64
        assert data["config_path"].endswith("config.json")

    async def test_source_detected_after_startup_backfill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """config=None + empty config file -> backfill populates -> detected."""
        monkeypatch.setenv("MARKITAI_CONFIG", str(tmp_path / "absent.json"))
        monkeypatch.delenv("MODEL", raising=False)

        async def fake_detect() -> list[dict[str, Any]]:
            return [
                {
                    "provider": "openai",
                    "model": "openai/gpt-detected",
                    "label": "OPENAI_API_KEY (environment)",
                    "requires_api_key": False,
                }
            ]

        monkeypatch.setattr(
            "markitai.serve.app._detect_provider_candidates", fake_detect
        )
        app = create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            configure_logging=False,
            config_path=tmp_path / "config.json",
        )
        async with _serve_client(app) as client:
            data = (await client.get("/api/settings/llm")).json()
        assert data["configured"] is False
        assert data["source"] == "detected"
        assert data["deployments"] == []
        assert [model["model"] for model in data["detected"]] == ["openai/gpt-detected"]

    async def test_local_provider_entry_with_stored_key_masks_null(
        self, tmp_path: Path
    ) -> None:
        """A key stored on a local-provider entry never surfaces in GET."""
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry("claude-agent/sonnet", api_key="sk-legacy-1234567890")
        ]
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            data = (await client.get("/api/settings/llm")).json()
        assert data["deployments"][0]["api_key_configured"] is False
        assert "sk-legacy-1234567890" not in json.dumps(data)


class TestAddLLMModel:
    """POST /api/settings/llm/models: add-entry + hot update semantics."""

    async def test_add_creates_config_file_with_0600(self, tmp_path: Path) -> None:
        config_path = tmp_path / "home" / ".markitai" / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            resp = await client.post(
                "/api/settings/llm/models",
                json={
                    "model_name": "primary",
                    "model": "openai/gpt-test",
                    "api_key": "sk-1234567890abcdef",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is True
        assert data["source"] == "config"
        entry = data["deployments"][0]
        assert entry["routing_group"] == "primary"
        assert entry["model"] == "openai/gpt-test"
        assert entry["api_key_configured"] is True
        assert entry["deployment_id"]
        assert "sk-1234567890abcdef" not in resp.text

        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        stored = on_disk["llm"]["model_list"][0]
        assert stored["model_name"] == "primary"
        assert stored["litellm_params"] == {
            "model": "openai/gpt-test",
            "api_key": "sk-1234567890abcdef",
        }
        assert stored["model_info"]["id"] == entry["deployment_id"]
        if os.name == "posix":
            assert (config_path.stat().st_mode & 0o777) == 0o600

    async def test_add_appends_and_preserves_unrelated_keys(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "output": {"dir": "./out"},
                    "custom_top": {"keep": True},
                    "llm": {
                        "enabled": True,
                        "concurrency": 3,
                        "model_list": [
                            {
                                "model_name": "default",
                                "litellm_params": {"model": "old/model", "weight": 2},
                                "custom_entry_key": "keep-me",
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            resp = await client.post(
                "/api/settings/llm/models",
                json={
                    "model_name": "extra",
                    "model": "openai/gpt-new",
                    "api_base": "env:MY_BASE",
                },
            )
        assert resp.status_code == 200

        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk["output"] == {"dir": "./out"}  # unrelated keys survive
        assert on_disk["custom_top"] == {"keep": True}
        assert on_disk["llm"]["enabled"] is True  # llm.enabled untouched
        assert on_disk["llm"]["concurrency"] == 3
        stored_models = on_disk["llm"]["model_list"]
        assert stored_models[0] == {
            "model_name": "default",
            "litellm_params": {"model": "old/model", "weight": 2},
            "custom_entry_key": "keep-me",
        }
        assert stored_models[1]["model_name"] == "extra"
        assert stored_models[1]["litellm_params"] == {
            "model": "openai/gpt-new",
            "api_base": "env:MY_BASE",
        }
        assert stored_models[1]["model_info"]["id"]
        if os.name == "posix":  # rewrite tightens permissions to owner-only
            assert (config_path.stat().st_mode & 0o777) == 0o600

    async def test_add_duplicate_routing_group_is_allowed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            first = await client.post(
                "/api/settings/llm/models",
                json={"model_name": "primary", "model": "openai/first"},
            )
            assert first.status_code == 200
            dup = await client.post(
                "/api/settings/llm/models",
                json={"model_name": "primary", "model": "openai/second"},
            )
            settings = (await client.get("/api/settings/llm")).json()
        assert dup.status_code == 200
        assert [model["routing_group"] for model in settings["deployments"]] == [
            "primary",
            "primary",
        ]
        assert [model["model"] for model in settings["deployments"]] == [
            "openai/first",
            "openai/second",
        ]

    async def test_add_local_provider_ignores_key_and_base(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            resp = await client.post(
                "/api/settings/llm/models",
                json={
                    "model_name": "local",
                    "model": "claude-agent/haiku",
                    "api_key": "sk-should-be-ignored",
                    "api_base": "https://ignored.local",
                },
            )
        assert resp.status_code == 200
        entry = resp.json()["deployments"][0]
        assert entry["api_key_configured"] is False
        assert entry["api_base_configured"] is False
        assert entry["api_base"] is None
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        stored = on_disk["llm"]["model_list"][0]
        assert stored["model_name"] == "local"
        assert stored["litellm_params"] == {"model": "claude-agent/haiku"}
        assert stored["model_info"]["id"]
        assert "sk-should-be-ignored" not in config_path.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "payload",
        [
            {"model": "openai/x"},  # model_name missing
            {"model_name": "a"},  # model missing
            {"model_name": "   ", "model": "openai/x"},  # blank model_name
            {"model_name": "a", "model": "   "},  # blank model
            {"model_name": "a", "model": "openai/x", "bogus": 1},  # unknown key
            {"model_name": "a", "model": "openai/x", "api_key": 123},  # wrong type
            {"model_name": "a", "model": "openai/x", "api_key": "sk…cdef"},  # mask
        ],
    )
    async def test_add_validation_is_422(
        self, tmp_path: Path, payload: dict[str, Any]
    ) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            resp = await client.post("/api/settings/llm/models", json=payload)
        assert resp.status_code == 422
        assert not config_path.exists()  # nothing written on validation failure

    async def test_add_hot_updates_capabilities(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            before = (await client.get("/api/capabilities")).json()
            assert before["llm"] == {
                "configured": False,
                "routable": False,
                "effective": False,
                "models": [],
            }
            resp = await client.post(
                "/api/settings/llm/models",
                json={"model_name": "primary", "model": "openai/gpt-test"},
            )
            assert resp.status_code == 200
            after = (await client.get("/api/capabilities")).json()
        assert after["llm"] == {
            "configured": True,
            "routable": False,
            "effective": False,
            "models": ["openai/gpt-test"],
        }


class TestUpdateLLMModel:
    """PUT /api/settings/llm/models/{model_name}: field-merge semantics."""

    @staticmethod
    async def _seed(client: httpx.AsyncClient) -> None:
        resp = await client.post(
            "/api/settings/llm/models",
            json={
                "model_name": "primary",
                "model": "openai/gpt-test",
                "api_key": "sk-1234567890abcdef",
                "api_base": "https://proxy.local/v1",
            },
        )
        assert resp.status_code == 200

    async def test_put_omitted_api_key_keeps_current_value(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await self._seed(client)
            resp = await client.put(
                "/api/settings/llm/models/primary",
                json={"model": "openai/gpt-next"},  # api_key/api_base omitted
            )
            settings = (await client.get("/api/settings/llm")).json()
        assert resp.status_code == 200
        entry = settings["deployments"][0]
        assert entry["routing_group"] == "primary"
        assert entry["model"] == "openai/gpt-next"
        assert entry["api_key_configured"] is True
        assert entry["api_base"] == "https://proxy.local"
        params = json.loads(config_path.read_text(encoding="utf-8"))["llm"][
            "model_list"
        ][0]["litellm_params"]
        assert params == {
            "model": "openai/gpt-next",
            "api_key": "sk-1234567890abcdef",  # literal key survives on disk
            "api_base": "https://proxy.local/v1",
        }

    async def test_put_explicit_null_clears_api_key(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await self._seed(client)
            resp = await client.put(
                "/api/settings/llm/models/primary",
                json={"api_key": None, "api_base": None},
            )
        assert resp.status_code == 200
        entry = resp.json()["deployments"][0]
        assert entry["api_key_configured"] is False
        assert entry["api_base"] is None
        params = json.loads(config_path.read_text(encoding="utf-8"))["llm"][
            "model_list"
        ][0]["litellm_params"]
        assert params == {"model": "openai/gpt-test"}

    async def test_put_new_string_replaces_api_key(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await self._seed(client)
            resp = await client.put(
                "/api/settings/llm/models/primary",
                json={"api_key": "env:NEW_KEY_VAR"},
            )
        assert resp.status_code == 200
        assert resp.json()["deployments"][0]["api_key_configured"] is True
        params = json.loads(config_path.read_text(encoding="utf-8"))["llm"][
            "model_list"
        ][0]["litellm_params"]
        assert params["api_key"] == "env:NEW_KEY_VAR"

    async def test_put_masked_value_writeback_is_422(self, tmp_path: Path) -> None:
        """Echoing the masked key from GET back must never corrupt the file."""
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await self._seed(client)
            before = config_path.read_text(encoding="utf-8")
            settings = (await client.get("/api/settings/llm")).json()
            assert "sk-1234567890abcdef" not in json.dumps(settings)
            resp = await client.put(
                "/api/settings/llm/models/primary", json={"api_key": "sk…cdef"}
            )
        assert resp.status_code == 422
        assert config_path.read_text(encoding="utf-8") == before

    async def test_put_unknown_model_name_is_404(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.put(
                "/api/settings/llm/models/nope", json={"model": "openai/x"}
            )
        assert resp.status_code == 404

    async def test_put_switch_to_local_provider_drops_stored_key(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await self._seed(client)
            resp = await client.put(
                "/api/settings/llm/models/primary",
                json={"model": "claude-agent/sonnet"},
            )
        assert resp.status_code == 200
        entry = resp.json()["deployments"][0]
        assert entry["model"] == "claude-agent/sonnet"
        assert entry["api_key_configured"] is False
        params = json.loads(config_path.read_text(encoding="utf-8"))["llm"][
            "model_list"
        ][0]["litellm_params"]
        assert params == {"model": "claude-agent/sonnet"}

    @pytest.mark.parametrize(
        "payload",
        [
            {"model": None},  # model cannot be cleared
            {"model": "   "},  # blank model
            {"bogus": 1},  # unknown key
            {"api_base": "https://x/…/v1"},  # mask char in api_base
        ],
    )
    async def test_put_validation_is_422(
        self, tmp_path: Path, payload: dict[str, Any]
    ) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await self._seed(client)
            before = config_path.read_text(encoding="utf-8")
            resp = await client.put("/api/settings/llm/models/primary", json=payload)
        assert resp.status_code == 422
        assert config_path.read_text(encoding="utf-8") == before


class TestDeleteLLMModel:
    """DELETE /api/settings/llm/models/{model_name}."""

    async def test_delete_removes_only_named_entry(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            for name, model in (("a", "openai/one"), ("b", "openai/two")):
                await client.post(
                    "/api/settings/llm/models",
                    json={"model_name": name, "model": model},
                )
            resp = await client.delete("/api/settings/llm/models/a")
        assert resp.status_code == 200
        data = resp.json()
        assert [model["routing_group"] for model in data["deployments"]] == ["b"]
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert [e["model_name"] for e in on_disk["llm"]["model_list"]] == ["b"]

    async def test_delete_last_entry_unconfigures(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            await client.post(
                "/api/settings/llm/models",
                json={"model_name": "only", "model": "openai/x"},
            )
            resp = await client.delete("/api/settings/llm/models/only")
            capabilities = (await client.get("/api/capabilities")).json()
        assert resp.status_code == 200
        data = resp.json()
        assert data["configured"] is False
        assert data["source"] == "none"
        assert data["deployments"] == []
        assert capabilities["llm"] == {
            "configured": False,
            "routable": False,
            "effective": False,
            "models": [],
        }
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk["llm"]["model_list"] == []

    async def test_delete_unknown_model_name_is_404(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.delete("/api/settings/llm/models/nope")
        assert resp.status_code == 404


class TestProviderConnections:
    async def test_local_cli_and_its_saved_model_share_one_provider_card(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.providers import discovery

        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry("claude-agent/sonnet"),
            _model_entry(
                "deepseek/deepseek-chat", "env:DEEPSEEK_API_KEY", model_name="a"
            ),
            _model_entry(
                "deepseek/deepseek-reasoner", "env:DEEPSEEK_API_KEY", model_name="b"
            ),
        ]

        async def fake_connections(
            _configured: list[ModelConfig], *, refresh: bool = False
        ) -> list[dict[str, Any]]:
            return [
                {
                    "id": "claude-agent",
                    "provider": "claude-agent",
                    "label": "Claude Code CLI",
                    "kind": "local_cli",
                    "status": "ready",
                    "source": "cli",
                    "default_model": "claude-agent/sonnet",
                    "supports_discovery": True,
                },
                {
                    "id": "configured:1",
                    "provider": "claude-agent",
                    "label": "Claude-Agent",
                    "kind": "configured",
                    "status": "ready",
                    "source": "config",
                    "supports_discovery": True,
                },
            ]

        monkeypatch.setattr(discovery, "detect_provider_connections", fake_connections)
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            response = await client.get("/api/settings/llm/providers?refresh=true")
        assert response.status_code == 200
        claude = [
            card
            for card in response.json()["providers"]
            if card["provider"] == "claude-agent"
        ]
        assert len(claude) == 1
        assert claude[0]["kind"] == "local_cli"
        assert claude[0]["supports_discovery"] is True
        deepseek = [
            card
            for card in response.json()["providers"]
            if card["provider"] == "deepseek"
        ]
        assert len(deepseek) == 1
        assert deepseek[0]["kind"] == "configured"
        assert deepseek[0]["label"] == "DeepSeek"


class TestProviderCredentialLifecycle:
    async def test_credentials_are_revealed_only_by_explicit_provider_request(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            revision = (await client.get("/api/settings/llm")).json()["revision"]
            added = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "deepseek/deepseek-chat",
                            "provider": "deepseek",
                            "api_key": "sk-visible-on-request",
                            "api_base": "https://proxy.example/v1",
                        }
                    ],
                },
            )
            assert added.status_code == 200
            cards_response = await client.get("/api/settings/llm/providers")
            card = next(
                card
                for card in cards_response.json()["providers"]
                if card["provider"] == "deepseek" and card["kind"] == "configured"
            )
            credentials = await client.get(
                f"/api/settings/llm/providers/{card['provider_id']}/credentials"
            )

        assert "sk-visible-on-request" not in cards_response.text
        assert credentials.status_code == 200
        assert credentials.headers["cache-control"] == "no-store"
        # An explicitly-saved base is returned verbatim; the default is offered
        # separately for the editor placeholder.
        assert credentials.json() == {
            "api_key": "sk-visible-on-request",
            "api_base": "https://proxy.example/v1",
            "api_base_placeholder": "https://api.deepseek.com",
        }

    async def test_provider_editor_gets_effective_default_api_base(
        self, tmp_path: Path
    ) -> None:
        """A key-only DeepSeek connection still has a visible effective URL."""
        async with _serve_client(_make_app(tmp_path)) as client:
            revision = (await client.get("/api/settings/llm")).json()["revision"]
            added = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "deepseek/deepseek-chat",
                            "provider": "deepseek",
                            "api_key": "sk-deepseek",
                        }
                    ],
                },
            )
            assert added.status_code == 200
            cards_response = await client.get("/api/settings/llm/providers")
            card = next(
                card
                for card in cards_response.json()["providers"]
                if card["provider"] == "deepseek"
                and card["kind"] == "configured"
            )
            credentials = await client.get(
                f"/api/settings/llm/providers/{card['provider_id']}/credentials"
            )

        assert card["api_base_configured"] is False
        assert card["api_base"] == "https://api.deepseek.com"
        # A key-only connection saves no base (stays on the provider default);
        # the editor field is empty with the default as its placeholder.
        assert credentials.json() == {
            "api_key": "sk-deepseek",
            "api_base": None,
            "api_base_placeholder": "https://api.deepseek.com",
        }

    async def test_legacy_embedded_credentials_migrate_before_last_model_delete(
        self, tmp_path: Path
    ) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry(
                "deepseek/deepseek-chat", "env:DEEPSEEK_API_KEY", model_name="a"
            ),
            _model_entry(
                "deepseek/deepseek-reasoner",
                "env:DEEPSEEK_API_KEY",
                model_name="b",
            ),
        ]
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, cfg, config_path=config_path)
        ) as client:
            payload = (await client.get("/api/settings/llm")).json()
            first_id = payload["deployments"][0]["deployment_id"]
            first = await client.delete(
                f"/api/settings/llm/deployments/{first_id}",
                params={"expected_revision": payload["revision"]},
            )
            assert first.status_code == 200
            payload = first.json()
            second_id = payload["deployments"][0]["deployment_id"]
            second = await client.delete(
                f"/api/settings/llm/deployments/{second_id}",
                params={"expected_revision": payload["revision"]},
            )
            assert second.status_code == 200
            cards = (await client.get("/api/settings/llm/providers")).json()[
                "providers"
            ]

        deepseek = next(
            card
            for card in cards
            if card["provider"] == "deepseek" and card["kind"] == "configured"
        )
        assert deepseek["model_count"] == 0
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))["llm"]
        assert on_disk["model_list"] == []
        assert on_disk["providers"][0]["api_key"] == "env:DEEPSEEK_API_KEY"

    async def test_deleting_all_models_keeps_provider_and_can_add_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.json"
        captured: dict[str, Any] = {}

        async def fake_discover(
            provider: str,
            *,
            api_key: str | None,
            api_base: str | None,
            refresh: bool,
        ) -> dict[str, Any]:
            captured.update(
                provider=provider,
                api_key=api_key,
                api_base=api_base,
                refresh=refresh,
            )
            return {
                "provider": provider,
                "status": "ok",
                "source": "test",
                "authoritative": True,
                "cached": False,
                "stale": False,
                "models": [],
            }

        monkeypatch.setattr(
            "markitai.providers.discovery.discover_models", fake_discover
        )
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            revision = (await client.get("/api/settings/llm")).json()["revision"]
            added = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "deepseek/deepseek-chat",
                            "provider": "deepseek",
                            "api_key": "sk-deepseek-secret",
                        },
                        {
                            "model_name": "default",
                            "model": "deepseek/deepseek-reasoner",
                            "provider": "deepseek",
                            "api_key": "sk-deepseek-secret",
                        },
                    ],
                },
            )
            payload = added.json()
            first, second = payload["deployments"]
            for deployment in (first, second):
                deleted = await client.delete(
                    f"/api/settings/llm/deployments/{deployment['deployment_id']}",
                    params={"expected_revision": payload["revision"]},
                )
                assert deleted.status_code == 200
                payload = deleted.json()

            cards = (await client.get("/api/settings/llm/providers")).json()[
                "providers"
            ]
            deepseek = next(
                card
                for card in cards
                if card["provider"] == "deepseek" and card["kind"] == "configured"
            )
            assert deepseek["model_count"] == 0
            assert deepseek["api_key_configured"] is True
            assert "sk-deepseek-secret" not in json.dumps(cards)

            discovered = await client.post(
                "/api/settings/llm/model-discovery",
                json={"provider": "deepseek", "provider_id": deepseek["provider_id"]},
            )
            assert discovered.status_code == 200
            readded = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": payload["revision"],
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "deepseek/deepseek-chat",
                            "provider": "deepseek",
                            "credential_provider_id": deepseek["provider_id"],
                        }
                    ],
                },
            )

        assert captured["api_key"] == "sk-deepseek-secret"
        assert readded.status_code == 200
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))["llm"]
        assert len(on_disk["providers"]) == 1
        assert on_disk["providers"][0]["api_key"] == "sk-deepseek-secret"
        assert on_disk["model_list"][0]["litellm_params"]["api_key"] == (
            "sk-deepseek-secret"
        )

    async def test_edit_provider_updates_models_and_delete_provider_clears_all(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            revision = (await client.get("/api/settings/llm")).json()["revision"]
            added = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "deepseek/deepseek-chat",
                            "provider": "deepseek",
                            "api_key": "sk-old-secret",
                        }
                    ],
                },
            )
            card = next(
                card
                for card in (
                    await client.get("/api/settings/llm/providers")
                ).json()["providers"]
                if card["provider"] == "deepseek" and card["kind"] == "configured"
            )
            edited = await client.patch(
                f"/api/settings/llm/providers/{card['provider_id']}",
                json={
                    "api_key": "sk-new-secret",
                    "api_base": "https://proxy.example/v1",
                    "expected_revision": added.json()["revision"],
                },
            )
            assert edited.status_code == 200
            deleted = await client.delete(
                f"/api/settings/llm/providers/{card['provider_id']}",
                params={"expected_revision": edited.json()["revision"]},
            )

        assert deleted.status_code == 200
        assert deleted.json()["deployments"] == []
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))["llm"]
        assert on_disk["providers"] == []
        assert on_disk["model_list"] == []
        text = config_path.read_text(encoding="utf-8")
        assert "sk-old-secret" not in text
        assert "sk-new-secret" not in text

    async def test_missing_provider_error_is_actionable(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/settings/llm/model-discovery",
                json={"provider": "deepseek", "provider_id": "gone"},
            )
        assert response.status_code == 404
        assert "Refresh the provider list" in response.json()["detail"]


class TestDetectedProviders:
    """GET /api/settings/llm/detected: detection candidates passthrough."""

    async def test_detected_returns_candidate_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        candidates = [
            {
                "provider": "claude-agent",
                "model": "claude-agent/sonnet",
                "label": "Claude Code CLI",
                "requires_api_key": False,
            },
            {
                "provider": "openai",
                "model": "openai/gpt-5.4-nano",
                "label": "OPENAI_API_KEY (environment)",
                "requires_api_key": False,
            },
        ]

        async def fake_detect() -> list[dict[str, Any]]:
            return candidates

        monkeypatch.setattr(
            "markitai.serve.app._detect_provider_candidates", fake_detect
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/settings/llm/detected")
        assert resp.status_code == 200
        assert resp.json() == candidates

    async def test_detected_env_provider_via_real_detector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real detector reports env-key providers without touching CLIs."""
        import shutil as shutil_module

        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(shutil_module, "which", lambda _b: None)  # no CLIs

        class _Unauthenticated:
            authenticated = False

        async def fake_check_auth(_self: Any, _provider: str) -> _Unauthenticated:
            return _Unauthenticated()

        monkeypatch.setattr(
            "markitai.providers.auth.AuthManager.check_auth", fake_check_auth
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/settings/llm/detected")
        assert resp.status_code == 200
        assert resp.json() == [
            {
                "provider": "gemini",
                "model": "gemini/gemini-3.1-flash-lite-preview",
                "label": "GEMINI_API_KEY (environment)",
                "requires_api_key": False,
            }
        ]


class TestLLMSettingsProbe:
    """POST /api/settings/llm/test: transient probe, uniform 200."""

    async def test_probe_success_with_minimal_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/settings/llm/test",
                json={"model": "openai/gpt-test", "api_key": "sk-1234567890abcdef"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "openai/gpt-test" in data["detail"]
        assert captured["model"] == "openai/gpt-test"
        assert captured["api_key"] == "sk-1234567890abcdef"
        assert captured["max_tokens"] == 16
        assert captured["messages"] == [
            {"role": "user", "content": "Reply with exactly OK."}
        ]
        assert captured["timeout"] == 15
        assert not (tmp_path / "config.json").exists()  # probe persists nothing

    async def test_probe_resolves_env_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKITAI_TEST_KEY", "sk-resolved-999999")
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/settings/llm/test",
                json={"model": "openai/gpt-test", "api_key": "env:MARKITAI_TEST_KEY"},
            )
        assert resp.json()["ok"] is True
        assert captured["api_key"] == "sk-resolved-999999"
        assert "sk-resolved-999999" not in resp.text

    async def test_probe_failure_is_sanitized_and_never_500(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def exploding_acompletion(**kwargs: Any) -> Any:
            raise RuntimeError(
                "Incorrect API key provided: sk-12345**********cdef "
                "(raw: sk-1234567890abcdef)\nsecond line with sk-1234567890abcdef"
            )

        monkeypatch.setattr("litellm.acompletion", exploding_acompletion)
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/settings/llm/test",
                json={"model": "openai/gpt-test", "api_key": "sk-1234567890abcdef"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data["detail"].startswith("RuntimeError:")
        assert "second line" not in data["detail"]  # first line only
        assert "sk-1234567890abcdef" not in resp.text  # full key scrubbed
        assert "sk-12345" not in resp.text  # partial echoes scrubbed too

    async def test_probe_missing_env_var_fails_without_calling_llm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MARKITAI_TEST_MISSING_VAR", raising=False)
        calls: list[Any] = []

        async def fake_acompletion(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/settings/llm/test",
                json={
                    "model": "openai/gpt-test",
                    "api_key": "env:MARKITAI_TEST_MISSING_VAR",
                },
            )
        data = resp.json()
        assert resp.status_code == 200
        assert data["ok"] is False
        assert "Environment variable not found" in data["detail"]
        assert calls == []

    async def test_probe_local_provider_without_sdk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "markitai.providers.is_local_provider_available", lambda _m: False
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.post(
                "/api/settings/llm/test", json={"model": "claude-agent/sonnet"}
            )
        data = resp.json()
        assert resp.status_code == 200
        assert data["ok"] is False
        assert "runtime support is missing" in data["detail"]
        assert "CLI login alone is not the SDK" in data["detail"]
        assert "uv sync --extra serve --extra claude-agent" in data["detail"]

    async def test_probe_without_key_uses_stored_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A saved-row test posts no api_key (the UI only has the mask);
        the probe must resolve the stored entry's credentials itself."""
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry(
                "openai/gpt-test",
                api_key="sk-stored-1234567890",
                api_base="https://proxy.example/v1",
            )
        ]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test",
                json={
                    "model": "openai/gpt-test",
                    "api_base": "https://proxy.example/v1",
                },
            )
        data = resp.json()
        assert resp.status_code == 200
        assert data["ok"] is True
        assert captured["api_key"] == "sk-stored-1234567890"
        assert captured["api_base"] == "https://proxy.example/v1"
        assert "sk-stored-1234567890" not in resp.text

    async def test_probe_explicit_key_wins_over_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [_model_entry("openai/gpt-test", api_key="sk-stored-1111")]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test",
                json={"model": "openai/gpt-test", "api_key": "sk-explicit-2222"},
            )
        assert resp.json()["ok"] is True
        assert captured["api_key"] == "sk-explicit-2222"

    async def test_probe_stored_key_failure_is_scrubbed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resolved stored key must be scrubbed from failure details."""

        async def exploding_acompletion(**kwargs: Any) -> Any:
            raise RuntimeError(f"Incorrect API key provided: {kwargs['api_key']}")

        monkeypatch.setattr("litellm.acompletion", exploding_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry("openai/gpt-test", api_key="sk-stored-1234567890")
        ]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test", json={"model": "openai/gpt-test"}
            )
        data = resp.json()
        assert resp.status_code == 200
        assert data["ok"] is False
        assert "sk-stored-1234567890" not in resp.text

    async def test_probe_unknown_model_stays_keyless(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [_model_entry("openai/other-model", api_key="sk-other-1")]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test", json={"model": "openai/gpt-test"}
            )
        assert resp.json()["ok"] is True
        assert "api_key" not in captured

    async def test_probe_model_name_reference_uses_stored_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A saved-row test posts only model_name; the probe must run with
        that entry's full stored params (the UI never has the real key)."""
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry(
                "openai/gpt-test",
                api_key="sk-stored-1234567890",
                api_base="https://proxy.example/v1",
                model_name="primary",
            )
        ]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test", json={"model_name": "primary"}
            )
        data = resp.json()
        assert resp.status_code == 200
        assert data["ok"] is True
        assert "openai/gpt-test" in data["detail"]
        assert captured["model"] == "openai/gpt-test"
        assert captured["api_key"] == "sk-stored-1234567890"
        assert captured["api_base"] == "https://proxy.example/v1"
        assert "sk-stored-1234567890" not in resp.text

    async def test_probe_model_name_reference_resolves_env_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MARKITAI_TEST_REF_KEY", "sk-ref-resolved-42")
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> dict[str, str]:
            captured.update(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry(
                "openai/gpt-test",
                api_key="env:MARKITAI_TEST_REF_KEY",
                model_name="primary",
            )
        ]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test", json={"model_name": "primary"}
            )
        assert resp.json()["ok"] is True
        assert captured["api_key"] == "sk-ref-resolved-42"
        assert "sk-ref-resolved-42" not in resp.text

    async def test_probe_unknown_model_name_is_422(self, tmp_path: Path) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [_model_entry("openai/gpt-test", model_name="primary")]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post(
                "/api/settings/llm/test", json={"model_name": "nope"}
            )
        assert resp.status_code == 422
        assert "nope" in resp.json()["detail"]

    @pytest.mark.parametrize(
        "payload",
        [
            {"model_name": "primary", "model": "openai/gpt-test"},  # ambiguous
            {"model_name": "primary", "api_key": "sk-x"},  # mixed forms
            {},  # neither form
            {"model_name": "   "},  # blank reference
        ],
    )
    async def test_probe_body_form_validation_is_422(
        self, tmp_path: Path, payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[Any] = []

        async def fake_acompletion(**kwargs: Any) -> Any:
            calls.append(kwargs)
            return {"id": "ok"}

        monkeypatch.setattr("litellm.acompletion", fake_acompletion)
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [_model_entry("openai/gpt-test", model_name="primary")]
        async with _serve_client(_make_app(tmp_path, cfg=cfg)) as client:
            resp = await client.post("/api/settings/llm/test", json=payload)
        assert resp.status_code == 422
        assert calls == []  # rejected before any probe runs


class TestDeploymentIdentityAndRevision:
    """V2 deployment routes use stable ids and optimistic concurrency."""

    async def test_duplicate_routing_group_crud_is_per_deployment(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            revision = (await client.get("/api/settings/llm")).json()["revision"]
            created = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [
                        {"model_name": "default", "model": "openai/first"},
                        {"model_name": "default", "model": "openai/second"},
                    ],
                },
            )
            assert created.status_code == 200
            payload = created.json()
            first, second = payload["deployments"]
            assert first["deployment_id"] != second["deployment_id"]

            updated = await client.patch(
                f"/api/settings/llm/deployments/{first['deployment_id']}",
                json={
                    "model": "openai/first-updated",
                    "expected_revision": payload["revision"],
                },
            )
            assert updated.status_code == 200
            updated_payload = updated.json()
            assert [item["model"] for item in updated_payload["deployments"]] == [
                "openai/first-updated",
                "openai/second",
            ]

            deleted = await client.delete(
                f"/api/settings/llm/deployments/{second['deployment_id']}",
                params={"expected_revision": updated_payload["revision"]},
            )
        assert deleted.status_code == 200
        assert [item["model"] for item in deleted.json()["deployments"]] == [
            "openai/first-updated"
        ]

    async def test_first_v2_mutation_backfills_all_legacy_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "legacy.json"
        config_path.write_text(
            json.dumps(
                {
                    "llm": {
                        "model_list": [
                            {
                                "model_name": "default",
                                "litellm_params": {"model": "openai/first"},
                            },
                            {
                                "model_name": "default",
                                "litellm_params": {"model": "openai/second"},
                            },
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("MARKITAI_CONFIG", str(config_path))
        app = create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            configure_logging=False,
        )
        async with _serve_client(app) as client:
            before = (await client.get("/api/settings/llm")).json()
            assert all(
                item["deployment_id"].startswith("legacy-")
                for item in before["deployments"]
            )
            response = await client.patch(
                f"/api/settings/llm/deployments/{before['deployments'][0]['deployment_id']}",
                json={"weight": 2, "expected_revision": before["revision"]},
            )
        assert response.status_code == 200
        after = response.json()
        assert all(
            not item["deployment_id"].startswith("legacy-")
            for item in after["deployments"]
        )
        stored = json.loads(config_path.read_text(encoding="utf-8"))["llm"][
            "model_list"
        ]
        assert all(entry["model_info"]["id"] for entry in stored)

    async def test_legacy_name_route_rejects_ambiguous_group(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            for model in ("openai/first", "openai/second"):
                assert (
                    await client.post(
                        "/api/settings/llm/models",
                        json={"model_name": "default", "model": model},
                    )
                ).status_code == 200
            update = await client.put(
                "/api/settings/llm/models/default", json={"weight": 2}
            )
            delete = await client.delete("/api/settings/llm/models/default")
        assert update.status_code == 409
        assert update.json()["detail"]["code"] == "ambiguous_legacy_model_name"
        assert delete.status_code == 409

    async def test_stale_revision_rejects_without_writing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            revision = (await client.get("/api/settings/llm")).json()["revision"]
            first = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [{"model_name": "default", "model": "openai/first"}],
                },
            )
            before = config_path.read_text(encoding="utf-8")
            stale = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": revision,
                    "deployments": [{"model_name": "default", "model": "openai/stale"}],
                },
            )
        assert first.status_code == 200
        assert stale.status_code == 409
        assert stale.json()["detail"]["code"] == "stale_revision"
        assert config_path.read_text(encoding="utf-8") == before


class TestSettingsOriginAndSecurity:
    async def test_markitai_config_is_the_write_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "from-env.json"
        config_path.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("MARKITAI_CONFIG", str(config_path))
        app = create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            configure_logging=False,
        )
        async with _serve_client(app) as client:
            settings = (await client.get("/api/settings/llm")).json()
            assert settings["config_origin"] == "environment"
            response = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": settings["revision"],
                    "deployments": [{"model_name": "default", "model": "openai/test"}],
                },
            )
        assert response.status_code == 200
        assert (
            json.loads(config_path.read_text(encoding="utf-8"))["llm"]["model_list"][0][
                "litellm_params"
            ]["model"]
            == "openai/test"
        )

    async def test_api_base_and_key_are_secret_free(self, tmp_path: Path) -> None:
        cfg = MarkitaiConfig()
        cfg.llm.model_list = [
            _model_entry(
                "openai/test",
                api_key="sk-super-secret-value",
                api_base="https://user:pass@proxy.example/v1?token=secret#frag",
            )
        ]
        async with _serve_client(_make_app(tmp_path, cfg)) as client:
            response = await client.get("/api/settings/llm")
        deployment = response.json()["deployments"][0]
        assert deployment["api_key_configured"] is True
        assert deployment["api_base"] == "https://proxy.example"
        assert "super-secret" not in response.text
        assert "user:pass" not in response.text
        assert "token=secret" not in response.text
        assert response.headers["cache-control"] == "no-store"

    async def test_settings_are_loopback_only(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 1234))
            async with httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1"
            ) as client:
                response = await client.get("/api/settings/llm")
        assert response.status_code == 403
        assert response.headers["cache-control"] == "no-store"


class TestDetectedRuntimeSeparation:
    async def test_persisting_one_model_keeps_other_detected_models_effective(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "detected.json"
        monkeypatch.setenv("MARKITAI_CONFIG", str(config_path))
        monkeypatch.delenv("MODEL", raising=False)

        async def fake_detect() -> list[dict[str, Any]]:
            return [
                {
                    "provider": "openai",
                    "model": "openai/detected",
                    "label": "OpenAI",
                    "requires_api_key": False,
                }
            ]

        monkeypatch.setattr(
            "markitai.serve.app._detect_provider_candidates", fake_detect
        )
        app = create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            configure_logging=False,
        )
        async with _serve_client(app) as client:
            before = (await client.get("/api/settings/llm")).json()
            added = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": before["revision"],
                    "deployments": [{"model_name": "default", "model": "ollama/local"}],
                },
            )
            capabilities = (await client.get("/api/capabilities")).json()
        assert added.status_code == 200
        payload = added.json()
        assert [item["model"] for item in payload["deployments"]] == ["ollama/local"]
        assert [item["model"] for item in payload["detected"]] == ["openai/detected"]
        assert capabilities["llm"]["models"] == [
            "ollama/local",
            "openai/detected",
        ]

    async def test_model_discovery_does_not_echo_draft_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_discover(
            provider: str,
            *,
            api_key: str | None,
            api_base: str | None,
            refresh: bool,
        ) -> dict[str, Any]:
            assert provider == "openai"
            assert api_key == "sk-draft-secret"
            assert api_base is None
            assert refresh is True
            return {
                "provider": provider,
                "status": "ok",
                "source": "live_api",
                "authoritative": True,
                "cached": False,
                "stale": False,
                "models": [],
            }

        monkeypatch.setattr(
            "markitai.providers.discovery.discover_models", fake_discover
        )
        async with _serve_client(_make_app(tmp_path)) as client:
            response = await client.post(
                "/api/settings/llm/model-discovery",
                json={
                    "provider": "openai",
                    "api_key": "sk-draft-secret",
                    "refresh": True,
                },
            )
        assert response.status_code == 200
        assert "sk-draft-secret" not in response.text
        assert response.headers["cache-control"] == "no-store"


class TestConfiguredConnectionReuse:
    async def test_batch_can_reuse_stored_credentials_without_returning_them(
        self, tmp_path: Path
    ) -> None:
        config_path = tmp_path / "config.json"
        async with _serve_client(
            _make_app(tmp_path, config_path=config_path)
        ) as client:
            seeded = await client.post(
                "/api/settings/llm/models",
                json={
                    "model_name": "default",
                    "model": "openai/first",
                    "api_key": "sk-stored-secret",
                    "api_base": "https://proxy.example/v1",
                },
            )
            payload = seeded.json()
            source_id = payload["deployments"][0]["deployment_id"]
            added = await client.post(
                "/api/settings/llm/deployments/batch",
                json={
                    "expected_revision": payload["revision"],
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "openai/second",
                            "credential_deployment_id": source_id,
                        }
                    ],
                },
            )
        assert added.status_code == 200
        assert "sk-stored-secret" not in added.text
        stored = json.loads(config_path.read_text(encoding="utf-8"))["llm"][
            "model_list"
        ]
        assert stored[1]["litellm_params"] == {
            "model": "openai/second",
            "api_key": "sk-stored-secret",
            "api_base": "https://proxy.example/v1",
        }
