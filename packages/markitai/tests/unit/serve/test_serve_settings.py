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
            transport=transport, base_url="http://test"
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
        assert data["models"] == [
            {
                "model_name": "default",
                "model": "gemini/gemini-2.5-flash",
                "api_key_masked": "env:GEMINI_API_KEY",
                "api_base": None,
            },
            {
                "model_name": "default",
                "model": "openai/gpt-4o-mini",
                "api_key_masked": "sk…cdef",
                "api_base": "https://proxy.local/v1",
            },
            {
                "model_name": "default",
                "model": "claude-agent/sonnet",
                "api_key_masked": None,
                "api_base": None,
            },
        ]
        assert "sk-1234567890abcdef" not in resp.text  # never the plaintext key

    async def test_source_none_for_empty_config(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.get("/api/settings/llm")
        data = resp.json()
        assert data == {
            "configured": False,
            "source": "none",
            "config_path": data["config_path"],
            "models": [],
        }
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
        assert data["configured"] is True
        assert data["source"] == "detected"
        assert [m["model"] for m in data["models"]] == ["openai/gpt-detected"]

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
        assert data["models"][0]["api_key_masked"] is None


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
        assert data["models"] == [
            {
                "model_name": "primary",
                "model": "openai/gpt-test",
                "api_key_masked": "sk…cdef",
                "api_base": None,
            }
        ]
        assert "sk-1234567890abcdef" not in resp.text

        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk == {
            "llm": {
                "model_list": [
                    {
                        "model_name": "primary",
                        "litellm_params": {
                            "model": "openai/gpt-test",
                            "api_key": "sk-1234567890abcdef",
                        },
                    }
                ]
            }
        }
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
        assert on_disk["llm"]["model_list"] == [
            {
                "model_name": "default",
                "litellm_params": {"model": "old/model", "weight": 2},
                "custom_entry_key": "keep-me",  # unknown entry keys survive
            },
            {
                "model_name": "extra",
                "litellm_params": {
                    "model": "openai/gpt-new",
                    "api_base": "env:MY_BASE",
                },
            },
        ]
        if os.name == "posix":  # rewrite tightens permissions to owner-only
            assert (config_path.stat().st_mode & 0o777) == 0o600

    async def test_add_duplicate_model_name_is_409(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        app = _make_app(tmp_path, config_path=config_path)
        async with _serve_client(app) as client:
            first = await client.post(
                "/api/settings/llm/models",
                json={"model_name": "primary", "model": "openai/first"},
            )
            assert first.status_code == 200
            before = config_path.read_text(encoding="utf-8")
            dup = await client.post(
                "/api/settings/llm/models",
                json={"model_name": "primary", "model": "openai/second"},
            )
            settings = (await client.get("/api/settings/llm")).json()
        assert dup.status_code == 409
        assert config_path.read_text(encoding="utf-8") == before  # nothing written
        assert [m["model"] for m in settings["models"]] == ["openai/first"]

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
        entry = resp.json()["models"][0]
        assert entry["api_key_masked"] is None
        assert entry["api_base"] is None
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk["llm"]["model_list"] == [
            {
                "model_name": "local",
                "litellm_params": {"model": "claude-agent/haiku"},
            }
        ]
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
            assert before["llm"] == {"configured": False, "models": []}
            resp = await client.post(
                "/api/settings/llm/models",
                json={"model_name": "primary", "model": "openai/gpt-test"},
            )
            assert resp.status_code == 200
            after = (await client.get("/api/capabilities")).json()
        assert after["llm"] == {"configured": True, "models": ["openai/gpt-test"]}


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
        assert settings["models"] == [
            {
                "model_name": "primary",
                "model": "openai/gpt-next",
                "api_key_masked": "sk…cdef",  # mask unchanged -> key kept
                "api_base": "https://proxy.local/v1",
            }
        ]
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
        entry = resp.json()["models"][0]
        assert entry["api_key_masked"] is None
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
        assert resp.json()["models"][0]["api_key_masked"] == "env:NEW_KEY_VAR"
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
            masked = (await client.get("/api/settings/llm")).json()["models"][0][
                "api_key_masked"
            ]
            resp = await client.put(
                "/api/settings/llm/models/primary", json={"api_key": masked}
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
        entry = resp.json()["models"][0]
        assert entry["model"] == "claude-agent/sonnet"
        assert entry["api_key_masked"] is None
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
        assert [m["model_name"] for m in data["models"]] == ["b"]
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
        assert data["models"] == []
        assert capabilities["llm"] == {"configured": False, "models": []}
        on_disk = json.loads(config_path.read_text(encoding="utf-8"))
        assert on_disk["llm"]["model_list"] == []

    async def test_delete_unknown_model_name_is_404(self, tmp_path: Path) -> None:
        async with _serve_client(_make_app(tmp_path)) as client:
            resp = await client.delete("/api/settings/llm/models/nope")
        assert resp.status_code == 404


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
        assert captured["max_tokens"] == 1
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
        assert "claude-agent SDK is not installed" in data["detail"]

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
