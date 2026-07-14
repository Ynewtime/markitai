from __future__ import annotations

import asyncio
from typing import Any

import pytest

from markitai.providers import discovery


@pytest.fixture(autouse=True)
def clear_discovery_caches() -> None:
    discovery._cache.clear()
    discovery._connection_cache.clear()
    discovery._locks.clear()


def install_http_response(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    captured: dict[str, Any],
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return body

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(
            self,
            url: str,
            *,
            headers: dict[str, str],
            params: dict[str, str],
        ) -> Response:
            captured.update(url=url, headers=headers, params=params)
            return Response()

    monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)


class TestDiscoveryRequirements:
    async def test_custom_endpoint_requires_explicit_api_base(self) -> None:
        result = await discovery.discover_models("custom")
        assert result["status"] == "unavailable"
        assert result["models"] == []
        assert "API base" in result["detail"]

    async def test_azure_requires_its_endpoint(self) -> None:
        result = await discovery.discover_models("azure")
        assert result["source"] == "live_api"
        assert result["models"] == []
        assert "endpoint" in result["detail"]

    async def test_unknown_provider_never_falls_back_to_static_catalog(self) -> None:
        result = await discovery.discover_models("unknown")
        assert result["status"] == "unavailable"
        assert result["source"] == "live_api"
        assert result["models"] == []

    @pytest.mark.parametrize(
        ("provider", "loader_name", "token_field"),
        [
            ("claude-agent", "_claude_oauth_credentials", "accessToken"),
            ("chatgpt", "_chatgpt_oauth_credentials", "access_token"),
        ],
    )
    async def test_oauth_providers_use_account_credentials_for_live_discovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        loader_name: str,
        token_field: str,
    ) -> None:
        from markitai.providers import auth

        monkeypatch.setattr(auth, loader_name, lambda: {token_field: "oauth-secret"})
        captured: dict[str, Any] = {}

        async def fake_http(
            next_provider: str,
            api_key: str | None,
            api_base: str | None,
            *,
            account_id: str | None = None,
        ) -> dict[str, Any]:
            captured.update(
                provider=next_provider,
                api_key=api_key,
                api_base=api_base,
                account_id=account_id,
            )
            return discovery._result(
                next_provider,
                status="ok",
                source="live_api",
                authoritative=True,
                models=[],
            )

        monkeypatch.setattr(discovery, "_http_models", fake_http)
        result = await discovery.discover_models(provider, refresh=True)
        assert result["source"] == "live_api"
        assert captured == {
            "provider": provider,
            "api_key": "oauth-secret",
            "api_base": None,
            "account_id": None,
        }
        assert "oauth-secret" not in str(result)

    async def test_expired_claude_token_is_refreshed_before_live_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.providers import auth

        monkeypatch.setattr(
            auth,
            "_claude_oauth_credentials",
            lambda: {
                "accessToken": "expired-secret",
                "refreshToken": "refresh-secret",
                "expiresAt": 0,
            },
        )
        stored: list[dict[str, Any]] = []
        monkeypatch.setattr(
            auth,
            "_store_claude_oauth_credentials",
            lambda credentials: stored.append(credentials) is None,
        )

        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    "access_token": "fresh-secret",
                    "refresh_token": "rotated-secret",
                    "expires_in": 3600,
                }

        class Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def post(self, url: str, *, json: dict[str, str]) -> Response:
                assert url == discovery._CLAUDE_OAUTH_TOKEN_URL
                assert json == {
                    "grant_type": "refresh_token",
                    "refresh_token": "refresh-secret",
                    "client_id": discovery._CLAUDE_OAUTH_CLIENT_ID,
                }
                return Response()

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)
        used_tokens: list[str | None] = []

        async def fake_http(
            provider: str, api_key: str | None, api_base: str | None
        ) -> dict[str, Any]:
            used_tokens.append(api_key)
            return discovery._result(
                provider,
                status="ok",
                source="live_api",
                authoritative=True,
                models=[discovery._candidate("claude-agent/claude-test")],
            )

        monkeypatch.setattr(discovery, "_http_models", fake_http)
        result = await discovery.discover_models("claude-agent", refresh=True)
        assert result["status"] == "ok"
        assert used_tokens == ["fresh-secret"]
        assert len(stored) == 1
        assert stored[0]["accessToken"] == "fresh-secret"
        assert stored[0]["refreshToken"] == "rotated-secret"
        assert "fresh-secret" not in str(result)


class TestLiveDiscovery:
    async def test_openai_compatible_base_keeps_its_v1_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {"data": [{"id": "gpt-test"}]}

        class Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get(
                self,
                url: str,
                *,
                headers: dict[str, str],
                params: dict[str, str],
            ) -> Response:
                captured.update(url=url, headers=headers, params=params)
                return Response()

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)
        result = await discovery._http_models(
            "openai", "sk-secret", "https://proxy.example/v1"
        )
        assert captured["url"] == "https://proxy.example/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-secret"}
        assert result["models"] == [
            {
                "model": "openai/gpt-test",
                "label": "gpt-test",
                "supports_vision": False,
            }
        ]
        assert "sk-secret" not in str(result)

    async def test_deepseek_uses_the_authoritative_live_model_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    "data": [
                        {"id": "deepseek-v4-flash"},
                        {"id": "deepseek-v4-pro"},
                    ]
                }

        class Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get(
                self,
                url: str,
                *,
                headers: dict[str, str],
                params: dict[str, str],
            ) -> Response:
                captured.update(url=url, headers=headers, params=params)
                return Response()

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)
        result = await discovery.discover_models(
            "deepseek", api_key="sk-secret", refresh=True
        )
        assert captured["url"] == "https://api.deepseek.com/models"
        assert [model["model"] for model in result["models"]] == [
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v4-pro",
        ]
        assert result["authoritative"] is True
        assert result["source"] == "live_api"

    async def test_openrouter_uses_its_live_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {
                "data": [
                    {
                        "id": "anthropic/claude-test",
                        "name": "Claude Test",
                        "architecture": {"input_modalities": ["text", "image"]},
                    }
                ]
            },
            captured,
        )
        result = await discovery._http_models("openrouter", "sk-secret", None)
        assert captured["url"] == "https://openrouter.ai/api/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-secret"}
        assert result["models"] == [
            {
                "model": "openrouter/anthropic/claude-test",
                "label": "Claude Test",
                "supports_vision": True,
            }
        ]
        assert result["authoritative"] is True

    @pytest.mark.parametrize(
        ("provider", "expected_headers", "expected_model"),
        [
            (
                "anthropic",
                {"x-api-key": "secret", "anthropic-version": "2023-06-01"},
                "anthropic/claude-test",
            ),
            (
                "claude-agent",
                {
                    "Authorization": "Bearer secret",
                    "anthropic-beta": "oauth-2025-04-20",
                    "anthropic-version": "2023-06-01",
                },
                "claude-agent/claude-test",
            ),
        ],
    )
    async def test_anthropic_connections_use_the_live_models_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        expected_headers: dict[str, str],
        expected_model: str,
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {"data": [{"id": "claude-test", "display_name": "Claude Test"}]},
            captured,
        )
        result = await discovery._http_models(provider, "secret", None)
        assert captured["url"] == "https://api.anthropic.com/v1/models"
        assert captured["headers"] == expected_headers
        assert result["models"] == [
            {
                "model": expected_model,
                "label": "Claude Test",
                "supports_vision": True,
            }
        ]

    async def test_chatgpt_uses_the_account_scoped_codex_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {
                "models": [
                    {
                        "slug": "gpt-visible",
                        "display_name": "GPT Visible",
                        "visibility": "list",
                    },
                    {
                        "slug": "gpt-hidden",
                        "display_name": "GPT Hidden",
                        "visibility": "hide",
                    },
                ]
            },
            captured,
        )
        result = await discovery._http_models(
            "chatgpt", "oauth-secret", None, account_id="account-1"
        )
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/models"
        assert captured["headers"] == {
            "Authorization": "Bearer oauth-secret",
            "ChatGPT-Account-Id": "account-1",
        }
        assert captured["params"] == {"client_version": "0.0.0"}
        assert result["models"] == [
            {
                "model": "chatgpt/gpt-visible",
                "label": "GPT Visible",
                "supports_vision": False,
            }
        ]
        assert "oauth-secret" not in str(result)

    async def test_gemini_filters_to_generate_content_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {
                "models": [
                    {
                        "name": "models/gemini-live",
                        "displayName": "Gemini Live",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/embedding-only",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            },
            captured,
        )
        result = await discovery._http_models("gemini", "secret", None)
        assert captured["url"] == (
            "https://generativelanguage.googleapis.com/v1beta/models"
        )
        assert captured["params"] == {"key": "secret", "pageSize": "1000"}
        assert [model["model"] for model in result["models"]] == ["gemini/gemini-live"]

    async def test_azure_fetches_live_regional_models_but_marks_them_partial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {"data": [{"id": "gpt-region-model"}]},
            captured,
        )
        result = await discovery._http_models(
            "azure", "secret", "https://resource.openai.azure.com"
        )
        assert captured["url"] == ("https://resource.openai.azure.com/openai/models")
        assert captured["headers"] == {"api-key": "secret"}
        assert captured["params"] == {"api-version": "2024-10-21"}
        assert result["models"][0]["model"] == "azure/gpt-region-model"
        assert result["status"] == "partial"
        assert result["authoritative"] is False

    @pytest.mark.parametrize(
        ("provider", "api_base", "body", "expected_url", "expected_model"),
        [
            (
                "ollama",
                None,
                {"models": [{"name": "qwen:latest"}]},
                "http://127.0.0.1:11434/api/tags",
                "ollama/qwen:latest",
            ),
            (
                "custom",
                "https://llm.example/v1",
                {"data": [{"id": "custom-live"}]},
                "https://llm.example/v1/models",
                "openai/custom-live",
            ),
        ],
    )
    async def test_local_and_compatible_endpoints_are_live(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        api_base: str | None,
        body: dict[str, Any],
        expected_url: str,
        expected_model: str,
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(monkeypatch, body, captured)
        result = await discovery._http_models(provider, "secret", api_base)
        assert captured["url"] == expected_url
        assert [model["model"] for model in result["models"]] == [expected_model]
        assert result["source"] == "live_api"


class TestDiscoveryCache:
    async def test_concurrent_requests_are_single_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        async def fake_http(
            provider: str, api_key: str | None, api_base: str | None
        ) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.02)
            return discovery._result(
                provider,
                status="ok",
                source="live_api",
                authoritative=True,
                models=[discovery._candidate("openai/test")],
            )

        monkeypatch.setattr(discovery, "_http_models", fake_http)
        first, second = await asyncio.gather(
            discovery.discover_models("openai", api_key="secret"),
            discovery.discover_models("openai", api_key="secret"),
        )
        assert calls == 1
        assert first["models"] == second["models"]

    async def test_failed_refresh_uses_recent_stale_value_without_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def success(
            provider: str, api_key: str | None, api_base: str | None
        ) -> dict[str, Any]:
            return discovery._result(
                provider,
                status="ok",
                source="live_api",
                authoritative=True,
                models=[discovery._candidate("openai/test")],
            )

        monkeypatch.setattr(discovery, "_http_models", success)
        await discovery.discover_models("openai", api_key="sk-never-return-this")

        async def fail(
            provider: str, api_key: str | None, api_base: str | None
        ) -> dict[str, Any]:
            raise RuntimeError(f"Authorization: Bearer {api_key}")

        monkeypatch.setattr(discovery, "_http_models", fail)
        result = await discovery.discover_models(
            "openai", api_key="sk-never-return-this", refresh=True
        )
        assert result["status"] == "partial"
        assert result["cached"] is True
        assert result["stale"] is True
        assert "sk-never-return-this" not in str(result)
        assert "Authorization" not in str(result)
