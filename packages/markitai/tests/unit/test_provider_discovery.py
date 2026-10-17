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
        content = b""  # the body's size is all discovery reads of it

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
        assert result["detail"] == "An API endpoint is required"
        assert result["reason"] == "provider_error"

    async def test_azure_requires_its_endpoint(self) -> None:
        result = await discovery.discover_models("azure")
        assert result["source"] == "live_api"
        assert result["models"] == []
        assert "endpoint" in result["detail"]

    async def test_unknown_provider_never_falls_back_to_static_catalog(self) -> None:
        result = await discovery.discover_models("unknown")
        assert result["status"] == "unavailable"
        # No list to load is no failure: model IDs are entered by hand
        assert result["source"] == "manual"
        assert "reason" not in result
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
            content = b""  # the body's size is all discovery reads of it

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
            content = b""  # the body's size is all discovery reads of it

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
            content = b""  # the body's size is all discovery reads of it

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    "data": [
                        {"id": "deepseek-flash"},
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
            "deepseek/deepseek-flash",
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


class TestProviderConnections:
    async def test_configured_claude_model_reuses_detected_cli_card(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.config import LiteLLMParams, ModelConfig
        from markitai.providers.auth import AuthManager

        monkeypatch.setattr(
            discovery.shutil,
            "which",
            lambda binary: "/usr/bin/claude" if binary == "claude" else None,
        )
        monkeypatch.setattr(
            discovery, "_local_runtime_available", lambda _provider: True
        )

        class Authenticated:
            authenticated = True

        class Unauthenticated:
            authenticated = False

        async def fake_auth(
            _self: AuthManager, provider: str, **_kwargs: Any
        ) -> Authenticated | Unauthenticated:
            return Authenticated() if provider == "claude-agent" else Unauthenticated()

        monkeypatch.setattr(AuthManager, "check_auth", fake_auth)
        configured = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="claude-agent/sonnet"),
            )
        ]
        cards = await discovery.detect_provider_connections(configured, refresh=True)
        claude = [card for card in cards if card["provider"] == "claude-agent"]
        assert len(claude) == 1
        assert claude[0]["kind"] == "local_cli"
        assert claude[0]["status"] == "ready"
        assert discovery.startup_model_candidates(claude)[0]["model"] == (
            "claude-agent/haiku"
        )

    async def test_authenticated_cli_without_sdk_is_not_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.providers.auth import AuthManager

        monkeypatch.setattr(
            discovery.shutil,
            "which",
            lambda binary: "/usr/bin/claude" if binary == "claude" else None,
        )
        monkeypatch.setattr(
            discovery, "_local_runtime_available", lambda _provider: False
        )

        class Status:
            authenticated = True

        async def fake_auth(
            _self: AuthManager, _provider: str, **_kwargs: Any
        ) -> Status:
            return Status()

        monkeypatch.setattr(AuthManager, "check_auth", fake_auth)
        cards = await discovery.detect_provider_connections(refresh=True)
        claude = next(card for card in cards if card["provider"] == "claude-agent")
        assert claude["status"] == "missing_dependency"
        assert claude["supports_discovery"] is False
        assert discovery.startup_model_candidates([claude]) == []


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


class TestConnectableProviderDiscovery:
    """The providers beyond markitai's first eight list models the
    OpenAI-compatible way, Cohere its own; the others take IDs by hand."""

    async def test_a_provider_lists_its_models_at_its_default_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {
                "data": [
                    {"id": "llama-3.3-70b-versatile"},
                    {"id": "whisper", "name": "W"},
                ]
            },
            captured,
        )
        result = await discovery._http_models("groq", "gsk-secret", None)
        assert captured["url"] == "https://api.groq.com/openai/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer gsk-secret"}
        assert result["status"] == "ok" and result["authoritative"] is True
        assert result["models"] == [
            {
                "model": "groq/llama-3.3-70b-versatile",
                "label": "llama-3.3-70b-versatile",
                "supports_vision": False,
            },
            {"model": "groq/whisper", "label": "W", "supports_vision": False},
        ]

    @pytest.mark.parametrize(
        ("provider", "api_base", "expected_url"),
        [
            (
                "groq",
                "https://proxy.example/openai/v1/",
                "https://proxy.example/openai/v1/models",
            ),
            # LiteLLM takes these hosts and adds /v1 itself
            ("oobabooga", None, "http://127.0.0.1:5000/v1/models"),
            (
                "heroku",
                "https://us.inference.heroku.com",
                "https://us.inference.heroku.com/v1/models",
            ),
            # The documented address, not LiteLLM's older default
            ("lm_studio", None, "http://localhost:1234/v1/models"),
            ("nebius", None, "https://api.tokenfactory.nebius.com/v1/models"),
            (
                "databricks",
                "https://adb-1.azuredatabricks.net/serving-endpoints",
                "https://adb-1.azuredatabricks.net/serving-endpoints/models",
            ),
        ],
    )
    async def test_the_list_sits_where_the_provider_serves_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        api_base: str | None,
        expected_url: str,
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(monkeypatch, {"data": [{"id": "m"}]}, captured)
        result = await discovery._http_models(provider, None, api_base)
        assert captured["url"] == expected_url
        assert captured["headers"] == {}
        assert [m["model"] for m in result["models"]] == [f"{provider}/m"]

    async def test_together_answers_with_a_bare_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            [
                {
                    "id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "display_name": "Llama",
                }
            ],  # type: ignore[arg-type]
            captured,
        )
        result = await discovery._http_models("together_ai", "key", None)
        assert captured["url"] == "https://api.together.ai/v1/models"
        assert result["models"] == [
            {
                "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "label": "Llama",
                "supports_vision": False,
            }
        ]

    async def test_cohere_lists_its_chat_models_from_its_own_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(
            monkeypatch,
            {
                "models": [{"name": "command-a-03-2025"}, {"name": "command-r7b"}],
                "next_page_token": "page-2",
            },
            captured,
        )
        result = await discovery._http_models("cohere", "co-secret", None)
        assert captured["url"] == "https://api.cohere.com/v1/models"
        assert captured["params"] == {"endpoint": "chat", "page_size": "1000"}
        assert captured["headers"] == {"Authorization": "Bearer co-secret"}
        assert [m["model"] for m in result["models"]] == [
            "cohere/command-a-03-2025",
            "cohere/command-r7b",
        ]
        # A further page exists: the list is shown, but not as complete
        assert (result["status"], result["authoritative"]) == ("partial", False)
        assert "first page" in result["detail"]

    @pytest.mark.parametrize(
        "provider", ["perplexity", "zai", "fireworks_ai", "bedrock"]
    )
    async def test_a_provider_without_a_model_list_sends_nothing(
        self, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(monkeypatch, {"data": []}, captured)
        result = await discovery.discover_models(provider, api_key="key")
        assert captured == {}
        assert result["status"] == "unavailable"
        assert result["source"] == "manual" and "reason" not in result
        assert "enter model IDs manually" in result["detail"]

    @pytest.mark.parametrize("provider", ["databricks", "azure_ai", "hosted_vllm"])
    async def test_a_server_of_ones_own_needs_its_address(
        self, monkeypatch: pytest.MonkeyPatch, provider: str
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(monkeypatch, {"data": []}, captured)
        result = await discovery.discover_models(provider, api_key="key")
        assert captured == {}
        assert (result["status"], result["detail"]) == (
            "unavailable",
            "An API endpoint is required",
        )

    async def test_a_failed_list_leaves_model_ids_by_hand_without_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Refusing:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Refusing:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get(self, url: str, **_kwargs: Any) -> Any:
                request = discovery.httpx.Request("GET", url)
                raise discovery.httpx.HTTPStatusError(
                    "401 for key gsk-never-shown",
                    request=request,
                    response=discovery.httpx.Response(401, request=request),
                )

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Refusing)
        result = await discovery.discover_models("groq", api_key="gsk-never-shown")
        assert result["status"] == "unavailable"
        assert result["models"] == []
        # The model test's reason (contract v1.10), not an exception's name
        assert result["reason"] == "key_rejected"
        assert result["detail"] == "Model discovery returned HTTP 401"
        assert "gsk-never-shown" not in str(result)

    @pytest.mark.parametrize(
        "api_base", [None, "https://api.anthropic.com", "https://api.anthropic.com/v1/"]
    )
    async def test_anthropic_lists_models_whether_or_not_its_address_has_v1(
        self, monkeypatch: pytest.MonkeyPatch, api_base: str | None
    ) -> None:
        captured: dict[str, Any] = {}
        install_http_response(monkeypatch, {"data": [{"id": "claude-x"}]}, captured)
        await discovery._http_models("anthropic", "sk-ant", api_base)
        assert captured["url"] == "https://api.anthropic.com/v1/models"


class TestConnectableProviderCards:
    async def test_every_connectable_provider_gets_a_built_in_card(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.providers.auth import AuthManager
        from markitai.providers.connectable import CONNECTABLE

        monkeypatch.setattr(discovery.shutil, "which", lambda _binary: None)
        monkeypatch.setattr(discovery, "_local_runtime_available", lambda _p: False)
        monkeypatch.setattr(
            discovery, "installed", lambda row: row.provider != "vertex_ai"
        )

        class Status:
            authenticated = False

        async def fake_auth(
            _self: AuthManager, _provider: str, **_kwargs: Any
        ) -> Status:
            return Status()

        monkeypatch.setattr(AuthManager, "check_auth", fake_auth)
        for row in CONNECTABLE:
            if row.key_variable:
                monkeypatch.delenv(row.key_variable, raising=False)
        cards = {
            card["provider"]: card
            for card in await discovery.detect_provider_connections(refresh=True)
            if card["kind"] == "common"
        }
        # ChatGPT has no SDK, so its card is always offered
        assert set(cards) == ({row.provider for row in CONNECTABLE} - {"vertex_ai"}) | {
            "chatgpt"
        }
        assert {card["id"] for card in cards.values()} == {
            f"common:{provider}" for provider in cards
        }
        status = {p: (c["status"], c["supports_discovery"]) for p, c in cards.items()}
        # A local server at its default address or a cloud's own credentials
        # may already work; keyed APIs need a key; some list no models
        assert status["lm_studio"] == ("unknown", True)
        assert status["ollama"] == ("unknown", True)
        assert status["bedrock"] == ("unknown", False)
        assert status["groq"] == ("needs_credentials", True)
        assert status["hosted_vllm"] == ("needs_credentials", True)
        assert status["perplexity"] == ("needs_credentials", False)
        assert cards["azure"]["label"] == "Azure OpenAI"

    async def test_an_installed_subscription_runtime_is_offered_before_sign_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.providers.auth import AuthManager

        monkeypatch.setattr(discovery.shutil, "which", lambda _binary: None)
        monkeypatch.setattr(
            discovery, "_local_runtime_available", lambda p: p == "claude-agent"
        )

        class Status:
            authenticated = False

        async def fake_auth(
            _self: AuthManager, _provider: str, **_kwargs: Any
        ) -> Status:
            return Status()

        monkeypatch.setattr(AuthManager, "check_auth", fake_auth)
        cards = await discovery.detect_provider_connections(refresh=True)
        subscriptions = {
            card["provider"]: card
            for card in cards
            if card["provider"] in {"claude-agent", "copilot", "chatgpt"}
        }
        # Copilot's runtime is not installed; ChatGPT has none to install,
        # so its card leads to "not signed in" and the login command
        assert set(subscriptions) == {"claude-agent", "chatgpt"}
        assert subscriptions["chatgpt"]["supports_discovery"] is True
        claude = subscriptions["claude-agent"]
        assert (claude["kind"], claude["status"], claude["key_optional"]) == (
            "common",
            "unknown",
            True,
        )
        # Never a startup default: nothing says it works yet
        assert discovery.startup_model_candidates([claude]) == []
