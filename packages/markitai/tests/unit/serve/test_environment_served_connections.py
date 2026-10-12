"""A connection with no key of its own is served by the server's key variable.

Owner report (2026-10-12): a hand-written ``deepseek/deepseek-flash`` entry
with ``DEEPSEEK_API_KEY`` set where ``markitai serve`` runs passed its model
test, yet Edit showed "API key: Not set" and the model list failed with
"model discovery failed (HTTPStatusError)": discovery sent no key while
tests and conversions carry the variable's. Both now resolve the key the
way LiteLLM does for the conversions, and a failed list says why with the
model test's reasons (contract v1.10).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig
from markitai.providers import discovery
from markitai.serve import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

_KEY = "sk-deepseek-from-the-environment-0042"


def _app(tmp_path: Path, *entries: ModelConfig) -> FastAPI:
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(tmp_path / "cache")
    cfg.llm.model_list = list(entries)
    return create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=cfg,
        configure_logging=False,
        config_path=tmp_path / "config.json",
    )


def _entry(
    model: str, api_key: str | None = None, api_base: str | None = None
) -> ModelConfig:
    return ModelConfig(
        model_name="default",
        litellm_params=LiteLLMParams(model=model, api_key=api_key, api_base=api_base),
    )


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            yield client


async def _legacy_card(client: httpx.AsyncClient, provider: str) -> dict[str, Any]:
    cards = (await client.get("/api/settings/llm/providers")).json()["providers"]
    return next(
        card
        for card in cards
        if card["kind"] == "configured" and card["provider"] == provider
    )


@pytest.fixture(autouse=True)
def _fresh_discovery_cache() -> None:
    discovery._cache.clear()
    discovery._locks.clear()


class _ModelsServer:
    """httpx.AsyncClient stand-in: records each request, answers *status*.

    It replaces httpx's own class, so it is installed once the test's client
    exists.
    """

    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status = status
        self.body = body if body is not None else {"data": [{"id": "deepseek-flash"}]}
        self.requests: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = self

        class Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get(self, url: str, **kwargs: Any) -> httpx.Response:
                server.requests.append({"url": url, **kwargs})
                request = httpx.Request("GET", url)
                return httpx.Response(server.status, json=server.body, request=request)

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)


class TestEditShowsTheEnvironmentKey:
    async def test_a_hand_written_entry_shows_its_key_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
        async with _client(_app(tmp_path, _entry("deepseek/deepseek-flash"))) as client:
            card = await _legacy_card(client, "deepseek")
            shown = (
                await client.get(
                    f"/api/settings/llm/providers/{card['provider_id']}/credentials"
                )
            ).json()
        # What the workbench renders as "From environment variable ..."
        assert shown["api_key"] == "env:DEEPSEEK_API_KEY"
        assert shown["api_key_masked"] == "env:DEEPSEEK_API_KEY"
        assert shown["api_key_source"] == "environment"
        assert _KEY not in str(shown)

    async def test_without_the_variable_the_key_is_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        async with _client(_app(tmp_path, _entry("deepseek/deepseek-flash"))) as client:
            card = await _legacy_card(client, "deepseek")
            shown = (
                await client.get(
                    f"/api/settings/llm/providers/{card['provider_id']}/credentials"
                )
            ).json()
        assert shown["api_key"] is None
        assert "api_key_source" not in shown

    async def test_a_stored_key_and_the_keyless_placeholder_stay_as_they_are(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
        app = _app(
            tmp_path,
            _entry("deepseek/deepseek-flash", api_key="sk-stored-in-config-9876"),
            _entry(
                "deepseek/deepseek-chat",
                api_key="markitai-no-key",
                api_base="http://127.0.0.1:9/v1",
            ),
        )
        async with _client(app) as client:
            cards = (await client.get("/api/settings/llm/providers")).json()
            shown = [
                (
                    await client.get(
                        f"/api/settings/llm/providers/{card['provider_id']}/credentials"
                    )
                ).json()
                for card in cards["providers"]
                if card["kind"] == "configured"
            ]
        assert [view["api_key"] for view in shown] == ["sk…9876", None]
        assert all("api_key_source" not in view for view in shown)


class TestDiscoveryCarriesTheEnvironmentKey:
    async def test_a_hand_written_entry_lists_models_with_the_variables_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
        server = _ModelsServer()
        async with _client(_app(tmp_path, _entry("deepseek/deepseek-flash"))) as client:
            card = await _legacy_card(client, "deepseek")
            server.install(monkeypatch)
            # What the workbench posts for a hand-written entry
            listed = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={
                        "provider": "deepseek",
                        "deployment_id": card["deployment_id"],
                        "refresh": True,
                    },
                )
            ).json()
        assert listed["status"] == "ok"
        assert [m["model"] for m in listed["models"]] == ["deepseek/deepseek-flash"]
        assert server.requests[0]["url"] == "https://api.deepseek.com/models"
        assert server.requests[0]["headers"]["Authorization"] == f"Bearer {_KEY}"
        assert _KEY not in str(listed)

    async def test_an_entry_with_its_own_address_lists_with_the_variables_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A hand-written entry naming an address: LiteLLM sends the server's
        # OPENAI_API_KEY there for its conversions, so its list does too
        monkeypatch.setenv("OPENAI_API_KEY", _KEY)
        server = _ModelsServer(body={"data": [{"id": "fake-model"}]})
        entry = _entry("openai/fake-hand", api_base="http://127.0.0.1:9/v1")
        async with _client(_app(tmp_path, entry)) as client:
            card = await _legacy_card(client, "openai")
            shown = (
                await client.get(
                    f"/api/settings/llm/providers/{card['provider_id']}/credentials"
                )
            ).json()
            server.install(monkeypatch)
            listed = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={
                        "provider": "openai",
                        "deployment_id": card["deployment_id"],
                        "api_base": shown["api_base"],
                        "refresh": True,
                    },
                )
            ).json()
        assert shown["api_key"] == "env:OPENAI_API_KEY"
        assert listed["status"] == "ok", listed
        assert [r["headers"].get("Authorization") for r in server.requests] == [
            f"Bearer {_KEY}"
        ]

    async def test_an_address_the_caller_names_never_gets_the_variables_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
        server = _ModelsServer(status=401, body={"error": {"message": "no key"}})
        async with _client(_app(tmp_path)) as client:
            server.install(monkeypatch)
            listed = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={"provider": "deepseek", "api_base": "https://elsewhere.test"},
                )
            ).json()
        assert all(
            "Authorization" not in request.get("headers", {})
            for request in server.requests
        )
        assert listed["status"] == "unavailable"
        assert _KEY not in str(listed)

    async def test_no_key_anywhere_is_no_key_and_nothing_is_sent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        server = _ModelsServer()
        async with _client(_app(tmp_path, _entry("deepseek/deepseek-flash"))) as client:
            card = await _legacy_card(client, "deepseek")
            server.install(monkeypatch)
            listed = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={
                        "provider": "deepseek",
                        "deployment_id": card["deployment_id"],
                    },
                )
            ).json()
        assert server.requests == []
        assert listed["status"] == "unavailable"
        assert listed["reason"] == "no_key"
        assert listed["key_variable"] == "DEEPSEEK_API_KEY"
        assert "DEEPSEEK_API_KEY is not set on the server" in listed["detail"]
        assert "HTTPStatusError" not in listed["detail"]


class TestDiscoveryFailureReasons:
    """A failed list answers the model test's reason, not an exception name."""

    @pytest.mark.parametrize(
        ("status", "body", "reason", "words"),
        [
            (
                401,
                {"error": {"message": "Authentication Fails (governor)"}},
                "key_rejected",
                "HTTP 401: Authentication Fails (governor)",
            ),
            (429, {"error": {"message": "slow down"}}, "rate_limited", "HTTP 429"),
            (503, {"message": "overloaded"}, "provider_error", "HTTP 503: overloaded"),
            (302, {}, "network", "HTTP 302"),
        ],
    )
    async def test_a_refusal_by_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        body: Any,
        reason: str,
        words: str,
    ) -> None:
        _ModelsServer(status=status, body=body).install(monkeypatch)
        listed = await discovery.discover_models("deepseek", api_key="sk-wrong-123456")
        assert listed["status"] == "unavailable"
        assert listed["reason"] == reason
        assert words in listed["detail"]
        assert "sk-wrong-123456" not in str(listed)

    async def test_a_keyless_refusal_is_a_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ModelsServer(status=401, body={}).install(monkeypatch)
        listed = await discovery.discover_models(
            "deepseek", key_variable="DEEPSEEK_API_KEY"
        )
        assert listed["reason"] == "no_key"
        assert listed["key_variable"] == "DEEPSEEK_API_KEY"

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (httpx.ConnectTimeout("slow"), "timeout"),
            (httpx.ConnectError("refused"), "network"),
            (ValueError("not json"), "provider_error"),
        ],
    )
    async def test_no_answer(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str
    ) -> None:
        async def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise error

        monkeypatch.setattr(discovery, "_http_models", fail)
        listed = await discovery.discover_models("deepseek", api_key="sk-x-123456")
        assert listed["reason"] == reason
        assert "Error" not in listed["detail"]


class TestDiscoveryDetailsAsContractV111:
    """The failure sentences both backends send (contract v1.11 section 8)."""

    @pytest.mark.parametrize(
        ("status", "body", "reason", "detail"),
        [
            (
                401,
                {"error": {"message": "Incorrect API key provided: sk-wrong-123456"}},
                "key_rejected",
                "Model discovery returned HTTP 401: Incorrect API key provided: …",
            ),
            (
                403,
                {"message": "first", "error": "second"},
                "key_rejected",
                "Model discovery returned HTTP 403: first",
            ),
            (
                429,
                {"error": "slow down"},
                "rate_limited",
                "Model discovery returned HTTP 429: slow down",
            ),
            (
                404,
                {"detail": "no such path"},
                "provider_error",
                "Model discovery returned HTTP 404: no such path",
            ),
            (500, {"other": 1}, "provider_error", "Model discovery returned HTTP 500"),
            (
                301,
                {},
                "network",
                "Model discovery request failed: the address redirected (HTTP 301), "
                "which discovery does not follow",
            ),
        ],
        ids=["401", "403", "429", "404", "500", "301"],
    )
    async def test_a_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        body: Any,
        reason: str,
        detail: str,
    ) -> None:
        _ModelsServer(status=status, body=body).install(monkeypatch)
        listed = await discovery.discover_models("deepseek", api_key="sk-wrong-123456")
        assert listed == {
            "provider": "deepseek",
            "status": "unavailable",
            "source": "live_api",
            "authoritative": False,
            "cached": False,
            "stale": False,
            "models": [],
            "detail": detail,
            "reason": reason,
        }

    async def test_a_text_answer_is_read_without_markup_and_cut_at_300(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = "<html><body><h1>Bad\ngateway</h1><p>" + "x" * 400 + "</p></body></html>"

        class Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get(self, url: str, **_kwargs: Any) -> httpx.Response:
                return httpx.Response(502, text=page, request=httpx.Request("GET", url))

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)
        listed = await discovery.discover_models("deepseek", api_key="sk-x-123456")
        words = listed["detail"].removeprefix("Model discovery returned HTTP 502: ")
        assert words.startswith("Bad gateway x")
        assert len(words) == 301 and words.endswith("…")
        assert "<" not in words

    @pytest.mark.parametrize(
        ("body", "detail"),
        [
            ({"object": "list"}, "Model discovery response has no model list"),
            ({"data": "none"}, "Model discovery response has no model list"),
        ],
        ids=["no-list", "not-a-list"],
    )
    async def test_an_answer_without_a_list(
        self, monkeypatch: pytest.MonkeyPatch, body: Any, detail: str
    ) -> None:
        _ModelsServer(body=body).install(monkeypatch)
        listed = await discovery.discover_models("deepseek", api_key="sk-x-123456")
        assert (listed["status"], listed["reason"], listed["detail"]) == (
            "unavailable",
            "provider_error",
            detail,
        )

    @pytest.mark.parametrize(
        ("content", "detail"),
        [
            (b"not json", "Model discovery response is not JSON"),
            (b" " * (2 * 1024 * 1024 + 1), "Model discovery response exceeds 2 MiB"),
        ],
        ids=["not-json", "oversized"],
    )
    async def test_an_unreadable_answer(
        self, monkeypatch: pytest.MonkeyPatch, content: bytes, detail: str
    ) -> None:
        class Client:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> Client:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get(self, url: str, **_kwargs: Any) -> httpx.Response:
                return httpx.Response(
                    200, content=content, request=httpx.Request("GET", url)
                )

        monkeypatch.setattr(discovery.httpx, "AsyncClient", Client)
        listed = await discovery.discover_models("deepseek", api_key="sk-x-123456")
        assert (listed["reason"], listed["detail"]) == ("provider_error", detail)

    @pytest.mark.parametrize(
        ("error", "reason", "detail"),
        [
            (httpx.ReadTimeout("slow"), "timeout", "Model discovery timed out"),
            (TimeoutError(), "timeout", "Model discovery timed out"),
            (
                httpx.ConnectError("boom", request=httpx.Request("GET", "http://x")),
                "network",
                "Model discovery request failed: boom",
            ),
            (
                RuntimeError("internal"),
                "provider_error",
                "Model discovery could not complete",
            ),
        ],
        ids=["read-timeout", "overall-timeout", "connect", "internal"],
    )
    async def test_no_answer_in_rusts_words(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
        reason: str,
        detail: str,
    ) -> None:
        async def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise error

        monkeypatch.setattr(discovery, "_http_models", fail)
        listed = await discovery.discover_models("deepseek", api_key="sk-x-123456")
        assert (listed["reason"], listed["detail"]) == (reason, detail)

    async def test_a_refused_connection_names_its_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            try:
                raise ConnectionRefusedError(111, "Connection refused")
            except ConnectionRefusedError as cause:
                raise httpx.ConnectError("All connection attempts failed") from cause

        monkeypatch.setattr(discovery, "_http_models", fail)
        listed = await discovery.discover_models("deepseek", api_key="sk-x-123456")
        assert listed["detail"] == "Model discovery request failed: connection refused"

    async def test_a_keyless_refusal_keeps_the_http_sentence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ModelsServer(status=401, body={"error": {"message": "no key"}}).install(
            monkeypatch
        )
        listed = await discovery.discover_models(
            "deepseek", key_variable="DEEPSEEK_API_KEY"
        )
        assert listed["reason"] == "no_key"
        assert listed["key_variable"] == "DEEPSEEK_API_KEY"
        assert listed["detail"] == "Model discovery returned HTTP 401: no key"

    async def test_an_address_missing_is_a_provider_error(self) -> None:
        listed = await discovery.discover_models("azure", api_key="k")
        assert (listed["reason"], listed["detail"]) == (
            "provider_error",
            "An API endpoint is required",
        )

    async def test_a_stale_list_stays_partial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ModelsServer().install(monkeypatch)
        await discovery.discover_models("deepseek", api_key="sk-x-123456")
        _ModelsServer(status=429, body={"error": "slow"}).install(monkeypatch)
        listed = await discovery.discover_models(
            "deepseek", api_key="sk-x-123456", refresh=True
        )
        assert (listed["status"], listed["stale"], listed["reason"]) == (
            "partial",
            True,
            "rate_limited",
        )
        assert listed["models"]


class TestNoKeyBeforeAnyRequest:
    async def test_an_unset_env_reference_names_its_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_DEEPSEEK_KEY", raising=False)
        server = _ModelsServer()
        entry = _entry("deepseek/deepseek-flash", api_key="env:MY_DEEPSEEK_KEY")
        async with _client(_app(tmp_path, entry)) as client:
            card = await _legacy_card(client, "deepseek")
            server.install(monkeypatch)
            listed = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={
                        "provider": "deepseek",
                        "deployment_id": card["deployment_id"],
                    },
                )
            ).json()
        assert server.requests == []
        assert listed["reason"] == "no_key"
        assert listed["key_variable"] == "MY_DEEPSEEK_KEY"
        assert (
            listed["detail"]
            == "MY_DEEPSEEK_KEY is not set on the server, so nothing was sent"
        )

    async def test_no_key_anywhere_is_said_in_the_contracts_words(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        server = _ModelsServer()
        async with _client(_app(tmp_path, _entry("deepseek/deepseek-flash"))) as client:
            card = await _legacy_card(client, "deepseek")
            server.install(monkeypatch)
            listed = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={"provider": "deepseek", "provider_id": card["provider_id"]},
                )
            ).json()
        assert server.requests == []
        assert listed["detail"] == (
            "DEEPSEEK_API_KEY is not set on the server, so nothing was sent"
        )

    async def test_a_keyless_entry_and_a_local_server_are_exempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        server = _ModelsServer(body={"models": []})
        entry = _entry(
            "deepseek/deepseek-flash",
            api_key="markitai-no-key",
            api_base="http://127.0.0.1:9/v1",
        )
        async with _client(_app(tmp_path, entry)) as client:
            card = await _legacy_card(client, "deepseek")
            server.install(monkeypatch)
            keyless = (
                await client.post(
                    "/api/settings/llm/model-discovery",
                    json={
                        "provider": "deepseek",
                        "deployment_id": card["deployment_id"],
                    },
                )
            ).json()
            local = (
                await client.post(
                    "/api/settings/llm/model-discovery", json={"provider": "ollama"}
                )
            ).json()
        assert keyless.get("reason") != "no_key"
        assert local.get("reason") != "no_key"
        assert len(server.requests) == 2


class TestOneProviderOneSavedRow:
    """Cards carry ``credential``; the environment card folds into a
    connection that already uses its variable (contract v1.11 section 8)."""

    async def _cards(
        self, tmp_path: Path, *entries: ModelConfig
    ) -> list[dict[str, Any]]:
        async with _client(_app(tmp_path, *entries)) as client:
            return (await client.get("/api/settings/llm/providers")).json()["providers"]

    async def test_every_provider_whose_key_variable_is_set_gets_a_card(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-env-1234")
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-1234")
        monkeypatch.setenv("MISTRAL_API_KEY", " ")
        cards = await self._cards(tmp_path)
        environment = {
            card["provider"]: card for card in cards if card["kind"] == "environment"
        }
        assert environment["groq"]["credential"] == "env:GROQ_API_KEY"
        assert environment["groq"]["source"] == "GROQ_API_KEY"
        assert environment["groq"]["id"] == "env:groq"
        assert environment["moonshot"]["credential"] == "env:MOONSHOT_API_KEY"
        assert "mistral" not in environment
        assert "gsk-env-1234" not in str(cards)

    async def test_a_hand_written_entry_names_the_variable_that_serves_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
        cards = await self._cards(tmp_path, _entry("deepseek/deepseek-flash"))
        deepseek = [card for card in cards if card["provider"] == "deepseek"]
        # One row: the environment card folds into the hand-written entry
        assert [card["kind"] for card in deepseek] == ["configured"]
        assert deepseek[0]["credential"] == "env:DEEPSEEK_API_KEY"

    async def test_an_entry_at_another_address_keeps_the_environment_card(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", _KEY)
        cards = await self._cards(
            tmp_path,
            _entry("deepseek/deepseek-flash", api_base="https://proxy.test/v1"),
        )
        deepseek = {
            card["kind"]: card for card in cards if card["provider"] == "deepseek"
        }
        assert set(deepseek) == {"configured", "environment"}
        assert deepseek["configured"]["credential"] == "env:DEEPSEEK_API_KEY"

    async def test_a_reference_is_named_as_written_and_a_stored_key_not_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", _KEY)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        cards = await self._cards(
            tmp_path,
            _entry("openai/gpt-x", api_key="env:OPENAI_API_KEY"),
            _entry("deepseek/deepseek-flash", api_key="sk-stored-in-config-9876"),
            _entry(
                "groq/llama",
                api_key="markitai-no-key",
                api_base="http://127.0.0.1:9/v1",
            ),
        )
        configured = {
            card["provider"]: card for card in cards if card["kind"] == "configured"
        }
        assert configured["openai"]["credential"] == "env:OPENAI_API_KEY"
        assert "credential" not in configured["deepseek"]
        assert "credential" not in configured["groq"]
        assert not any(
            card["kind"] == "environment" and card["provider"] == "openai"
            for card in cards
        )
