"""Every Claude Messages request goes out on Anthropic API version 2026-10-21.

On 2026-10-21 a request with ``temperature``, ``top_p`` or ``top_k`` is
refused on every model, so none may carry them, whatever set them. These
tests capture the request LiteLLM sends (an httpx mock transport, no
network) for each route markitai can configure, and the Message Batches
client and the model list markitai calls itself.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from markitai.config import LiteLLMParams, LLMConfig, ModelConfig
from markitai.constants import ANTHROPIC_API_VERSION
from markitai.llm import anthropic_api
from markitai.llm.anthropic_api import (
    install_litellm_api_version,
    sampling_notice,
    uses_messages_api,
    without_sampling,
)
from markitai.notices import capture_task_notices

_MESSAGE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "OK"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_HI = [{"role": "user", "content": "Reply with exactly OK."}]


class _Capture:
    """An async LiteLLM HTTP client that records requests and answers OK."""

    def __init__(self) -> None:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        self.requests: list[httpx.Request] = []
        self.client = AsyncHTTPHandler()
        self.client.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self._answer)
        )

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=_MESSAGE)

    @property
    def last(self) -> tuple[httpx.Request, dict[str, Any]]:
        request = self.requests[-1]
        return request, json.loads(request.content)


@pytest.fixture
def capture() -> _Capture:
    install_litellm_api_version()
    return _Capture()


@pytest.fixture(autouse=True)
def _fresh_notices() -> Iterator[None]:
    """The warning is said once per process: forget earlier tests'."""
    anthropic_api._reported.clear()
    yield
    anthropic_api._reported.clear()


async def _send(capture: _Capture, **kwargs: Any) -> tuple[httpx.Request, dict]:
    import litellm

    await litellm.acompletion(
        messages=_HI, max_tokens=16, client=capture.client, **kwargs
    )
    return capture.last


def _sampling(body: dict[str, Any]) -> dict[str, Any]:
    return {k: body[k] for k in ("temperature", "top_p", "top_k") if k in body}


class TestLiteLLMRoutes:
    async def test_claude_api_header(self, capture: _Capture) -> None:
        request, body = await _send(
            capture, model="anthropic/claude-sonnet-4-6", api_key="sk-ant-test"
        )

        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
        assert "anthropic_version" not in body
        assert _sampling(body) == {}

    async def test_sampling_passed_to_litellm_is_not_sent(
        self, capture: _Capture
    ) -> None:
        """A value LiteLLM is handed anyway is dropped at its last step."""
        with capture_task_notices() as notices:
            _, body = await _send(
                capture,
                model="anthropic/claude-sonnet-4-6",
                api_key="sk-ant-test",
                temperature=0.2,
                top_p=0.5,
                top_k=3,
            )

        assert _sampling(body) == {}
        assert body["max_tokens"] == 16
        assert notices == [
            sampling_notice("claude-sonnet-4-6", ["temperature", "top_p", "top_k"])
        ]

    async def test_gateway_address_gets_the_same_header(
        self, capture: _Capture
    ) -> None:
        request, _ = await _send(
            capture,
            model="anthropic/claude-sonnet-4-6",
            api_key="gateway-key",
            api_base="https://gateway.example.test",
        )

        assert request.url.host == "gateway.example.test"
        assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION

    async def test_microsoft_foundry(self, capture: _Capture) -> None:
        request, body = await _send(
            capture,
            model="azure_ai/claude-sonnet-4-6",
            api_key="foundry-key",
            api_base="https://example.services.ai.azure.com/anthropic",
            temperature=0.3,
        )

        assert request.url.path == "/anthropic/v1/messages"
        assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
        assert _sampling(body) == {}

    async def test_vertex_ai_body_version(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vertex reads the version from the body (it was vertex-2023-10-16)."""
        from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

        # LiteLLM's partner route imports the Vertex SDK and asks Google for
        # a token; neither is needed to shape the request
        vertexai = types.ModuleType("vertexai")
        vertexai.preview = types.SimpleNamespace(language_models=None)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vertexai", vertexai)
        monkeypatch.setattr(
            VertexBase,
            "_ensure_access_token",
            lambda *_args, **_kwargs: ("token", "project"),
        )

        request, body = await _send(
            capture,
            model="vertex_ai/claude-sonnet-4-6",
            vertex_project="project",
            vertex_location="global",
            temperature=0.4,
        )

        assert request.url.path.endswith("claude-sonnet-4-6:rawPredict")
        assert body["anthropic_version"] == ANTHROPIC_API_VERSION
        assert _sampling(body) == {}

    async def test_bedrock_invoke_keeps_its_version(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """InvokeModel is not versioned by 2026-10-21: left as LiteLLM sends it."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "example-secret")
        monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

        request, body = await _send(
            capture, model="bedrock/invoke/anthropic.claude-sonnet-4-6"
        )

        assert request.url.host == "bedrock-runtime.us-east-1.amazonaws.com"
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert "anthropic-version" not in request.headers

    def test_claude_platform_on_aws_header(self) -> None:
        from litellm.llms.bedrock.claude_platform.transformation import (
            BedrockClaudePlatformConfig,
        )

        install_litellm_api_version()
        headers = BedrockClaudePlatformConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-6",
            messages=_HI,  # type: ignore[arg-type]
            optional_params={},
            litellm_params={"workspace_id": "wrkspc_test"},
            api_key="aws-key",
        )

        assert headers["anthropic-version"] == ANTHROPIC_API_VERSION

    def test_install_is_idempotent(self) -> None:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        install_litellm_api_version()
        wrapped = AnthropicModelInfo.get_anthropic_headers
        install_litellm_api_version()

        assert AnthropicModelInfo.get_anthropic_headers is wrapped


class TestConfiguredSampling:
    """temperature/top_p/top_k in a model_list entry: never sent, said once."""

    def _router(self, *params: LiteLLMParams) -> Any:
        from markitai.llm import LLMProcessor

        models = [ModelConfig(model_name="default", litellm_params=p) for p in params]
        return LLMProcessor(LLMConfig(model_list=models))._create_router()

    async def test_claude_entry_warns_once_and_sends_none(
        self, capture: _Capture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm.llms.anthropic.chat.handler as anthropic_handler

        monkeypatch.setattr(
            anthropic_handler, "get_async_httpx_client", lambda **_: capture.client
        )
        params = LiteLLMParams.model_validate(
            {
                "model": "anthropic/claude-sonnet-4-6",
                "api_key": "sk-ant-test",
                "temperature": 0.2,
                "extra_body": {"top_k": 5},
            }
        )

        with capture_task_notices() as notices:
            router = self._router(params)
            self._router(params)  # a second router (the vision pool) says nothing
            await router.acompletion("default", _HI, max_tokens=16)

        request, body = capture.last
        assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
        assert _sampling(body) == {}
        assert "extra_body" not in body
        assert notices == [
            "temperature, top_k not sent to anthropic/claude-sonnet-4-6: "
            "Anthropic API 2026-10-21 rejects temperature, top_p and top_k"
        ]

    def test_other_providers_say_nothing(self) -> None:
        with capture_task_notices() as notices:
            self._router(
                LiteLLMParams.model_validate(
                    {"model": "openai/gpt-5-mini", "api_key": "k", "temperature": 0}
                ),
                LiteLLMParams.model_validate(
                    {
                        "model": "bedrock/anthropic.claude-sonnet-4-6",
                        "temperature": 0,
                    }
                ),
            )

        assert notices == []

    def test_the_settings_are_remembered_not_kept(self) -> None:
        params = LiteLLMParams.model_validate(
            {"model": "anthropic/claude-haiku-4-5", "top_p": 0.9, "top_k": 4}
        )

        assert params.sampling_settings == ("top_p", "top_k")
        assert "top_p" not in params.model_dump()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic/claude-sonnet-4-6", True),
        ("azure_ai/claude-opus-4-8", True),
        ("vertex_ai/claude-haiku-4-5@20251001", True),
        ("bedrock/claude_platform/claude-opus-5-5", True),
        ("bedrock/anthropic.claude-opus-4-7", False),  # Converse
        ("bedrock/invoke/anthropic.claude-opus-4-7", False),
        ("vertex_ai/gemini-3-pro", False),
        ("openrouter/anthropic/claude-sonnet-4-6", False),
        ("claude-agent/sonnet", False),
    ],
)
def test_uses_messages_api(model: str, expected: bool) -> None:
    assert uses_messages_api(model) is expected


class TestMessageBatches:
    def test_client_sends_the_version(self) -> None:
        from markitai.llm.batch_api import BatchCredentials, _anthropic_client

        client = _anthropic_client(BatchCredentials(api_key="sk-ant-test"))

        assert client.default_headers["anthropic-version"] == ANTHROPIC_API_VERSION
        assert client.default_headers["X-Api-Key"] == "sk-ant-test"

    def test_request_params_carry_no_sampling(self) -> None:
        with capture_task_notices() as notices:
            params = without_sampling(
                {"model": "claude-haiku-4-5", "max_tokens": 64, "temperature": 0},
                "claude-haiku-4-5",
            )

        assert params == {"model": "claude-haiku-4-5", "max_tokens": 64}
        assert notices == [sampling_notice("claude-haiku-4-5", ["temperature"])]
