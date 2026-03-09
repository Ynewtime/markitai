"""Tests for ChatGPT provider (Responses API wrapper).

Tests the ChatGPTProvider which bypasses LiteLLM's broken chatgpt/ routing
and calls the Responses API directly via httpx.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


class TestConvertMessages:
    """Test conversion from Chat Completions messages to Responses API input."""

    def _make_provider(self) -> Any:
        from markitai.providers.chatgpt import ChatGPTProvider

        return ChatGPTProvider()

    def test_simple_user_message(self) -> None:
        provider = self._make_provider()
        messages = [{"role": "user", "content": "Hello"}]
        instructions, result = provider._convert_messages(messages)
        assert instructions == ""
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_extracted_to_instructions(self) -> None:
        """System messages should be extracted into instructions, not input."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        instructions, result = provider._convert_messages(messages)
        assert instructions == "You are helpful."
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_multi_turn_conversation(self) -> None:
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        instructions, result = provider._convert_messages(messages)
        assert instructions == "Be concise."
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Hi"}
        assert result[1] == {"role": "assistant", "content": "Hello!"}
        assert result[2] == {"role": "user", "content": "How are you?"}

    def test_multiple_system_messages_joined(self) -> None:
        """Multiple system messages should be concatenated."""
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        instructions, result = provider._convert_messages(messages)
        assert instructions == "Rule 1\n\nRule 2"
        assert len(result) == 1

    def test_vision_message_with_base64_image(self) -> None:
        provider = self._make_provider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"},
                    },
                ],
            }
        ]
        instructions, result = provider._convert_messages(messages)
        assert instructions == ""
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {
            "type": "input_text",
            "text": "Describe this image",
        }
        assert msg["content"][1]["type"] == "input_image"
        assert (
            msg["content"][1]["image_url"] == "data:image/png;base64,iVBORw0KGgoAAAANS"
        )

    def test_empty_messages(self) -> None:
        provider = self._make_provider()
        instructions, result = provider._convert_messages([])
        assert instructions == ""
        assert result == []

    def test_message_with_empty_content(self) -> None:
        provider = self._make_provider()
        messages = [{"role": "user", "content": ""}]
        instructions, result = provider._convert_messages(messages)
        assert instructions == ""
        assert result == [{"role": "user", "content": ""}]

    def test_mixed_text_and_image_parts(self) -> None:
        provider = self._make_provider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                    },
                ],
            }
        ]
        instructions, result = provider._convert_messages(messages)
        assert instructions == ""
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 3
        assert content[0] == {"type": "input_text", "text": "First part"}
        assert content[1] == {"type": "input_text", "text": "Second part"}
        assert content[2]["type"] == "input_image"


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Test authentication via LiteLLM's Authenticator."""

    def _make_provider(self) -> Any:
        from markitai.providers.chatgpt import ChatGPTProvider

        return ChatGPTProvider()

    def test_successful_token_retrieval(self) -> None:
        provider = self._make_provider()
        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "test-token-123"
        provider._authenticator = mock_auth

        authenticator = provider._get_authenticator()
        token = authenticator.get_access_token()
        assert token == "test-token-123"

    @patch(
        "markitai.providers.chatgpt._import_authenticator",
        side_effect=ImportError("chatgpt module not found"),
    )
    def test_authenticator_import_error(self, mock_import: MagicMock) -> None:
        from markitai.providers.chatgpt import ChatGPTProvider
        from markitai.providers.errors import AuthenticationError

        provider = ChatGPTProvider()
        with pytest.raises(AuthenticationError, match="not available"):
            provider._get_authenticator()

    def test_get_authenticator_returns_cached_instance(self) -> None:
        """Once set, _get_authenticator returns the same instance."""
        from markitai.providers.chatgpt import ChatGPTProvider

        provider = ChatGPTProvider()
        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "cached-token"
        provider._authenticator = mock_auth

        result = provider._get_authenticator()
        assert result is mock_auth

    async def test_acompletion_auth_error_raises_authentication_error(self) -> None:
        from markitai.providers.chatgpt import ChatGPTProvider
        from markitai.providers.errors import AuthenticationError

        provider = ChatGPTProvider()
        mock_auth = MagicMock()
        mock_auth.get_access_token.side_effect = Exception("Token refresh failed")
        mock_auth.get_api_base.return_value = "https://chatgpt.com/backend-api/codex"
        provider._authenticator = mock_auth

        with pytest.raises(AuthenticationError, match="Token refresh failed"):
            await provider.acompletion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "Hello"}],
            )


# ---------------------------------------------------------------------------
# API call tests (mock streaming httpx)
# ---------------------------------------------------------------------------


def _make_sse_lines(
    text: str = "Hello from ChatGPT",
    model: str = "codex-mini",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[str]:
    """Build SSE lines simulating a ChatGPT Responses API stream."""
    completed_response = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "model": model,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            },
        }
    )
    # Simulate delta events then the completed event
    delta = json.dumps({"type": "response.output_text.delta", "delta": text})
    return [
        f"data: {delta}",
        f"data: {completed_response}",
    ]


def _mock_stream_client(
    sse_lines: list[str],
    status_code: int = 200,
    error_body: str | None = None,
) -> AsyncMock:
    """Create a mock httpx client that simulates client.stream() → SSE."""

    async def _aiter_lines() -> Any:  # noqa: ANN401
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.status_code = status_code

    if status_code >= 400:
        # Error path: aread returns the error body
        mock_response.aread = AsyncMock(
            return_value=(error_body or "error").encode("utf-8")
        )
    else:
        mock_response.aiter_lines = _aiter_lines

    # Build async context manager for client.stream()
    stream_cm = AsyncMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=None)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=stream_cm)
    return mock_client


class TestAPICall:
    """Test the full acompletion flow with mocked streaming httpx."""

    def _make_provider(self) -> Any:
        from markitai.providers.chatgpt import ChatGPTProvider

        return ChatGPTProvider()

    def _mock_auth(self) -> MagicMock:
        mock = MagicMock()
        mock.get_access_token.return_value = "test-token"
        mock.get_api_base.return_value = "https://chatgpt.com/backend-api/codex"
        mock.get_account_id.return_value = None
        return mock

    async def test_successful_completion(self) -> None:
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()

        sse_lines = _make_sse_lines()
        provider._client = _mock_stream_client(sse_lines)

        result = await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hello"}],
        )

        assert result.choices[0].message.content == "Hello from ChatGPT"
        assert result.choices[0].finish_reason == "stop"

        # Verify stream was called with correct URL
        provider._client.stream.assert_called_once()
        call_args = provider._client.stream.call_args
        assert call_args[0][0] == "POST"
        assert "/responses" in call_args[0][1]
        body = call_args[1]["json"]
        assert body["model"] == "codex-mini"
        assert body["store"] is False
        assert body["stream"] is True

    async def test_instructions_sent_from_system_message(self) -> None:
        """System message should be sent as 'instructions' field in request body."""
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        await provider.acompletion(
            "chatgpt/codex-mini",
            [
                {"role": "system", "content": "You are a markdown expert."},
                {"role": "user", "content": "Clean this text."},
            ],
        )

        body = provider._client.stream.call_args[1]["json"]
        assert body["instructions"] == "You are a markdown expert."
        for msg in body["input"]:
            assert msg["role"] != "system"

    async def test_default_instructions_when_no_system(self) -> None:
        """Should use default instructions when no system message is provided."""
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hello"}],
        )

        body = provider._client.stream.call_args[1]["json"]
        assert body["instructions"] == "You are a helpful assistant."

    async def test_model_prefix_stripped(self) -> None:
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        await provider.acompletion(
            "chatgpt/gpt-4o",
            [{"role": "user", "content": "Hi"}],
        )

        body = provider._client.stream.call_args[1]["json"]
        assert body["model"] == "gpt-4o"

    async def test_unsupported_params_not_sent(self) -> None:
        """max_tokens, temperature, top_p should NOT be in the request body."""
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hi"}],
            max_tokens=1000,
            temperature=0.5,
            top_p=0.9,
        )

        body = provider._client.stream.call_args[1]["json"]
        assert "max_tokens" not in body
        assert "temperature" not in body
        assert "top_p" not in body

    async def test_codex_headers_sent(self) -> None:
        """Codex-specific headers (originator, session_id, user-agent) are sent."""
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hi"}],
        )

        headers = provider._client.stream.call_args[1]["headers"]
        assert headers["originator"] == "codex_cli_rs"
        assert "session_id" in headers
        assert "codex_cli_rs" in headers["user-agent"]
        assert headers["accept"] == "text/event-stream"

    async def test_include_reasoning_in_request(self) -> None:
        """Request body should include reasoning.encrypted_content."""
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hi"}],
        )

        body = provider._client.stream.call_args[1]["json"]
        assert "include" in body
        assert "reasoning.encrypted_content" in body["include"]

    async def test_api_error_400_raises_provider_error(self) -> None:
        from markitai.providers.errors import ProviderError

        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(
            [], status_code=400, error_body='{"detail": "Bad Request"}'
        )

        with pytest.raises(ProviderError, match="Bad Request"):
            await provider.acompletion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "Hello"}],
            )

    async def test_api_403_raises_authentication_error(self) -> None:
        from markitai.providers.errors import AuthenticationError

        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(
            [], status_code=403, error_body='{"detail": "Forbidden"}'
        )

        with pytest.raises(AuthenticationError):
            await provider.acompletion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "Hello"}],
            )

    async def test_network_error_raises_provider_error(self) -> None:
        from markitai.providers.errors import ProviderError

        provider = self._make_provider()
        provider._authenticator = self._mock_auth()

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        provider._client = mock_client

        with pytest.raises(ProviderError, match="Connection refused"):
            await provider.acompletion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "Hello"}],
            )

    async def test_response_includes_usage_data(self) -> None:
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(
            _make_sse_lines(input_tokens=100, output_tokens=50)
        )

        result = await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hello"}],
        )

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150

    async def test_response_model_includes_prefix(self) -> None:
        """The model in ModelResponse should keep the full chatgpt/ prefix."""
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        result = await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hello"}],
        )

        assert result.model == "chatgpt/codex-mini"

    async def test_hidden_params_cost_tracking(self) -> None:
        provider = self._make_provider()
        provider._authenticator = self._mock_auth()
        provider._client = _mock_stream_client(_make_sse_lines())

        result = await provider.acompletion(
            "chatgpt/codex-mini",
            [{"role": "user", "content": "Hello"}],
        )

        assert hasattr(result, "_hidden_params")
        assert "total_cost_usd" in result._hidden_params


# ---------------------------------------------------------------------------
# Sync completion tests
# ---------------------------------------------------------------------------


class TestSyncCompletion:
    """Test the synchronous completion wrapper."""

    def test_completion_calls_acompletion(self) -> None:
        from markitai.providers.chatgpt import ChatGPTProvider

        provider = ChatGPTProvider()
        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "test-token"
        mock_auth.get_api_base.return_value = "https://chatgpt.com/backend-api/codex"
        mock_auth.get_account_id.return_value = None
        provider._authenticator = mock_auth

        # Mock at the sync_completion level since it wraps asyncio.run
        with patch("markitai.providers.chatgpt.sync_completion") as mock_sync:
            mock_sync.return_value = MagicMock()
            provider.completion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "Hello"}],
            )
            mock_sync.assert_called_once()


# ---------------------------------------------------------------------------
# OAuth UX tests
# ---------------------------------------------------------------------------


class TestChatGPTOAuthUX:
    """Tests for ChatGPT auth guard in acompletion().

    Device Code Flow interception moved to _login_chatgpt() in auth.py.
    acompletion() now guards against blocking Device Code Flow by checking
    for the auth file on first use and delegating interactive auth to
    preflight / `markitai auth`.
    """

    async def test_no_auth_file_raises_error(self) -> None:
        """acompletion() raises AuthenticationError when auth file missing."""
        from markitai.providers.chatgpt import ChatGPTProvider
        from markitai.providers.errors import AuthenticationError

        provider = ChatGPTProvider()
        # _authenticator is None → triggers auth file guard

        with (
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(AuthenticationError, match="not authenticated"),
        ):
            await provider.acompletion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "test"}],
            )

    async def test_cached_authenticator_skips_file_check(self) -> None:
        """When _authenticator is already set, auth file check is skipped."""
        from markitai.providers.chatgpt import ChatGPTProvider

        provider = ChatGPTProvider()

        mock_auth = MagicMock()
        mock_auth.get_access_token.return_value = "cached-token"
        mock_auth.get_api_base.return_value = "https://chatgpt.com/backend-api/codex"
        mock_auth.get_account_id.return_value = None
        provider._authenticator = mock_auth

        with patch.object(
            provider, "_stream_response", new_callable=AsyncMock
        ) as mock_stream:
            mock_stream.return_value = ("result", 10, 5)
            result = await provider.acompletion(
                "chatgpt/codex-mini",
                [{"role": "user", "content": "test"}],
            )

        assert result.choices[0].message.content == "result"
        mock_auth.get_access_token.assert_called_once()
