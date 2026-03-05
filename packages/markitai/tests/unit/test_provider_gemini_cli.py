"""Unit tests for the Gemini CLI provider.

Tests cover credential loading, message conversion, request building,
API calls, token refresh, and dependency checking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_creds_json(
    *,
    access_token: str = "ya29.test-access-token",
    refresh_token: str = "1//test-refresh-token",
    expiry_date: int | None = None,
    client_id: str = "test-client-id",
    client_secret: str = "test-client-secret",
) -> dict[str, Any]:
    """Build a credentials JSON dict matching ~/.gemini/oauth_creds.json format."""
    data: dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    if expiry_date is not None:
        data["expiry_date"] = expiry_date
    return data


def _gemini_api_response(
    text: str = "Hello from Gemini!",
    prompt_tokens: int = 10,
    candidates_tokens: int = 20,
    *,
    wrapped: bool = True,
) -> dict[str, Any]:
    """Build a mock Gemini API response.

    Args:
        wrapped: If True, wraps in Code Assist envelope (default).
    """
    inner = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "totalTokenCount": prompt_tokens + candidates_tokens,
        },
    }
    if wrapped:
        return {"response": inner, "traceId": "test-trace", "metadata": {}}
    return inner


def _make_mock_creds_class() -> MagicMock:
    """Create a mock for google.oauth2.credentials.Credentials class."""
    mock_cls = MagicMock()

    def _create_creds(**kwargs: Any) -> MagicMock:
        creds = MagicMock()
        creds.token = kwargs.get("token")
        creds.refresh_token = kwargs.get("refresh_token")
        creds.client_id = kwargs.get("client_id")
        creds.client_secret = kwargs.get("client_secret")
        creds.expiry = None
        creds.valid = True
        creds.expired = False
        return creds

    mock_cls.side_effect = _create_creds
    return mock_cls


# ===================================================================
# Credential loading tests
# ===================================================================


class TestLoadCredentials:
    """Tests for _load_credentials method."""

    def test_loads_valid_credentials(self, tmp_path: Path) -> None:
        """Loads valid credentials from oauth_creds.json."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_creds_json()))

        provider = GeminiCLIProvider()
        mock_creds_cls = _make_mock_creds_class()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch(
                "markitai.providers.gemini_cli._google_oauth2_credentials"
            ) as mock_mod,
        ):
            mock_mod.Credentials = mock_creds_cls
            result = provider._load_credentials()

        assert result is not None
        assert result.token == "ya29.test-access-token"
        assert result.refresh_token == "1//test-refresh-token"

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """Returns None when the credentials file does not exist."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        creds_file = tmp_path / "nonexistent.json"
        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
        ):
            result = provider._load_credentials()

        assert result is None

    def test_returns_none_when_file_is_invalid_json(self, tmp_path: Path) -> None:
        """Returns None when the file contains invalid JSON."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text("not valid json {{{")

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
        ):
            result = provider._load_credentials()

        assert result is None

    def test_parses_expiry_date_milliseconds(self, tmp_path: Path) -> None:
        """Parses expiry_date as milliseconds Unix timestamp (Gemini CLI format)."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        # 2026-01-01 00:00:00 UTC in milliseconds
        expiry_ms = 1767225600000
        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_creds_json(expiry_date=expiry_ms)))

        provider = GeminiCLIProvider()
        mock_creds_cls = _make_mock_creds_class()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch(
                "markitai.providers.gemini_cli._google_oauth2_credentials"
            ) as mock_mod,
        ):
            mock_mod.Credentials = mock_creds_cls
            result = provider._load_credentials()

        assert result is not None
        # expiry should have been set on the mock creds
        assert result.expiry is not None

    def test_handles_missing_refresh_token(self, tmp_path: Path) -> None:
        """Loads credentials even when refresh_token is absent."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        data = _make_creds_json()
        del data["refresh_token"]
        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(data))

        provider = GeminiCLIProvider()
        mock_creds_cls = _make_mock_creds_class()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch(
                "markitai.providers.gemini_cli._google_oauth2_credentials"
            ) as mock_mod,
        ):
            mock_mod.Credentials = mock_creds_cls
            result = provider._load_credentials()

        assert result is not None
        # refresh_token should be None since it was not in the JSON
        assert result.refresh_token is None


# ===================================================================
# Message conversion tests
# ===================================================================


class TestConvertMessages:
    """Tests for _convert_messages method."""

    def test_simple_user_message(self) -> None:
        """Converts a simple user text message."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        contents, system = provider._convert_messages(messages)

        assert system is None
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"] == [{"text": "Hello"}]

    def test_system_message_extracted(self) -> None:
        """System messages are extracted into systemInstruction."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
        ]
        contents, system = provider._convert_messages(messages)

        assert system is not None
        assert system["parts"] == [{"text": "You are a helper."}]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_message_mapped_to_model(self) -> None:
        """Assistant messages are mapped to role 'model'."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        contents, system = provider._convert_messages(messages)

        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hello!"}]
        assert contents[2]["role"] == "user"

    def test_mixed_system_user_assistant(self) -> None:
        """Mixed system + user + assistant messages are properly structured."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Summarize X"},
            {"role": "assistant", "content": "X is about Y"},
            {"role": "user", "content": "More details"},
        ]
        contents, system = provider._convert_messages(messages)

        assert system is not None
        assert system["parts"] == [{"text": "Be concise."}]
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"

    def test_vision_with_base64_image(self) -> None:
        """Base64 images are converted to inlineData parts."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."},
                    },
                ],
            }
        ]
        contents, system = provider._convert_messages(messages)

        assert len(contents) == 1
        parts = contents[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == {"text": "Describe this image"}
        assert parts[1] == {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": "/9j/4AAQ...",
            }
        }

    def test_empty_messages(self) -> None:
        """Empty messages list returns empty contents and no system."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        contents, system = provider._convert_messages([])

        assert contents == []
        assert system is None

    def test_multiple_system_messages_merged(self) -> None:
        """Multiple system messages are merged into one systemInstruction."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [
            {"role": "system", "content": "Rule 1."},
            {"role": "system", "content": "Rule 2."},
            {"role": "user", "content": "Hello"},
        ]
        contents, system = provider._convert_messages(messages)

        assert system is not None
        # Both system texts should be present
        texts = [p["text"] for p in system["parts"]]
        assert "Rule 1." in texts
        assert "Rule 2." in texts


# ===================================================================
# Request building tests
# ===================================================================


class TestBuildRequest:
    """Tests for _build_request method."""

    def test_proper_ca_generate_content_structure(self) -> None:
        """Request follows CAGenerateContentRequest structure."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        request = provider._build_request("gemini-2.5-pro", messages)

        assert "model" in request
        assert "request" in request
        assert "contents" in request["request"]

    def test_model_name_prefix_stripping(self) -> None:
        """gemini-cli/ prefix is stripped from model name."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        request = provider._build_request("gemini-cli/gemini-2.5-pro", messages)

        assert request["model"] == "gemini-2.5-pro"

    def test_generation_config_from_kwargs(self) -> None:
        """Generation config params are passed through."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        request = provider._build_request(
            "gemini-2.5-pro",
            messages,
            temperature=0.5,
            max_tokens=1024,
        )

        gen_config = request["request"]["generationConfig"]
        assert gen_config["temperature"] == 0.5
        assert gen_config["maxOutputTokens"] == 1024

    def test_system_instruction_included(self) -> None:
        """System messages produce systemInstruction in request."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        request = provider._build_request("gemini-2.5-pro", messages)

        assert "systemInstruction" in request["request"]
        assert request["request"]["systemInstruction"]["parts"] == [
            {"text": "Be helpful."}
        ]

    def test_no_system_instruction_when_none(self) -> None:
        """No systemInstruction when there are no system messages."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        request = provider._build_request("gemini-2.5-pro", messages)

        assert "systemInstruction" not in request["request"]

    def test_project_included_when_provided(self) -> None:
        """project field is included in request when provided."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        request = provider._build_request(
            "gemini-2.5-pro", messages, project="my-project-123"
        )

        assert request["project"] == "my-project-123"

    def test_project_omitted_when_none(self) -> None:
        """project field is NOT included when not provided."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        messages = [{"role": "user", "content": "Hello"}]
        request = provider._build_request("gemini-2.5-pro", messages)

        assert "project" not in request


# ===================================================================
# Project discovery tests
# ===================================================================


class TestProjectDiscovery:
    """Tests for _get_project_id method."""

    async def test_discovers_project_id(self) -> None:
        """Discovers project ID from loadCodeAssist endpoint."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cloudProject": "test-project-42"}

        with patch("markitai.providers.gemini_cli.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider._get_project_id("test-token")

        assert result == "test-project-42"

    async def test_caches_project_id(self) -> None:
        """Project ID is cached after first discovery."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        provider._project_id = "cached-project"

        # Should return cached value without making any HTTP calls
        result = await provider._get_project_id("test-token")
        assert result == "cached-project"

    async def test_returns_none_on_failure(self) -> None:
        """Returns None when discovery fails."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}

        with patch("markitai.providers.gemini_cli.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider._get_project_id("test-token")

        assert result is None


# ===================================================================
# API call tests (mock httpx)
# ===================================================================


class TestACompletion:
    """Tests for acompletion method."""

    async def test_successful_completion(self) -> None:
        """Successful API call returns proper ModelResponse."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_api_response(
            text="Test response",
            prompt_tokens=15,
            candidates_tokens=25,
        )
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider.acompletion(
                "gemini-cli/gemini-2.5-pro",
                [{"role": "user", "content": "Hello"}],
            )

        assert result.choices[0].message.content == "Test response"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 25
        assert result.usage.total_tokens == 40

    async def test_retries_on_429(self) -> None:
        """429 response triggers retry with wait, eventually succeeds."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        # First call returns 429, second returns 200
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = '{"error":{"message":"reset after 1s."}}'

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = _gemini_api_response(text="After retry")

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
            patch("markitai.providers.gemini_cli.asyncio") as mock_asyncio,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[mock_429, mock_200])
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client
            mock_asyncio.sleep = AsyncMock()

            result = await provider.acompletion(
                "gemini-cli/gemini-2.5-flash-lite",
                [{"role": "user", "content": "Hello"}],
            )

        assert result.choices[0].message.content == "After retry"
        mock_asyncio.sleep.assert_called_once_with(2.0)  # 1s + 1s buffer

    async def test_429_exhausted_raises_quota_error(self) -> None:
        """Raises QuotaError after MAX_429_RETRIES attempts."""
        from markitai.providers.errors import QuotaError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = '{"error":{"message":"reset after 5s."}}'

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
            patch("markitai.providers.gemini_cli.asyncio") as mock_asyncio,
        ):
            mock_client = AsyncMock()
            # All calls return 429
            mock_client.post = AsyncMock(return_value=mock_429)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client
            mock_asyncio.sleep = AsyncMock()

            with pytest.raises(QuotaError):
                await provider.acompletion(
                    "gemini-cli/gemini-2.5-flash-lite",
                    [{"role": "user", "content": "Hello"}],
                )

    async def test_auth_failure_raises_authentication_error(self) -> None:
        """401 response raises AuthenticationError."""
        from markitai.providers.errors import AuthenticationError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with (
            patch.object(provider, "_get_access_token", return_value="bad-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(AuthenticationError) as exc_info:
                await provider.acompletion(
                    "gemini-cli/gemini-2.5-pro",
                    [{"role": "user", "content": "Hello"}],
                )

            assert exc_info.value.provider == "gemini-cli"

    async def test_api_error_raises_provider_error(self) -> None:
        """Non-401 API errors raise ProviderError."""
        from markitai.providers.errors import ProviderError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(ProviderError) as exc_info:
                await provider.acompletion(
                    "gemini-cli/gemini-2.5-pro",
                    [{"role": "user", "content": "Hello"}],
                )

            assert exc_info.value.provider == "gemini-cli"

    async def test_response_parsing_with_usage_metadata(self) -> None:
        """Usage metadata is correctly parsed into ModelResponse."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_api_response(
            prompt_tokens=100,
            candidates_tokens=200,
        )
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider.acompletion(
                "gemini-cli/gemini-2.5-pro",
                [{"role": "user", "content": "Hello"}],
            )

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 200
        assert result.usage.total_tokens == 300

    async def test_model_name_in_response(self) -> None:
        """Response model field preserves full model ID with prefix."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _gemini_api_response()
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider.acompletion(
                "gemini-cli/gemini-2.5-pro",
                [{"role": "user", "content": "Hello"}],
            )

        # Full model ID with prefix is preserved for tracking
        assert "gemini-cli" in result.model


# ===================================================================
# Token refresh tests
# ===================================================================


class TestGetAccessToken:
    """Tests for _get_access_token method."""

    async def test_returns_valid_token(self) -> None:
        """Returns access_token when credentials are valid."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_creds = MagicMock()
        mock_creds.token = "ya29.valid-token"
        mock_creds.valid = True
        mock_creds.expired = False

        with patch.object(provider, "_load_credentials", return_value=mock_creds):
            token = await provider._get_access_token()

        assert token == "ya29.valid-token"

    async def test_refreshes_expired_token(self) -> None:
        """Expired token triggers refresh."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_creds = MagicMock()
        mock_creds.token = "ya29.expired-token"
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "1//refresh"

        # After refresh, token becomes valid
        def fake_refresh(request: Any) -> None:
            mock_creds.token = "ya29.refreshed-token"
            mock_creds.valid = True
            mock_creds.expired = False

        mock_creds.refresh.side_effect = fake_refresh

        with (
            patch.object(provider, "_load_credentials", return_value=mock_creds),
            patch.object(provider, "_save_credentials"),
            patch("markitai.providers.gemini_cli._google_auth_requests") as mock_req,
        ):
            mock_req.Request.return_value = MagicMock()
            token = await provider._get_access_token()

        assert token == "ya29.refreshed-token"
        mock_creds.refresh.assert_called_once()

    async def test_no_creds_triggers_oauth_flow(self) -> None:
        """Missing credentials triggers OAuth flow."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_new_creds = MagicMock()
        mock_new_creds.token = "ya29.new-oauth-token"

        with (
            patch.object(provider, "_load_credentials", return_value=None),
            patch.object(provider, "_run_oauth_flow", return_value=mock_new_creds),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
        ):
            token = await provider._get_access_token()

        assert token == "ya29.new-oauth-token"

    async def test_refresh_failure_triggers_oauth_flow(self) -> None:
        """If refresh fails, falls back to OAuth flow."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_creds = MagicMock()
        mock_creds.token = "ya29.old"
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "1//refresh"
        mock_creds.refresh.side_effect = Exception("Refresh failed")

        mock_new_creds = MagicMock()
        mock_new_creds.token = "ya29.fresh-from-oauth"

        with (
            patch.object(provider, "_load_credentials", return_value=mock_creds),
            patch.object(provider, "_run_oauth_flow", return_value=mock_new_creds),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch("markitai.providers.gemini_cli._google_auth_requests") as mock_req,
        ):
            mock_req.Request.return_value = MagicMock()
            token = await provider._get_access_token()

        assert token == "ya29.fresh-from-oauth"


# ===================================================================
# OAuth flow tests
# ===================================================================


class TestOAuthFlow:
    """Tests for _run_oauth_flow method."""

    def test_oauth_flow_saves_credentials(self, tmp_path: Path) -> None:
        """OAuth flow saves credentials to disk after login."""
        from datetime import UTC, datetime

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        creds_file = tmp_path / "oauth_creds.json"

        mock_creds = MagicMock()
        mock_creds.token = "ya29.oauth-token"
        mock_creds.refresh_token = "1//new-refresh"
        mock_creds.client_id = "test-client-id"
        mock_creds.client_secret = "test-secret"
        mock_creds.expiry = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._OAUTHLIB_AVAILABLE", True),
            patch("markitai.providers.gemini_cli._InstalledAppFlow") as mock_flow_cls,
        ):
            mock_flow_cls.from_client_config.return_value = mock_flow
            result = provider._run_oauth_flow()

        assert result is mock_creds
        # Credentials file should have been written
        assert creds_file.exists()
        saved = json.loads(creds_file.read_text())
        assert saved["access_token"] == "ya29.oauth-token"
        assert saved["refresh_token"] == "1//new-refresh"
        # expiry_date should be in milliseconds (Gemini CLI format)
        assert "expiry_date" in saved
        assert isinstance(saved["expiry_date"], int)
        assert "expiry" not in saved  # should NOT use ISO format


# ===================================================================
# Dependency check tests
# ===================================================================


class TestDependencyChecks:
    """Tests for optional dependency handling."""

    def test_load_credentials_returns_raw_token_without_google_auth(
        self, tmp_path: Path
    ) -> None:
        """Returns _RawToken when google-auth is missing but token exists."""
        from markitai.providers.gemini_cli import GeminiCLIProvider, _RawToken

        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(_make_creds_json()))

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", False),
        ):
            result = provider._load_credentials()

        assert isinstance(result, _RawToken)
        assert result.token == "ya29.test-access-token"

    def test_load_credentials_returns_none_without_google_auth_and_no_file(
        self, tmp_path: Path
    ) -> None:
        """Returns None when google-auth missing AND no credential file."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(
                    lambda _self: tmp_path / "nonexistent.json"
                ),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", False),
        ):
            result = provider._load_credentials()

        assert result is None

    async def test_get_access_token_raises_when_no_auth_and_no_creds(
        self, tmp_path: Path
    ) -> None:
        """SDKNotAvailableError when google-auth missing and no cached token."""
        from markitai.providers.errors import SDKNotAvailableError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(
                    lambda _self: tmp_path / "nonexistent.json"
                ),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", False),
        ):
            with pytest.raises(SDKNotAvailableError) as exc_info:
                await provider._get_access_token()

            assert "gemini-cli" in exc_info.value.install_command

    def test_google_auth_oauthlib_not_available_during_oauth(self) -> None:
        """Helpful error when google-auth-oauthlib is missing during OAuth."""
        from markitai.providers.errors import SDKNotAvailableError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        with patch("markitai.providers.gemini_cli._OAUTHLIB_AVAILABLE", False):
            with pytest.raises(SDKNotAvailableError) as exc_info:
                provider._run_oauth_flow()

            assert "gemini-cli" in exc_info.value.install_command


# ===================================================================
# Sync completion tests
# ===================================================================


class TestSyncCompletion:
    """Tests for the sync completion() wrapper."""

    def test_completion_delegates_to_acompletion(self) -> None:
        """Sync completion calls acompletion under the hood."""

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_response = MagicMock()

        # Patch asyncio at the top-level module used by sync_completion
        with patch.object(
            provider,
            "acompletion",
            new_callable=lambda: AsyncMock(return_value=mock_response),
        ):
            # sync_completion uses asyncio.run() internally
            result = provider.completion(
                "gemini-cli/gemini-2.5-pro",
                [{"role": "user", "content": "Hello"}],
            )

        assert result is mock_response


# ===================================================================
# Constants tests
# ===================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_code_assist_endpoint_defined(self) -> None:
        """CODE_ASSIST_ENDPOINT constant is defined."""
        from markitai.providers.gemini_cli import CODE_ASSIST_ENDPOINT

        assert "cloudcode-pa.googleapis.com" in CODE_ASSIST_ENDPOINT

    def test_gemini_cli_client_id_defined(self) -> None:
        """GEMINI_CLI_CLIENT_ID constant is defined."""
        from markitai.providers.gemini_cli import GEMINI_CLI_CLIENT_ID

        assert GEMINI_CLI_CLIENT_ID.endswith(".apps.googleusercontent.com")

    def test_gemini_cli_scopes_defined(self) -> None:
        """GEMINI_CLI_SCOPES constant is defined."""
        from markitai.providers.gemini_cli import GEMINI_CLI_SCOPES

        assert isinstance(GEMINI_CLI_SCOPES, list)
        assert len(GEMINI_CLI_SCOPES) >= 1
        assert any("cloud-platform" in s for s in GEMINI_CLI_SCOPES)

    def test_parse_retry_after_from_error(self) -> None:
        """Parses retry wait time from 429 error message."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        assert (
            GeminiCLIProvider._parse_retry_after(
                '{"error":{"message":"quota will reset after 57s."}}'
            )
            == 58.0
        )  # 57 + 1 buffer

    def test_parse_retry_after_fallback(self) -> None:
        """Falls back to 60s when error message can't be parsed."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        assert GeminiCLIProvider._parse_retry_after("unknown error") == 60.0

    def test_parse_response_wrapped_format(self) -> None:
        """Response wrapped in Code Assist envelope is parsed correctly."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        data = _gemini_api_response(text="wrapped text", wrapped=True)
        result = provider._parse_response(data, "gemini-cli/gemini-2.5-flash")

        assert result.choices[0].message.content == "wrapped text"

    def test_parse_response_unwrapped_format(self) -> None:
        """Response without envelope (plain Gemini format) also works."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        data = _gemini_api_response(text="raw text", wrapped=False)
        result = provider._parse_response(data, "gemini-cli/gemini-2.5-flash")

        assert result.choices[0].message.content == "raw text"

    def test_creds_path_points_to_gemini_dir(self) -> None:
        """GEMINI_CLI_CREDS_PATH is under ~/.gemini/."""
        from markitai.providers.gemini_cli import GEMINI_CLI_CREDS_PATH

        assert ".gemini" in str(GEMINI_CLI_CREDS_PATH)
        assert str(GEMINI_CLI_CREDS_PATH).endswith("oauth_creds.json")
