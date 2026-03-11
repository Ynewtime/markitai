"""Unit tests for the Gemini CLI provider.

Tests cover credential loading, message conversion, request building,
API calls, token refresh, and dependency checking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.providers.gemini_cli import GeminiCredentialRecord

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


def _make_managed_creds_json(
    *,
    access_token: str = "ya29.managed-token",
    refresh_token: str = "1//managed-refresh-token",
    expiry_date: int | None = None,
    client_id: str = "managed-client-id",
    client_secret: str = "managed-client-secret",
    email: str = "gemini@example.com",
    project_id: str | None = "demo-project",
    auth_mode: str | None = "google-one",
    source: str = "markitai",
) -> dict[str, Any]:
    """Build a Markitai-managed credentials JSON dict."""
    data = _make_creds_json(
        access_token=access_token,
        refresh_token=refresh_token,
        expiry_date=expiry_date,
        client_id=client_id,
        client_secret=client_secret,
    )
    data["email"] = email
    data["project_id"] = project_id
    data["auth_mode"] = auth_mode
    data["source"] = source
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
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        provider = GeminiCLIProvider()
        mock_creds_cls = _make_mock_creds_class()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()
        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        provider = GeminiCLIProvider()
        mock_creds_cls = _make_mock_creds_class()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        provider = GeminiCLIProvider()
        mock_creds_cls = _make_mock_creds_class()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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

    def test_prefers_markitai_managed_credentials_over_shared_cli(
        self, tmp_path: Path
    ) -> None:
        """Managed profile should win over ~/.gemini/oauth_creds.json."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        managed_dir = tmp_path / ".markitai" / "auth"
        managed_dir.mkdir(parents=True)
        managed_file = managed_dir / "gemini-profile.json"
        managed_file.write_text(
            json.dumps(_make_managed_creds_json(access_token="ya29.managed-preferred")),
            encoding="utf-8",
        )
        active_file = managed_dir / "gemini-current.json"
        active_file.write_text(
            json.dumps({"credential_path": str(managed_file)}),
            encoding="utf-8",
        )

        shared_file = tmp_path / ".gemini" / "oauth_creds.json"
        shared_file.parent.mkdir()
        shared_file.write_text(
            json.dumps(_make_creds_json(access_token="ya29.shared-fallback")),
            encoding="utf-8",
        )

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: managed_dir),
            ),
            patch.object(
                type(provider),
                "_active_profile_path",
                new_callable=lambda: property(lambda _self: active_file),
            ),
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: shared_file),
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", False),
        ):
            result = provider._load_credentials()

        assert result is not None
        assert result.token == "ya29.managed-preferred"


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

    async def test_discovers_project_id(self, tmp_path: Path) -> None:
        """Discovers project ID from loadCodeAssist endpoint."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cloudProject": "test-project-42"}

        with (
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
            ),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider._get_project_id("test-token")

        assert result == "test-project-42"

    async def test_caches_project_id(self, tmp_path: Path) -> None:
        """Project ID is cached after first discovery."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        provider._project_id = "cached-project"
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        with patch.object(
            type(provider),
            "_managed_auth_dir",
            new_callable=lambda: property(lambda _self: empty_auth_dir),
        ):
            # Should return cached value without making any HTTP calls
            result = await provider._get_project_id("test-token")
        assert result == "cached-project"

    async def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        """Returns None when discovery fails."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {}

        with (
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
            ),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            result = await provider._get_project_id("test-token")

        assert result is None

    async def test_uses_bound_project_from_managed_profile(
        self, tmp_path: Path
    ) -> None:
        """Managed profiles should use their bound project without discovery."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        managed_dir = tmp_path / ".markitai" / "auth"
        managed_dir.mkdir(parents=True)
        managed_file = managed_dir / "gemini-profile.json"
        managed_file.write_text(
            json.dumps(_make_managed_creds_json(project_id="bound-project")),
            encoding="utf-8",
        )
        active_file = managed_dir / "gemini-current.json"
        active_file.write_text(
            json.dumps({"credential_path": str(managed_file)}),
            encoding="utf-8",
        )

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: managed_dir),
            ),
            patch.object(
                type(provider),
                "_active_profile_path",
                new_callable=lambda: property(lambda _self: active_file),
            ),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            result = await provider._get_project_id("test-token")

        assert result == "bound-project"
        mock_httpx.AsyncClient.assert_not_called()


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

    async def test_429_raises_quota_error_immediately(self) -> None:
        """GeminiCLI should raise QuotaError on first 429 (no internal retries).

        With MAX_429_RETRIES=0, the provider does not retry internally.
        The QuotaError is raised immediately so the Router can set a cooldown
        and route subsequent requests to other models.
        """
        from markitai.providers.errors import QuotaError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = '{"error":{"message":"reset after 57s."}}'

        with (
            patch.object(provider, "_get_access_token", return_value="test-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_429)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(QuotaError) as exc_info:
                await provider.acompletion(
                    "gemini-cli/gemini-2.5-flash-lite",
                    [{"role": "user", "content": "Hello"}],
                )

            # Only one HTTP call (no retries)
            assert mock_client.post.call_count == 1
            # Error message includes the response text so Router can parse
            # retry-after time via its regex r"(\d+)\s*s"
            assert "reset after 57s" in str(exc_info.value)

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

    async def test_auth_failure_clears_cached_token(self) -> None:
        """401 responses should invalidate the in-memory token cache."""
        from markitai.providers.errors import AuthenticationError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        provider._cached_token = "stale-token"
        provider._token_expiry = time.monotonic() + 3000

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with (
            patch.object(provider, "_get_access_token", return_value="stale-token"),
            patch.object(provider, "_get_project_id", return_value=None),
            patch("markitai.providers.gemini_cli.httpx") as mock_httpx,
        ):
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(AuthenticationError):
                await provider.acompletion(
                    "gemini-cli/gemini-2.5-pro",
                    [{"role": "user", "content": "Hello"}],
                )

        assert provider._cached_token is None
        assert provider._token_expiry == 0.0

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
        from markitai.providers.gemini_cli import (
            GeminiCLIProvider,
            GeminiCredentialRecord,
        )

        provider = GeminiCLIProvider()
        record = GeminiCredentialRecord(
            path=Path("/tmp/valid.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )

        mock_creds = MagicMock()
        mock_creds.token = "ya29.valid-token"
        mock_creds.valid = True
        mock_creds.expired = False

        with (
            patch.object(
                provider,
                "_get_credential_payload_candidates",
                return_value=[(record, {"access_token": mock_creds.token})],
            ),
            patch.object(
                provider, "_build_credentials_from_data", return_value=mock_creds
            ),
        ):
            token = await provider._get_access_token()

        assert token == "ya29.valid-token"

    async def test_refreshes_expired_token(self) -> None:
        """Expired token triggers refresh."""
        from markitai.providers.gemini_cli import (
            GeminiCLIProvider,
            GeminiCredentialRecord,
        )

        provider = GeminiCLIProvider()
        record = GeminiCredentialRecord(
            path=Path("/tmp/expired.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )

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
            patch.object(
                provider,
                "_get_credential_payload_candidates",
                return_value=[(record, {"access_token": mock_creds.token})],
            ),
            patch.object(
                provider, "_build_credentials_from_data", return_value=mock_creds
            ),
            patch.object(provider, "_save_credentials"),
            patch("markitai.providers.gemini_cli._google_auth_requests") as mock_req,
        ):
            mock_req.Request.return_value = MagicMock()
            token = await provider._get_access_token()

        assert token == "ya29.refreshed-token"
        mock_creds.refresh.assert_called_once()

    async def test_no_creds_raises_auth_error(self) -> None:
        """Missing credentials raises AuthenticationError (no inline OAuth)."""
        from markitai.providers.errors import AuthenticationError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        with (
            patch.object(
                provider, "_get_credential_payload_candidates", return_value=[]
            ),
            pytest.raises(AuthenticationError, match="No valid Gemini credentials"),
        ):
            await provider._get_access_token()

    async def test_refresh_failure_raises_auth_error(self) -> None:
        """If all credentials fail to refresh, raises AuthenticationError."""
        from markitai.providers.errors import AuthenticationError
        from markitai.providers.gemini_cli import (
            GeminiCLIProvider,
            GeminiCredentialRecord,
        )

        provider = GeminiCLIProvider()
        record = GeminiCredentialRecord(
            path=Path("/tmp/oauth-fallback.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )

        mock_creds = MagicMock()
        mock_creds.token = "ya29.old"
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "1//refresh"
        mock_creds.refresh.side_effect = Exception("Refresh failed")

        with (
            patch.object(
                provider,
                "_get_credential_payload_candidates",
                return_value=[(record, {"access_token": mock_creds.token})],
            ),
            patch.object(
                provider, "_build_credentials_from_data", return_value=mock_creds
            ),
            patch("markitai.providers.gemini_cli._google_auth_requests") as mock_req,
        ):
            mock_req.Request.return_value = MagicMock()
            with pytest.raises(
                AuthenticationError, match="No valid Gemini credentials"
            ):
                await provider._get_access_token()


# ===================================================================
# Token caching / concurrency tests
# ===================================================================


class TestTokenCaching:
    """Tests for token caching and concurrent safety (T4)."""

    async def test_falls_back_to_shared_credentials_when_managed_refresh_fails(
        self,
    ) -> None:
        """Managed refresh failures should not block valid shared CLI creds."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        managed_record = GeminiCredentialRecord(
            path=Path("/tmp/managed.json"),
            source="markitai",
            email="managed@example.com",
            project_id="managed-project",
            auth_mode="google-one",
        )
        shared_record = GeminiCredentialRecord(
            path=Path("/tmp/shared.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )
        managed_payload = _make_managed_creds_json(access_token="ya29.stale-managed")
        shared_payload = _make_creds_json(access_token="ya29.shared-fallback")

        managed_creds = MagicMock()
        managed_creds.token = "ya29.stale-managed"
        managed_creds.valid = False
        managed_creds.expired = True
        managed_creds.refresh_token = "1//managed-refresh"
        managed_creds.refresh.side_effect = Exception("refresh revoked")

        shared_creds = MagicMock()
        shared_creds.token = "ya29.shared-fallback"
        shared_creds.valid = True
        shared_creds.expired = False

        with (
            patch.object(
                provider,
                "_get_credential_payload_candidates",
                return_value=[
                    (managed_record, managed_payload),
                    (shared_record, shared_payload),
                ],
            ),
            patch.object(
                provider,
                "_build_credentials_from_data",
                side_effect=[managed_creds, shared_creds],
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch(
                "markitai.providers.gemini_cli._google_auth_requests"
            ) as mock_requests,
        ):
            mock_requests.Request.return_value = object()
            token = await provider._get_access_token()

        assert token == "ya29.shared-fallback"
        assert provider._cached_token_source == str(shared_record.path)

    async def test_token_cached_across_calls(self) -> None:
        """Second call should use cached token, not reload from disk."""
        import time as _time

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        with (
            patch.object(
                provider, "_get_credential_payload_candidates", return_value=[]
            ),
            patch.object(
                provider,
                "_acquire_token",
                new_callable=AsyncMock,
                return_value=("ya29.cached-token", _time.monotonic() + 3000, None),
            ) as mock_acquire,
        ):
            token1 = await provider._get_access_token()
            token2 = await provider._get_access_token()

        assert token1 == "ya29.cached-token"
        assert token2 == "ya29.cached-token"
        # _acquire_token should be called only once; second call
        # hits the in-memory cache.
        mock_acquire.assert_awaited_once()

    async def test_expired_cache_reloads(self) -> None:
        """Token is reloaded after cache expiry."""
        import time as _time

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()

        with (
            patch.object(
                provider, "_get_credential_payload_candidates", return_value=[]
            ),
            patch.object(
                provider,
                "_acquire_token",
                new_callable=AsyncMock,
                side_effect=[
                    ("ya29.fresh", _time.monotonic() + 3000, None),
                    ("ya29.refreshed", _time.monotonic() + 3000, None),
                ],
            ) as mock_acquire,
        ):
            # First call populates cache
            await provider._get_access_token()
            assert mock_acquire.await_count == 1

            # Simulate cache expiry by backdating _token_expiry
            provider._token_expiry = _time.monotonic() - 1

            # Second call should reload
            await provider._get_access_token()
            assert mock_acquire.await_count == 2

    async def test_cache_expiry_tracks_real_token_expiry(self) -> None:
        """In-memory cache should not outlive the credential expiry."""
        from datetime import UTC, datetime, timedelta

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        record = GeminiCredentialRecord(
            path=Path("/tmp/short-lived.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )

        mock_creds = MagicMock()
        mock_creds.token = "ya29.short-lived"
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.expiry = datetime.now(UTC) + timedelta(seconds=120)

        before = time.monotonic()
        with (
            patch.object(
                provider,
                "_get_credential_payload_candidates",
                return_value=[(record, {"access_token": mock_creds.token})],
            ),
            patch.object(
                provider, "_build_credentials_from_data", return_value=mock_creds
            ),
        ):
            token = await provider._get_access_token()

        remaining = provider._token_expiry - before
        assert token == "ya29.short-lived"
        assert 0 < remaining < 3000
        assert remaining <= 120

    async def test_raw_token_cache_uses_expiry_date_metadata(self) -> None:
        """Raw tokens should still respect expiry_date from the credential payload."""
        from datetime import UTC, datetime, timedelta

        from markitai.providers.gemini_cli import (
            GeminiCLIProvider,
            _RawToken,
        )

        provider = GeminiCLIProvider()
        expiry = datetime.now(UTC) + timedelta(seconds=180)
        record = GeminiCredentialRecord(
            path=Path("/tmp/gemini.json"),
            source="shared",
            email=None,
            project_id=None,
            auth_mode=None,
        )
        payload = {
            "access_token": "ya29.raw",
            "expiry_date": int(expiry.timestamp() * 1000),
        }

        before = time.monotonic()
        with (
            patch.object(
                provider,
                "_get_credential_payload_candidates",
                return_value=[(record, payload)],
            ),
            patch.object(
                provider,
                "_build_credentials_from_data",
                return_value=_RawToken("ya29.raw"),
            ),
        ):
            token = await provider._get_access_token()

        remaining = provider._token_expiry - before
        assert token == "ya29.raw"
        assert 0 < remaining < 3000
        assert remaining <= 180


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

    async def test_alogin_saves_managed_profile_and_marks_active(
        self, tmp_path: Path
    ) -> None:
        """Gemini login should save a Markitai-managed profile and activate it."""
        from datetime import UTC, datetime

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        managed_dir = tmp_path / ".markitai" / "auth"
        active_file = managed_dir / "gemini-current.json"

        mock_creds = MagicMock()
        mock_creds.token = "ya29.oauth-token"
        mock_creds.refresh_token = "1//new-refresh"
        mock_creds.client_id = "test-client-id"
        mock_creds.client_secret = "test-secret"
        mock_creds.expiry = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

        with (
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: managed_dir),
            ),
            patch.object(
                type(provider),
                "_active_profile_path",
                new_callable=lambda: property(lambda _self: active_file),
            ),
            patch.object(
                provider, "_create_oauth_credentials", return_value=mock_creds
            ),
            patch.object(
                provider,
                "_fetch_user_email",
                new=AsyncMock(return_value="me@example.com"),
            ),
            patch.object(
                provider,
                "_resolve_login_project",
                new=AsyncMock(return_value="demo-project"),
            ),
        ):
            record = await provider.alogin(mode="google-one")

        assert record.email == "me@example.com"
        assert record.project_id == "demo-project"
        assert record.source == "markitai"
        assert record.path.exists()
        active = json.loads(active_file.read_text(encoding="utf-8"))
        assert active == {"credential_path": str(record.path)}

    async def test_alogin_saves_credentials_when_project_resolution_fails(
        self, tmp_path: Path
    ) -> None:
        """alogin() should save credentials even if project resolution fails.

        When _resolve_login_project() raises AuthenticationError, the OAuth
        token and refresh_token should still be persisted to disk so subsequent
        LLM calls can use them.
        """
        from datetime import UTC, datetime

        from markitai.providers.errors import AuthenticationError
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        managed_dir = tmp_path / ".markitai" / "auth"
        active_file = managed_dir / "gemini-current.json"

        mock_creds = MagicMock()
        mock_creds.token = "ya29.oauth-token"
        mock_creds.refresh_token = "1//new-refresh"
        mock_creds.client_id = "test-client-id"
        mock_creds.client_secret = "test-secret"
        mock_creds.expiry = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

        with (
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: managed_dir),
            ),
            patch.object(
                type(provider),
                "_active_profile_path",
                new_callable=lambda: property(lambda _self: active_file),
            ),
            patch.object(
                provider, "_create_oauth_credentials", return_value=mock_creds
            ),
            patch.object(
                provider,
                "_fetch_user_email",
                new=AsyncMock(return_value="me@example.com"),
            ),
            patch.object(
                provider,
                "_resolve_login_project",
                new=AsyncMock(
                    side_effect=AuthenticationError(
                        "Failed to resolve a Gemini Code Assist project.",
                        provider="gemini-cli",
                    )
                ),
            ),
        ):
            record = await provider.alogin(mode="google-one")

        # Credentials should be saved even without project resolution
        assert record.email == "me@example.com"
        assert record.project_id is None
        assert record.source == "markitai"
        assert record.path.exists()

        # Active profile should point to saved credentials
        active = json.loads(active_file.read_text(encoding="utf-8"))
        assert active == {"credential_path": str(record.path)}

        # Saved file should have the token data
        saved = json.loads(record.path.read_text(encoding="utf-8"))
        assert saved["access_token"] == "ya29.oauth-token"
        assert saved["refresh_token"] == "1//new-refresh"
        assert saved["email"] == "me@example.com"


# ===================================================================
# OAuth scope tests
# ===================================================================


class TestOAuthScopes:
    """Verify OAuth scopes include openid to prevent scope-mismatch errors.

    Google's OAuth server automatically injects 'openid' when userinfo.*
    scopes are requested.  oauthlib strictly checks returned vs requested
    scopes and raises "Scope has changed" if they differ.
    """

    def test_scopes_include_openid(self) -> None:
        """GEMINI_CLI_SCOPES must include 'openid'."""
        from markitai.providers.gemini_cli import GEMINI_CLI_SCOPES

        assert "openid" in GEMINI_CLI_SCOPES

    def test_create_oauth_sets_relax_env(self) -> None:
        """_create_oauth_credentials sets OAUTHLIB_RELAX_TOKEN_SCOPE."""
        import os

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = MagicMock(
            token="ya29.test", refresh_token="1//test"
        )

        with (
            patch("markitai.providers.gemini_cli._OAUTHLIB_AVAILABLE", True),
            patch("markitai.providers.gemini_cli._InstalledAppFlow") as mock_cls,
            patch("markitai.providers.gemini_cli.suppress_stdout"),
        ):
            mock_cls.from_client_config.return_value = mock_flow

            # Capture env state during the call
            captured_env: dict[str, str | None] = {}
            original_run = mock_flow.run_local_server

            def capture_env(*args: Any, **kwargs: Any) -> Any:
                captured_env["OAUTHLIB_RELAX_TOKEN_SCOPE"] = os.environ.get(
                    "OAUTHLIB_RELAX_TOKEN_SCOPE"
                )
                return original_run(*args, **kwargs)

            mock_flow.run_local_server = capture_env
            provider._create_oauth_credentials()

        assert captured_env.get("OAUTHLIB_RELAX_TOKEN_SCOPE") == "1"


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
        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()

        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(lambda _self: creds_file),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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

        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()
        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(
                    lambda _self: tmp_path / "nonexistent.json"
                ),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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

        empty_auth_dir = tmp_path / "empty_auth"
        empty_auth_dir.mkdir()
        provider = GeminiCLIProvider()

        with (
            patch.object(
                type(provider),
                "_creds_path",
                new_callable=lambda: property(
                    lambda _self: tmp_path / "nonexistent.json"
                ),
            ),
            patch.object(
                type(provider),
                "_managed_auth_dir",
                new_callable=lambda: property(lambda _self: empty_auth_dir),
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


# ===================================================================
# OAuth UX tests
# ===================================================================


class TestOAuthUX:
    """Tests for OAuth flow UX improvements (Rich output, stdout suppression)."""

    def test_oauth_suppresses_stdout(self, tmp_path: Path) -> None:
        """run_local_server stdout output is captured and not leaked."""
        import sys
        from io import StringIO

        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        creds_file = tmp_path / "oauth_creds.json"

        mock_creds = MagicMock()
        mock_creds.token = "ya29.oauth-token"
        mock_creds.refresh_token = "1//refresh"
        mock_creds.client_id = "test-client-id"
        mock_creds.client_secret = "test-secret"
        mock_creds.expiry = None

        def fake_run_local_server(**kwargs: Any) -> MagicMock:
            # Simulate google-auth-oauthlib printing to stdout
            print("Please visit this URL to authorize this application:")
            print("https://accounts.google.com/o/oauth2/auth?redirect=...")
            return mock_creds

        mock_flow = MagicMock()
        mock_flow.run_local_server = fake_run_local_server

        # Capture actual stdout to verify nothing leaks
        captured_stdout = StringIO()
        original_stdout = sys.stdout

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
            sys.stdout = captured_stdout
            try:
                provider._run_oauth_flow()
            finally:
                sys.stdout = original_stdout

        # The raw "Please visit this URL" should NOT appear in stdout
        leaked = captured_stdout.getvalue()
        assert "Please visit this URL" not in leaked

    def test_oauth_shows_success_message(self, tmp_path: Path) -> None:
        """OAuth flow shows success message after completion."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        creds_file = tmp_path / "oauth_creds.json"

        mock_creds = MagicMock()
        mock_creds.token = "ya29.oauth-token"
        mock_creds.refresh_token = "1//refresh"
        mock_creds.client_id = "test-client-id"
        mock_creds.client_secret = "test-secret"
        mock_creds.expiry = None

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
            patch("markitai.providers.gemini_cli.show_oauth_start") as mock_start,
            patch("markitai.providers.gemini_cli.show_oauth_success") as mock_success,
        ):
            mock_flow_cls.from_client_config.return_value = mock_flow
            provider._run_oauth_flow()

        mock_start.assert_called_once_with("gemini-cli")
        mock_success.assert_called_once()
        call_kwargs = mock_success.call_args
        assert call_kwargs[0][0] == "gemini-cli"


# ===================================================================
# Token refresh retry tests
# ===================================================================


class TestTokenRefreshRetry:
    """Tests for token refresh retry with exponential backoff."""

    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """Token refresh retries on transient error and succeeds."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        record = GeminiCredentialRecord(
            path=Path("/tmp/creds.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )
        payload = _make_creds_json()

        mock_creds = MagicMock()
        mock_creds.token = "ya29.refreshed"
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "1//refresh"
        mock_creds.expiry = None

        # First refresh fails (transient SSL error), second succeeds
        call_count = 0

        def refresh_side_effect(request: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("SSLEOFError: EOF occurred")
            # Second call succeeds — creds.token is already set
            mock_creds.valid = True
            mock_creds.expired = False

        mock_creds.refresh.side_effect = refresh_side_effect

        with (
            patch.object(
                provider,
                "_build_credentials_from_data",
                return_value=mock_creds,
            ),
            patch.object(provider, "_save_credentials"),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch(
                "markitai.providers.gemini_cli._google_auth_requests"
            ) as mock_requests,
            patch("markitai.providers.gemini_cli.asyncio.sleep") as mock_sleep,
        ):
            mock_sleep.return_value = None
            mock_requests.Request.return_value = object()
            result = await provider._try_credentials_candidate(record, payload)

        assert result is not None
        assert result[0] == "ya29.refreshed"
        # Should have retried once
        assert call_count == 2
        # Should have slept between retries
        mock_sleep.assert_called_once()

    async def test_retry_exhausted_returns_none(self) -> None:
        """Returns None after all retry attempts are exhausted."""
        from markitai.providers.gemini_cli import GeminiCLIProvider

        provider = GeminiCLIProvider()
        record = GeminiCredentialRecord(
            path=Path("/tmp/creds.json"),
            source="gemini-cli",
            email=None,
            project_id=None,
            auth_mode=None,
        )
        payload = _make_creds_json()

        mock_creds = MagicMock()
        mock_creds.token = "ya29.stale"
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "1//refresh"

        # All refresh attempts fail
        mock_creds.refresh.side_effect = Exception("SSLEOFError: persistent")

        with (
            patch.object(
                provider,
                "_build_credentials_from_data",
                return_value=mock_creds,
            ),
            patch("markitai.providers.gemini_cli._GOOGLE_AUTH_AVAILABLE", True),
            patch(
                "markitai.providers.gemini_cli._google_auth_requests"
            ) as mock_requests,
            patch("markitai.providers.gemini_cli.asyncio.sleep") as mock_sleep,
        ):
            mock_sleep.return_value = None
            mock_requests.Request.return_value = object()
            result = await provider._try_credentials_candidate(record, payload)

        assert result is None
        # Should have tried 3 times (1 original + 2 retries)
        assert mock_creds.refresh.call_count == 3
