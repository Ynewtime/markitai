"""Unit tests for provider authentication manager.

These tests cover the authentication status checking and caching functionality
for local LLM providers (claude-agent and copilot).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.providers.auth import AuthStatus


class TestEmailFromJwt:
    """Tests for _email_from_jwt helper."""

    def test_extracts_email_from_valid_jwt(self) -> None:
        """Extracts email from a well-formed JWT id_token."""
        import base64

        from markitai.providers.auth import _email_from_jwt

        payload = (
            base64.urlsafe_b64encode(
                json.dumps({"email": "user@example.com", "sub": "123"}).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        token = f"header.{payload}.signature"
        assert _email_from_jwt(token) == "user@example.com"

    def test_returns_none_for_no_email(self) -> None:
        """Returns None when JWT has no email field."""
        import base64

        from markitai.providers.auth import _email_from_jwt

        payload = (
            base64.urlsafe_b64encode(json.dumps({"sub": "123"}).encode())
            .rstrip(b"=")
            .decode()
        )
        token = f"header.{payload}.signature"
        assert _email_from_jwt(token) is None

    def test_returns_none_for_invalid_token(self) -> None:
        """Returns None for non-JWT strings."""
        from markitai.providers.auth import _email_from_jwt

        assert _email_from_jwt("not-a-jwt") is None
        assert _email_from_jwt("") is None


class TestAuthStatus:
    """Tests for AuthStatus dataclass."""

    def test_auth_status_has_required_fields(self) -> None:
        """Test that AuthStatus has all required fields."""
        from markitai.providers.auth import AuthStatus

        status = AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user="test@example.com",
            expires_at=None,
            error=None,
        )
        assert status.provider == "claude-agent"
        assert status.authenticated is True
        assert status.user == "test@example.com"
        assert status.expires_at is None
        assert status.error is None

    def test_auth_status_is_frozen(self) -> None:
        """Test that AuthStatus is immutable (frozen)."""
        from markitai.providers.auth import AuthStatus

        status = AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Not authenticated",
        )
        with pytest.raises(AttributeError):
            status.authenticated = True  # type: ignore[misc]

    def test_auth_status_with_expiration(self) -> None:
        """Test AuthStatus with token expiration."""
        from markitai.providers.auth import AuthStatus

        expires = datetime.now(UTC) + timedelta(hours=1)
        status = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="user@github.com",
            expires_at=expires,
            error=None,
        )
        assert status.expires_at == expires

    def test_auth_status_unauthenticated_with_error(self) -> None:
        """Test AuthStatus for unauthenticated state with error."""
        from markitai.providers.auth import AuthStatus

        status = AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Token expired",
        )
        assert status.authenticated is False
        assert status.error == "Token expired"


class TestAuthManagerSingleton:
    """Tests for AuthManager singleton pattern."""

    def test_auth_manager_singleton_returns_same_instance(self) -> None:
        """Test that AuthManager returns the same instance."""
        from markitai.providers.auth import AuthManager

        manager1 = AuthManager()
        manager2 = AuthManager()
        assert manager1 is manager2

    def test_auth_manager_has_cache(self) -> None:
        """Test that AuthManager has a cache dictionary."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()
        assert hasattr(manager, "_cache")
        assert isinstance(manager._cache, dict)


class TestAuthManagerCheckAuth:
    """Tests for AuthManager.check_auth method."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton and cache before each test."""
        from markitai.providers.auth import AuthManager

        # Clear the singleton instance
        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_copilot_authenticated(self, tmp_path: Path) -> None:
        """Test check_auth for copilot when authenticated via config file."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock config file
        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "logged_in_users": [
                        {"host": "https://github.com", "login": "testuser"}
                    ]
                }
            )
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("copilot")

        assert status.provider == "copilot"
        assert status.authenticated is True
        assert status.user == "testuser"

    @pytest.mark.asyncio
    async def test_check_auth_copilot_not_authenticated(self, tmp_path: Path) -> None:
        """Test check_auth for copilot when not authenticated."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock config file with no logged in users
        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"logged_in_users": []}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("copilot")

        assert status.authenticated is False

    @pytest.mark.asyncio
    async def test_check_auth_claude_authenticated(self, tmp_path: Path) -> None:
        """Test check_auth for claude when authenticated."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock credentials file
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        # Token expires in 1 hour
        expires_at = int((datetime.now().timestamp() + 3600) * 1000)
        creds_file.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "subscriptionType": "max",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("markitai.providers.auth._claude_cli_email", return_value=None),
        ):
            status = await manager.check_auth("claude-agent")

        assert status.provider == "claude-agent"
        assert status.authenticated is True
        assert "max" in (status.user or "")

    @pytest.mark.asyncio
    async def test_check_auth_claude_token_expired_no_refresh(
        self, tmp_path: Path
    ) -> None:
        """Test check_auth for claude when token is expired and no refresh token."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock credentials file with expired token, no refresh token
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        # Token expired 1 hour ago
        expires_at = int((datetime.now().timestamp() - 3600) * 1000)
        creds_file.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "subscriptionType": "max",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("claude-agent")

        assert status.authenticated is False
        assert "expired" in (status.error or "").lower()

    @pytest.mark.asyncio
    async def test_check_auth_claude_token_expired_with_refresh(
        self, tmp_path: Path
    ) -> None:
        """Test check_auth for claude when access token is expired but refresh token exists.

        Claude CLI auto-refreshes tokens, so expired access token + valid refresh
        token should still be considered authenticated.
        """
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        # Access token expired 1 hour ago, but refresh token present
        expires_at = int((datetime.now().timestamp() - 3600) * 1000)
        creds_file.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "refreshToken": "refresh-token-abc",
                        "subscriptionType": "max",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("markitai.providers.auth._claude_cli_email", return_value=None),
        ):
            status = await manager.check_auth("claude-agent")

        assert status.authenticated is True
        assert "max" in (status.user or "")


class TestAuthManagerUnknownProvider:
    """Tests for unknown provider handling."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_unknown_provider(self) -> None:
        """Unknown provider should return unauthenticated with error."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()
        status = await manager.check_auth("nonexistent-provider")

        assert status.authenticated is False
        assert "Unknown provider" in (status.error or "")
        assert status.provider == "nonexistent-provider"


class TestAuthManagerNoConfigFile:
    """Tests for scenarios where config files don't exist."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_copilot_no_config(self, tmp_path: Path) -> None:
        """Test check_auth when Copilot config file doesn't exist."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("copilot")

        assert status.authenticated is False
        assert "not found" in (status.error or "").lower()

    @pytest.mark.asyncio
    async def test_check_auth_claude_no_credentials(self, tmp_path: Path) -> None:
        """Test check_auth when Claude credentials file doesn't exist."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("claude-agent")

        assert status.authenticated is False
        assert "not found" in (status.error or "").lower()


class TestAuthManagerCache:
    """Tests for AuthManager cache behavior."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_uses_cache(self, tmp_path: Path) -> None:
        """Test that check_auth uses cached result on second call."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock config file
        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"logged_in_users": [{"login": "testuser"}]}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status1 = await manager.check_auth("copilot")
            # Modify the config file
            config_file.write_text(json.dumps({"logged_in_users": []}))
            status2 = await manager.check_auth("copilot")

        # Second call should return cached result (still authenticated)
        assert status1.authenticated is True
        assert status2.authenticated is True

    @pytest.mark.asyncio
    async def test_check_auth_force_refresh_bypasses_cache(
        self, tmp_path: Path
    ) -> None:
        """Test that force_refresh bypasses the cache."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock config file
        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"logged_in_users": [{"login": "testuser"}]}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status1 = await manager.check_auth("copilot")
            # Modify the config file
            config_file.write_text(json.dumps({"logged_in_users": []}))
            status2 = await manager.check_auth("copilot", force_refresh=True)

        # Second call with force_refresh should read new config
        assert status1.authenticated is True
        assert status2.authenticated is False

    def test_clear_cache_single_provider(self) -> None:
        """Test clear_cache for a single provider."""
        from markitai.providers.auth import AuthManager, AuthStatus

        manager = AuthManager()
        manager._cache["copilot"] = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="test",
            expires_at=None,
            error=None,
        )
        manager._cache["claude-agent"] = AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user="test",
            expires_at=None,
            error=None,
        )

        manager.clear_cache("copilot")

        assert "copilot" not in manager._cache
        assert "claude-agent" in manager._cache

    def test_clear_cache_all_providers(self) -> None:
        """Test clear_cache for all providers."""
        from markitai.providers.auth import AuthManager, AuthStatus

        manager = AuthManager()
        manager._cache["copilot"] = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="test",
            expires_at=None,
            error=None,
        )
        manager._cache["claude-agent"] = AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user="test",
            expires_at=None,
            error=None,
        )

        manager.clear_cache()

        assert len(manager._cache) == 0


class TestResolutionHints:
    """Tests for resolution hints."""

    def test_get_auth_resolution_hint_copilot(self) -> None:
        """Test resolution hint for copilot provider."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("copilot")
        assert "copilot" in hint.lower()
        assert "auth" in hint.lower() or "login" in hint.lower()

    def test_get_auth_resolution_hint_claude(self) -> None:
        """Test resolution hint for claude-agent provider."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("claude-agent")
        assert "claude" in hint.lower()
        assert "auth" in hint.lower() or "login" in hint.lower()

    def test_resolution_hint_is_platform_aware(self) -> None:
        """Test that resolution hints use platform-specific install commands."""
        from markitai.providers.auth import _build_resolution_hint

        with patch("markitai.providers.auth.sys") as mock_sys:
            mock_sys.platform = "win32"
            hint = _build_resolution_hint("claude-agent")
            assert "install.ps1" in hint or "iex" in hint

            mock_sys.platform = "linux"
            hint = _build_resolution_hint("claude-agent")
            assert "install.sh" in hint

    def test_get_auth_resolution_hint_unknown_provider(self) -> None:
        """Test resolution hint for unknown provider."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("unknown-provider")
        assert hint is not None
        assert len(hint) > 0

    def test_copilot_hint_mentions_gh_token(self) -> None:
        """Resolution hint for copilot should mention GH_TOKEN as alternative."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("copilot")
        assert "GH_TOKEN" in hint or "GITHUB_TOKEN" in hint

    def test_claude_hint_mentions_cloud_env_vars(self) -> None:
        """Resolution hint for claude should mention cloud provider env vars."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("claude-agent")
        assert "CLAUDE_CODE_USE_BEDROCK" in hint


class TestSDKAvailabilityHelpers:
    """Tests for SDK availability helper functions."""

    def test_is_copilot_sdk_available_when_installed(self) -> None:
        """Test _is_copilot_sdk_available returns True when SDK is installed."""
        from markitai.providers.auth import _is_copilot_sdk_available

        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert _is_copilot_sdk_available() is True

    def test_is_copilot_sdk_available_when_not_installed(self) -> None:
        """Test _is_copilot_sdk_available returns False when SDK is not installed."""
        from markitai.providers.auth import _is_copilot_sdk_available

        with patch("importlib.util.find_spec", return_value=None):
            assert _is_copilot_sdk_available() is False

    def test_is_claude_agent_sdk_available_when_installed(self) -> None:
        """Test _is_claude_agent_sdk_available returns True when SDK is installed."""
        from markitai.providers.auth import _is_claude_agent_sdk_available

        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert _is_claude_agent_sdk_available() is True

    def test_is_claude_agent_sdk_available_when_not_installed(self) -> None:
        """Test _is_claude_agent_sdk_available returns False when not installed."""
        from markitai.providers.auth import _is_claude_agent_sdk_available

        with patch("importlib.util.find_spec", return_value=None):
            assert _is_claude_agent_sdk_available() is False


class TestConfigFileAuth:
    """Tests for config file based authentication checks."""

    def test_check_copilot_config_auth_authenticated(self, tmp_path: Path) -> None:
        """Test _check_copilot_config_auth when authenticated."""
        from markitai.providers.auth import _check_copilot_config_auth

        # Create mock config file
        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "logged_in_users": [
                        {"host": "https://github.com", "login": "testuser"}
                    ]
                }
            )
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_copilot_config_auth()

        assert status.provider == "copilot"
        assert status.authenticated is True
        assert status.user == "testuser"

    def test_check_copilot_config_auth_not_authenticated(self, tmp_path: Path) -> None:
        """Test _check_copilot_config_auth when not authenticated."""
        from markitai.providers.auth import _check_copilot_config_auth

        # Create mock config file with empty users
        config_dir = tmp_path / ".copilot"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps({"logged_in_users": []}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_copilot_config_auth()

        assert status.authenticated is False

    def test_check_claude_credentials_auth_authenticated(self, tmp_path: Path) -> None:
        """Test _check_claude_credentials_auth when authenticated."""
        from markitai.providers.auth import _check_claude_credentials_auth

        # Create mock credentials file
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        expires_at = int((datetime.now().timestamp() + 3600) * 1000)
        creds_file.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "subscriptionType": "pro",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("markitai.providers.auth._claude_cli_email", return_value=None),
        ):
            status = _check_claude_credentials_auth()

        assert status.provider == "claude-agent"
        assert status.authenticated is True
        assert "pro" in (status.user or "")

    def test_claude_auth_extracts_email_via_cli(self, tmp_path: Path) -> None:
        """Claude auth extracts email by running `claude auth status`."""
        from markitai.providers.auth import _check_claude_credentials_auth

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        expires_at = int((datetime.now().timestamp() + 3600) * 1000)
        (claude_dir / ".credentials.json").write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "subscriptionType": "max",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "markitai.providers.auth._claude_cli_email",
                return_value="user@example.com",
            ),
        ):
            status = _check_claude_credentials_auth()

        assert status.authenticated is True
        assert status.user == "user@example.com"

    def test_claude_auth_falls_back_to_subscription_when_cli_fails(
        self, tmp_path: Path
    ) -> None:
        """Claude auth falls back to subscription type when CLI is unavailable."""
        from markitai.providers.auth import _check_claude_credentials_auth

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        expires_at = int((datetime.now().timestamp() + 3600) * 1000)
        (claude_dir / ".credentials.json").write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "subscriptionType": "pro",
                        "expiresAt": expires_at,
                    }
                }
            )
        )

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("markitai.providers.auth._claude_cli_email", return_value=None),
        ):
            status = _check_claude_credentials_auth()

        assert status.authenticated is True
        assert status.user == "subscription: pro"

    def test_check_claude_credentials_auth_no_token(self, tmp_path: Path) -> None:
        """Test _check_claude_credentials_auth when no token."""
        from markitai.providers.auth import _check_claude_credentials_auth

        # Create mock credentials file without token
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        creds_file.write_text(json.dumps({"claudeAiOauth": {}}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_claude_credentials_auth()

        assert status.authenticated is False
        assert "No access token" in (status.error or "")


class TestCopilotEnvVarAuth:
    """Tests for Copilot authentication via GH_TOKEN / GITHUB_TOKEN env vars.

    The Copilot CLI supports authenticating via personal access tokens set in
    GH_TOKEN or GITHUB_TOKEN environment variables (with "Copilot Requests"
    permission). The auth pre-check should detect these as valid auth.
    """

    def test_copilot_auth_detects_gh_token(self, tmp_path: Path) -> None:
        """GH_TOKEN env var should be detected as valid Copilot auth."""
        from markitai.providers.auth import _check_copilot_config_auth

        # No config file exists, but GH_TOKEN is set
        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"GH_TOKEN": "ghp_test123"}, clear=False),
        ):
            status = _check_copilot_config_auth()

        assert status.authenticated is True
        assert status.provider == "copilot"

    def test_copilot_auth_detects_github_token(self, tmp_path: Path) -> None:
        """GITHUB_TOKEN env var should be detected as valid Copilot auth."""
        from markitai.providers.auth import _check_copilot_config_auth

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test456"}, clear=False),
        ):
            status = _check_copilot_config_auth()

        assert status.authenticated is True
        assert status.provider == "copilot"

    def test_copilot_auth_gh_token_priority_over_missing_config(
        self, tmp_path: Path
    ) -> None:
        """GH_TOKEN should authenticate even when config file is absent."""
        from markitai.providers.auth import _check_copilot_config_auth

        # Explicitly no .copilot directory
        assert not (tmp_path / ".copilot").exists()

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"GH_TOKEN": "ghp_valid"}, clear=False),
        ):
            status = _check_copilot_config_auth()

        assert status.authenticated is True
        assert status.error is None


class TestClaudeEnvVarAuth:
    """Tests for Claude authentication via Bedrock/Vertex/Foundry env vars.

    Claude Code CLI supports authenticating via cloud provider env vars:
    - CLAUDE_CODE_USE_BEDROCK (AWS Bedrock)
    - CLAUDE_CODE_USE_VERTEX (Google Vertex AI)
    - CLAUDE_CODE_USE_FOUNDRY (Azure Foundry)
    The auth pre-check should detect these as valid auth.
    """

    def test_claude_auth_detects_bedrock_env_var(self, tmp_path: Path) -> None:
        """CLAUDE_CODE_USE_BEDROCK=1 should be detected as valid Claude auth."""
        from markitai.providers.auth import _check_claude_credentials_auth

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"CLAUDE_CODE_USE_BEDROCK": "1"}, clear=False),
        ):
            status = _check_claude_credentials_auth()

        assert status.authenticated is True
        assert status.provider == "claude-agent"

    def test_claude_auth_detects_vertex_env_var(self, tmp_path: Path) -> None:
        """CLAUDE_CODE_USE_VERTEX=1 should be detected as valid Claude auth."""
        from markitai.providers.auth import _check_claude_credentials_auth

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"CLAUDE_CODE_USE_VERTEX": "1"}, clear=False),
        ):
            status = _check_claude_credentials_auth()

        assert status.authenticated is True
        assert status.provider == "claude-agent"

    def test_claude_auth_detects_foundry_env_var(self, tmp_path: Path) -> None:
        """CLAUDE_CODE_USE_FOUNDRY=1 should be detected as valid Claude auth."""
        from markitai.providers.auth import _check_claude_credentials_auth

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"CLAUDE_CODE_USE_FOUNDRY": "1"}, clear=False),
        ):
            status = _check_claude_credentials_auth()

        assert status.authenticated is True
        assert status.provider == "claude-agent"

    def test_claude_auth_env_var_overrides_missing_credentials(
        self, tmp_path: Path
    ) -> None:
        """Bedrock env var should authenticate even without credentials file."""
        from markitai.providers.auth import _check_claude_credentials_auth

        assert not (tmp_path / ".claude").exists()

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch.dict("os.environ", {"CLAUDE_CODE_USE_BEDROCK": "1"}, clear=False),
        ):
            status = _check_claude_credentials_auth()

        assert status.authenticated is True
        assert status.error is None


class TestChatGPTAuth:
    """Tests for ChatGPT authentication checks."""

    def test_chatgpt_auth_authenticated(self, tmp_path: Path) -> None:
        """Test _check_chatgpt_auth when token exists."""
        from markitai.providers.auth import _check_chatgpt_auth

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        auth_file = auth_dir / "auth.json"
        auth_file.write_text(json.dumps({"access_token": "test-token-123"}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_chatgpt_auth()

        assert status.provider == "chatgpt"
        assert status.authenticated is True

    def test_chatgpt_auth_extracts_email_from_id_token(self, tmp_path: Path) -> None:
        """Test _check_chatgpt_auth extracts email from id_token JWT."""
        import base64

        from markitai.providers.auth import _check_chatgpt_auth

        payload = (
            base64.urlsafe_b64encode(json.dumps({"email": "user@example.com"}).encode())
            .rstrip(b"=")
            .decode()
        )
        id_token = f"header.{payload}.signature"

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text(
            json.dumps({"access_token": "tok", "id_token": id_token})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_chatgpt_auth()

        assert status.user == "user@example.com"

    def test_chatgpt_auth_no_file(self, tmp_path: Path) -> None:
        """Test _check_chatgpt_auth when auth file doesn't exist."""
        from markitai.providers.auth import _check_chatgpt_auth

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_chatgpt_auth()

        assert status.authenticated is False
        assert "not found" in (status.error or "").lower()

    def test_chatgpt_auth_no_token(self, tmp_path: Path) -> None:
        """Test _check_chatgpt_auth when file exists but no token."""
        from markitai.providers.auth import _check_chatgpt_auth

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        auth_file = auth_dir / "auth.json"
        auth_file.write_text(json.dumps({}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_chatgpt_auth()

        assert status.authenticated is False
        assert "No access token" in (status.error or "")

    def test_chatgpt_auth_invalid_json(self, tmp_path: Path) -> None:
        """Test _check_chatgpt_auth when file has invalid JSON."""
        from markitai.providers.auth import _check_chatgpt_auth

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        auth_file = auth_dir / "auth.json"
        auth_file.write_text("not json")

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_chatgpt_auth()

        assert status.authenticated is False
        assert "Failed to read" in (status.error or "")


class TestGeminiCLIManagedAuth:
    """Tests for Markitai-managed Gemini auth profiles."""

    def _write_managed_profile(
        self,
        home: Path,
        *,
        email: str = "gemini@example.com",
        project_id: str = "demo-project",
        auth_mode: str = "google-one",
    ) -> Path:
        """Create a Markitai-managed Gemini auth profile."""
        auth_dir = home / ".markitai" / "auth"
        auth_dir.mkdir(parents=True)
        profile_path = auth_dir / "gemini-profile.json"
        profile_data: dict[str, Any] = {
            "access_token": "managed-access-token",
            "refresh_token": "managed-refresh-token",
            "expiry_date": 1767225600000,
            "email": email,
            "project_id": project_id,
            "auth_mode": auth_mode,
            "source": "markitai",
        }
        profile_path.write_text(json.dumps(profile_data), encoding="utf-8")
        (auth_dir / "gemini-current.json").write_text(
            json.dumps({"credential_path": str(profile_path)}),
            encoding="utf-8",
        )
        return profile_path

    def test_check_gemini_cli_auth_prefers_markitai_managed_profile(
        self, tmp_path: Path
    ) -> None:
        """Managed Gemini auth should override shared Gemini CLI credentials."""
        from markitai.providers.auth import _check_gemini_cli_auth

        managed_path = self._write_managed_profile(tmp_path)
        shared_dir = tmp_path / ".gemini"
        shared_dir.mkdir()
        (shared_dir / "oauth_creds.json").write_text(
            json.dumps({"access_token": "shared-access-token"}),
            encoding="utf-8",
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.authenticated is True
        assert status.user == "gemini@example.com"
        assert status.details == {
            "source": "markitai",
            "project_id": "demo-project",
            "auth_mode": "google-one",
            "credential_path": str(managed_path),
        }

    def test_check_gemini_cli_auth_falls_back_to_shared_cli_credentials(
        self, tmp_path: Path
    ) -> None:
        """Fallback to ~/.gemini/oauth_creds.json when no managed profile exists."""
        from markitai.providers.auth import _check_gemini_cli_auth

        shared_dir = tmp_path / ".gemini"
        shared_dir.mkdir()
        shared_path = shared_dir / "oauth_creds.json"
        shared_path.write_text(
            json.dumps({"access_token": "shared-access-token"}),
            encoding="utf-8",
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.authenticated is True
        assert status.user == "gemini-cli"
        assert status.details == {
            "source": "gemini-cli",
            "project_id": None,
            "auth_mode": None,
            "credential_path": str(shared_path),
        }


class TestGeminiCLIAuth:
    """Tests for Gemini CLI authentication checks."""

    def test_gemini_cli_auth_authenticated(self, tmp_path: Path) -> None:
        """Test _check_gemini_cli_auth when credentials exist."""
        from markitai.providers.auth import _check_gemini_cli_auth

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        creds_file = gemini_dir / "oauth_creds.json"
        creds_file.write_text(
            json.dumps({"access_token": "ya29.xxx", "refresh_token": "1//xxx"})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.provider == "gemini-cli"
        assert status.authenticated is True

    def test_gemini_cli_auth_no_file(self, tmp_path: Path) -> None:
        """Test _check_gemini_cli_auth when credentials file doesn't exist."""
        from markitai.providers.auth import _check_gemini_cli_auth

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.authenticated is False
        assert "not found" in (status.error or "").lower()

    def test_gemini_cli_auth_no_token(self, tmp_path: Path) -> None:
        """Test _check_gemini_cli_auth when file exists but no token."""
        from markitai.providers.auth import _check_gemini_cli_auth

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        creds_file = gemini_dir / "oauth_creds.json"
        creds_file.write_text(json.dumps({"refresh_token": "1//xxx"}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.authenticated is False
        assert "No access token" in (status.error or "")

    def test_gemini_cli_auth_invalid_json(self, tmp_path: Path) -> None:
        """Test _check_gemini_cli_auth when file has invalid JSON."""
        from markitai.providers.auth import _check_gemini_cli_auth

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        creds_file = gemini_dir / "oauth_creds.json"
        creds_file.write_text("{broken")

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.authenticated is False
        assert "Failed to read" in (status.error or "")

    def test_gemini_cli_shared_creds_extracts_email_from_id_token(
        self, tmp_path: Path
    ) -> None:
        """Shared creds with id_token should show email instead of 'gemini-cli'."""
        import base64

        from markitai.providers.auth import _check_gemini_cli_auth

        payload = (
            base64.urlsafe_b64encode(
                json.dumps({"email": "gemini-user@gmail.com"}).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        id_token = f"header.{payload}.signature"

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "oauth_creds.json").write_text(
            json.dumps({"access_token": "ya29.xxx", "id_token": id_token})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_gemini_cli_auth()

        assert status.authenticated is True
        assert status.user == "gemini-user@gmail.com"


class TestAuthManagerNewProviders:
    """Tests for AuthManager with chatgpt and gemini-cli providers."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_chatgpt(self, tmp_path: Path) -> None:
        """Test AuthManager.check_auth for chatgpt provider."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        auth_dir = tmp_path / ".config" / "litellm" / "chatgpt"
        auth_dir.mkdir(parents=True)
        (auth_dir / "auth.json").write_text(json.dumps({"access_token": "tok"}))

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("chatgpt")

        assert status.provider == "chatgpt"
        assert status.authenticated is True

    @pytest.mark.asyncio
    async def test_check_auth_gemini_cli(self, tmp_path: Path) -> None:
        """Test AuthManager.check_auth for gemini-cli provider."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        gemini_dir = tmp_path / ".gemini"
        gemini_dir.mkdir()
        (gemini_dir / "oauth_creds.json").write_text(
            json.dumps({"access_token": "ya29.xxx"})
        )

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("gemini-cli")

        assert status.provider == "gemini-cli"
        assert status.authenticated is True


class TestNewResolutionHints:
    """Tests for resolution hints for new providers."""

    def test_chatgpt_resolution_hint(self) -> None:
        """Test resolution hint for chatgpt provider."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("chatgpt")
        assert "chatgpt" in hint.lower() or "oauth" in hint.lower()

    def test_gemini_cli_resolution_hint(self) -> None:
        """Test resolution hint for gemini-cli provider."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("gemini-cli")
        assert "gemini" in hint.lower()


class TestAttemptLogin:
    """Tests for attempt_login() dispatcher and login functions."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    async def test_copilot_login_calls_subprocess(self) -> None:
        """attempt_login('copilot') shells out to 'copilot login'."""
        from markitai.providers.auth import attempt_login

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        with (
            patch(
                "markitai.providers.auth._resolve_cli_path",
                return_value="/usr/bin/copilot",
            ),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
            patch(
                "markitai.providers.auth._check_copilot_config_auth",
                return_value=AuthStatus(
                    provider="copilot",
                    authenticated=True,
                    user="testuser",
                    expires_at=None,
                    error=None,
                ),
            ),
        ):
            result = await attempt_login("copilot")

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "copilot" in args[0]
        assert "login" in args
        assert result.authenticated is True

    async def test_copilot_login_cli_not_found(self) -> None:
        """attempt_login('copilot') returns error if CLI not found."""
        from markitai.providers.auth import attempt_login

        with patch("markitai.providers.auth._resolve_cli_path", return_value=None):
            result = await attempt_login("copilot")

        assert result.authenticated is False
        assert "not found" in (result.error or "").lower()

    async def test_claude_login_calls_subprocess(self) -> None:
        """attempt_login('claude-agent') shells out to 'claude auth login'."""
        from markitai.providers.auth import attempt_login

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        with (
            patch(
                "markitai.providers.auth._resolve_cli_path",
                return_value="/usr/bin/claude",
            ),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
            patch(
                "markitai.providers.auth._check_claude_credentials_auth",
                return_value=AuthStatus(
                    provider="claude-agent",
                    authenticated=True,
                    user="subscription: max",
                    expires_at=None,
                    error=None,
                ),
            ),
        ):
            result = await attempt_login("claude-agent")

        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert "claude" in args[0]
        assert result.authenticated is True

    async def test_gemini_login_calls_alogin(self) -> None:
        """attempt_login('gemini-cli') calls GeminiCLIProvider.alogin()."""
        from markitai.providers.auth import attempt_login

        mock_record = MagicMock()
        mock_record.email = "test@example.com"

        with (
            patch("markitai.providers.gemini_cli.GeminiCLIProvider") as MockProvider,
            patch(
                "markitai.providers.auth._check_gemini_cli_auth",
                return_value=AuthStatus(
                    provider="gemini-cli",
                    authenticated=True,
                    user="test@example.com",
                    expires_at=None,
                    error=None,
                ),
            ),
        ):
            MockProvider.return_value.alogin = AsyncMock(return_value=mock_record)
            result = await attempt_login("gemini-cli")

        assert result.authenticated is True

    async def test_chatgpt_login_returns_auto_login_info(self) -> None:
        """attempt_login('chatgpt') returns info about auto-login."""
        from markitai.providers.auth import attempt_login

        result = await attempt_login("chatgpt")
        assert result.provider == "chatgpt"
        assert result.details is not None
        assert result.details.get("auto_login") is True

    async def test_unknown_provider_returns_error(self) -> None:
        """attempt_login with unknown provider returns error."""
        from markitai.providers.auth import attempt_login

        result = await attempt_login("nonexistent")
        assert result.authenticated is False
        assert "unknown" in (result.error or "").lower()

    async def test_subprocess_failure_returns_error(self) -> None:
        """Login handles subprocess failures gracefully."""
        from markitai.providers.auth import attempt_login

        mock_proc = AsyncMock()
        mock_proc.returncode = 1

        with (
            patch(
                "markitai.providers.auth._resolve_cli_path",
                return_value="/usr/bin/copilot",
            ),
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
        ):
            result = await attempt_login("copilot")

        assert result.authenticated is False
        assert "failed" in (result.error or "").lower()


class TestCanAttemptLogin:
    """Tests for can_attempt_login() guard function."""

    def test_gemini_cli_with_oauthlib(self) -> None:
        """Returns True when google_auth_oauthlib is importable."""
        from markitai.providers.auth import can_attempt_login

        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert can_attempt_login("gemini-cli") is True

    def test_gemini_cli_without_oauthlib(self) -> None:
        """Returns False when google_auth_oauthlib is not installed."""
        from markitai.providers.auth import can_attempt_login

        with patch("importlib.util.find_spec", return_value=None):
            assert can_attempt_login("gemini-cli") is False

    def test_claude_agent_with_cli(self) -> None:
        """Returns True when claude CLI is found."""
        from markitai.providers.auth import can_attempt_login

        with patch(
            "markitai.providers.auth._resolve_cli_path", return_value="/usr/bin/claude"
        ):
            assert can_attempt_login("claude-agent") is True

    def test_claude_agent_without_cli(self) -> None:
        """Returns False when claude CLI is not found."""
        from markitai.providers.auth import can_attempt_login

        with patch("markitai.providers.auth._resolve_cli_path", return_value=None):
            assert can_attempt_login("claude-agent") is False

    def test_copilot_with_cli(self) -> None:
        """Returns True when copilot CLI is found."""
        from markitai.providers.auth import can_attempt_login

        with patch(
            "markitai.providers.auth._resolve_cli_path", return_value="/usr/bin/copilot"
        ):
            assert can_attempt_login("copilot") is True

    def test_copilot_without_cli(self) -> None:
        """Returns False when copilot CLI is not found."""
        from markitai.providers.auth import can_attempt_login

        with patch("markitai.providers.auth._resolve_cli_path", return_value=None):
            assert can_attempt_login("copilot") is False

    def test_chatgpt_always_true(self) -> None:
        """ChatGPT auto-authenticates, always returns True."""
        from markitai.providers.auth import can_attempt_login

        assert can_attempt_login("chatgpt") is True

    def test_unknown_provider(self) -> None:
        """Unknown providers return False."""
        from markitai.providers.auth import can_attempt_login

        assert can_attempt_login("nonexistent") is False
