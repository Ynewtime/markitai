"""Unit tests for provider authentication manager.

These tests cover the authentication status checking and caching functionality
for local LLM providers (claude-agent and copilot).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
        assert "No logged in users" in (status.error or "")

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

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = await manager.check_auth("claude-agent")

        assert status.provider == "claude-agent"
        assert status.authenticated is True
        assert "max" in (status.user or "")

    @pytest.mark.asyncio
    async def test_check_auth_claude_token_expired(self, tmp_path: Path) -> None:
        """Test check_auth for claude when token is expired."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Create mock credentials file with expired token
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

    def test_get_auth_resolution_hint_unknown_provider(self) -> None:
        """Test resolution hint for unknown provider."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("unknown-provider")
        assert hint is not None
        assert len(hint) > 0


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

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = _check_claude_credentials_auth()

        assert status.provider == "claude-agent"
        assert status.authenticated is True
        assert "pro" in (status.user or "")

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
