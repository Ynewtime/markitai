"""Unit tests for provider authentication manager.

These tests cover the authentication status checking and caching functionality
for local LLM providers (claude-agent and copilot).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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
    async def test_check_auth_copilot_authenticated(self) -> None:
        """Test check_auth for copilot when authenticated via SDK."""
        from markitai.providers.auth import AuthManager, AuthStatus

        manager = AuthManager()

        mock_auth_status = MagicMock()
        mock_auth_status.authenticated = True
        mock_auth_status.user = "user@github.com"
        mock_auth_status.expires_at = None

        with (
            patch(
                "markitai.providers.auth._is_copilot_sdk_available", return_value=True
            ),
            patch(
                "markitai.providers.auth._check_copilot_sdk_auth",
                new_callable=AsyncMock,
                return_value=AuthStatus(
                    provider="copilot",
                    authenticated=True,
                    user="user@github.com",
                    expires_at=None,
                    error=None,
                ),
            ),
        ):
            status = await manager.check_auth("copilot")

        assert status.provider == "copilot"
        assert status.authenticated is True
        assert status.user == "user@github.com"

    @pytest.mark.asyncio
    async def test_check_auth_copilot_not_authenticated(self) -> None:
        """Test check_auth for copilot when not authenticated."""
        from markitai.providers.auth import AuthManager, AuthStatus

        manager = AuthManager()

        with (
            patch(
                "markitai.providers.auth._is_copilot_sdk_available", return_value=True
            ),
            patch(
                "markitai.providers.auth._check_copilot_sdk_auth",
                new_callable=AsyncMock,
                return_value=AuthStatus(
                    provider="copilot",
                    authenticated=False,
                    user=None,
                    expires_at=None,
                    error="Not authenticated",
                ),
            ),
        ):
            status = await manager.check_auth("copilot")

        assert status.authenticated is False
        assert status.error == "Not authenticated"

    @pytest.mark.asyncio
    async def test_check_auth_claude_authenticated(self) -> None:
        """Test check_auth for claude when authenticated."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with (
            patch(
                "markitai.providers.auth._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.auth._run_claude_doctor",
                new_callable=AsyncMock,
                return_value=(True, None),
            ),
        ):
            status = await manager.check_auth("claude-agent")

        assert status.provider == "claude-agent"
        assert status.authenticated is True

    @pytest.mark.asyncio
    async def test_check_auth_claude_not_authenticated(self) -> None:
        """Test check_auth for claude when not authenticated."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with (
            patch(
                "markitai.providers.auth._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.auth._run_claude_doctor",
                new_callable=AsyncMock,
                return_value=(False, "Authentication required"),
            ),
        ):
            status = await manager.check_auth("claude-agent")

        assert status.authenticated is False
        assert status.error is not None


class TestAuthManagerSDKNotInstalled:
    """Tests for SDK not installed scenarios."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_copilot_sdk_not_installed(self) -> None:
        """Test check_auth when Copilot SDK is not installed."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with patch(
            "markitai.providers.auth._is_copilot_sdk_available", return_value=False
        ):
            status = await manager.check_auth("copilot")

        assert status.authenticated is False
        assert status.error is not None
        assert "SDK" in status.error or "not installed" in status.error.lower()

    @pytest.mark.asyncio
    async def test_check_auth_claude_sdk_not_installed(self) -> None:
        """Test check_auth when Claude Agent SDK is not installed."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with patch(
            "markitai.providers.auth._is_claude_agent_sdk_available", return_value=False
        ):
            status = await manager.check_auth("claude-agent")

        assert status.authenticated is False
        assert status.error is not None
        assert "SDK" in status.error or "not installed" in status.error.lower()


class TestAuthManagerCache:
    """Tests for AuthManager cache behavior."""

    @pytest.fixture(autouse=True)
    def reset_auth_manager(self) -> None:
        """Reset AuthManager singleton before each test."""
        from markitai.providers.auth import AuthManager

        AuthManager._instance = None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_check_auth_uses_cache(self) -> None:
        """Test that check_auth uses cached result on second call."""
        from markitai.providers.auth import AuthManager, AuthStatus

        manager = AuthManager()

        mock_check = AsyncMock(
            return_value=AuthStatus(
                provider="copilot",
                authenticated=True,
                user="user@github.com",
                expires_at=None,
                error=None,
            )
        )

        with (
            patch(
                "markitai.providers.auth._is_copilot_sdk_available", return_value=True
            ),
            patch("markitai.providers.auth._check_copilot_sdk_auth", mock_check),
        ):
            status1 = await manager.check_auth("copilot")
            status2 = await manager.check_auth("copilot")

        # Should only call the check function once (cached on second call)
        assert mock_check.call_count == 1
        assert status1.authenticated is True
        assert status2.authenticated is True

    @pytest.mark.asyncio
    async def test_check_auth_force_refresh_bypasses_cache(self) -> None:
        """Test that force_refresh bypasses the cache."""
        from markitai.providers.auth import AuthManager, AuthStatus

        manager = AuthManager()

        mock_check = AsyncMock(
            return_value=AuthStatus(
                provider="copilot",
                authenticated=True,
                user="user@github.com",
                expires_at=None,
                error=None,
            )
        )

        with (
            patch(
                "markitai.providers.auth._is_copilot_sdk_available", return_value=True
            ),
            patch("markitai.providers.auth._check_copilot_sdk_auth", mock_check),
        ):
            await manager.check_auth("copilot")
            await manager.check_auth("copilot", force_refresh=True)

        # Should call the check function twice (force refresh)
        assert mock_check.call_count == 2

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


class TestClaudeDoctor:
    """Tests for _run_claude_doctor helper."""

    @pytest.mark.asyncio
    async def test_run_claude_doctor_success(self) -> None:
        """Test _run_claude_doctor when claude doctor succeeds."""
        from markitai.providers.auth import _run_claude_doctor

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"All checks passed", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            success, error = await _run_claude_doctor()

        assert success is True
        assert error is None

    @pytest.mark.asyncio
    async def test_run_claude_doctor_failure(self) -> None:
        """Test _run_claude_doctor when claude doctor fails."""
        from markitai.providers.auth import _run_claude_doctor

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Authentication failed")
        )

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_process,
        ):
            success, error = await _run_claude_doctor()

        assert success is False
        assert error is not None

    @pytest.mark.asyncio
    async def test_run_claude_doctor_command_not_found(self) -> None:
        """Test _run_claude_doctor when claude command is not found."""
        from markitai.providers.auth import _run_claude_doctor

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("claude not found"),
        ):
            success, error = await _run_claude_doctor()

        assert success is False
        assert error is not None
        assert "not found" in error.lower() or "not installed" in error.lower()


class TestCopilotSDKAuth:
    """Tests for _check_copilot_sdk_auth helper."""

    @pytest.mark.asyncio
    async def test_check_copilot_sdk_auth_authenticated(self) -> None:
        """Test _check_copilot_sdk_auth when authenticated."""
        from markitai.providers.auth import _check_copilot_sdk_auth

        mock_client = MagicMock()
        mock_auth_status = MagicMock()
        mock_auth_status.authenticated = True
        mock_auth_status.user = "user@github.com"
        mock_auth_status.expires_at = None
        mock_client.get_auth_status = MagicMock(return_value=mock_auth_status)

        with (
            patch.dict(
                "sys.modules", {"copilot": MagicMock(Client=lambda: mock_client)}
            ),
            patch(
                "markitai.providers.auth._get_copilot_client",
                return_value=mock_client,
            ),
        ):
            status = await _check_copilot_sdk_auth()

        assert status.provider == "copilot"
        assert status.authenticated is True
        assert status.user == "user@github.com"

    @pytest.mark.asyncio
    async def test_check_copilot_sdk_auth_not_authenticated(self) -> None:
        """Test _check_copilot_sdk_auth when not authenticated."""
        from markitai.providers.auth import _check_copilot_sdk_auth

        mock_client = MagicMock()
        mock_auth_status = MagicMock()
        mock_auth_status.authenticated = False
        mock_auth_status.user = None
        mock_auth_status.expires_at = None
        mock_client.get_auth_status = MagicMock(return_value=mock_auth_status)

        with patch(
            "markitai.providers.auth._get_copilot_client",
            return_value=mock_client,
        ):
            status = await _check_copilot_sdk_auth()

        assert status.authenticated is False
