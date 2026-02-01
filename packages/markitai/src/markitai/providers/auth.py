"""Authentication manager for local LLM providers.

This module provides authentication status checking and caching for local
providers (claude-agent and copilot). It enables checking authentication
status before making API calls and provides actionable resolution hints.

Usage:
    from markitai.providers.auth import AuthManager

    manager = AuthManager()
    status = await manager.check_auth("copilot")
    if not status.authenticated:
        print(f"Error: {status.error}")
        print(get_auth_resolution_hint("copilot"))
"""

from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class AuthStatus:
    """Authentication status for a provider.

    This is a frozen (immutable) dataclass containing the result of
    an authentication check.

    Attributes:
        provider: Provider name (e.g., "claude-agent", "copilot")
        authenticated: Whether the user is authenticated
        user: Username or email if available
        expires_at: Token expiration time if known
        error: Error message if not authenticated
    """

    provider: str
    authenticated: bool
    user: str | None
    expires_at: datetime | None
    error: str | None


# Resolution hints for each provider
_RESOLUTION_HINTS: dict[str, str] = {
    "claude-agent": (
        "Run 'claude auth login' to authenticate with Claude Code CLI.\n"
        "If Claude Code CLI is not installed, install it with:\n"
        "  curl -fsSL https://claude.ai/install.sh | bash"
    ),
    "copilot": (
        "Run 'copilot auth login' to authenticate with GitHub Copilot.\n"
        "If Copilot CLI is not installed, install it with:\n"
        "  curl -fsSL https://gh.io/copilot-install | bash"
    ),
}

_DEFAULT_RESOLUTION_HINT = "Please authenticate with the provider CLI."


def get_auth_resolution_hint(provider: str) -> str:
    """Get a user-friendly resolution hint for authentication issues.

    Args:
        provider: Provider name (e.g., "claude-agent", "copilot")

    Returns:
        Resolution hint string with instructions to authenticate
    """
    return _RESOLUTION_HINTS.get(provider, _DEFAULT_RESOLUTION_HINT)


def _is_copilot_sdk_available() -> bool:
    """Check if the Copilot SDK is installed.

    Returns:
        True if the copilot package is available
    """
    return importlib.util.find_spec("copilot") is not None


def _is_claude_agent_sdk_available() -> bool:
    """Check if the Claude Agent SDK is installed.

    Returns:
        True if the claude_agent_sdk package is available
    """
    return importlib.util.find_spec("claude_agent_sdk") is not None


def _get_copilot_client() -> object:
    """Get a Copilot SDK client instance.

    Returns:
        Copilot Client instance

    Raises:
        ImportError: If copilot SDK is not installed
    """
    import copilot

    return copilot.Client()


async def _check_copilot_sdk_auth() -> AuthStatus:
    """Check authentication status via Copilot SDK.

    Returns:
        AuthStatus with authentication result
    """
    try:
        client = _get_copilot_client()
        auth_status = client.get_auth_status()  # type: ignore[union-attr]

        return AuthStatus(
            provider="copilot",
            authenticated=auth_status.authenticated,
            user=getattr(auth_status, "user", None),
            expires_at=getattr(auth_status, "expires_at", None),
            error=None if auth_status.authenticated else "Not authenticated",
        )
    except Exception as e:
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Failed to check auth status: {e}",
        )


async def _run_claude_doctor() -> tuple[bool, str | None]:
    """Run 'claude doctor' to check Claude Code CLI health.

    Returns:
        Tuple of (success, error_message)
        - success: True if claude doctor passed
        - error_message: Error message if failed, None if success
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "claude",
            "doctor",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return (True, None)
        else:
            error_msg = stderr.decode().strip() or stdout.decode().strip()
            return (False, error_msg or "claude doctor failed")

    except FileNotFoundError:
        return (False, "Claude Code CLI not found. Please install it first.")
    except Exception as e:
        return (False, f"Failed to run claude doctor: {e}")


class AuthManager:
    """Singleton manager for provider authentication status.

    This class provides cached authentication checking for local providers.
    It uses a singleton pattern to ensure consistent caching across the
    application.

    Usage:
        manager = AuthManager()
        status = await manager.check_auth("copilot")
        if not status.authenticated:
            print(status.error)
    """

    _instance: AuthManager | None = None

    def __new__(cls) -> AuthManager:
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def __init__(self) -> None:
        """Initialize the AuthManager.

        Note: The cache is initialized in __new__ to preserve state
        across multiple __init__ calls on the singleton.
        """
        # _cache is initialized in __new__
        pass

    @property
    def _cache(self) -> dict[str, AuthStatus]:
        """Get the authentication status cache."""
        return self.__dict__.get("_cache", {})

    @_cache.setter
    def _cache(self, value: dict[str, AuthStatus]) -> None:
        """Set the authentication status cache."""
        self.__dict__["_cache"] = value

    async def check_auth(
        self, provider: str, force_refresh: bool = False
    ) -> AuthStatus:
        """Check authentication status for a provider.

        Results are cached by default. Use force_refresh=True to bypass cache.

        Args:
            provider: Provider name ("claude-agent" or "copilot")
            force_refresh: If True, bypass cache and check fresh

        Returns:
            AuthStatus with authentication result
        """
        # Check cache first (unless force_refresh)
        if not force_refresh and provider in self._cache:
            return self._cache[provider]

        # Perform the actual check
        status = await self._check_provider(provider)

        # Cache the result
        self._cache[provider] = status

        return status

    async def _check_provider(self, provider: str) -> AuthStatus:
        """Internal method to check authentication for a specific provider.

        Args:
            provider: Provider name

        Returns:
            AuthStatus with authentication result
        """
        if provider == "copilot":
            return await self._check_copilot()
        elif provider == "claude-agent":
            return await self._check_claude()
        else:
            return AuthStatus(
                provider=provider,
                authenticated=False,
                user=None,
                expires_at=None,
                error=f"Unknown provider: {provider}",
            )

    async def _check_copilot(self) -> AuthStatus:
        """Check authentication status for Copilot provider.

        Returns:
            AuthStatus with authentication result
        """
        if not _is_copilot_sdk_available():
            return AuthStatus(
                provider="copilot",
                authenticated=False,
                user=None,
                expires_at=None,
                error="Copilot SDK not installed. Run: uv add github-copilot-sdk",
            )

        return await _check_copilot_sdk_auth()

    async def _check_claude(self) -> AuthStatus:
        """Check authentication status for Claude Agent provider.

        Returns:
            AuthStatus with authentication result
        """
        if not _is_claude_agent_sdk_available():
            return AuthStatus(
                provider="claude-agent",
                authenticated=False,
                user=None,
                expires_at=None,
                error="Claude Agent SDK not installed. Run: uv add claude-agent-sdk",
            )

        success, error = await _run_claude_doctor()

        return AuthStatus(
            provider="claude-agent",
            authenticated=success,
            user=None,  # claude doctor doesn't return user info
            expires_at=None,
            error=error,
        )

    def clear_cache(self, provider: str | None = None) -> None:
        """Clear the authentication status cache.

        Args:
            provider: If specified, only clear cache for this provider.
                     If None, clear cache for all providers.
        """
        if provider is None:
            self._cache.clear()
        elif provider in self._cache:
            del self._cache[provider]


__all__ = [
    "AuthStatus",
    "AuthManager",
    "get_auth_resolution_hint",
    "_is_copilot_sdk_available",
    "_is_claude_agent_sdk_available",
    "_check_copilot_sdk_auth",
    "_run_claude_doctor",
]
