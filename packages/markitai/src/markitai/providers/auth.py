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

import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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


# Platform-specific install commands for CLI tools
_CLI_INSTALL_COMMANDS: dict[str, dict[str, str]] = {
    "claude": {
        "darwin": "curl -fsSL https://claude.ai/install.sh | bash",
        "linux": "curl -fsSL https://claude.ai/install.sh | bash",
        "win32": "irm https://claude.ai/install.ps1 | iex",
    },
    "copilot": {
        "darwin": "curl -fsSL https://gh.io/copilot-install | bash",
        "linux": "curl -fsSL https://gh.io/copilot-install | bash",
        "win32": "winget install GitHub.Copilot",
    },
}


def _get_cli_install_cmd(tool: str) -> str:
    """Get platform-specific CLI install command."""
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    commands = _CLI_INSTALL_COMMANDS.get(tool, {})
    return commands.get(platform, commands.get("linux", f"Install {tool} CLI"))


def _build_resolution_hint(provider: str) -> str:
    """Build platform-aware resolution hint for a provider."""
    if provider == "claude-agent":
        return (
            "Run 'claude auth login' to authenticate with Claude Code CLI.\n"
            "Alternatively, set a cloud provider env var:\n"
            "  CLAUDE_CODE_USE_BEDROCK=1, CLAUDE_CODE_USE_VERTEX=1, "
            "or CLAUDE_CODE_USE_FOUNDRY=1\n"
            "If Claude Code CLI is not installed, install it with:\n"
            f"  {_get_cli_install_cmd('claude')}"
        )
    elif provider == "copilot":
        return (
            "Run 'copilot auth login' to authenticate with GitHub Copilot.\n"
            "Alternatively, set GH_TOKEN or GITHUB_TOKEN env var "
            "(requires 'Copilot Requests' permission).\n"
            "If Copilot CLI is not installed, install it with:\n"
            f"  {_get_cli_install_cmd('copilot')}"
        )
    elif provider == "chatgpt":
        return (
            "ChatGPT provider uses OAuth Device Code Flow.\n"
            "Run any chatgpt/ model to trigger automatic login,\n"
            "or install and authenticate via: pip install litellm && litellm --model chatgpt/gpt-5.2"
        )
    elif provider == "gemini-cli":
        return (
            "Install Gemini CLI and run 'gemini login' to authenticate,\n"
            "or markitai will trigger OAuth login automatically on first use.\n"
            "Requires: uv add 'markitai\\[gemini-cli]'"
        )
    return "Please authenticate with the provider CLI."


def get_auth_resolution_hint(provider: str) -> str:
    """Get a user-friendly, platform-aware resolution hint for authentication issues.

    Args:
        provider: Provider name (e.g., "claude-agent", "copilot")

    Returns:
        Resolution hint string with instructions to authenticate
    """
    return _build_resolution_hint(provider)


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


def _check_copilot_config_auth() -> AuthStatus:
    """Check Copilot authentication by reading config file or env vars.

    Checks (in order):
    1. GH_TOKEN / GITHUB_TOKEN environment variables (personal access token)
    2. ~/.copilot/config.json for logged_in_users

    Returns:
        AuthStatus with authentication result
    """
    # Check env var auth first (GH_TOKEN / GITHUB_TOKEN)
    import os

    gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if gh_token:
        return AuthStatus(
            provider="copilot",
            authenticated=True,
            user="token",
            expires_at=None,
            error=None,
        )

    config_path = Path.home() / ".copilot" / "config.json"

    if not config_path.exists():
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Config file not found (~/.copilot/config.json)",
        )

    try:
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        logged_in_users = config.get("logged_in_users", [])

        if logged_in_users:
            # Get first logged in user
            user_info = logged_in_users[0]
            username = user_info.get("login", "unknown")
            return AuthStatus(
                provider="copilot",
                authenticated=True,
                user=username,
                expires_at=None,
                error=None,
            )
        else:
            return AuthStatus(
                provider="copilot",
                authenticated=False,
                user=None,
                expires_at=None,
                error="No logged in users found",
            )
    except Exception as e:
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Failed to read config: {e}",
        )


def _check_claude_credentials_auth() -> AuthStatus:
    """Check Claude authentication by reading credentials file or env vars.

    Checks (in order):
    1. CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX / CLAUDE_CODE_USE_FOUNDRY
       environment variables (cloud provider auth)
    2. ~/.claude/.credentials.json for OAuth tokens

    Returns:
        AuthStatus with authentication result
    """
    # Check cloud provider env var auth first
    import os

    cloud_providers = {
        "CLAUDE_CODE_USE_BEDROCK": "bedrock",
        "CLAUDE_CODE_USE_VERTEX": "vertex",
        "CLAUDE_CODE_USE_FOUNDRY": "foundry",
    }
    for env_var, cloud_name in cloud_providers.items():
        if os.environ.get(env_var):
            return AuthStatus(
                provider="claude-agent",
                authenticated=True,
                user=f"cloud: {cloud_name}",
                expires_at=None,
                error=None,
            )

    credentials_path = Path.home() / ".claude" / ".credentials.json"

    if not credentials_path.exists():
        return AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Credentials file not found (~/.claude/.credentials.json)",
        )

    try:
        creds: dict[str, Any] = json.loads(credentials_path.read_text(encoding="utf-8"))
        oauth_data = creds.get("claudeAiOauth", {})

        if oauth_data.get("accessToken"):
            # Check expiration
            expires_at_ms = oauth_data.get("expiresAt")
            expires_at = None
            is_expired = False

            if expires_at_ms:
                expires_at = datetime.fromtimestamp(expires_at_ms / 1000)
                is_expired = datetime.now() > expires_at

            if is_expired:
                # Claude CLI uses OAuth with refresh tokens — an expired access
                # token does NOT mean auth is broken if a refresh token exists.
                # The CLI refreshes automatically on next use.
                has_refresh = bool(oauth_data.get("refreshToken"))
                if not has_refresh:
                    return AuthStatus(
                        provider="claude-agent",
                        authenticated=False,
                        user=None,
                        expires_at=expires_at,
                        error="Token expired (no refresh token)",
                    )

            # Get subscription type as user info
            subscription = oauth_data.get("subscriptionType", "unknown")
            return AuthStatus(
                provider="claude-agent",
                authenticated=True,
                user=f"subscription: {subscription}",
                expires_at=expires_at,
                error=None,
            )
        else:
            return AuthStatus(
                provider="claude-agent",
                authenticated=False,
                user=None,
                expires_at=None,
                error="No access token found",
            )
    except Exception as e:
        return AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Failed to read credentials: {e}",
        )


def _check_chatgpt_auth() -> AuthStatus:
    """Check ChatGPT authentication by reading LiteLLM's chatgpt auth file.

    Checks ~/.config/litellm/chatgpt/auth.json for access_token.

    Returns:
        AuthStatus with authentication result
    """
    auth_path = Path.home() / ".config" / "litellm" / "chatgpt" / "auth.json"

    if not auth_path.exists():
        return AuthStatus(
            provider="chatgpt",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Auth file not found (~/.config/litellm/chatgpt/auth.json)",
        )

    try:
        data: dict[str, Any] = json.loads(auth_path.read_text(encoding="utf-8"))
        access_token = data.get("access_token")

        if access_token:
            return AuthStatus(
                provider="chatgpt",
                authenticated=True,
                user="chatgpt",
                expires_at=None,
                error=None,
            )
        else:
            return AuthStatus(
                provider="chatgpt",
                authenticated=False,
                user=None,
                expires_at=None,
                error="No access token found",
            )
    except Exception as e:
        return AuthStatus(
            provider="chatgpt",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Failed to read auth file: {e}",
        )


def _check_gemini_cli_auth() -> AuthStatus:
    """Check Gemini CLI authentication by reading OAuth credentials.

    Checks ~/.gemini/oauth_creds.json for access_token.

    Returns:
        AuthStatus with authentication result
    """
    creds_path = Path.home() / ".gemini" / "oauth_creds.json"

    if not creds_path.exists():
        return AuthStatus(
            provider="gemini-cli",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Credentials file not found (~/.gemini/oauth_creds.json)",
        )

    try:
        data: dict[str, Any] = json.loads(creds_path.read_text(encoding="utf-8"))
        access_token = data.get("access_token")

        if access_token:
            return AuthStatus(
                provider="gemini-cli",
                authenticated=True,
                user="gemini-cli",
                expires_at=None,
                error=None,
            )
        else:
            return AuthStatus(
                provider="gemini-cli",
                authenticated=False,
                user=None,
                expires_at=None,
                error="No access token found",
            )
    except Exception as e:
        return AuthStatus(
            provider="gemini-cli",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Failed to read credentials: {e}",
        )


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
            return self._check_copilot()
        elif provider == "claude-agent":
            return self._check_claude()
        elif provider == "chatgpt":
            return _check_chatgpt_auth()
        elif provider == "gemini-cli":
            return _check_gemini_cli_auth()
        else:
            return AuthStatus(
                provider=provider,
                authenticated=False,
                user=None,
                expires_at=None,
                error=f"Unknown provider: {provider}",
            )

    def _check_copilot(self) -> AuthStatus:
        """Check authentication status for Copilot provider.

        Checks ~/.copilot/config.json for logged_in_users.

        Returns:
            AuthStatus with authentication result
        """
        return _check_copilot_config_auth()

    def _check_claude(self) -> AuthStatus:
        """Check authentication status for Claude Agent provider.

        Checks ~/.claude/.credentials.json for valid access token.

        Returns:
            AuthStatus with authentication result
        """
        return _check_claude_credentials_auth()

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
    "_check_copilot_config_auth",
    "_check_claude_credentials_auth",
    "_check_chatgpt_auth",
    "_check_gemini_cli_auth",
]
