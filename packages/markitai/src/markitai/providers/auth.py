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

import base64
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


def _email_from_google_userinfo(access_token: str) -> str | None:
    """Fetch email from Google userinfo API using an access token.

    Args:
        access_token: A valid Google OAuth2 access token.

    Returns:
        Email address if available, None on any failure.
    """
    try:
        import httpx

        resp = httpx.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            params={"alt": "json"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            email = data.get("email")
            return email if isinstance(email, str) and email else None
    except Exception:
        pass
    return None


def _email_from_jwt(id_token: str) -> str | None:
    """Extract email from a JWT id_token without external libraries.

    Args:
        id_token: A JWT string (header.payload.signature).

    Returns:
        Email address if found, None otherwise.
    """
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        # Add base64 padding
        payload += "=" * (4 - len(payload) % 4)
        data: dict[str, Any] = json.loads(base64.urlsafe_b64decode(payload))
        email = data.get("email")
        return email if isinstance(email, str) and email else None
    except Exception:
        return None


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
        details: Provider-specific metadata for diagnostics/UI
    """

    provider: str
    authenticated: bool
    user: str | None
    expires_at: datetime | None
    error: str | None
    details: dict[str, Any] | None = None


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
            "Run 'copilot login' to authenticate with GitHub Copilot.\n"
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
            "Run 'markitai auth gemini login' to create a managed Gemini profile,\n"
            "or run 'gemini login' to reuse your shared Gemini CLI credentials.\n"
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
            details={"source": "env", "verification": "credentials-only"},
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
                details={"source": "config", "verification": "credentials-only"},
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


def _resolve_cli_path(command: str) -> str | None:
    """Resolve a CLI path and reject mismatched executables.

    This prevents accidental execution of unrelated binaries when tests or
    environments monkeypatch ``shutil.which()`` too broadly.
    """
    import shutil

    cli_path = shutil.which(command)
    if not cli_path:
        return None

    cli_name = Path(cli_path).name.lower()
    expected_names = {command.lower()}
    if sys.platform.startswith("win"):
        expected_names.add(f"{command.lower()}.exe")

    if cli_name not in expected_names:
        return None

    return cli_path


def _claude_cli_email() -> str | None:
    """Try to extract email from `claude auth status` JSON output.

    Runs `claude auth status` with a short timeout.  Returns the email
    string on success, or ``None`` if the CLI is unavailable / fails.
    """
    import subprocess

    cli_path = _resolve_cli_path("claude")
    if not cli_path:
        return None
    try:
        proc = subprocess.run(
            [cli_path, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        data: dict[str, Any] = json.loads(proc.stdout)
        email = data.get("email")
        return email if isinstance(email, str) and email else None
    except Exception:
        return None


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

            # Try to get email from `claude auth status` CLI
            email = _claude_cli_email()
            subscription = oauth_data.get("subscriptionType", "unknown")
            user = email if email else f"subscription: {subscription}"
            return AuthStatus(
                provider="claude-agent",
                authenticated=True,
                user=user,
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
            # Try to extract email from id_token JWT
            user = _email_from_jwt(data.get("id_token", "")) or "chatgpt"
            return AuthStatus(
                provider="chatgpt",
                authenticated=True,
                user=user,
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

    Checks Markitai-managed Gemini profiles first, then falls back to
    ~/.gemini/oauth_creds.json.

    Returns:
        AuthStatus with authentication result
    """
    home = Path.home()
    managed_dir = home / ".markitai" / "auth"
    active_profile_path = managed_dir / "gemini-current.json"
    shared_creds_path = home / ".gemini" / "oauth_creds.json"

    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _expires_at(data: dict[str, Any]) -> datetime | None:
        expiry_date_ms = data.get("expiry_date")
        if not expiry_date_ms:
            return None
        try:
            return datetime.fromtimestamp(expiry_date_ms / 1000, tz=UTC)
        except (TypeError, ValueError, OSError):
            return None

    def _status_from_data(
        data: dict[str, Any],
        *,
        source: str,
        path: Path,
    ) -> AuthStatus | None:
        access_token = data.get("access_token")
        if not access_token:
            return None

        email = data.get("email")
        if not email:
            # Fall back to extracting email from id_token JWT
            email = _email_from_jwt(data.get("id_token", ""))
        if not email and isinstance(access_token, str):
            # Fall back to Google userinfo API
            email = _email_from_google_userinfo(access_token)
        project_id = data.get("project_id")
        auth_mode = data.get("auth_mode")
        user = email if isinstance(email, str) and email else "gemini-cli"
        return AuthStatus(
            provider="gemini-cli",
            authenticated=True,
            user=user,
            expires_at=_expires_at(data),
            error=None,
            details={
                "source": source,
                "project_id": (
                    str(project_id) if isinstance(project_id, str) else None
                ),
                "auth_mode": str(auth_mode) if isinstance(auth_mode, str) else None,
                "credential_path": str(path),
            },
        )

    active_data = _read_json(active_profile_path)
    if active_data:
        credential_path = active_data.get("credential_path")
        if isinstance(credential_path, str) and credential_path:
            managed_path = Path(credential_path)
            try:
                managed_path.resolve().relative_to(managed_dir.resolve())
            except ValueError:
                logger.warning(
                    "[Auth] credential_path points outside managed dir, "
                    f"ignoring: {managed_path}"
                )
                managed_path = None  # type: ignore[assignment]
            if managed_path is not None and managed_path.exists():
                managed_data = _read_json(managed_path)
                if managed_data:
                    status = _status_from_data(
                        managed_data,
                        source="markitai",
                        path=managed_path,
                    )
                    if status is not None:
                        return status

    if managed_dir.exists():
        managed_profiles = sorted(
            (
                path
                for path in managed_dir.glob("gemini-*.json")
                if path.name != active_profile_path.name
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for managed_path in managed_profiles:
            managed_data = _read_json(managed_path)
            if managed_data:
                status = _status_from_data(
                    managed_data,
                    source="markitai",
                    path=managed_path,
                )
                if status is not None:
                    return status

    if not shared_creds_path.exists():
        return AuthStatus(
            provider="gemini-cli",
            authenticated=False,
            user=None,
            expires_at=None,
            error=(
                "Credentials file not found (~/.markitai/auth or "
                "~/.gemini/oauth_creds.json)"
            ),
        )

    shared_data = _read_json(shared_creds_path)
    if shared_data is None:
        return AuthStatus(
            provider="gemini-cli",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Failed to read credentials: invalid JSON",
        )
    status = _status_from_data(
        shared_data,
        source="gemini-cli",
        path=shared_creds_path,
    )
    if status is not None:
        return status
    return AuthStatus(
        provider="gemini-cli",
        authenticated=False,
        user=None,
        expires_at=None,
        error="No access token found",
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


async def _login_copilot() -> AuthStatus:
    """Run `copilot login` interactively.

    Returns:
        AuthStatus after the login attempt.
    """
    import asyncio

    cli_path = _resolve_cli_path("copilot")
    if not cli_path:
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Copilot CLI not found in PATH",
        )

    proc = await asyncio.create_subprocess_exec(cli_path, "login")
    await proc.wait()

    if proc.returncode != 0:
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Login failed (exit code {proc.returncode})",
        )

    AuthManager().clear_cache("copilot")
    return _check_copilot_config_auth()


async def _login_claude_agent() -> AuthStatus:
    """Run `claude auth login` interactively.

    Returns:
        AuthStatus after the login attempt.
    """
    import asyncio

    cli_path = _resolve_cli_path("claude")
    if not cli_path:
        return AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Claude CLI not found in PATH",
        )

    proc = await asyncio.create_subprocess_exec(cli_path, "auth", "login")
    await proc.wait()

    if proc.returncode != 0:
        return AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Login failed (exit code {proc.returncode})",
        )

    AuthManager().clear_cache("claude-agent")
    return _check_claude_credentials_auth()


async def _login_gemini_cli() -> AuthStatus:
    """Run Gemini OAuth login via GeminiCLIProvider.alogin().

    Returns:
        AuthStatus after the login attempt.
    """
    from markitai.providers.gemini_cli import GeminiCLIProvider

    try:
        provider = GeminiCLIProvider()
        await provider.alogin()
    except Exception as e:
        return AuthStatus(
            provider="gemini-cli",
            authenticated=False,
            user=None,
            expires_at=None,
            error=str(e),
        )

    AuthManager().clear_cache("gemini-cli")
    return _check_gemini_cli_auth()


async def _login_chatgpt() -> AuthStatus:
    """Trigger ChatGPT Device Code Flow authentication.

    Calls LiteLLM's Authenticator.get_access_token() which handles the
    full device code flow: request code → display prompt → poll for
    completion → exchange for tokens → save to disk.

    The DeviceCodeInterceptor captures stdout from the authenticator and
    re-displays the device code in Rich format on stderr.

    Returns:
        AuthStatus after the login attempt.
    """
    try:
        from litellm.llms.chatgpt.authenticator import Authenticator
    except ImportError:
        return AuthStatus(
            provider="chatgpt",
            authenticated=False,
            user=None,
            expires_at=None,
            error="LiteLLM chatgpt authenticator not available",
        )

    import sys

    from markitai.providers.oauth_display import (
        DeviceCodeInterceptor,
        show_oauth_success,
    )

    authenticator = Authenticator()
    interceptor = DeviceCodeInterceptor()
    original_stdout = sys.stdout
    sys.stdout = interceptor  # type: ignore[assignment]
    try:
        authenticator.get_access_token()
    except Exception as e:
        return AuthStatus(
            provider="chatgpt",
            authenticated=False,
            user=None,
            expires_at=None,
            error=str(e),
        )
    finally:
        sys.stdout = original_stdout

    if interceptor.displayed:
        show_oauth_success("chatgpt")

    AuthManager().clear_cache("chatgpt")
    return _check_chatgpt_auth()


def can_attempt_login(provider: str) -> bool:
    """Check if interactive login is possible for a provider.

    Returns False when required dependencies (CLIs or libraries)
    are missing, so the caller can skip the login prompt and show
    an install hint instead.

    Args:
        provider: Provider name (e.g., "gemini-cli", "claude-agent")

    Returns:
        True if login can be attempted.
    """
    if provider == "gemini-cli":
        return importlib.util.find_spec("google_auth_oauthlib") is not None
    if provider == "claude-agent":
        return _resolve_cli_path("claude") is not None
    if provider == "copilot":
        return _resolve_cli_path("copilot") is not None
    if provider == "chatgpt":
        try:
            from litellm.llms.chatgpt.authenticator import (
                Authenticator,  # noqa: F401  # pyright: ignore[reportUnusedImport]
            )

            return True
        except ImportError:
            return False
    return False


async def attempt_login(provider: str) -> AuthStatus:
    """Attempt interactive login for a provider.

    Dispatches to the appropriate login flow:
    - copilot: subprocess `copilot login`
    - claude-agent: subprocess `claude auth login`
    - gemini-cli: programmatic OAuth via GeminiCLIProvider.alogin()
    - chatgpt: Device Code Flow via LiteLLM Authenticator

    Args:
        provider: Provider name (e.g., "copilot", "claude-agent")

    Returns:
        AuthStatus after the login attempt.
    """
    if provider == "copilot":
        return await _login_copilot()
    elif provider == "claude-agent":
        return await _login_claude_agent()
    elif provider == "gemini-cli":
        return await _login_gemini_cli()
    elif provider == "chatgpt":
        return await _login_chatgpt()
    else:
        return AuthStatus(
            provider=provider,
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Unknown provider: {provider}",
        )


__all__ = [
    "AuthStatus",
    "AuthManager",
    "get_auth_resolution_hint",
    "attempt_login",
    "can_attempt_login",
    "_is_copilot_sdk_available",
    "_is_claude_agent_sdk_available",
    "_check_copilot_config_auth",
    "_check_claude_credentials_auth",
    "_check_chatgpt_auth",
    "_check_gemini_cli_auth",
]
