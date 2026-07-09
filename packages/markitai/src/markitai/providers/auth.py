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
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


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
            "Run 'markitai auth claude login' (or 'claude auth login') to "
            "authenticate with Claude Code CLI.\n"
            "Alternatively, set a cloud provider env var:\n"
            "  CLAUDE_CODE_USE_BEDROCK=1, CLAUDE_CODE_USE_VERTEX=1, "
            "or CLAUDE_CODE_USE_FOUNDRY=1\n"
            "If Claude Code CLI is not installed, install it with:\n"
            f"  {_get_cli_install_cmd('claude')}"
        )
    elif provider == "copilot":
        return (
            "Run 'markitai auth copilot login' (or 'copilot login') to "
            "authenticate with GitHub Copilot.\n"
            "Alternatively, set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN "
            "env var (requires 'Copilot Requests' permission).\n"
            "If Copilot CLI is not installed, install it with:\n"
            f"  {_get_cli_install_cmd('copilot')}"
        )
    elif provider == "chatgpt":
        return (
            "Run 'markitai auth chatgpt login' to authenticate via the "
            "OAuth Device Code Flow (visit the URL, enter the code).\n"
            "Login is also triggered automatically on the first chatgpt/ "
            "model call."
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

    Checks (in order, matching `copilot login --help`'s own documented
    precedence): COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN, then
    ~/.copilot/config.json for logged_in_users.

    Returns:
        AuthStatus with authentication result
    """
    # Check env var auth first (COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN —
    # this is the CLI's own documented order of precedence)
    import os

    gh_token = (
        os.environ.get("COPILOT_GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )
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


def _claude_cli_auth_status() -> dict[str, Any] | None:
    """Query `claude auth status` and return the parsed JSON payload.

    Runs `claude auth status` with a short timeout.  Returns the parsed
    dict on success, or ``None`` if the CLI is unavailable / fails /
    emits invalid JSON.
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
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        data: Any = json.loads(proc.stdout)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _claude_cli_email() -> str | None:
    """Try to extract email from `claude auth status` JSON output.

    Returns the email string on success, or ``None`` if the CLI is
    unavailable / fails.
    """
    data = _claude_cli_auth_status()
    if not data:
        return None
    email = data.get("email")
    return email if isinstance(email, str) and email else None


def _check_claude_cli_login() -> AuthStatus | None:
    """Check Claude authentication via `claude auth status`.

    On macOS the Claude Code CLI stores OAuth tokens in the system
    Keychain rather than ``~/.claude/.credentials.json``, so the CLI
    itself is the source of truth when the credentials file is absent.

    Returns:
        Authenticated AuthStatus if the CLI reports a logged-in session,
        or ``None`` if the CLI is unavailable or not logged in.
    """
    data = _claude_cli_auth_status()
    if not data or not data.get("loggedIn"):
        return None

    email = data.get("email")
    subscription = data.get("subscriptionType")
    if not isinstance(subscription, str) or not subscription:
        subscription = "unknown"
    user = (
        email if isinstance(email, str) and email else f"subscription: {subscription}"
    )
    auth_method = data.get("authMethod")
    return AuthStatus(
        provider="claude-agent",
        authenticated=True,
        user=user,
        expires_at=None,
        error=None,
        details={
            "source": "cli",
            "subscription": subscription,
            "auth_method": auth_method if isinstance(auth_method, str) else None,
        },
    )


def _check_claude_credentials_auth() -> AuthStatus:
    """Check Claude authentication via env vars, credentials file, or CLI.

    Checks (in order):
    1. CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX / CLAUDE_CODE_USE_FOUNDRY
       environment variables (cloud provider auth)
    2. ~/.claude/.credentials.json for OAuth tokens
    3. `claude auth status` CLI fallback (covers macOS Keychain storage,
       where no credentials file exists even when logged in)

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
                details={"source": "env"},
            )

    file_status = _check_claude_credentials_file()
    if file_status.authenticated:
        return file_status

    # Credentials file missing or unusable — ask the CLI directly.
    # On macOS, Claude Code stores OAuth tokens in the Keychain, so a
    # logged-in user may have no ~/.claude/.credentials.json at all.
    cli_status = _check_claude_cli_login()
    if cli_status is not None:
        return cli_status

    return file_status


def _check_claude_credentials_file() -> AuthStatus:
    """Check Claude authentication by reading ~/.claude/.credentials.json.

    Returns:
        AuthStatus with authentication result
    """
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
                details={
                    "source": "credentials-file",
                    "subscription": (
                        subscription if isinstance(subscription, str) else None
                    ),
                },
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


def _clear_stale_chatgpt_device_code_request() -> None:
    """Clear incomplete ChatGPT device-code cooldown state before login.

    LiteLLM records ``device_code_requested_at`` before the device-code flow
    completes. If the user aborts before tokens are written, a later
    ``get_access_token()`` call can spend several minutes silently waiting for
    a token that will never arrive. When the user explicitly asks to log in
    again, drop that stale cooldown marker and start a fresh device-code flow.
    """
    auth_path = Path.home() / ".config" / "litellm" / "chatgpt" / "auth.json"
    if not auth_path.exists():
        return

    try:
        data: dict[str, Any] = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(
            "[Auth] Failed to read ChatGPT auth file for cooldown reset: {}", e
        )
        return

    if data.get("access_token") or "device_code_requested_at" not in data:
        return

    data.pop("device_code_requested_at", None)
    try:
        auth_path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        logger.debug("[Auth] Failed to clear stale ChatGPT cooldown marker: {}", e)
    else:
        logger.debug("[Auth] Cleared stale ChatGPT cooldown marker")


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

    Always runs with inherited stdio so the CLI sees a real TTY —
    required for credential storage to work correctly.

    Returns:
        AuthStatus after the login attempt.
    """
    import asyncio

    from markitai.providers.oauth_display import show_login_start

    cli_path = _resolve_cli_path("copilot")
    if not cli_path:
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Copilot CLI not found in PATH",
        )

    show_login_start("copilot")
    proc = await asyncio.create_subprocess_exec(cli_path, "login")
    returncode = await proc.wait()

    AuthManager().clear_cache("copilot")
    config_status = _check_copilot_config_auth()
    if config_status.authenticated:
        return config_status

    if returncode != 0:
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Login failed (exit code {returncode})",
        )
    return config_status


async def _login_claude_agent() -> AuthStatus:
    """Run `claude auth login` interactively.

    Always runs with inherited stdio so the CLI sees a real TTY.

    Returns:
        AuthStatus after the login attempt.
    """
    import asyncio

    from markitai.providers.oauth_display import show_login_start

    cli_path = _resolve_cli_path("claude")
    if not cli_path:
        return AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Claude CLI not found in PATH",
        )

    show_login_start("claude-agent")
    proc = await asyncio.create_subprocess_exec(cli_path, "auth", "login")
    returncode = await proc.wait()

    AuthManager().clear_cache("claude-agent")
    config_status = _check_claude_credentials_auth()
    if config_status.authenticated:
        return config_status

    if returncode != 0:
        return AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error=f"Login failed (exit code {returncode})",
        )
    return config_status


async def _login_chatgpt() -> AuthStatus:
    """Trigger ChatGPT Device Code Flow authentication.

    Calls LiteLLM's Authenticator.get_access_token() which handles the
    full device code flow: request code -> display prompt -> poll for
    completion -> exchange for tokens -> save to disk.

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

    _clear_stale_chatgpt_device_code_request()
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
        provider: Provider name (e.g., "claude-agent", "copilot")

    Returns:
        True if login can be attempted.
    """
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
    - copilot: subprocess ``copilot login``
    - claude-agent: subprocess ``claude auth login``
    - chatgpt: Device Code Flow via LiteLLM Authenticator

    Args:
        provider: Provider name (e.g., "copilot", "claude-agent").

    Returns:
        AuthStatus after the login attempt.
    """
    if provider == "copilot":
        return await _login_copilot()
    elif provider == "claude-agent":
        return await _login_claude_agent()
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
]
