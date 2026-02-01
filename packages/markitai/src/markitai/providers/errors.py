"""Structured error classes for local LLM providers.

This module provides a hierarchy of error classes for handling various
failure modes in local providers (claude-agent, copilot). Each error
type indicates whether it can be retried and provides resolution hints
for users.

Error Hierarchy:
    ProviderError (base)
    ├── AuthenticationError (not retryable, requires user action)
    ├── QuotaError (not retryable, billing/subscription issue)
    ├── ProviderTimeoutError (may be retried with adjusted timeout)
    └── SDKNotAvailableError (SDK not installed)

Usage:
    try:
        result = await provider.acompletion(model, messages)
    except AuthenticationError as e:
        print(f"Auth failed for {e.provider}: {e.resolution_hint}")
    except QuotaError as e:
        print(f"Quota exceeded: {e.resolution_hint}")
    except ProviderTimeoutError as e:
        print(f"Timeout after {e.timeout_seconds}s, retry with longer timeout")
    except SDKNotAvailableError as e:
        print(f"Install SDK: {e.install_command}")
    except ProviderError as e:
        print(f"Provider {e.provider} error: {e}")
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for all provider-related errors.

    All provider errors carry a `provider` attribute identifying which
    provider (e.g., "claude-agent", "copilot") raised the error.

    Attributes:
        provider: The provider name (e.g., "claude-agent", "copilot")
        retryable: Whether this error type may be retried
    """

    __slots__ = ("provider", "retryable")

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retryable: bool = False,
    ) -> None:
        """Initialize ProviderError.

        Args:
            message: Error description
            provider: Provider name (e.g., "claude-agent", "copilot")
            retryable: Whether this error may be retried (default: False)
        """
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class AuthenticationError(ProviderError):
    """Error raised when authentication fails.

    This error is NOT retryable because it requires user action to resolve
    (e.g., running `claude auth login` or `copilot auth login`).

    Attributes:
        provider: The provider name
        resolution_hint: Suggested action to resolve the error
        retryable: Always False for authentication errors
    """

    __slots__ = ("resolution_hint",)

    # Default resolution hints per provider
    _DEFAULT_HINTS: dict[str, str] = {
        "claude-agent": "Run 'claude auth login' to authenticate with Claude Code CLI",
        "copilot": "Run 'copilot auth login' to authenticate with GitHub Copilot",
    }

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        resolution_hint: str | None = None,
    ) -> None:
        """Initialize AuthenticationError.

        Args:
            message: Error description
            provider: Provider name (e.g., "claude-agent", "copilot")
            resolution_hint: Custom hint for resolving the error.
                If not provided, uses provider-specific default.
        """
        super().__init__(message, provider=provider, retryable=False)
        self.resolution_hint = (
            resolution_hint
            if resolution_hint is not None
            else self._DEFAULT_HINTS.get(
                provider, "Please authenticate with the provider CLI"
            )
        )


class QuotaError(ProviderError):
    """Error raised when quota or billing limits are exceeded.

    This error is NOT retryable because it requires user action to resolve
    (e.g., upgrading subscription, adding payment method).

    Attributes:
        provider: The provider name
        resolution_hint: Suggested action to resolve the error
        retryable: Always False for quota errors
    """

    __slots__ = ("resolution_hint",)

    # Default resolution hints per provider
    _DEFAULT_HINTS: dict[str, str] = {
        "claude-agent": "Check your Claude subscription status or upgrade your plan",
        "copilot": "Check your GitHub Copilot subscription or billing settings",
    }

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        resolution_hint: str | None = None,
    ) -> None:
        """Initialize QuotaError.

        Args:
            message: Error description
            provider: Provider name (e.g., "claude-agent", "copilot")
            resolution_hint: Custom hint for resolving the error.
                If not provided, uses provider-specific default.
        """
        super().__init__(message, provider=provider, retryable=False)
        self.resolution_hint = (
            resolution_hint
            if resolution_hint is not None
            else self._DEFAULT_HINTS.get(
                provider, "Check your subscription quota and billing status"
            )
        )


class ProviderTimeoutError(ProviderError):
    """Error raised when a provider request times out.

    This error MAY be retried, potentially with an adjusted timeout value.
    The timeout_seconds attribute indicates the timeout that was used.

    Attributes:
        provider: The provider name
        timeout_seconds: The timeout value in seconds that was exceeded
        retryable: Always True for timeout errors
    """

    __slots__ = ("timeout_seconds",)

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        timeout_seconds: int | float,
    ) -> None:
        """Initialize ProviderTimeoutError.

        Args:
            message: Error description
            provider: Provider name (e.g., "claude-agent", "copilot")
            timeout_seconds: The timeout value that was exceeded
        """
        super().__init__(message, provider=provider, retryable=True)
        self.timeout_seconds = timeout_seconds


class SDKNotAvailableError(ProviderError):
    """Error raised when a required SDK is not installed.

    This error is NOT retryable because the SDK needs to be installed
    before the operation can succeed.

    Attributes:
        provider: The provider name
        install_command: Command to install the required SDK
        retryable: Always False for SDK availability errors
    """

    __slots__ = ("install_command",)

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        install_command: str,
    ) -> None:
        """Initialize SDKNotAvailableError.

        Args:
            message: Error description
            provider: Provider name (e.g., "claude-agent", "copilot")
            install_command: Command to install the required SDK
        """
        super().__init__(message, provider=provider, retryable=False)
        self.install_command = install_command


__all__ = [
    "ProviderError",
    "AuthenticationError",
    "QuotaError",
    "ProviderTimeoutError",
    "SDKNotAvailableError",
]
