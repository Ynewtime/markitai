"""Unit tests for provider error classes.

These error classes provide structured error handling for local LLM providers
(claude-agent and copilot), enabling better error messages and appropriate
retry behavior.
"""

from __future__ import annotations

import pytest


class TestProviderError:
    """Tests for base ProviderError class."""

    def test_provider_error_has_provider_attribute(self) -> None:
        """Test that ProviderError has a provider attribute."""
        from markitai.providers.errors import ProviderError

        error = ProviderError("Test error", provider="claude-agent")
        assert error.provider == "claude-agent"

    def test_provider_error_has_message(self) -> None:
        """Test that ProviderError message is accessible."""
        from markitai.providers.errors import ProviderError

        error = ProviderError("Test error message", provider="copilot")
        assert str(error) == "Test error message"
        assert error.args[0] == "Test error message"

    def test_provider_error_inherits_from_exception(self) -> None:
        """Test that ProviderError inherits from Exception."""
        from markitai.providers.errors import ProviderError

        error = ProviderError("Test", provider="test")
        assert isinstance(error, Exception)


class TestAuthenticationError:
    """Tests for AuthenticationError class."""

    def test_authentication_error_inherits_from_provider_error(self) -> None:
        """Test that AuthenticationError inherits from ProviderError."""
        from markitai.providers.errors import AuthenticationError, ProviderError

        error = AuthenticationError("Auth failed", provider="claude-agent")
        assert isinstance(error, ProviderError)

    def test_authentication_error_has_provider(self) -> None:
        """Test that AuthenticationError has provider attribute."""
        from markitai.providers.errors import AuthenticationError

        error = AuthenticationError("Auth failed", provider="copilot")
        assert error.provider == "copilot"

    def test_authentication_error_default_resolution_hint(self) -> None:
        """Test AuthenticationError has default resolution hint."""
        from markitai.providers.errors import AuthenticationError

        error = AuthenticationError("Not authenticated", provider="claude-agent")
        assert error.resolution_hint is not None
        assert (
            "auth" in error.resolution_hint.lower()
            or "login" in error.resolution_hint.lower()
        )

    def test_authentication_error_custom_resolution_hint(self) -> None:
        """Test AuthenticationError with custom resolution hint."""
        from markitai.providers.errors import AuthenticationError

        custom_hint = "Run 'claude auth login' to authenticate"
        error = AuthenticationError(
            "Not authenticated",
            provider="claude-agent",
            resolution_hint=custom_hint,
        )
        assert error.resolution_hint == custom_hint

    def test_authentication_error_not_retryable(self) -> None:
        """Test that AuthenticationError indicates it's not retryable."""
        from markitai.providers.errors import AuthenticationError

        error = AuthenticationError("Auth failed", provider="claude-agent")
        # Not retryable because user action is required
        assert error.retryable is False


class TestQuotaError:
    """Tests for QuotaError class."""

    def test_quota_error_inherits_from_provider_error(self) -> None:
        """Test that QuotaError inherits from ProviderError."""
        from markitai.providers.errors import ProviderError, QuotaError

        error = QuotaError("Quota exceeded", provider="copilot")
        assert isinstance(error, ProviderError)

    def test_quota_error_has_provider(self) -> None:
        """Test that QuotaError has provider attribute."""
        from markitai.providers.errors import QuotaError

        error = QuotaError("Quota exceeded", provider="claude-agent")
        assert error.provider == "claude-agent"

    def test_quota_error_has_resolution_hint(self) -> None:
        """Test that QuotaError has resolution hint."""
        from markitai.providers.errors import QuotaError

        error = QuotaError("Quota exceeded", provider="copilot")
        assert error.resolution_hint is not None
        # Should mention subscription, upgrade, or billing
        hint_lower = error.resolution_hint.lower()
        assert any(
            word in hint_lower
            for word in ["subscription", "upgrade", "billing", "plan", "quota"]
        )

    def test_quota_error_custom_resolution_hint(self) -> None:
        """Test QuotaError with custom resolution hint."""
        from markitai.providers.errors import QuotaError

        custom_hint = "Please upgrade your subscription"
        error = QuotaError(
            "Quota exceeded",
            provider="copilot",
            resolution_hint=custom_hint,
        )
        assert error.resolution_hint == custom_hint

    def test_quota_error_not_retryable(self) -> None:
        """Test that QuotaError indicates it's not retryable."""
        from markitai.providers.errors import QuotaError

        error = QuotaError("Quota exceeded", provider="copilot")
        # Not retryable because billing/subscription action is required
        assert error.retryable is False


class TestProviderTimeoutError:
    """Tests for ProviderTimeoutError class."""

    def test_timeout_error_inherits_from_provider_error(self) -> None:
        """Test that ProviderTimeoutError inherits from ProviderError."""
        from markitai.providers.errors import ProviderError, ProviderTimeoutError

        error = ProviderTimeoutError(
            "Request timed out", provider="claude-agent", timeout_seconds=120
        )
        assert isinstance(error, ProviderError)

    def test_timeout_error_has_provider(self) -> None:
        """Test that ProviderTimeoutError has provider attribute."""
        from markitai.providers.errors import ProviderTimeoutError

        error = ProviderTimeoutError(
            "Request timed out", provider="copilot", timeout_seconds=60
        )
        assert error.provider == "copilot"

    def test_timeout_error_has_timeout_seconds(self) -> None:
        """Test that ProviderTimeoutError has timeout_seconds attribute."""
        from markitai.providers.errors import ProviderTimeoutError

        error = ProviderTimeoutError(
            "Request timed out", provider="claude-agent", timeout_seconds=120
        )
        assert error.timeout_seconds == 120

    def test_timeout_error_may_be_retried(self) -> None:
        """Test that ProviderTimeoutError indicates it may be retried."""
        from markitai.providers.errors import ProviderTimeoutError

        error = ProviderTimeoutError(
            "Request timed out", provider="claude-agent", timeout_seconds=60
        )
        # Timeouts may be retried with adjusted timeout
        assert error.retryable is True


class TestSDKNotAvailableError:
    """Tests for SDKNotAvailableError class."""

    def test_sdk_not_available_error_inherits_from_provider_error(self) -> None:
        """Test that SDKNotAvailableError inherits from ProviderError."""
        from markitai.providers.errors import ProviderError, SDKNotAvailableError

        error = SDKNotAvailableError(
            "SDK not installed",
            provider="claude-agent",
            install_command="uv add claude-agent-sdk",
        )
        assert isinstance(error, ProviderError)

    def test_sdk_not_available_error_has_provider(self) -> None:
        """Test that SDKNotAvailableError has provider attribute."""
        from markitai.providers.errors import SDKNotAvailableError

        error = SDKNotAvailableError(
            "SDK not installed",
            provider="copilot",
            install_command="uv add github-copilot-sdk",
        )
        assert error.provider == "copilot"

    def test_sdk_not_available_error_has_install_command(self) -> None:
        """Test that SDKNotAvailableError has install_command attribute."""
        from markitai.providers.errors import SDKNotAvailableError

        error = SDKNotAvailableError(
            "SDK not installed",
            provider="claude-agent",
            install_command="uv add claude-agent-sdk",
        )
        assert error.install_command == "uv add claude-agent-sdk"

    def test_sdk_not_available_error_not_retryable(self) -> None:
        """Test that SDKNotAvailableError indicates it's not retryable."""
        from markitai.providers.errors import SDKNotAvailableError

        error = SDKNotAvailableError(
            "SDK not installed",
            provider="copilot",
            install_command="uv add github-copilot-sdk",
        )
        # Not retryable because SDK needs to be installed
        assert error.retryable is False


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_provider_error(self) -> None:
        """Test that all error classes inherit from ProviderError."""
        from markitai.providers.errors import (
            AuthenticationError,
            ProviderError,
            ProviderTimeoutError,
            QuotaError,
            SDKNotAvailableError,
        )

        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(QuotaError, ProviderError)
        assert issubclass(ProviderTimeoutError, ProviderError)
        assert issubclass(SDKNotAvailableError, ProviderError)

    def test_errors_can_be_caught_as_provider_error(self) -> None:
        """Test that specific errors can be caught as ProviderError."""
        from markitai.providers.errors import (
            AuthenticationError,
            ProviderError,
            ProviderTimeoutError,
            QuotaError,
            SDKNotAvailableError,
        )

        errors = [
            AuthenticationError("Auth failed", provider="test"),
            QuotaError("Quota exceeded", provider="test"),
            ProviderTimeoutError("Timeout", provider="test", timeout_seconds=60),
            SDKNotAvailableError(
                "Not installed", provider="test", install_command="uv add test"
            ),
        ]

        for error in errors:
            try:
                raise error
            except ProviderError as caught:
                assert caught.provider == "test"
            except Exception:
                pytest.fail(f"{type(error).__name__} should be caught as ProviderError")


class TestErrorMessages:
    """Tests for error message formatting."""

    def test_provider_error_str_representation(self) -> None:
        """Test ProviderError string representation."""
        from markitai.providers.errors import ProviderError

        error = ProviderError("Something went wrong", provider="test-provider")
        assert "Something went wrong" in str(error)

    def test_authentication_error_str_representation(self) -> None:
        """Test AuthenticationError string representation."""
        from markitai.providers.errors import AuthenticationError

        error = AuthenticationError(
            "Not authenticated",
            provider="claude-agent",
            resolution_hint="Run claude auth login",
        )
        error_str = str(error)
        assert "Not authenticated" in error_str

    def test_sdk_not_available_error_str_representation(self) -> None:
        """Test SDKNotAvailableError string representation."""
        from markitai.providers.errors import SDKNotAvailableError

        error = SDKNotAvailableError(
            "Claude Agent SDK not installed",
            provider="claude-agent",
            install_command="uv add claude-agent-sdk",
        )
        error_str = str(error)
        assert "Claude Agent SDK not installed" in error_str
