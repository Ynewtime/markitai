"""Unit tests for provider error classes.

These error classes provide structured error handling for local LLM providers
(claude-agent and copilot), enabling better error messages and appropriate
retry behavior.
"""

from __future__ import annotations


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
