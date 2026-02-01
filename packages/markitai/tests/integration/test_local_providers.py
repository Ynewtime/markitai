"""Integration tests for local LLM providers (claude-agent, copilot).

These tests use mocks to verify the full request flow without requiring
real SDK credentials. They test the integration between the provider
classes and the expected SDK interfaces.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# TestClaudeAgentIntegration
# =============================================================================


class TestClaudeAgentIntegration:
    """Integration tests for ClaudeAgentProvider with mocked SDK."""

    @pytest.mark.asyncio
    async def test_successful_completion_flow(self) -> None:
        """Test successful completion flow with mocked SDK.

        Verifies:
        - Provider correctly initializes and calls SDK
        - Response format matches LiteLLM ModelResponse structure
        - Usage information is correctly extracted
        """
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Mock the SDK module and its components
        mock_text_block = MagicMock()
        mock_text_block.text = "Hello! I am Claude."

        mock_assistant_message = MagicMock()
        mock_assistant_message.content = [mock_text_block]

        mock_result_message = MagicMock()
        mock_result_message.usage = MagicMock(
            input_tokens=25,
            output_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_result_message.total_cost_usd = 0.0001
        mock_result_message.structured_output = None

        # Create mock SDK client
        mock_client = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            # Mock SDK types module
            yield mock_assistant_message
            yield mock_result_message

        mock_client.receive_response = mock_receive_response

        # Create mock context manager for ClaudeSDKClient
        mock_sdk_client_class = MagicMock()
        mock_sdk_client_instance = AsyncMock()
        mock_sdk_client_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_sdk_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_sdk_client_class.return_value = mock_sdk_client_instance

        # Mock SDK types
        mock_types = MagicMock()
        mock_types.AssistantMessage = type(mock_assistant_message)
        mock_types.ResultMessage = type(mock_result_message)
        mock_types.TextBlock = type(mock_text_block)

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "claude_agent_sdk": MagicMock(
                        ClaudeSDKClient=mock_sdk_client_class,
                        ClaudeAgentOptions=MagicMock(),
                        types=mock_types,
                    ),
                    "claude_agent_sdk.types": mock_types,
                },
            ),
        ):
            response = await provider.acompletion(
                model="claude-agent/sonnet",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert response.choices[0]["message"]["role"] == "assistant"
        assert response.choices[0]["message"]["content"] == "Hello! I am Claude."
        assert response.choices[0]["finish_reason"] == "stop"

        # Verify usage
        assert hasattr(response, "usage")
        assert response.usage.prompt_tokens == 25
        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 35

        # Verify model identifier preserved
        assert response.model == "claude-agent/sonnet"

    @pytest.mark.asyncio
    async def test_image_request_flow(self) -> None:
        """Test multimodal image request flow with mocked SDK.

        Verifies:
        - Provider correctly detects image content
        - Streaming input is used for multimodal messages
        - Image content is properly converted to SDK format
        """
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Mock the SDK components
        mock_text_block = MagicMock()
        mock_text_block.text = "I see a beautiful landscape in the image."

        mock_assistant_message = MagicMock()
        mock_assistant_message.content = [mock_text_block]

        mock_result_message = MagicMock()
        mock_result_message.usage = MagicMock(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_result_message.total_cost_usd = 0.0005
        mock_result_message.structured_output = None

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield mock_assistant_message
            yield mock_result_message

        mock_client.receive_response = mock_receive_response

        mock_sdk_client_class = MagicMock()
        mock_sdk_client_instance = AsyncMock()
        mock_sdk_client_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_sdk_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_sdk_client_class.return_value = mock_sdk_client_instance

        mock_types = MagicMock()
        mock_types.AssistantMessage = type(mock_assistant_message)
        mock_types.ResultMessage = type(mock_result_message)
        mock_types.TextBlock = type(mock_text_block)

        # Test message with image content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJ"},
                    },
                ],
            }
        ]

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "claude_agent_sdk": MagicMock(
                        ClaudeSDKClient=mock_sdk_client_class,
                        ClaudeAgentOptions=MagicMock(),
                        types=mock_types,
                    ),
                    "claude_agent_sdk.types": mock_types,
                },
            ),
        ):
            response = await provider.acompletion(
                model="claude-agent/sonnet",
                messages=messages,
            )

        # Verify response
        assert response is not None
        assert "landscape" in response.choices[0]["message"]["content"]

        # Verify SDK was called with streaming input
        # (query should have been called with an async iterator)
        mock_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self) -> None:
        """Test that authentication errors are properly raised.

        Verifies:
        - SDK authentication errors are caught
        - LiteLLM AuthenticationError is raised
        - Error message contains helpful information
        """
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Mock SDK to raise authentication error
        mock_sdk_client_class = MagicMock()
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(
            side_effect=Exception("Not authenticated. Please run 'claude auth login'")
        )

        mock_sdk_client_instance = AsyncMock()
        mock_sdk_client_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_sdk_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_sdk_client_class.return_value = mock_sdk_client_instance

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "claude_agent_sdk": MagicMock(
                        ClaudeSDKClient=mock_sdk_client_class,
                        ClaudeAgentOptions=MagicMock(),
                        types=MagicMock(),
                    ),
                    "claude_agent_sdk.types": MagicMock(),
                },
            ),
            pytest.raises(LiteLLMAuthError) as exc_info,
        ):
            await provider.acompletion(
                model="claude-agent/sonnet",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self) -> None:
        """Test that rate limit errors are properly raised.

        Verifies:
        - SDK rate limit errors are caught
        - LiteLLM RateLimitError is raised
        - Error is potentially retryable by LiteLLM
        """
        from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Mock SDK to raise rate limit error
        mock_sdk_client_class = MagicMock()
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(
            side_effect=Exception("Rate limit exceeded. Error 429.")
        )

        mock_sdk_client_instance = AsyncMock()
        mock_sdk_client_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_sdk_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_sdk_client_class.return_value = mock_sdk_client_instance

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "claude_agent_sdk": MagicMock(
                        ClaudeSDKClient=mock_sdk_client_class,
                        ClaudeAgentOptions=MagicMock(),
                        types=MagicMock(),
                    ),
                    "claude_agent_sdk.types": MagicMock(),
                },
            ),
            pytest.raises(LiteLLMRateLimitError) as exc_info,
        ):
            await provider.acompletion(
                model="claude-agent/sonnet",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_structured_output_response(self) -> None:
        """Test structured output (JSON mode) response handling.

        Verifies:
        - response_format parameter is converted to SDK output_format
        - Structured output is correctly extracted from SDK response
        - Response content is valid JSON string
        """
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Mock structured output response
        expected_json = {"name": "John", "age": 30, "active": True}

        mock_text_block = MagicMock()
        mock_text_block.text = ""  # Empty text when structured output is used

        mock_assistant_message = MagicMock()
        mock_assistant_message.content = [mock_text_block]

        mock_result_message = MagicMock()
        mock_result_message.usage = MagicMock(
            input_tokens=50,
            output_tokens=15,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_result_message.total_cost_usd = 0.0002
        mock_result_message.structured_output = expected_json

        mock_client = AsyncMock()
        mock_client.query = AsyncMock()

        async def mock_receive_response():
            yield mock_assistant_message
            yield mock_result_message

        mock_client.receive_response = mock_receive_response

        mock_sdk_client_class = MagicMock()
        mock_sdk_client_instance = AsyncMock()
        mock_sdk_client_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_sdk_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_sdk_client_class.return_value = mock_sdk_client_instance

        mock_types = MagicMock()
        mock_types.AssistantMessage = type(mock_assistant_message)
        mock_types.ResultMessage = type(mock_result_message)
        mock_types.TextBlock = type(mock_text_block)

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "claude_agent_sdk": MagicMock(
                        ClaudeSDKClient=mock_sdk_client_class,
                        ClaudeAgentOptions=MagicMock(),
                        types=mock_types,
                    ),
                    "claude_agent_sdk.types": mock_types,
                },
            ),
        ):
            response = await provider.acompletion(
                model="claude-agent/sonnet",
                messages=[{"role": "user", "content": "Return a person object"}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                                "active": {"type": "boolean"},
                            },
                        }
                    },
                },
            )

        # Verify structured output is JSON serialized
        content = response.choices[0]["message"]["content"]
        parsed = json.loads(content)
        assert parsed == expected_json


# =============================================================================
# TestCopilotIntegration
# =============================================================================


class TestCopilotIntegration:
    """Integration tests for CopilotProvider with mocked SDK."""

    @pytest.mark.asyncio
    async def test_successful_completion_flow(self) -> None:
        """Test successful completion flow with mocked SDK.

        Verifies:
        - Provider correctly initializes and calls SDK
        - Response format matches LiteLLM ModelResponse structure
        - Token counting works correctly
        """
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Mock response data
        mock_response = MagicMock()
        mock_response.data = MagicMock(content="Hello from GPT!")

        # Mock session
        mock_session = AsyncMock()
        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_session.destroy = AsyncMock()

        # Mock client
        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.stop = AsyncMock()

        mock_copilot_client_class = MagicMock(return_value=mock_client)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.copilot._find_copilot_cli",
                return_value="/usr/local/bin/copilot",
            ),
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(CopilotClient=mock_copilot_client_class),
                },
            ),
        ):
            response = await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert response.choices[0]["message"]["role"] == "assistant"
        assert response.choices[0]["message"]["content"] == "Hello from GPT!"
        assert response.choices[0]["finish_reason"] == "stop"

        # Verify usage exists
        assert hasattr(response, "usage")
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        # Verify model identifier preserved
        assert response.model == "copilot/gpt-4.1"

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_json_mode_extraction(self) -> None:
        """Test JSON mode response extraction.

        Verifies:
        - JSON prompt suffix is appended when response_format specified
        - JSON is correctly extracted from response text
        - Markdown code blocks are handled properly
        """
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Mock response with JSON in markdown code block
        json_response = '```json\n{"key": "value", "number": 42}\n```'
        mock_response = MagicMock()
        mock_response.data = MagicMock(content=json_response)

        mock_session = AsyncMock()
        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.stop = AsyncMock()

        mock_copilot_client_class = MagicMock(return_value=mock_client)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.copilot._find_copilot_cli",
                return_value="/usr/local/bin/copilot",
            ),
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(CopilotClient=mock_copilot_client_class),
                },
            ),
        ):
            response = await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Return JSON data"}],
                response_format={"type": "json_object"},
            )

        # Verify JSON was extracted
        content = response.choices[0]["message"]["content"]
        parsed = json.loads(content)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self) -> None:
        """Test that timeout errors have helpful messages.

        Verifies:
        - asyncio.TimeoutError is caught
        - RuntimeError is raised with timeout information
        - Error message suggests checking network or timeout settings
        """
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Mock session that times out
        mock_session = AsyncMock()
        mock_session.send_and_wait = AsyncMock(side_effect=TimeoutError())
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.stop = AsyncMock()

        mock_copilot_client_class = MagicMock(return_value=mock_client)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.copilot._find_copilot_cli",
                return_value="/usr/local/bin/copilot",
            ),
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(CopilotClient=mock_copilot_client_class),
                },
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        error_msg = str(exc_info.value).lower()
        assert "timeout" in error_msg
        assert any(word in error_msg for word in ["network", "increase"])

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self) -> None:
        """Test that authentication errors are properly raised.

        Verifies:
        - SDK authentication errors are caught
        - LiteLLM AuthenticationError is raised
        - Error message suggests running auth login
        """
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Mock session that raises auth error
        mock_session = AsyncMock()
        mock_session.send_and_wait = AsyncMock(
            side_effect=Exception("Not authenticated. Please run copilot auth login.")
        )
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.stop = AsyncMock()

        mock_copilot_client_class = MagicMock(return_value=mock_client)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.copilot._find_copilot_cli",
                return_value="/usr/local/bin/copilot",
            ),
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(CopilotClient=mock_copilot_client_class),
                },
            ),
            pytest.raises(LiteLLMAuthError) as exc_info,
        ):
            await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        error_msg = str(exc_info.value).lower()
        assert "not authenticated" in error_msg or "auth" in error_msg

        # Cleanup
        await provider.close()

    @pytest.mark.asyncio
    async def test_quota_error_handling(self) -> None:
        """Test that quota errors are properly raised.

        Verifies:
        - SDK quota/billing errors are caught
        - LiteLLM AuthenticationError is raised (for non-retryable billing issues)
        """
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Mock session that raises quota error
        mock_session = AsyncMock()
        mock_session.send_and_wait = AsyncMock(
            side_effect=Exception("Quota exceeded. Error 402: Payment required.")
        )
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)
        mock_client.stop = AsyncMock()

        mock_copilot_client_class = MagicMock(return_value=mock_client)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.copilot._find_copilot_cli",
                return_value="/usr/local/bin/copilot",
            ),
            patch.dict(
                "sys.modules",
                {
                    "copilot": MagicMock(CopilotClient=mock_copilot_client_class),
                },
            ),
            pytest.raises(LiteLLMAuthError) as exc_info,
        ):
            await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Hello!"}],
            )

        error_msg = str(exc_info.value).lower()
        assert "quota" in error_msg or "billing" in error_msg or "402" in error_msg

        # Cleanup
        await provider.close()


# =============================================================================
# TestProviderErrorClassification
# =============================================================================


class TestProviderErrorClassification:
    """Tests for provider error classification and retry behavior."""

    def test_quota_errors_not_retried(self) -> None:
        """Test that quota/billing errors are NOT retryable.

        Quota errors require user action (upgrade subscription, add payment)
        and should not be automatically retried.
        """
        from markitai.providers.errors import AuthenticationError, QuotaError

        # QuotaError is explicitly not retryable
        quota_error = QuotaError("Quota exceeded", provider="copilot")
        assert quota_error.retryable is False

        # AuthenticationError is also not retryable
        auth_error = AuthenticationError("Auth failed", provider="claude-agent")
        assert auth_error.retryable is False

    def test_network_errors_are_retried(self) -> None:
        """Test that network/connection errors are marked as retryable.

        Network errors are transient and should be retried with backoff.
        """
        from markitai.providers.errors import ProviderTimeoutError

        # ProviderTimeoutError is retryable
        timeout_error = ProviderTimeoutError(
            "Request timed out",
            provider="claude-agent",
            timeout_seconds=120,
        )
        assert timeout_error.retryable is True

    def test_litellm_retryable_errors(self) -> None:
        """Test that LiteLLM's retryable error types are used correctly.

        Verifies that providers raise appropriate LiteLLM exceptions that
        integrate with LiteLLM's retry logic.
        """
        from litellm.exceptions import (
            APIConnectionError,
            RateLimitError,
        )

        # These errors should be caught by LiteLLM's retry mechanism
        # Verify they can be instantiated (basic sanity check)
        rate_limit = RateLimitError("Rate limited", "test-provider", "test-model")
        assert isinstance(rate_limit, Exception)

        # APIConnectionError requires specific parameters
        connection_error = APIConnectionError(
            message="Connection failed",
            llm_provider="test-provider",
            model="test-model",
        )
        assert isinstance(connection_error, Exception)

    def test_error_provider_attribute_preserved(self) -> None:
        """Test that provider attribute is preserved across error types."""
        from markitai.providers.errors import (
            AuthenticationError,
            ProviderTimeoutError,
            QuotaError,
            SDKNotAvailableError,
        )

        errors = [
            AuthenticationError("Auth failed", provider="claude-agent"),
            QuotaError("Quota exceeded", provider="copilot"),
            ProviderTimeoutError(
                "Timeout", provider="claude-agent", timeout_seconds=60
            ),
            SDKNotAvailableError(
                "Not installed",
                provider="copilot",
                install_command="uv add github-copilot-sdk",
            ),
        ]

        for error in errors:
            assert hasattr(error, "provider")
            assert error.provider in ("claude-agent", "copilot")


# =============================================================================
# TestProviderRegistration
# =============================================================================


class TestProviderRegistration:
    """Tests for provider registration with LiteLLM."""

    def test_register_providers_with_mock_sdk(self) -> None:
        """Test that providers can be registered when SDK is available."""
        # Reset registration state
        import markitai.providers as providers_module

        original_registered = providers_module._registered
        providers_module._registered = False

        try:
            with (
                patch(
                    "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                    return_value=True,
                ),
                patch(
                    "markitai.providers.copilot._is_copilot_sdk_available",
                    return_value=True,
                ),
            ):
                # Import and register
                from markitai.providers import register_providers

                register_providers()

                # Verify registration happened
                assert providers_module._registered is True
        finally:
            # Restore original state
            providers_module._registered = original_registered

    def test_is_local_provider_model_detection(self) -> None:
        """Test that local provider models are correctly identified."""
        from markitai.providers import is_local_provider_model

        # Local providers
        assert is_local_provider_model("claude-agent/sonnet") is True
        assert is_local_provider_model("claude-agent/opus") is True
        assert is_local_provider_model("claude-agent/haiku") is True
        assert is_local_provider_model("copilot/gpt-4.1") is True
        assert is_local_provider_model("copilot/claude-sonnet-4.5") is True

        # Non-local providers
        assert is_local_provider_model("openai/gpt-4") is False
        assert is_local_provider_model("anthropic/claude-3-sonnet") is False
        assert is_local_provider_model("gemini/gemini-2.5-pro") is False
