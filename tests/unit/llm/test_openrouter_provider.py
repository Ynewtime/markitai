"""Tests for OpenRouter LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markit.llm.base import LLMMessage
from markit.llm.openrouter import OpenRouterProvider


class TestOpenRouterProviderInit:
    """Tests for OpenRouterProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("markit.llm.openai.AsyncOpenAI") as mock_client:
            provider = OpenRouterProvider(api_key="or-test")

            assert provider.name == "openrouter"
            assert provider.model == "google/gemini-3-flash-preview"
            assert provider._use_max_completion_tokens is False
            mock_client.assert_called_once_with(
                api_key="or-test",
                base_url="https://openrouter.ai/api/v1",
                timeout=120,
                max_retries=3,
            )

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("markit.llm.openai.AsyncOpenAI") as mock_client:
            provider = OpenRouterProvider(
                api_key="or-custom",
                model="anthropic/claude-3-opus",
                base_url="https://custom.openrouter.ai/api/v1",
                timeout=60,
                max_retries=5,
            )

            assert provider.model == "anthropic/claude-3-opus"
            mock_client.assert_called_once_with(
                api_key="or-custom",
                base_url="https://custom.openrouter.ai/api/v1",
                timeout=60,
                max_retries=5,
            )

    def test_init_default_base_url(self):
        """Test that default base URL is set when not provided."""
        with patch("markit.llm.openai.AsyncOpenAI") as mock_client:
            OpenRouterProvider(api_key="or-test", base_url=None)

            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"


class TestOpenRouterProviderComplete:
    """Tests for OpenRouterProvider.complete method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenRouterProvider(api_key="or-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_complete_uses_max_tokens(self, provider):
        """Test that OpenRouter uses max_tokens instead of max_completion_tokens."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
        mock_response.model = "google/gemini-3-flash-preview"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        await provider.complete(messages, max_tokens=100)

        # Verify max_tokens was used (not max_completion_tokens)
        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello!"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "anthropic/claude-3-sonnet"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages)

        assert response.content == "Hello!"
        assert response.model == "anthropic/claude-3-sonnet"


class TestOpenRouterProviderStream:
    """Tests for OpenRouterProvider.stream method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenRouterProvider(api_key="or-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_stream_uses_max_tokens(self, provider):
        """Test that streaming uses max_tokens."""

        async def mock_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hi"))])

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        async for _ in provider.stream(messages, max_tokens=50):
            pass

        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs


class TestOpenRouterProviderValidate:
    """Tests for OpenRouterProvider.validate method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenRouterProvider(api_key="or-test")
            provider.client = MagicMock()
            return provider

    @pytest.mark.asyncio
    async def test_validate_success(self, provider):
        """Test successful validation."""
        result = await provider.validate()

        # OpenRouter validate just checks if client exists
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_no_client(self, provider):
        """Test validation when client is None."""
        provider.client = None

        result = await provider.validate()

        assert result is False


class TestOpenRouterProviderInheritance:
    """Tests for OpenRouterProvider inheritance from OpenAIProvider."""

    def test_inherits_from_openai_provider(self):
        """Test that OpenRouterProvider inherits from OpenAIProvider."""
        from markit.llm.openai import OpenAIProvider

        assert issubclass(OpenRouterProvider, OpenAIProvider)

    def test_name_is_openrouter(self):
        """Test that provider name is 'openrouter'."""
        assert OpenRouterProvider.name == "openrouter"

    def test_uses_max_tokens_flag(self):
        """Test that _use_max_completion_tokens is False."""
        assert OpenRouterProvider._use_max_completion_tokens is False
