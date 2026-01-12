"""Tests for OpenAI LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import LLMMessage, ResponseFormat
from markit.llm.openai import OpenAIProvider


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("markit.llm.openai.AsyncOpenAI") as mock_client:
            provider = OpenAIProvider(api_key="sk-test")

            assert provider.model == "gpt-5.2"
            assert provider._use_max_completion_tokens is True
            mock_client.assert_called_once_with(
                api_key="sk-test",
                base_url=None,
                timeout=120,
                max_retries=3,
            )

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("markit.llm.openai.AsyncOpenAI") as mock_client:
            provider = OpenAIProvider(
                api_key="sk-custom",
                model="gpt-4o",
                base_url="https://custom.api.com",
                timeout=60,
                max_retries=5,
            )

            assert provider.model == "gpt-4o"
            mock_client.assert_called_once_with(
                api_key="sk-custom",
                base_url="https://custom.api.com",
                timeout=60,
                max_retries=5,
            )


class TestOpenAIProviderComplete:
    """Tests for OpenAIProvider.complete method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenAIProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello!"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "gpt-5.2"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages, temperature=0.5, max_tokens=100)

        assert response.content == "Hello!"
        assert response.model == "gpt-5.2"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_uses_max_completion_tokens(self, provider):
        """Test that OpenAI provider uses max_completion_tokens."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)
        mock_response.model = "gpt-5.2"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        await provider.complete(messages, max_tokens=100)

        # Check that max_completion_tokens was used (not max_tokens)
        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_without_usage(self, provider):
        """Test completion when usage is not returned."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi"), finish_reason="stop")]
        mock_response.usage = None
        mock_response.model = "gpt-5.2"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages)

        assert response.content == "Hi"
        assert response.usage is None

    @pytest.mark.asyncio
    async def test_complete_with_custom_model(self, provider):
        """Test completion with model override."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hi"), finish_reason="stop")]
        mock_response.usage = None
        mock_response.model = "gpt-4o"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        await provider.complete(messages, model="gpt-4o")

        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_complete_api_error(self, provider):
        """Test completion with API error."""
        provider.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError) as exc:
            await provider.complete(messages)

        assert "API error" in str(exc.value)

    @pytest.mark.asyncio
    async def test_complete_rate_limit_error(self, provider):
        """Test completion with rate limit error."""
        provider.client.chat.completions.create = AsyncMock(
            side_effect=Exception("rate_limit_exceeded")
        )

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(RateLimitError):
            await provider.complete(messages)


class TestOpenAIProviderStream:
    """Tests for OpenAIProvider.stream method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenAIProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_stream_success(self, provider):
        """Test successful streaming."""

        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
            ]
            for chunk in chunks:
                yield chunk

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_uses_max_completion_tokens(self, provider):
        """Test that streaming uses max_completion_tokens."""

        async def mock_stream():
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hi"))])

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        async for _ in provider.stream(messages, max_tokens=50):
            pass

        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_stream_api_error(self, provider):
        """Test streaming with API error."""
        provider.client.chat.completions.create = AsyncMock(side_effect=Exception("Stream Error"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError):
            async for _ in provider.stream(messages):
                pass


class TestOpenAIProviderAnalyzeImage:
    """Tests for OpenAIProvider.analyze_image method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenAIProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, provider):
        """Test successful image analysis."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="A cat"), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        mock_response.model = "gpt-5.2"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Describe this image",
            image_format="png",
        )

        assert response.content == "A cat"
        assert response.usage.prompt_tokens == 100

    @pytest.mark.asyncio
    async def test_analyze_image_with_json_response_format(self, provider):
        """Test image analysis with JSON response format."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"description": "A cat"}'), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        mock_response.model = "gpt-5.2"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response_format = ResponseFormat(type="json_object")
        response = await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Describe this image",
            response_format=response_format,
        )

        assert response.content == '{"description": "A cat"}'
        # Verify response_format was passed
        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_analyze_image_with_json_schema(self, provider):
        """Test image analysis with JSON schema."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"type": "photo"}'), finish_reason="stop")
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        mock_response.model = "gpt-5.2"
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        schema = {"type": "object", "properties": {"type": {"type": "string"}}}
        response_format = ResponseFormat(type="json_schema", json_schema=schema, strict=True)
        await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Describe this image",
            response_format=response_format,
        )

        call_kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True


class TestOpenAIProviderBuildResponseFormat:
    """Tests for OpenAIProvider._build_response_format method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            return OpenAIProvider(api_key="sk-test")

    def test_build_json_object(self, provider):
        """Test building json_object format."""
        response_format = ResponseFormat(type="json_object")
        result = provider._build_response_format(response_format)

        assert result == {"type": "json_object"}

    def test_build_json_schema(self, provider):
        """Test building json_schema format."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_format = ResponseFormat(type="json_schema", json_schema=schema, strict=True)
        result = provider._build_response_format(response_format)

        assert result["type"] == "json_schema"
        assert result["json_schema"]["name"] == "image_analysis"
        assert result["json_schema"]["strict"] is True
        assert result["json_schema"]["schema"] == schema

    def test_build_text_format(self, provider):
        """Test building text format."""
        response_format = ResponseFormat(type="text")
        result = provider._build_response_format(response_format)

        assert result == {"type": "text"}


class TestOpenAIProviderValidate:
    """Tests for OpenAIProvider.validate method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            provider = OpenAIProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_validate_success(self, provider):
        """Test successful validation."""
        provider.client.models.list = AsyncMock(return_value=[])

        result = await provider.validate()

        assert result is True
        provider.client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_failure(self, provider):
        """Test validation failure."""
        provider.client.models.list = AsyncMock(side_effect=Exception("Invalid API key"))

        result = await provider.validate()

        assert result is False


class TestOpenAIProviderConvertMessages:
    """Tests for OpenAIProvider._convert_messages method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.openai.AsyncOpenAI"):
            return OpenAIProvider(api_key="sk-test")

    def test_convert_simple_messages(self, provider):
        """Test converting simple text messages."""
        messages = [
            LLMMessage.system("You are helpful"),
            LLMMessage.user("Hello"),
            LLMMessage.assistant("Hi there!"),
        ]

        result = provider._convert_messages(messages)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi there!"}

    def test_convert_multimodal_message(self, provider):
        """Test converting multimodal message with image."""
        message = LLMMessage.user_with_image("Describe this", b"fake_image", "png")

        result = provider._convert_messages([message])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Describe this"
        assert result[0]["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in result[0]["content"][1]["image_url"]["url"]
