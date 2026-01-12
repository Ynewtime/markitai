"""Tests for Anthropic LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markit.exceptions import LLMError, RateLimitError
from markit.llm.anthropic import IMAGE_ANALYSIS_TOOL, AnthropicProvider
from markit.llm.base import LLMMessage, ResponseFormat


class TestAnthropicProviderInit:
    """Tests for AnthropicProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("markit.llm.anthropic.AsyncAnthropic") as mock_client:
            provider = AnthropicProvider(api_key="sk-ant-test")

            assert provider.model == "claude-sonnet-4-5"
            mock_client.assert_called_once_with(
                api_key="sk-ant-test",
                base_url=None,
                timeout=120,
                max_retries=3,
            )

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("markit.llm.anthropic.AsyncAnthropic") as mock_client:
            provider = AnthropicProvider(
                api_key="sk-custom",
                model="claude-opus-4-5",
                base_url="https://custom.anthropic.com",
                timeout=60,
                max_retries=5,
            )

            assert provider.model == "claude-opus-4-5"
            mock_client.assert_called_once_with(
                api_key="sk-custom",
                base_url="https://custom.anthropic.com",
                timeout=60,
                max_retries=5,
            )


class TestAnthropicProviderComplete:
    """Tests for AnthropicProvider.complete method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude!")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "end_turn"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages, temperature=0.5, max_tokens=100)

        assert response.content == "Hello from Claude!"
        assert response.model == "claude-sonnet-4-5"
        assert response.finish_reason == "end_turn"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, provider):
        """Test completion with system message."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi!")]
        mock_response.usage = MagicMock(input_tokens=15, output_tokens=3)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "end_turn"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            LLMMessage.system("You are helpful"),
            LLMMessage.user("Hi"),
        ]
        await provider.complete(messages)

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"
        # User messages should not include system message
        assert len(call_kwargs["messages"]) == 1

    @pytest.mark.asyncio
    async def test_complete_default_max_tokens(self, provider):
        """Test completion uses default max_tokens of 4096."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hi")]
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=2)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "end_turn"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        await provider.complete(messages)

        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_complete_api_error(self, provider):
        """Test completion with API error."""
        provider.client.messages.create = AsyncMock(side_effect=Exception("API Error"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError) as exc:
            await provider.complete(messages)

        assert "API error" in str(exc.value)

    @pytest.mark.asyncio
    async def test_complete_rate_limit_error(self, provider):
        """Test completion with rate limit error."""
        provider.client.messages.create = AsyncMock(side_effect=Exception("rate_limit_exceeded"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(RateLimitError):
            await provider.complete(messages)


class TestAnthropicProviderStream:
    """Tests for AnthropicProvider.stream method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_stream_success(self, provider):
        """Test successful streaming."""

        async def mock_text_stream():
            yield "Hello"
            yield " world"
            yield "!"

        mock_stream = MagicMock()
        mock_stream.text_stream = mock_text_stream()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        provider.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [LLMMessage.user("Hi")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_system_message(self, provider):
        """Test streaming with system message."""

        async def mock_text_stream():
            yield "Hi"

        mock_stream = MagicMock()
        mock_stream.text_stream = mock_text_stream()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        provider.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [
            LLMMessage.system("Be helpful"),
            LLMMessage.user("Hi"),
        ]
        async for _ in provider.stream(messages):
            pass

        call_kwargs = provider.client.messages.stream.call_args.kwargs
        assert call_kwargs["system"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_stream_api_error(self, provider):
        """Test streaming with API error."""
        provider.client.messages.stream = MagicMock(side_effect=Exception("Stream Error"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError):
            async for _ in provider.stream(messages):
                pass


class TestAnthropicProviderAnalyzeImage:
    """Tests for AnthropicProvider.analyze_image method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_analyze_image_without_response_format(self, provider):
        """Test image analysis without structured output."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A beautiful sunset")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=20)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "end_turn"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        response = await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Describe this image",
            image_format="png",
        )

        assert response.content == "A beautiful sunset"
        assert response.usage.prompt_tokens == 100

    @pytest.mark.asyncio
    async def test_analyze_image_with_tool_use(self, provider):
        """Test image analysis with Tool Use for structured output."""
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {
            "alt_text": "A cat",
            "detailed_description": "A fluffy cat",
            "image_type": "photo",
        }

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_response.usage = MagicMock(input_tokens=150, output_tokens=50)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "tool_use"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        response_format = ResponseFormat(type="json_object")
        response = await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Analyze this image",
            response_format=response_format,
        )

        # Response should be JSON from tool use
        assert "alt_text" in response.content
        assert "A cat" in response.content

        # Verify tool_choice was used
        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"]["type"] == "tool"

    @pytest.mark.asyncio
    async def test_analyze_image_with_custom_schema(self, provider):
        """Test image analysis with custom JSON schema."""
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {"category": "animal"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]
        mock_response.usage = MagicMock(input_tokens=150, output_tokens=30)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "tool_use"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        custom_schema = {"type": "object", "properties": {"category": {"type": "string"}}}
        response_format = ResponseFormat(type="json_schema", json_schema=custom_schema)
        await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Categorize this image",
            response_format=response_format,
        )

        call_kwargs = provider.client.messages.create.call_args.kwargs
        # Custom tool name should be used
        assert call_kwargs["tool_choice"]["name"] == "output_structured_data"
        assert call_kwargs["tools"][0]["input_schema"] == custom_schema

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, provider):
        """Test image analysis with error."""
        provider.client.messages.create = AsyncMock(side_effect=Exception("Image analysis failed"))

        with pytest.raises(LLMError) as exc:
            await provider.analyze_image(
                image_data=b"fake_image_data",
                prompt="Describe this",
            )

        assert "image analysis error" in str(exc.value)


class TestAnthropicProviderConvertMessage:
    """Tests for AnthropicProvider._convert_anthropic_message method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.anthropic.AsyncAnthropic"):
            return AnthropicProvider(api_key="sk-test")

    def test_convert_simple_message(self, provider):
        """Test converting simple text message."""
        message = LLMMessage.user("Hello")
        result = provider._convert_anthropic_message(message)

        assert result == {"role": "user", "content": "Hello"}

    def test_convert_multimodal_message(self, provider):
        """Test converting multimodal message with image."""
        message = LLMMessage.user_with_image("Describe this", b"fake_image", "jpeg")
        result = provider._convert_anthropic_message(message)

        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Describe this"
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["type"] == "base64"
        assert result["content"][1]["source"]["media_type"] == "image/jpeg"


class TestAnthropicProviderValidate:
    """Tests for AnthropicProvider.validate method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.anthropic.AsyncAnthropic"):
            provider = AnthropicProvider(api_key="sk-test")
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_validate_success(self, provider):
        """Test successful validation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)
        mock_response.model = "claude-sonnet-4-5"
        mock_response.stop_reason = "end_turn"
        provider.client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.validate()

        assert result is True
        # Verify minimal API call was made
        call_kwargs = provider.client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1

    @pytest.mark.asyncio
    async def test_validate_failure(self, provider):
        """Test validation failure."""
        provider.client.messages.create = AsyncMock(side_effect=Exception("Invalid API key"))

        result = await provider.validate()

        assert result is False


class TestImageAnalysisTool:
    """Tests for IMAGE_ANALYSIS_TOOL constant."""

    def test_tool_has_required_fields(self):
        """Test that IMAGE_ANALYSIS_TOOL has required fields."""
        assert "name" in IMAGE_ANALYSIS_TOOL
        assert "description" in IMAGE_ANALYSIS_TOOL
        assert "input_schema" in IMAGE_ANALYSIS_TOOL

    def test_tool_schema_has_required_properties(self):
        """Test that tool schema has required image analysis properties."""
        schema = IMAGE_ANALYSIS_TOOL["input_schema"]
        assert "alt_text" in schema["properties"]
        assert "detailed_description" in schema["properties"]
        assert "image_type" in schema["properties"]
