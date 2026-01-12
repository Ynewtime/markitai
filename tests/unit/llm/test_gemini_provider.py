"""Tests for Gemini LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import LLMMessage, ResponseFormat
from markit.llm.gemini import GeminiProvider


class TestGeminiProviderInit:
    """Tests for GeminiProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("markit.llm.gemini.genai.Client") as mock_client:
            provider = GeminiProvider(api_key="gemini-key")

            assert provider.model == "gemini-3-flash-preview"
            mock_client.assert_called_once()
            # Check that timeout was converted to milliseconds
            call_kwargs = mock_client.call_args.kwargs
            assert call_kwargs["api_key"] == "gemini-key"

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("markit.llm.gemini.genai.Client") as mock_client:
            provider = GeminiProvider(
                api_key="custom-key",
                model="gemini-3-pro-preview",
                timeout=60,
            )

            assert provider.model == "gemini-3-pro-preview"
            mock_client.assert_called_once()


class TestGeminiProviderComplete:
    """Tests for GeminiProvider.complete method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.gemini.genai.Client"):
            provider = GeminiProvider(api_key="test-key")
            provider.client = MagicMock()
            provider.client.aio = MagicMock()
            provider.client.aio.models = MagicMock()
            return provider

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10,
            candidates_token_count=5,
        )
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages, temperature=0.5, max_tokens=100)

        assert response.content == "Hello from Gemini!"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_without_usage(self, provider):
        """Test completion when usage metadata is not available."""
        mock_response = MagicMock()
        mock_response.text = "Hi"
        mock_response.usage_metadata = None
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages)

        assert response.content == "Hi"
        assert response.usage is None

    @pytest.mark.asyncio
    async def test_complete_empty_response(self, provider):
        """Test completion with empty response."""
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.usage_metadata = None
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages)

        assert response.content == ""

    @pytest.mark.asyncio
    async def test_complete_api_error(self, provider):
        """Test completion with API error."""
        provider.client.aio.models.generate_content = AsyncMock(side_effect=Exception("API Error"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError) as exc:
            await provider.complete(messages)

        assert "API error" in str(exc.value)

    @pytest.mark.asyncio
    async def test_complete_rate_limit_error(self, provider):
        """Test completion with rate limit error."""
        provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("rate limit exceeded")
        )

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(RateLimitError):
            await provider.complete(messages)


class TestGeminiProviderStream:
    """Tests for GeminiProvider.stream method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.gemini.genai.Client"):
            provider = GeminiProvider(api_key="test-key")
            provider.client = MagicMock()
            provider.client.aio = MagicMock()
            provider.client.aio.models = MagicMock()
            return provider

    @pytest.mark.asyncio
    async def test_stream_success(self, provider):
        """Test successful streaming."""

        async def mock_stream():
            chunks = [
                MagicMock(text="Hello"),
                MagicMock(text=" world"),
                MagicMock(text="!"),
            ]
            for chunk in chunks:
                yield chunk

        provider.client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, provider):
        """Test streaming with some empty chunks."""

        async def mock_stream():
            chunks = [
                MagicMock(text="Hello"),
                MagicMock(text=None),  # Empty chunk
                MagicMock(text="!"),
            ]
            for chunk in chunks:
                yield chunk

        provider.client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", "!"]

    @pytest.mark.asyncio
    async def test_stream_api_error(self, provider):
        """Test streaming with API error."""
        provider.client.aio.models.generate_content_stream = AsyncMock(
            side_effect=Exception("Stream Error")
        )

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError):
            async for _ in provider.stream(messages):
                pass


class TestGeminiProviderAnalyzeImage:
    """Tests for GeminiProvider.analyze_image method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.gemini.genai.Client"):
            provider = GeminiProvider(api_key="test-key")
            provider.client = MagicMock()
            provider.client.aio = MagicMock()
            provider.client.aio.models = MagicMock()
            return provider

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, provider):
        """Test successful image analysis."""
        mock_response = MagicMock()
        mock_response.text = "A beautiful landscape"
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=20,
        )
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("markit.llm.gemini.types.Part") as mock_part:
            mock_part.from_bytes = MagicMock(return_value="image_part")
            mock_part.from_text = MagicMock(return_value="text_part")

            response = await provider.analyze_image(
                image_data=b"fake_image_data",
                prompt="Describe this image",
                image_format="png",
            )

        assert response.content == "A beautiful landscape"
        assert response.usage.prompt_tokens == 100

    @pytest.mark.asyncio
    async def test_analyze_image_with_json_format(self, provider):
        """Test image analysis with JSON response format."""
        mock_response = MagicMock()
        mock_response.text = '{"description": "A cat"}'
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100,
            candidates_token_count=20,
        )
        provider.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        with patch("markit.llm.gemini.types.Part") as mock_part:
            mock_part.from_bytes = MagicMock(return_value="image_part")
            mock_part.from_text = MagicMock(return_value="text_part")

            response_format = ResponseFormat(type="json_object")
            response = await provider.analyze_image(
                image_data=b"fake_image_data",
                prompt="Describe this image as JSON",
                response_format=response_format,
            )

        assert response.content == '{"description": "A cat"}'

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, provider):
        """Test image analysis with error."""
        provider.client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("Image analysis failed")
        )

        with patch("markit.llm.gemini.types.Part") as mock_part:
            mock_part.from_bytes = MagicMock(return_value="image_part")
            mock_part.from_text = MagicMock(return_value="text_part")

            with pytest.raises(LLMError) as exc:
                await provider.analyze_image(
                    image_data=b"fake_image_data",
                    prompt="Describe this",
                )

        assert "image analysis error" in str(exc.value)


class TestGeminiProviderBuildGenerationConfig:
    """Tests for GeminiProvider._build_generation_config method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.gemini.genai.Client"):
            return GeminiProvider(api_key="test-key")

    def test_build_config_no_format(self, provider):
        """Test building config without response format."""
        with patch("markit.llm.gemini.types.GenerateContentConfig") as mock_config:
            provider._build_generation_config(None)
            mock_config.assert_called_once_with()

    def test_build_config_json_object(self, provider):
        """Test building config with json_object format."""
        with patch("markit.llm.gemini.types.GenerateContentConfig") as mock_config:
            response_format = ResponseFormat(type="json_object")
            provider._build_generation_config(response_format)

            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args.kwargs
            assert call_kwargs["response_mime_type"] == "application/json"

    def test_build_config_json_schema(self, provider):
        """Test building config with json_schema format."""
        with patch("markit.llm.gemini.types.GenerateContentConfig") as mock_config:
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            response_format = ResponseFormat(type="json_schema", json_schema=schema)
            provider._build_generation_config(response_format)

            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args.kwargs
            assert call_kwargs["response_mime_type"] == "application/json"
            assert call_kwargs["response_schema"] == schema

    def test_build_config_text_format(self, provider):
        """Test building config with text format."""
        with patch("markit.llm.gemini.types.GenerateContentConfig") as mock_config:
            response_format = ResponseFormat(type="text")
            provider._build_generation_config(response_format)

            mock_config.assert_called_once_with()


class TestGeminiProviderConvertToContents:
    """Tests for GeminiProvider._convert_to_gemini_contents method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.gemini.genai.Client"):
            return GeminiProvider(api_key="test-key")

    def test_convert_simple_messages(self, provider):
        """Test converting simple text messages."""
        with patch("markit.llm.gemini.types.Part") as mock_part:
            mock_part.from_text = MagicMock(side_effect=lambda text: f"part:{text}")

            messages = [
                LLMMessage.system("You are helpful"),
                LLMMessage.user("Hello"),
                LLMMessage.assistant("Hi!"),
            ]

            result = provider._convert_to_gemini_contents(messages)

            # Should have 3 parts
            assert len(result) == 3
            # System message should be prefixed
            assert "part:[System]:" in result[0]

    def test_convert_multimodal_message(self, provider):
        """Test converting multimodal message with image."""
        with patch("markit.llm.gemini.types.Part") as mock_part:
            mock_part.from_text = MagicMock(side_effect=lambda text: f"text:{text}")
            mock_part.from_bytes = MagicMock(
                side_effect=lambda data=None, mime_type=None: f"image:{mime_type}"  # noqa: ARG005
            )

            message = LLMMessage.user_with_image("Describe this", b"fake_image", "png")

            result = provider._convert_to_gemini_contents([message])

            # Should have text and image parts
            assert len(result) == 2


class TestGeminiProviderValidate:
    """Tests for GeminiProvider.validate method."""

    @pytest.mark.asyncio
    async def test_validate_success(self):
        """Test successful validation."""
        with patch("markit.llm.gemini.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            provider = GeminiProvider(api_key="test-key")
            result = await provider.validate()

            # Gemini validate just checks if client exists
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_no_client(self):
        """Test validation when client is None."""
        with patch("markit.llm.gemini.genai.Client"):
            provider = GeminiProvider(api_key="test-key")
            provider.client = None  # type: ignore[assignment]

            result = await provider.validate()

            assert result is False
