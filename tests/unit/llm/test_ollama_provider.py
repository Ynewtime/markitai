"""Tests for Ollama LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markit.exceptions import LLMError
from markit.llm.base import LLMMessage, ResponseFormat
from markit.llm.ollama import OllamaProvider


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        with patch("markit.llm.ollama.ollama.AsyncClient") as mock_client:
            provider = OllamaProvider()

            assert provider.model == "llama3.2-vision"
            mock_client.assert_called_once_with(
                host="http://localhost:11434",
                timeout=120.0,
            )

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch("markit.llm.ollama.ollama.AsyncClient") as mock_client:
            provider = OllamaProvider(
                model="mistral",
                host="http://custom:11434",
                timeout=60,
            )

            assert provider.model == "mistral"
            mock_client.assert_called_once_with(
                host="http://custom:11434",
                timeout=60.0,
            )


class TestOllamaProviderComplete:
    """Tests for OllamaProvider.complete method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.ollama.ollama.AsyncClient"):
            provider = OllamaProvider()
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_complete_success(self, provider):
        """Test successful completion."""
        provider.client.chat = AsyncMock(
            return_value={
                "message": {"content": "Hello from Ollama!"},
                "model": "llama3.2-vision",
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        )

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages, temperature=0.5, max_tokens=100)

        assert response.content == "Hello from Ollama!"
        assert response.model == "llama3.2-vision"
        assert response.finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_with_options(self, provider):
        """Test completion passes correct options."""
        provider.client.chat = AsyncMock(
            return_value={
                "message": {"content": "Hi"},
                "model": "llama3.2-vision",
            }
        )

        messages = [LLMMessage.user("Hi")]
        await provider.complete(messages, temperature=0.8, max_tokens=200)

        call_kwargs = provider.client.chat.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.8
        assert call_kwargs["options"]["num_predict"] == 200

    @pytest.mark.asyncio
    async def test_complete_without_usage(self, provider):
        """Test completion when usage info is not available."""
        provider.client.chat = AsyncMock(
            return_value={
                "message": {"content": "Hi"},
                "model": "llama3.2-vision",
            }
        )

        messages = [LLMMessage.user("Hi")]
        response = await provider.complete(messages)

        assert response.content == "Hi"
        assert response.usage is None

    @pytest.mark.asyncio
    async def test_complete_api_error(self, provider):
        """Test completion with API error."""
        provider.client.chat = AsyncMock(side_effect=Exception("Connection refused"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError) as exc:
            await provider.complete(messages)

        assert "Ollama API error" in str(exc.value)


class TestOllamaProviderStream:
    """Tests for OllamaProvider.stream method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.ollama.ollama.AsyncClient"):
            provider = OllamaProvider()
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_stream_success(self, provider):
        """Test successful streaming."""

        async def mock_stream():
            chunks = [
                {"message": {"content": "Hello"}},
                {"message": {"content": " world"}},
                {"message": {"content": "!"}},
            ]
            for chunk in chunks:
                yield chunk

        provider.client.chat = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_options(self, provider):
        """Test streaming passes correct options."""

        async def mock_stream():
            yield {"message": {"content": "Hi"}}

        provider.client.chat = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        async for _ in provider.stream(messages, temperature=0.9, max_tokens=50):
            pass

        call_kwargs = provider.client.chat.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["options"]["temperature"] == 0.9
        assert call_kwargs["options"]["num_predict"] == 50

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, provider):
        """Test streaming with empty chunks."""

        async def mock_stream():
            chunks = [
                {"message": {"content": "Hello"}},
                {"message": {}},  # Empty content
                {"message": {"content": "!"}},
            ]
            for chunk in chunks:
                yield chunk

        provider.client.chat = AsyncMock(return_value=mock_stream())

        messages = [LLMMessage.user("Hi")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", "!"]

    @pytest.mark.asyncio
    async def test_stream_api_error(self, provider):
        """Test streaming with API error."""
        provider.client.chat = AsyncMock(side_effect=Exception("Stream Error"))

        messages = [LLMMessage.user("Hi")]
        with pytest.raises(LLMError):
            async for _ in provider.stream(messages):
                pass


class TestOllamaProviderAnalyzeImage:
    """Tests for OllamaProvider.analyze_image method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.ollama.ollama.AsyncClient"):
            provider = OllamaProvider()
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, provider):
        """Test successful image analysis."""
        provider.client.chat = AsyncMock(
            return_value={
                "message": {"content": "A cute cat"},
                "model": "llama3.2-vision",
                "done_reason": "stop",
                "prompt_eval_count": 100,
                "eval_count": 10,
            }
        )

        response = await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Describe this image",
            image_format="png",
        )

        assert response.content == "A cute cat"
        assert response.usage.prompt_tokens == 100

        # Verify image was passed as base64
        call_kwargs = provider.client.chat.call_args.kwargs
        assert "images" in call_kwargs["messages"][0]
        assert len(call_kwargs["messages"][0]["images"]) == 1

    @pytest.mark.asyncio
    async def test_analyze_image_with_json_format(self, provider):
        """Test image analysis with JSON response format."""
        provider.client.chat = AsyncMock(
            return_value={
                "message": {"content": '{"description": "A cat"}'},
                "model": "llama3.2-vision",
            }
        )

        response_format = ResponseFormat(type="json_object")
        await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Describe this image as JSON",
            response_format=response_format,
        )

        call_kwargs = provider.client.chat.call_args.kwargs
        assert call_kwargs["format"] == "json"

    @pytest.mark.asyncio
    async def test_analyze_image_with_json_schema(self, provider):
        """Test image analysis with JSON schema."""
        provider.client.chat = AsyncMock(
            return_value={
                "message": {"content": '{"type": "animal"}'},
                "model": "llama3.2-vision",
            }
        )

        schema = {"type": "object", "properties": {"type": {"type": "string"}}}
        response_format = ResponseFormat(type="json_schema", json_schema=schema)
        await provider.analyze_image(
            image_data=b"fake_image_data",
            prompt="Categorize this image",
            response_format=response_format,
        )

        call_kwargs = provider.client.chat.call_args.kwargs
        assert call_kwargs["format"] == schema

    @pytest.mark.asyncio
    async def test_analyze_image_error(self, provider):
        """Test image analysis with error."""
        provider.client.chat = AsyncMock(side_effect=Exception("Analysis failed"))

        with pytest.raises(LLMError) as exc:
            await provider.analyze_image(
                image_data=b"fake_image_data",
                prompt="Describe this",
            )

        assert "image analysis error" in str(exc.value)


class TestOllamaProviderBuildFormat:
    """Tests for OllamaProvider._build_format method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.ollama.ollama.AsyncClient"):
            return OllamaProvider()

    def test_build_json_object(self, provider):
        """Test building json_object format."""
        response_format = ResponseFormat(type="json_object")
        result = provider._build_format(response_format)

        assert result == "json"

    def test_build_json_schema(self, provider):
        """Test building json_schema format."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response_format = ResponseFormat(type="json_schema", json_schema=schema)
        result = provider._build_format(response_format)

        assert result == schema

    def test_build_text_format(self, provider):
        """Test building text format falls back to json."""
        response_format = ResponseFormat(type="text")
        result = provider._build_format(response_format)

        assert result == "json"


class TestOllamaProviderConvertMessages:
    """Tests for OllamaProvider._convert_messages_ollama method."""

    @pytest.fixture
    def provider(self):
        """Create a provider."""
        with patch("markit.llm.ollama.ollama.AsyncClient"):
            return OllamaProvider()

    def test_convert_simple_messages(self, provider):
        """Test converting simple text messages."""
        messages = [
            LLMMessage.system("You are helpful"),
            LLMMessage.user("Hello"),
            LLMMessage.assistant("Hi!"),
        ]

        result = provider._convert_messages_ollama(messages)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi!"}

    def test_convert_multimodal_message(self, provider):
        """Test converting multimodal message with image."""
        message = LLMMessage.user_with_image("Describe this", b"fake_image", "png")

        result = provider._convert_messages_ollama([message])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Describe this"
        assert "images" in result[0]
        assert len(result[0]["images"]) == 1


class TestOllamaProviderValidate:
    """Tests for OllamaProvider.validate method."""

    @pytest.fixture
    def provider(self):
        """Create a provider with mocked client."""
        with patch("markit.llm.ollama.ollama.AsyncClient"):
            provider = OllamaProvider()
            provider.client = AsyncMock()
            return provider

    @pytest.mark.asyncio
    async def test_validate_success(self, provider):
        """Test successful validation."""
        mock_model = MagicMock()
        mock_model.model = "llama3.2-vision:latest"

        mock_response = MagicMock()
        mock_response.models = [mock_model]

        provider.client.list = AsyncMock(return_value=mock_response)

        result = await provider.validate()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_model_not_found(self, provider):
        """Test validation when model is not available."""
        mock_model = MagicMock()
        mock_model.model = "mistral:latest"

        mock_response = MagicMock()
        mock_response.models = [mock_model]

        provider.client.list = AsyncMock(return_value=mock_response)

        result = await provider.validate()

        assert result is False

    @pytest.mark.asyncio
    async def test_validate_connection_error(self, provider):
        """Test validation with connection error."""
        provider.client.list = AsyncMock(side_effect=Exception("Connection refused"))

        result = await provider.validate()

        assert result is False
