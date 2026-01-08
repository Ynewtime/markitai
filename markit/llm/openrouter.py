"""OpenRouter LLM provider implementation."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, TokenUsage
from markit.utils.logging import get_logger

log = get_logger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider using OpenAI-compatible interface."""

    name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "google/gemini-3-flash-preview",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (default: google/gemini-3-flash-preview)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using OpenRouter API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        try:
            converted_messages = self._convert_messages(messages)

            # Build request params, excluding None values (OpenAI-compatible API rejects null)
            request_params: dict = {
                "model": kwargs.get("model", self.model),
                "messages": converted_messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Add any extra kwargs (excluding 'model' which is already handled)
            for k, v in kwargs.items():
                if k != "model" and v is not None:
                    request_params[k] = v

            response = await self.client.chat.completions.create(
                **request_params  # type: ignore[arg-type]
            )

            choice = response.choices[0]
            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

            log.debug(
                "OpenRouter completion",
                model=response.model,
                tokens=usage.total_tokens if usage else 0,
            )

            return LLMResponse(
                content=choice.message.content or "",
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason or "unknown",
            )

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "rate limit" in error_str:
                raise RateLimitError() from e
            log.error("OpenRouter API error", error=str(e))
            raise LLMError(f"OpenRouter API error: {e}") from e

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion from OpenRouter API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Chunks of the response
        """
        try:
            converted_messages = self._convert_messages(messages)

            # Build request params, excluding None values (OpenAI-compatible API rejects null)
            request_params: dict = {
                "model": kwargs.get("model", self.model),
                "messages": converted_messages,
                "temperature": temperature,
                "stream": True,
            }
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # Add any extra kwargs (excluding 'model' which is already handled)
            for k, v in kwargs.items():
                if k != "model" and v is not None:
                    request_params[k] = v

            stream = await self.client.chat.completions.create(
                **request_params  # type: ignore[arg-type]
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "rate limit" in error_str:
                raise RateLimitError() from e
            log.error("OpenRouter streaming error", error=str(e))
            raise LLMError(f"OpenRouter streaming error: {e}") from e

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs,
    ) -> LLMResponse:
        """Analyze an image using OpenRouter Vision API.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for image analysis
            image_format: Format of the image
            **kwargs: Additional arguments

        Returns:
            LLM response with image analysis
        """
        message = LLMMessage.user_with_image(prompt, image_data, image_format)
        return await self.complete([message], **kwargs)

    async def validate(self) -> bool:
        """Validate the OpenRouter provider configuration.

        Note: This only checks that the client is configured, not that credentials
        are valid. Actual API errors will be caught during first use.
        """
        # Just verify the client was created successfully
        # Actual API validation (listing models) is expensive and slows down startup
        return self.client is not None
