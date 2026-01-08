"""OpenAI LLM provider implementation."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from markit.exceptions import LLMError, RateLimitError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, TokenUsage
from markit.utils.logging import get_logger

log = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider using official SDK."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-5.2)
            base_url: Optional custom base URL (for proxies/compatible APIs)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using OpenAI API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to the API

        Returns:
            LLM response
        """
        try:
            converted_messages = self._convert_messages(messages)

            # Build request params, excluding None values (OpenAI API rejects null)
            request_params: dict[str, Any] = {
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

            # Note: messages dict format is compatible at runtime
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
                "OpenAI completion",
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
            log.error("OpenAI API error", error=str(e))
            raise LLMError(f"OpenAI API error: {e}") from e

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from OpenAI API.

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

            # Build request params, excluding None values (OpenAI API rejects null)
            request_params: dict[str, Any] = {
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
            log.error("OpenAI streaming error", error=str(e))
            raise LLMError(f"OpenAI streaming error: {e}") from e

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs: Any,
    ) -> LLMResponse:
        """Analyze an image using OpenAI Vision API.

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
        """Validate the OpenAI provider configuration.

        Returns:
            True if the provider is properly configured
        """
        try:
            # Make a minimal API call to verify credentials
            await self.client.models.list()
            return True
        except Exception as e:
            log.warning("OpenAI validation failed", error=str(e))
            return False
