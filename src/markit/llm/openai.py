"""OpenAI LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, ResponseFormat, TokenUsage
from markit.utils.logging import generate_request_id, get_logger

log = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider using official SDK."""

    name = "openai"

    # GPT-5.2+ models require max_completion_tokens instead of max_tokens
    # Subclasses can override this for OpenAI-compatible APIs that use max_tokens
    _use_max_completion_tokens = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2",
        base_url: str | None = None,
        timeout: int = 120,
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
        request_id = generate_request_id()
        model_name = kwargs.get("model", self.model)
        start_time = time.perf_counter()

        log.debug(
            "Sending LLM request",
            provider=self.name,
            model=model_name,
            request_id=request_id,
        )

        try:
            converted_messages = self._convert_messages(messages)

            # Build request params, excluding None values (OpenAI API rejects null)
            request_params: dict[str, Any] = {
                "model": model_name,
                "messages": converted_messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                # GPT-5.2+ models require max_completion_tokens instead of max_tokens
                # OpenAI-compatible APIs (like OpenRouter) use standard max_tokens
                token_key = (
                    "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
                )
                request_params[token_key] = max_tokens

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

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log.debug(
                "LLM response received",
                provider=self.name,
                model=response.model,
                request_id=request_id,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                duration_ms=duration_ms,
            )

            return LLMResponse(
                content=choice.message.content or "",
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason or "unknown",
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log.warning(
                "LLM request failed",
                provider=self.name,
                model=model_name,
                request_id=request_id,
                duration_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
            )
            self._handle_api_error(e, "API", log)

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
                # GPT-5.2+ models require max_completion_tokens instead of max_tokens
                # OpenAI-compatible APIs (like OpenRouter) use standard max_tokens
                token_key = (
                    "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
                )
                request_params[token_key] = max_tokens

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
            self._handle_api_error(e, "streaming", log)

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
            **kwargs: Additional arguments including:
                - response_format: ResponseFormat for structured output

        Returns:
            LLM response with image analysis
        """
        message = LLMMessage.user_with_image(prompt, image_data, image_format)

        # Handle response_format for JSON mode
        response_format: ResponseFormat | None = kwargs.pop("response_format", None)
        if response_format:
            kwargs["response_format"] = self._build_response_format(response_format)

        return await self.complete([message], **kwargs)

    def _build_response_format(self, response_format: ResponseFormat) -> dict[str, Any]:
        """Build OpenAI response_format parameter from ResponseFormat.

        Args:
            response_format: ResponseFormat configuration

        Returns:
            OpenAI API response_format dict
        """
        if response_format.type == "json_object":
            return {"type": "json_object"}
        elif response_format.type == "json_schema" and response_format.json_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "image_analysis",
                    "strict": response_format.strict,
                    "schema": response_format.json_schema,
                },
            }
        else:
            return {"type": "text"}

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
