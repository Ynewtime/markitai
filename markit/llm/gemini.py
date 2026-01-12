"""Google Gemini LLM provider implementation."""

import time
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from markit.exceptions import LLMError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, ResponseFormat, TokenUsage
from markit.utils.logging import generate_request_id, get_logger

log = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider using official SDK."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3-flash-preview",
        timeout: int = 120,
        max_retries: int = 3,  # noqa: ARG002
    ) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use (default: gemini-3-flash-preview)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries (reserved for future use)
        """
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=timeout * 1000,  # Convert to milliseconds
            ),
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Gemini API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        request_id = generate_request_id()
        model_name = kwargs.get("model", self.model)
        start_time = time.perf_counter()

        log.debug(
            "Sending LLM request",
            provider="gemini",
            model=model_name,
            request_id=request_id,
        )

        try:
            # Convert messages to Gemini format
            contents = self._convert_to_gemini_contents(messages)

            # Build generation config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Note: SDK accepts list[Part] at runtime despite type hints
            response = await self.client.aio.models.generate_content(
                model=model_name,
                contents=contents,  # type: ignore[arg-type]
                config=config,
            )

            content = ""
            if response.text:
                content = response.text

            # Extract usage if available
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                    completion_tokens=response.usage_metadata.candidates_token_count or 0,
                )

            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log.debug(
                "LLM response received",
                provider="gemini",
                model=model_name,
                request_id=request_id,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                duration_ms=duration_ms,
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=model_name,
                finish_reason="stop",
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log.warning(
                "LLM request failed",
                provider="gemini",
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
        """Stream a completion from Gemini API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Chunks of the response
        """
        try:
            contents = self._convert_to_gemini_contents(messages)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # generate_content_stream returns an async iterator
            stream = await self.client.aio.models.generate_content_stream(
                model=kwargs.get("model", self.model),
                contents=contents,  # type: ignore[arg-type]
                config=config,
            )

            async for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            self._handle_api_error(e, "streaming", log)

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs: Any,
    ) -> LLMResponse:
        """Analyze an image using Gemini Vision API.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for image analysis
            image_format: Format of the image
            **kwargs: Additional arguments including:
                - response_format: ResponseFormat for structured output

        Returns:
            LLM response with image analysis
        """
        try:
            # Create image part using from_bytes with keyword arguments
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type=f"image/{image_format}",
            )

            # Create text part using from_text with keyword argument
            text_part = types.Part.from_text(text=prompt)

            # Use list of Parts
            contents = [image_part, text_part]

            # Build generation config with optional JSON mode
            response_format: ResponseFormat | None = kwargs.pop("response_format", None)
            config = self._build_generation_config(response_format)

            response = await self.client.aio.models.generate_content(
                model=kwargs.get("model", self.model),
                contents=contents,  # type: ignore[arg-type]
                config=config,
            )

            content = ""
            if response.text:
                content = response.text

            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                    completion_tokens=response.usage_metadata.candidates_token_count or 0,
                )

            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                finish_reason="stop",
            )

        except Exception as e:
            log.error("Gemini image analysis error", error=str(e))
            raise LLMError(f"Gemini image analysis error: {e}") from e

    def _build_generation_config(
        self, response_format: ResponseFormat | None = None
    ) -> types.GenerateContentConfig:
        """Build Gemini generation config with optional JSON mode.

        Args:
            response_format: Optional ResponseFormat for structured output

        Returns:
            Gemini GenerateContentConfig
        """
        if response_format and response_format.type in ("json_object", "json_schema"):
            # Enable JSON mode with optional schema
            config_kwargs: dict[str, Any] = {
                "response_mime_type": "application/json",
            }
            if response_format.json_schema:
                config_kwargs["response_schema"] = response_format.json_schema
            return types.GenerateContentConfig(**config_kwargs)
        else:
            return types.GenerateContentConfig()

    def _convert_to_gemini_contents(self, messages: list[LLMMessage]) -> list[types.Part]:
        """Convert LLMMessages to Gemini Parts format.

        Uses list[Part] format for better type compatibility with the SDK.
        For multi-turn conversations, we concatenate all message content.
        """
        parts: list[types.Part] = []

        for msg in messages:
            if msg.role == "system":
                # Prepend system message as text (Gemini doesn't have system role)
                if isinstance(msg.content, str):
                    parts.append(types.Part.from_text(text=f"[System]: {msg.content}\n\n"))
                continue

            if isinstance(msg.content, str):
                # Add role prefix for multi-turn context
                prefix = "[User]: " if msg.role == "user" else "[Assistant]: "
                parts.append(types.Part.from_text(text=f"{prefix}{msg.content}"))
            else:
                # Handle multimodal content
                for part in msg.content:
                    if part.type == "text" and part.text:
                        parts.append(types.Part.from_text(text=part.text))
                    elif (
                        part.type == "image_url"
                        and part.image_url
                        and part.image_url.startswith("data:")
                    ):
                        # Handle base64 data URLs
                        import base64

                        header, b64_data = part.image_url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        image_bytes = base64.b64decode(b64_data)
                        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        return parts

    async def validate(self) -> bool:
        """Validate the Gemini provider configuration.

        Note: This only checks that the client is configured, not that credentials
        are valid. Actual API errors will be caught during first use.
        """
        # Just verify the client was created successfully
        # Actual API validation is expensive and slows down startup
        return self.client is not None
