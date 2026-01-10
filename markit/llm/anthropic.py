"""Anthropic Claude LLM provider implementation."""

import base64
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from markit.exceptions import LLMError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, TokenUsage
from markit.utils.logging import get_logger

log = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider using official SDK."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-5)
            base_url: Optional custom base URL (for proxies)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.model = model
        self.client = AsyncAnthropic(
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
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using Anthropic API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (required for Anthropic)
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        try:
            # Separate system message from others (Anthropic handles system differently)
            system_message = None
            user_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                else:
                    user_messages.append(self._convert_anthropic_message(msg))

            # Build create params, filtering out None values (Anthropic API doesn't accept None)
            create_params = {
                "model": kwargs.get("model", self.model),
                "messages": user_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_message:
                create_params["system"] = system_message
            # Add kwargs, excluding 'model' and None values
            for k, v in kwargs.items():
                if k != "model" and v is not None:
                    create_params[k] = v

            response = await self.client.messages.create(**create_params)

            content = ""
            if response.content:
                content = response.content[0].text

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

            log.debug(
                "Anthropic completion",
                model=response.model,
                tokens=usage.total_tokens,
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.stop_reason or "unknown",
            )

        except Exception as e:
            self._handle_api_error(e, "API", log)

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion from Anthropic API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Chunks of the response
        """
        try:
            # Separate system message
            system_message = None
            user_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                else:
                    user_messages.append(self._convert_anthropic_message(msg))

            # Build stream params, filtering out None values (Anthropic API doesn't accept None)
            stream_params = {
                "model": kwargs.get("model", self.model),
                "messages": user_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }
            if system_message:
                stream_params["system"] = system_message
            # Add kwargs, excluding 'model' and None values
            for k, v in kwargs.items():
                if k != "model" and v is not None:
                    stream_params[k] = v

            async with self.client.messages.stream(**stream_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            self._handle_api_error(e, "streaming", log)

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs,
    ) -> LLMResponse:
        """Analyze an image using Anthropic Vision API.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for image analysis
            image_format: Format of the image
            **kwargs: Additional arguments

        Returns:
            LLM response with image analysis
        """
        b64_image = base64.b64encode(image_data).decode("utf-8")
        media_type = f"image/{image_format}"

        # Use the raw Anthropic format for images
        user_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            response = await self.client.messages.create(
                model=kwargs.get("model", self.model),
                messages=user_messages,
                max_tokens=kwargs.get("max_tokens", 4096),
            )

            content = ""
            if response.content:
                content = response.content[0].text

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.model,
                finish_reason=response.stop_reason or "unknown",
            )

        except Exception as e:
            log.error("Anthropic image analysis error", error=str(e))
            raise LLMError(f"Anthropic image analysis error: {e}") from e

    def _convert_anthropic_message(self, msg: LLMMessage) -> dict:
        """Convert LLMMessage to Anthropic format."""
        if isinstance(msg.content, str):
            return {"role": msg.role, "content": msg.content}

        # Handle multimodal content
        content_parts = []
        for part in msg.content:
            if part.type == "text":
                content_parts.append({"type": "text", "text": part.text})
            elif part.type == "image_url" and part.image_url:
                # Extract base64 data from data URL
                if part.image_url.startswith("data:"):
                    # Parse data URL
                    header, b64_data = part.image_url.split(",", 1)
                    media_type = header.split(";")[0].split(":")[1]
                    content_parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        }
                    )

        return {"role": msg.role, "content": content_parts}

    async def validate(self) -> bool:
        """Validate the Anthropic provider configuration."""
        try:
            # Make a minimal API call
            await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            log.warning("Anthropic validation failed", error=str(e))
            return False
