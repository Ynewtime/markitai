"""Ollama LLM provider implementation."""

import base64
from collections.abc import AsyncIterator
from typing import Any

import ollama

from markit.exceptions import LLMError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse, ResponseFormat, TokenUsage
from markit.utils.logging import get_logger

log = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama API provider for local LLM models."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2-vision",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            model: Model to use (default: llama3.2-vision)
            host: Ollama server host URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.client = ollama.AsyncClient(host=host, timeout=float(timeout))

    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a completion using Ollama API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Returns:
            LLM response
        """
        try:
            converted_messages = self._convert_messages_ollama(messages)

            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            response = await self.client.chat(
                model=kwargs.get("model", self.model),
                messages=converted_messages,
                options=options,
            )

            content = response.get("message", {}).get("content", "")

            # Ollama provides token counts
            usage = None
            if "prompt_eval_count" in response or "eval_count" in response:
                usage = TokenUsage(
                    prompt_tokens=response.get("prompt_eval_count", 0),
                    completion_tokens=response.get("eval_count", 0),
                )

            log.debug(
                "Ollama completion",
                model=self.model,
                tokens=usage.total_tokens if usage else 0,
            )

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.get("model", self.model),
                finish_reason=response.get("done_reason", "stop"),
            )

        except Exception as e:
            log.error("Ollama API error", error=str(e))
            raise LLMError(f"Ollama API error: {e}") from e

    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion from Ollama API.

        Args:
            messages: List of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Chunks of the response
        """
        try:
            converted_messages = self._convert_messages_ollama(messages)

            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            async for chunk in await self.client.chat(
                model=kwargs.get("model", self.model),
                messages=converted_messages,
                options=options,
                stream=True,
            ):
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]

        except Exception as e:
            log.error("Ollama streaming error", error=str(e))
            raise LLMError(f"Ollama streaming error: {e}") from e

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        _image_format: str = "png",  # Ollama auto-detects format
        **kwargs: Any,
    ) -> LLMResponse:
        """Analyze an image using Ollama Vision model.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for image analysis
            _image_format: Format of the image (unused, Ollama auto-detects)
            **kwargs: Additional arguments including:
                - response_format: ResponseFormat for structured output

        Returns:
            LLM response with image analysis
        """
        try:
            # Ollama expects base64 encoded images
            b64_image = base64.b64encode(image_data).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image],
                }
            ]

            # Build request kwargs with optional JSON mode
            response_format: ResponseFormat | None = kwargs.pop("response_format", None)
            request_kwargs: dict[str, Any] = {
                "model": kwargs.get("model", self.model),
                "messages": messages,
            }

            # Add format parameter for JSON mode
            if response_format:
                request_kwargs["format"] = self._build_format(response_format)

            response = await self.client.chat(**request_kwargs)

            content = response.get("message", {}).get("content", "")

            usage = None
            if "prompt_eval_count" in response or "eval_count" in response:
                usage = TokenUsage(
                    prompt_tokens=response.get("prompt_eval_count", 0),
                    completion_tokens=response.get("eval_count", 0),
                )

            return LLMResponse(
                content=content,
                usage=usage,
                model=response.get("model", self.model),
                finish_reason=response.get("done_reason", "stop"),
            )

        except Exception as e:
            log.error("Ollama image analysis error", error=str(e))
            raise LLMError(f"Ollama image analysis error: {e}") from e

    def _build_format(self, response_format: ResponseFormat) -> str | dict[str, Any]:
        """Build Ollama format parameter from ResponseFormat.

        Args:
            response_format: ResponseFormat configuration

        Returns:
            "json" or JSON schema dict
        """
        if response_format.type == "json_schema" and response_format.json_schema:
            return response_format.json_schema
        else:
            # Use simple JSON mode
            return "json"

    def _convert_messages_ollama(self, messages: list[LLMMessage]) -> list[dict]:
        """Convert LLMMessages to Ollama format."""
        result = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Handle multimodal - extract images
                text_parts = []
                images = []
                for part in msg.content:
                    if part.type == "text":
                        text_parts.append(part.text)
                    elif part.type == "image_url" and part.image_url:
                        if part.image_url.startswith("data:"):
                            # Extract base64 from data URL
                            _, b64_data = part.image_url.split(",", 1)
                            images.append(b64_data)

                message = {"role": msg.role, "content": " ".join(text_parts)}
                if images:
                    message["images"] = images
                result.append(message)

        return result

    async def validate(self) -> bool:
        """Validate the Ollama provider configuration."""
        try:
            # Check if the model is available
            response = await self.client.list()
            # New ollama library returns typed objects, not dicts
            model_names = [m.model for m in response.models]
            is_valid = any(self.model in name for name in model_names)
            if not is_valid:
                log.warning(
                    "Ollama model not found",
                    requested=self.model,
                    available=model_names,
                )
            return is_valid
        except Exception as e:
            log.warning("Ollama validation failed", error=str(e))
            return False
