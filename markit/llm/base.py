"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from markit.utils.logging import BoundLogger


@dataclass
class ResponseFormat:
    """Configuration for structured output format.

    Supports:
    - OpenAI: response_format={"type": "json_object"} or json_schema
    - Gemini: response_mime_type="application/json" with response_schema
    - Ollama: format="json" or format=schema
    - Anthropic: Tool Use with tool_choice (forces structured output)
    """

    type: Literal["json_object", "json_schema", "text"] = "json_object"
    json_schema: dict[str, Any] | None = None
    # If True, use strict mode (OpenAI) - guarantees 100% schema match
    strict: bool = False


# Pre-defined JSON schema for image analysis
IMAGE_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "alt_text": {
            "type": "string",
            "description": "Short description for image alt text (max 50 chars)",
        },
        "detailed_description": {
            "type": "string",
            "description": "Detailed description of the image content",
        },
        "detected_text": {
            "type": ["string", "null"],
            "description": "Text visible in the image, null if none",
        },
        "image_type": {
            "type": "string",
            "enum": [
                "diagram",
                "chart",
                "graph",
                "table",
                "screenshot",
                "photo",
                "illustration",
                "logo",
                "icon",
                "formula",
                "code",
                "other",
            ],
            "description": "Type classification of the image",
        },
        "knowledge_meta": {
            "type": ["object", "null"],
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named entities in the image",
                },
                "relationships": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity relationships",
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topic tags",
                },
                "domain": {
                    "type": ["string", "null"],
                    "description": "Domain classification",
                },
            },
        },
    },
    "required": ["alt_text", "detailed_description", "image_type"],
}


@dataclass
class TokenUsage:
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ContentPart:
    """A part of a message content (for multimodal)."""

    type: str  # "text" or "image_url"
    text: str | None = None
    image_url: str | None = None  # Base64 data URL or HTTP URL
    image_data: bytes | None = None  # Raw image bytes


@dataclass
class LLMMessage:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str | list[ContentPart]

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "LLMMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def user_with_image(
        cls, text: str, image_data: bytes, image_format: str = "png"
    ) -> "LLMMessage":
        """Create a user message with an image."""
        import base64

        b64_image = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:image/{image_format};base64,{b64_image}"

        return cls(
            role="user",
            content=[
                ContentPart(type="text", text=text),
                ContentPart(type="image_url", image_url=image_url),
            ],
        )


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    usage: TokenUsage | None
    model: str
    finish_reason: str
    estimated_cost: float | None = None  # Estimated cost in USD (if cost config available)


@dataclass
class LLMTaskResultWithStats:
    """Wrapper for LLM task results that carries statistics.

    This allows pipeline components to return both their business result
    and the LLM statistics for tracking.
    """

    result: Any  # The actual task result (ImageAnalysis, str, etc.)
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0

    @classmethod
    def from_response(cls, result: Any, response: "LLMResponse") -> "LLMTaskResultWithStats":
        """Create from an LLMResponse.

        Args:
            result: The business result (e.g., ImageAnalysis)
            response: The LLM response containing stats

        Returns:
            LLMTaskResultWithStats with extracted statistics
        """
        return cls(
            result=result,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            estimated_cost=response.estimated_cost or 0.0,
        )


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific arguments

        Returns:
            LLM response
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion as an async generator.

        Implementations should be async generators (async def with yield).

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific arguments

        Yields:
            Chunks of the response
        """
        # This yield is required for the type checker to recognize this as an async generator
        # Subclasses will override this method completely
        yield ""  # pragma: no cover
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs: Any,
    ) -> LLMResponse:
        """Analyze an image.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for image analysis
            image_format: Format of the image (png, jpeg, etc.)
            **kwargs: Provider-specific arguments

        Returns:
            LLM response with image analysis
        """
        ...

    async def validate(self) -> bool:
        """Validate the provider configuration.

        Returns:
            True if the provider is properly configured
        """
        return True

    def _handle_api_error(
        self,
        error: Exception,
        operation: str,
        log: "BoundLogger",
    ) -> None:
        """Handle API errors with consistent rate limit detection and logging.

        This method checks for rate limit errors and raises the appropriate exception.
        It should be called from exception handlers in provider implementations.

        Args:
            error: The caught exception
            operation: Description of the operation that failed (e.g., "API", "streaming")
            log: The logger instance to use for error logging

        Raises:
            RateLimitError: If the error indicates a rate limit
            LLMError: For all other errors
        """
        from markit.exceptions import LLMError, RateLimitError

        error_str = str(error).lower()
        if "rate_limit" in error_str or "rate limit" in error_str:
            raise RateLimitError() from error
        log.error(f"{self.name} {operation} error", error=str(error))
        raise LLMError(f"{self.name} {operation} error: {error}") from error

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert LLMMessage objects to provider-specific format.

        Args:
            messages: List of LLMMessage objects

        Returns:
            List of message dictionaries
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg.content, str):
                result.append({"role": msg.role, "content": msg.content})
            else:
                # Multimodal content
                content_parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if part.type == "text":
                        content_parts.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.image_url},
                            }
                        )
                result.append({"role": msg.role, "content": content_parts})
        return result
