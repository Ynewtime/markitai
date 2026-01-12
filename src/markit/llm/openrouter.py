"""OpenRouter LLM provider implementation."""

from markit.llm.openai import OpenAIProvider
from markit.utils.logging import get_logger

log = get_logger(__name__)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter API provider using OpenAI-compatible interface.

    Inherits from OpenAIProvider since OpenRouter uses the OpenAI-compatible API.
    The main differences are:
    - Default base_url is https://openrouter.ai/api/v1
    - Uses max_tokens instead of max_completion_tokens
    - Different default model
    """

    name = "openrouter"

    # OpenRouter uses standard max_tokens, not max_completion_tokens
    _use_max_completion_tokens = False

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "google/gemini-3-flash-preview",
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (default: google/gemini-3-flash-preview)
            base_url: Optional custom base URL (default: https://openrouter.ai/api/v1)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://openrouter.ai/api/v1",
            timeout=timeout,
            max_retries=max_retries,
        )

    async def validate(self) -> bool:
        """Validate the OpenRouter provider configuration.

        Note: This only checks that the client is configured, not that credentials
        are valid. Actual API errors will be caught during first use.
        """
        # Just verify the client was created successfully
        # Actual API validation (listing models) is expensive and slows down startup
        return self.client is not None
