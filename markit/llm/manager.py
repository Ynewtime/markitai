"""LLM Provider Manager with fallback support."""

import asyncio
import os
from typing import Literal

from markit.config.settings import LLMProviderConfig
from markit.exceptions import LLMError, ProviderNotFoundError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from markit.utils.logging import get_logger

log = get_logger(__name__)

ProviderType = Literal["openai", "anthropic", "gemini", "ollama", "openrouter"]


class ProviderManager:
    """Manages multiple LLM providers with fallback support."""

    def __init__(self, configs: list[LLMProviderConfig] | None = None) -> None:
        """Initialize the provider manager.

        Args:
            configs: List of provider configurations (in priority order)
        """
        self.configs = configs or []
        self._providers: dict[str, BaseLLMProvider] = {}
        self._valid_providers: list[str] = []
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize and validate all configured providers."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            for config in self.configs:
                try:
                    provider = self._create_provider(config)
                    if await self._validate_provider(provider, config):
                        self._providers[config.provider] = provider
                        self._valid_providers.append(config.provider)
                        log.info(f"Provider {config.provider} initialized successfully")
                except Exception as e:
                    log.warning(f"Provider {config.provider} initialization failed: {e}")

            if not self._valid_providers:
                log.warning("No valid LLM providers available")

            self._initialized = True

    def _create_provider(self, config: LLMProviderConfig) -> BaseLLMProvider:
        """Create a provider instance from configuration."""
        # Get API key from config or environment
        api_key = config.api_key or self._get_api_key_from_env(config.provider)

        if config.provider == "openai":
            from markit.llm.openai import OpenAIProvider

            return OpenAIProvider(
                api_key=api_key,
                model=config.model,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

        elif config.provider == "anthropic":
            from markit.llm.anthropic import AnthropicProvider

            return AnthropicProvider(
                api_key=api_key,
                model=config.model,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

        elif config.provider == "gemini":
            from markit.llm.gemini import GeminiProvider

            return GeminiProvider(
                api_key=api_key,
                model=config.model,
            )

        elif config.provider == "ollama":
            from markit.llm.ollama import OllamaProvider

            return OllamaProvider(
                model=config.model,
                host=config.base_url or "http://localhost:11434",
            )

        elif config.provider == "openrouter":
            from markit.llm.openrouter import OpenRouterProvider

            return OpenRouterProvider(
                api_key=api_key,
                model=config.model,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def _get_api_key_from_env(self, provider: str) -> str | None:
        """Get API key from environment variables."""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var = env_vars.get(provider)
        return os.environ.get(env_var) if env_var else None

    async def _validate_provider(
        self, provider: BaseLLMProvider, config: LLMProviderConfig
    ) -> bool:
        """Validate that a provider is properly configured."""
        # Check for required API key
        if config.provider in ("openai", "anthropic", "openrouter"):
            api_key = config.api_key or self._get_api_key_from_env(config.provider)
            if not api_key:
                log.warning(f"{config.provider} requires an API key")
                return False

        if config.provider == "gemini":
            api_key = config.api_key or self._get_api_key_from_env(config.provider)
            if not api_key:
                log.warning("Gemini requires GOOGLE_API_KEY")
                return False

        # Optionally validate connection
        try:
            return await provider.validate()
        except Exception as e:
            log.warning(f"Provider validation failed: {e}")
            return True  # Allow provider if validation call fails

    def get_default(self) -> BaseLLMProvider:
        """Get the default (first valid) provider.

        Returns:
            The default provider

        Raises:
            ProviderNotFoundError: If no valid provider is available
        """
        if not self._valid_providers:
            raise ProviderNotFoundError("No valid LLM provider available")
        return self._providers[self._valid_providers[0]]

    def get_provider(self, name: str) -> BaseLLMProvider:
        """Get a specific provider by name.

        Args:
            name: Provider name

        Returns:
            The requested provider

        Raises:
            ProviderNotFoundError: If the provider is not available
        """
        if name not in self._providers:
            raise ProviderNotFoundError(f"Provider '{name}' is not available")
        return self._providers[name]

    async def complete_with_fallback(
        self,
        messages: list[LLMMessage],
        **kwargs,
    ) -> LLMResponse:
        """Make a completion request with automatic fallback.

        Tries each provider in order until one succeeds.

        Args:
            messages: List of messages
            **kwargs: Additional arguments passed to the provider

        Returns:
            LLM response

        Raises:
            LLMError: If all providers fail
        """
        if not self._initialized:
            await self.initialize()

        if not self._valid_providers:
            raise ProviderNotFoundError("No valid LLM provider available")

        errors = []

        for provider_name in self._valid_providers:
            provider = self._providers[provider_name]
            try:
                log.debug(f"Trying provider: {provider_name}")
                return await provider.complete(messages, **kwargs)
            except Exception as e:
                log.warning(f"Provider {provider_name} failed: {e}")
                errors.append((provider_name, e))
                continue

        # All providers failed
        error_details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise LLMError(f"All providers failed: {error_details}")

    async def analyze_image_with_fallback(
        self,
        image_data: bytes,
        prompt: str,
        image_format: str = "png",
        **kwargs,
    ) -> LLMResponse:
        """Analyze an image with automatic fallback.

        Args:
            image_data: Raw image bytes
            prompt: Prompt for analysis
            image_format: Image format
            **kwargs: Additional arguments

        Returns:
            LLM response

        Raises:
            LLMError: If all providers fail
        """
        if not self._initialized:
            await self.initialize()

        if not self._valid_providers:
            raise ProviderNotFoundError("No valid LLM provider available")

        errors = []

        for provider_name in self._valid_providers:
            provider = self._providers[provider_name]
            try:
                log.debug(f"Trying image analysis with provider: {provider_name}")
                return await provider.analyze_image(image_data, prompt, image_format, **kwargs)
            except Exception as e:
                log.warning(f"Provider {provider_name} image analysis failed: {e}")
                errors.append((provider_name, e))
                continue

        error_details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise LLMError(f"All providers failed for image analysis: {error_details}")

    @property
    def available_providers(self) -> list[str]:
        """Return list of available provider names."""
        return self._valid_providers.copy()

    @property
    def has_providers(self) -> bool:
        """Check if any providers are available."""
        return len(self._valid_providers) > 0
