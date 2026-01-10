"""LLM Provider Manager with fallback support."""

import asyncio
import os

from markit.config.settings import (
    LLMConfig,
    LLMProviderConfig,
)
from markit.exceptions import LLMError, ProviderNotFoundError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from markit.utils.logging import get_logger

log = get_logger(__name__)


class ProviderManager:
    """Manages multiple LLM providers with fallback and load balancing."""

    def __init__(self, llm_config: LLMConfig | list[LLMProviderConfig] | None = None):
        """Initialize provider manager.

        Args:
            llm_config: LLM configuration object or legacy list of provider configs
        """
        self.config = llm_config or LLMConfig()
        # Support legacy list of configs if passed directly (for tests backward compat)
        if isinstance(llm_config, list):
            self.config = LLMConfig(providers=llm_config)

        # Ensure self.config is LLMConfig (handle None case above where it becomes LLMConfig())
        if not isinstance(self.config, LLMConfig):
            # Should be handled by logic above but for type checker:
            self.config = LLMConfig()

        self._providers: dict[str, BaseLLMProvider] = {}
        self._valid_providers: list[str] = []
        self._provider_capabilities: dict[str, list[str]] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._current_index = 0

    async def initialize(self) -> None:
        """Initialize all configured providers."""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            # 1. Process Legacy Providers
            for config in self.config.providers:
                try:
                    # Use name as key if provided, else provider_model
                    provider_key = config.name or f"{config.provider}_{config.model}"
                    provider = self._create_provider(config)
                    if await self._validate_provider(provider, config):
                        self._providers[provider_key] = provider
                        self._valid_providers.append(provider_key)

                        # Optimistic default
                        caps = (
                            config.capabilities
                            if config.capabilities is not None
                            else ["text", "vision"]
                        )
                        self._provider_capabilities[provider_key] = caps

                        log.info(
                            f"Provider {provider_key} initialized successfully", capabilities=caps
                        )
                except Exception as e:
                    log.warning(f"Failed to initialize provider {config.provider}: {e}")

            # 2. Process New Model Configs
            cred_map = {c.id: c for c in self.config.credentials}

            for model_config in self.config.models:
                try:
                    cred = cred_map.get(model_config.credential_id)
                    if not cred:
                        log.warning(
                            f"Credential '{model_config.credential_id}' not found for model '{model_config.name}'"
                        )
                        continue

                    # Synthesize a ProviderConfig from Credential + Model
                    # This allows reusing _create_provider logic
                    synthetic_config = LLMProviderConfig(
                        provider=cred.provider,
                        model=model_config.model,
                        name=model_config.name,
                        api_key=cred.api_key,
                        api_key_env=cred.api_key_env,
                        base_url=cred.base_url,
                        timeout=model_config.timeout,
                        max_retries=model_config.max_retries,
                        capabilities=model_config.capabilities,
                    )

                    provider_key = model_config.name  # Unique name required
                    provider = self._create_provider(synthetic_config)

                    if await self._validate_provider(provider, synthetic_config):
                        self._providers[provider_key] = provider
                        self._valid_providers.append(provider_key)

                        caps = (
                            model_config.capabilities
                            if model_config.capabilities is not None
                            else ["text", "vision"]
                        )
                        self._provider_capabilities[provider_key] = caps

                        log.info(
                            f"Model {provider_key} initialized successfully", capabilities=caps
                        )

                except Exception as e:
                    log.warning(f"Failed to initialize model {model_config.name}: {e}")

            if not self._valid_providers:
                log.warning("No valid LLM providers available. LLM features will be disabled.")

            self._initialized = True

    def _create_provider(self, config: LLMProviderConfig) -> BaseLLMProvider:
        """Create a provider instance from configuration."""
        # Get API key: explicit > env var from config > default env var
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
        if not api_key:
            api_key = self._get_api_key_from_env(config.provider)

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
        # Helper to get effective API key
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
        if not api_key:
            api_key = self._get_api_key_from_env(config.provider)

        # Check for required API key
        if config.provider in ("openai", "anthropic", "openrouter"):
            if not api_key:
                log.warning(f"{config.provider} requires an API key")
                return False

        if config.provider == "gemini":
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

    def has_capability(self, capability: str) -> bool:
        """Check if any initialized provider has the specified capability.

        Args:
            capability: Capability to check (e.g. "vision", "text")

        Returns:
            True if at least one provider supports the capability
        """
        for provider_name in self._valid_providers:
            caps = self._provider_capabilities.get(provider_name, [])
            if capability in caps:
                return True
        return False

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
        provider_count = len(self._valid_providers)

        # Feature 2: Round Robin Load Balancing
        # Start from current index and rotate
        start_index = self._current_index
        # Move index for next request to ensure distribution
        self._current_index = (self._current_index + 1) % provider_count

        for i in range(provider_count):
            idx = (start_index + i) % provider_count
            provider_name = self._valid_providers[idx]
            provider = self._providers[provider_name]
            try:
                # Only log when retrying (not first attempt) or multiple providers
                if i > 0:
                    log.debug(f"Retrying with provider: {provider_name}")
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

        # Filter providers that support vision
        vision_providers = [
            p for p in self._valid_providers if "vision" in self._provider_capabilities.get(p, [])
        ]

        if not vision_providers:
            raise LLMError("No provider with 'vision' capability available for image analysis")

        errors = []
        provider_count = len(vision_providers)

        # Feature 2: Round Robin Load Balancing (across vision-capable providers)
        start_index = self._current_index
        # Move index for next request
        self._current_index = (self._current_index + 1) % provider_count

        for i in range(provider_count):
            idx = (start_index + i) % provider_count
            provider_name = vision_providers[idx]
            provider = self._providers[provider_name]
            try:
                # Only log when retrying (not first attempt)
                if i > 0:
                    log.debug(f"Retrying image analysis with provider: {provider_name}")
                return await provider.analyze_image(image_data, prompt, image_format, **kwargs)
            except Exception as e:
                log.warning(f"Provider {provider_name} image analysis failed: {e}")

                # Diagnostic warning for configuration guidance
                warning_msg = (
                    f"If provider '{provider_name}' is text-only, please explicitly set "
                    f"capabilities=['text'] in markit.toml to skip it for image tasks."
                )
                log.warning(warning_msg)

                errors.append((provider_name, f"{e}. Hint: {warning_msg}"))
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
