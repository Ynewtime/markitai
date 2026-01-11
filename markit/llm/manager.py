"""LLM Provider Manager with fallback support."""

import asyncio
import contextlib
import os
from dataclasses import dataclass, field

from markit.config.constants import MIN_VALID_RESPONSE_LENGTH
from markit.config.settings import (
    LLMConfig,
    LLMProviderConfig,
    ModelCostConfig,
    ValidationConfig,
)
from markit.exceptions import LLMError, LLMTimeoutError, ProviderNotFoundError
from markit.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ProviderState:
    """State for a single provider (supports lazy initialization)."""

    config: LLMProviderConfig
    capabilities: list[str] = field(default_factory=list)
    cost: ModelCostConfig | None = None
    provider: BaseLLMProvider | None = None
    initialized: bool = False
    valid: bool | None = None  # None = not yet validated


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
        self._provider_costs: dict[str, ModelCostConfig] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._current_index = 0

        # Lazy loading support
        self._provider_states: dict[str, ProviderState] = {}
        self._configs_loaded = False
        self._provider_init_locks: dict[str, asyncio.Lock] = {}

        # Track last successful provider per capability for optimized routing
        self._last_successful_provider: dict[str, str] = {}

    async def _load_configs(self) -> None:
        """Load configurations without initializing providers (lazy loading support)."""
        if self._configs_loaded:
            return

        async with self._init_lock:
            if self._configs_loaded:
                return

            # 1. Process Legacy Providers
            for config in self.config.providers:
                provider_key = config.name or f"{config.provider}_{config.model}"
                caps = (
                    config.capabilities if config.capabilities is not None else ["text", "vision"]
                )

                self._provider_states[provider_key] = ProviderState(
                    config=config,
                    capabilities=caps,
                )
                self._provider_capabilities[provider_key] = caps

            # 2. Process New Model Configs
            cred_map = {c.id: c for c in self.config.credentials}

            for model_config in self.config.models:
                cred = cred_map.get(model_config.credential_id)
                if not cred:
                    log.warning(
                        f"Credential '{model_config.credential_id}' not found for model '{model_config.name}'"
                    )
                    continue

                # Synthesize a ProviderConfig from Credential + Model
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

                provider_key = model_config.name
                caps = (
                    model_config.capabilities
                    if model_config.capabilities is not None
                    else ["text", "vision"]
                )

                self._provider_states[provider_key] = ProviderState(
                    config=synthetic_config,
                    capabilities=caps,
                    cost=model_config.cost,
                )
                self._provider_capabilities[provider_key] = caps

                if model_config.cost:
                    self._provider_costs[provider_key] = model_config.cost

            self._configs_loaded = True

    async def _ensure_provider_initialized(self, provider_name: str) -> bool:
        """Ensure a specific provider is initialized (lazy loading).

        Args:
            provider_name: Name of the provider to initialize

        Returns:
            True if provider is valid and ready, False otherwise
        """
        state = self._provider_states.get(provider_name)
        if not state:
            return False

        if state.initialized:
            return state.valid is True

        # Get or create lock for this provider
        if provider_name not in self._provider_init_locks:
            self._provider_init_locks[provider_name] = asyncio.Lock()

        async with self._provider_init_locks[provider_name]:
            # Double-check after acquiring lock
            if state.initialized:
                return state.valid is True

            try:
                provider = self._create_provider(state.config)
                if await self._validate_provider(provider, state.config):
                    state.provider = provider
                    state.valid = True
                    self._providers[provider_name] = provider
                    if provider_name not in self._valid_providers:
                        self._valid_providers.append(provider_name)
                    log.info(
                        f"Provider {provider_name} initialized on demand",
                        capabilities=state.capabilities,
                    )
                else:
                    state.valid = False
                    log.warning(f"Provider {provider_name} validation failed")
            except Exception as e:
                state.valid = False
                log.warning(f"Failed to initialize provider {provider_name}: {e}")

            state.initialized = True
            return state.valid is True

    async def initialize(
        self,
        required_capabilities: list[str] | None = None,
        preload_all: bool = False,
        lazy: bool = False,
    ) -> None:
        """Initialize providers.

        Args:
            required_capabilities: Only preload providers with these capabilities
            preload_all: If True, initialize all providers (legacy behavior)
            lazy: If True, only load configs without network validation (fast startup).
                  Providers will be validated on-demand when first used.
        """
        # Always load configs first
        await self._load_configs()

        # Lazy mode: just mark as initialized without network validation
        if lazy:
            self._initialized = True
            log.debug("Lazy initialization complete, providers will be validated on-demand")
            return

        if self._initialized and not required_capabilities:
            return

        async with self._init_lock:
            if preload_all or (not required_capabilities and not self._initialized):
                # Legacy behavior: initialize all providers
                for provider_name in self._provider_states:
                    await self._ensure_provider_initialized(provider_name)
            elif required_capabilities:
                # Selective preloading: only initialize providers with required capabilities
                for provider_name, state in self._provider_states.items():
                    if any(cap in state.capabilities for cap in required_capabilities):
                        await self._ensure_provider_initialized(provider_name)

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
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

        elif config.provider == "gemini":
            from markit.llm.gemini import GeminiProvider

            return GeminiProvider(
                api_key=api_key,
                model=config.model,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

        elif config.provider == "ollama":
            from markit.llm.ollama import OllamaProvider

            return OllamaProvider(
                model=config.model,
                host=config.base_url or "http://localhost:11434",
                timeout=config.timeout,
            )

        elif config.provider == "openrouter":
            from markit.llm.openrouter import OpenRouterProvider

            return OpenRouterProvider(
                api_key=api_key,
                model=config.model,
                base_url=config.base_url,
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

    def _get_validation_config(self) -> ValidationConfig:
        """Get validation configuration from LLMConfig."""
        if hasattr(self.config, "validation") and self.config.validation:
            return self.config.validation
        return ValidationConfig()

    def _handle_validation_failure(self, message: str, action: str) -> bool:
        """Handle validation failure based on configured action.

        Args:
            message: Error message
            action: Action to take ("warn", "skip", "fail")

        Returns:
            True to allow provider, False to skip

        Raises:
            LLMError: If action is "fail"
        """
        if action == "fail":
            raise LLMError(f"Provider validation failed: {message}")
        elif action == "skip":
            log.warning(f"Skipping provider: {message}")
            return False
        else:  # warn
            log.warning(f"Provider validation warning: {message}")
            return True

    async def _validate_provider(
        self, provider: BaseLLMProvider, config: LLMProviderConfig
    ) -> bool:
        """Validate that a provider is properly configured.

        Respects validation configuration for retries and failure handling.
        """
        validation_config = self._get_validation_config()

        # Skip validation if disabled
        if not validation_config.enabled:
            log.debug(f"Validation disabled, assuming {config.name or config.model} is valid")
            return True

        # Helper to get effective API key
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
        if not api_key:
            api_key = self._get_api_key_from_env(config.provider)

        # Check for required API key (always enforced)
        if config.provider in ("openai", "anthropic", "openrouter"):
            if not api_key:
                return self._handle_validation_failure(
                    f"{config.provider} requires an API key",
                    validation_config.on_failure,
                )

        if config.provider == "gemini":
            if not api_key:
                return self._handle_validation_failure(
                    "Gemini requires GOOGLE_API_KEY",
                    validation_config.on_failure,
                )

        # Connection validation with retries
        for attempt in range(validation_config.retry_count + 1):
            try:
                if await provider.validate():
                    return True
            except Exception as e:
                if attempt < validation_config.retry_count:
                    log.debug(f"Validation attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(1)
                else:
                    return self._handle_validation_failure(
                        f"Validation failed after {validation_config.retry_count + 1} attempts: {e}",
                        validation_config.on_failure,
                    )

        return True  # Should not reach here

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
        """Check if any configured provider has the specified capability.

        This method checks configured providers (not just validated ones),
        allowing the pipeline to know what capabilities are available before
        providers are actually connected. A provider is considered to have
        a capability if:
        1. It's configured with that capability, AND
        2. It hasn't explicitly failed validation (state.valid != False)

        Args:
            capability: Capability to check (e.g. "vision", "text")

        Returns:
            True if at least one provider supports the capability
        """
        # Check configured providers (supports lazy loading)
        for _provider_name, state in self._provider_states.items():
            # Skip providers that have explicitly failed validation
            if state.valid is False:
                continue
            if capability in state.capabilities:
                return True

        # Fallback: also check validated providers for backward compatibility
        for provider_name in self._valid_providers:
            caps = self._provider_capabilities.get(provider_name, [])
            if capability in caps:
                return True

        return False

    def calculate_cost(self, provider_name: str, response: LLMResponse) -> float | None:
        """Calculate cost for a response based on token usage and cost config.

        Args:
            provider_name: Name of the provider
            response: LLM response with usage info

        Returns:
            Estimated cost in USD, or None if cost config not available
        """
        if not response.usage:
            return None

        cost_config = self._provider_costs.get(provider_name)
        if not cost_config:
            return None

        input_cost = (response.usage.prompt_tokens / 1_000_000) * cost_config.input_per_1m
        output_cost = (response.usage.completion_tokens / 1_000_000) * cost_config.output_per_1m

        return input_cost + output_cost

    def _filter_providers_by_capability(
        self,
        required: str | None,
        preferred: str | None,
        include_uninitialized: bool = False,
    ) -> list[str]:
        """Filter providers by capability requirements.

        Priority:
        1. Providers with only the required capability (lowest cost)
        2. Providers with required + preferred capabilities
        3. Providers with required capability (may have other capabilities)

        Args:
            required: Required capability (e.g. "text", "vision")
            preferred: Preferred additional capability
            include_uninitialized: If True, include providers that haven't been initialized yet

        Returns:
            Filtered list of provider names
        """
        # Determine the source list based on whether to include uninitialized
        if include_uninitialized:
            source = list(self._provider_states.keys())
        else:
            source = self._valid_providers.copy()

        if not required:
            return source

        # Filter providers that have the required capability
        capable = [p for p in source if required in self._provider_capabilities.get(p, [])]

        if not capable:
            return []

        if not preferred:
            # Prioritize providers with only the required capability (cost optimization)
            exact_match = [
                p for p in capable if self._provider_capabilities.get(p, []) == [required]
            ]
            if exact_match:
                return exact_match + [p for p in capable if p not in exact_match]
            return capable

        # With preferred: prioritize providers that have the preferred capability
        has_preferred = [p for p in capable if preferred in self._provider_capabilities.get(p, [])]
        no_preferred = [p for p in capable if p not in has_preferred]

        return has_preferred + no_preferred

    async def complete_with_fallback(
        self,
        messages: list[LLMMessage],
        required_capability: str | None = None,
        prefer_capability: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Make a completion request with automatic fallback.

        Tries each provider in order until one succeeds. Supports lazy loading.

        Args:
            messages: List of messages
            required_capability: Required capability (e.g. "text", "vision")
            prefer_capability: Preferred capability for prioritization
            **kwargs: Additional arguments passed to the provider

        Returns:
            LLM response

        Raises:
            LLMError: If all providers fail
        """
        # Load configs first (lazy loading)
        await self._load_configs()

        # Filter providers by capability (include uninitialized for lazy loading)
        candidates = self._filter_providers_by_capability(
            required_capability, prefer_capability, include_uninitialized=True
        )

        if not candidates:
            cap_desc = required_capability or "any"
            raise ProviderNotFoundError(f"No provider with '{cap_desc}' capability available")

        errors = []
        provider_count = len(candidates)

        # Round Robin Load Balancing
        start_index = self._current_index % provider_count
        self._current_index = (self._current_index + 1) % max(provider_count, 1)

        for i in range(provider_count):
            idx = (start_index + i) % provider_count
            provider_name = candidates[idx]

            # Lazy initialize the provider if needed
            if provider_name not in self._providers:
                if not await self._ensure_provider_initialized(provider_name):
                    errors.append((provider_name, "Initialization failed"))
                    continue

            provider = self._providers[provider_name]
            try:
                if i > 0:
                    log.debug(f"Retrying with provider: {provider_name}")
                response = await provider.complete(messages, **kwargs)
                response.estimated_cost = self.calculate_cost(provider_name, response)
                return response
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
        """Analyze an image with automatic fallback. Supports lazy loading.

        Optimizes provider selection by prioritizing last successful provider
        to avoid unnecessary initialization of multiple providers.

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
        # Load configs first (lazy loading)
        await self._load_configs()

        # Filter providers that support vision (include uninitialized for lazy loading)
        vision_candidates = self._filter_providers_by_capability(
            "vision", None, include_uninitialized=True
        )

        if not vision_candidates:
            raise LLMError("No provider with 'vision' capability available for image analysis")

        # Prioritize last successful provider to avoid unnecessary initialization
        if "vision" in self._last_successful_provider:
            preferred = self._last_successful_provider["vision"]
            if preferred in vision_candidates:
                vision_candidates.remove(preferred)
                vision_candidates.insert(0, preferred)
                log.debug(f"Prioritizing last successful vision provider: {preferred}")

        errors = []
        provider_count = len(vision_candidates)

        for i in range(provider_count):
            provider_name = vision_candidates[i]

            # Lazy initialize the provider if needed
            if provider_name not in self._providers:
                if not await self._ensure_provider_initialized(provider_name):
                    errors.append((provider_name, "Initialization failed"))
                    continue

            provider = self._providers[provider_name]
            try:
                if i > 0:
                    log.debug(f"Retrying image analysis with provider: {provider_name}")
                response = await provider.analyze_image(image_data, prompt, image_format, **kwargs)
                response.estimated_cost = self.calculate_cost(provider_name, response)

                # Track successful provider for future requests
                self._last_successful_provider["vision"] = provider_name
                return response
            except Exception as e:
                log.warning(
                    "Provider failed for image analysis, triggering fallback",
                    provider=provider_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                warning_msg = (
                    f"If provider '{provider_name}' is text-only, please explicitly set "
                    f"capabilities=['text'] in markit.yaml to skip it for image tasks."
                )
                log.warning(warning_msg)

                errors.append((provider_name, f"{e}. Hint: {warning_msg}"))
                continue

        error_details = "; ".join(f"{name}: {err}" for name, err in errors)
        raise LLMError(f"All providers failed for image analysis: {error_details}")

    async def complete_with_concurrent_fallback(
        self,
        messages: list[LLMMessage],
        timeout: int | None = None,
        required_capability: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Make a completion request with concurrent fallback on timeout.

        Smart concurrent fallback mechanism:
        1. Start primary model request
        2. Wait concurrent_fallback_timeout seconds
        3. If primary hasn't responded, start fallback model (don't cancel primary)
        4. Primary and fallback race, first successful response wins
        5. Cancel the other task after winner is determined
        6. Absolute timeout (max_request_timeout) force interrupts all tasks

        Args:
            messages: List of messages
            timeout: Timeout in seconds before starting concurrent fallback (overrides config)
            required_capability: Required capability filter
            **kwargs: Additional arguments passed to the provider

        Returns:
            LLM response

        Raises:
            LLMError: If all providers fail
            LLMTimeoutError: If absolute timeout is reached
        """
        await self._load_configs()

        # Get timeouts from config
        fallback_timeout = timeout
        if fallback_timeout is None:
            fallback_timeout = self.config.concurrent_fallback_timeout
        max_timeout = self.config.max_request_timeout

        # Check if concurrent fallback is enabled
        if not self.config.concurrent_fallback_enabled:
            # Fall back to standard sequential fallback
            return await self.complete_with_fallback(
                messages, required_capability=required_capability, **kwargs
            )

        candidates = self._filter_providers_by_capability(
            required_capability, None, include_uninitialized=True
        )

        if not candidates:
            cap_desc = required_capability or "any"
            raise ProviderNotFoundError(f"No provider with '{cap_desc}' capability available")

        # Prioritize last successful provider
        capability_key = required_capability or "text"
        if capability_key in self._last_successful_provider:
            preferred = self._last_successful_provider[capability_key]
            if preferred in candidates:
                candidates.remove(preferred)
                candidates.insert(0, preferred)
                log.debug(f"Prioritizing last successful provider: {preferred}")

        primary_name = candidates[0]

        # Ensure primary is initialized
        if primary_name not in self._providers:
            if not await self._ensure_provider_initialized(primary_name):
                raise LLMError(f"Failed to initialize primary provider: {primary_name}")

        primary_provider = self._providers[primary_name]

        # Create primary task
        primary_task = asyncio.create_task(primary_provider.complete(messages, **kwargs))

        try:
            # Wait for primary with shield to prevent cancellation
            response = await asyncio.wait_for(
                asyncio.shield(primary_task), timeout=fallback_timeout
            )

            # Validate response
            if self._validate_response(response):
                response.estimated_cost = self.calculate_cost(primary_name, response)
                self._last_successful_provider[capability_key] = primary_name
                return response
            else:
                log.warning(f"Primary model {primary_name} returned invalid response")
                raise LLMError("Primary model returned invalid response")

        except TimeoutError:
            # Primary exceeded timeout, check for fallback
            fallback_candidates = [c for c in candidates if c != primary_name]

            if not fallback_candidates:
                log.warning(
                    f"Primary model {primary_name} exceeded {fallback_timeout}s, "
                    "no fallback available, waiting with absolute timeout..."
                )
                try:
                    remaining_timeout = max_timeout - fallback_timeout
                    response = await asyncio.wait_for(primary_task, timeout=remaining_timeout)
                    response.estimated_cost = self.calculate_cost(primary_name, response)
                    self._last_successful_provider[capability_key] = primary_name
                    return response
                except TimeoutError:
                    primary_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await primary_task
                    raise LLMTimeoutError(max_timeout, primary_name) from None

            # Select and initialize fallback
            fallback_name = fallback_candidates[0]
            if fallback_name not in self._providers:
                if not await self._ensure_provider_initialized(fallback_name):
                    log.warning(
                        f"Failed to initialize fallback {fallback_name}, waiting for primary"
                    )
                    try:
                        remaining_timeout = max_timeout - fallback_timeout
                        response = await asyncio.wait_for(primary_task, timeout=remaining_timeout)
                        response.estimated_cost = self.calculate_cost(primary_name, response)
                        self._last_successful_provider[capability_key] = primary_name
                        return response
                    except TimeoutError:
                        primary_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await primary_task
                        raise LLMTimeoutError(max_timeout, primary_name) from None

            fallback_provider = self._providers[fallback_name]

            log.warning(
                f"Primary model {primary_name} exceeded {fallback_timeout}s, "
                f"starting fallback {fallback_name} concurrently"
            )

            # Start fallback task
            fallback_task = asyncio.create_task(fallback_provider.complete(messages, **kwargs))

            # Wait for either to complete with absolute timeout
            remaining_timeout = max_timeout - fallback_timeout
            try:
                done, pending = await asyncio.wait(
                    [primary_task, fallback_task],
                    timeout=remaining_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                # Cleanup on unexpected error
                for task in [primary_task, fallback_task]:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                raise

            # Handle absolute timeout
            if not done:
                log.error(f"All models exceeded absolute timeout ({max_timeout}s)")
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                raise LLMTimeoutError(max_timeout) from None

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Get result from completed task
            completed_task = done.pop()
            winner = "primary" if completed_task is primary_task else "fallback"
            winner_name = primary_name if winner == "primary" else fallback_name
            log.info(f"Concurrent fallback completed, winner: {winner} ({winner_name})")

            try:
                response = completed_task.result()
                if self._validate_response(response):
                    response.estimated_cost = self.calculate_cost(winner_name, response)
                    self._last_successful_provider[capability_key] = winner_name
                    return response
                else:
                    raise LLMError("Winner returned invalid response")
            except Exception as e:
                log.warning(
                    "Winner task failed, checking other task",
                    winner=winner_name,
                    error=str(e),
                )
                # Winner failed, but we already cancelled pending tasks
                raise LLMError(f"Concurrent fallback failed: {e}") from e

    def _validate_response(self, response: LLMResponse) -> bool:
        """Validate that a response is valid and usable.

        Args:
            response: LLM response to validate

        Returns:
            True if response is valid
        """
        if not response:
            return False
        if not response.content:
            return False
        return len(response.content.strip()) >= MIN_VALID_RESPONSE_LENGTH

    @property
    def available_providers(self) -> list[str]:
        """Return list of available provider names."""
        return self._valid_providers.copy()

    @property
    def has_providers(self) -> bool:
        """Check if any providers are available."""
        return len(self._valid_providers) > 0
