"""LLM integration module using LiteLLM Router."""

from __future__ import annotations

import asyncio
import base64
import copy
import random
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Suppress LiteLLM's async logging warnings (coroutine never awaited)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'Logging.async_success_handler' was never awaited",
    category=RuntimeWarning,
)

import litellm
from litellm import completion_cost

# Suppress LiteLLM's "Provider List" debug messages for custom providers
litellm.suppress_debug_info = True
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices
from loguru import logger

if TYPE_CHECKING:
    from markitai.config import LLMConfig, PromptsConfig

from markitai.constants import (
    DEFAULT_IO_CONCURRENCY,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
)
from markitai.llm import content
from markitai.llm.cache import ContentCache, PersistentCache
from markitai.llm.document import DocumentMixin
from markitai.llm.types import (
    LLMResponse,
    LLMRuntime,
)
from markitai.llm.vision import VisionMixin
from markitai.prompts import PromptManager
from markitai.utils.text import format_error_message

# =============================================================================
# Local Provider Support
# =============================================================================
# LiteLLM Router does not support custom providers (claude-agent/, copilot/).
# We need a wrapper class that uses litellm.acompletion() directly for these.


class LocalProviderWrapper:
    """Wrapper for local provider models that mimics Router interface.

    LiteLLM Router's get_llm_provider() does not recognize custom providers,
    so we need to call litellm.acompletion() directly for local provider models.

    Implements simple-shuffle load balancing strategy using weighted random selection.
    For image requests, prioritizes models with confirmed image support.
    """

    # Models with confirmed image/vision support via their provider.
    # Note: Copilot has a ~2000px dimension limit, but CopilotProvider
    # handles resizing automatically via _resize_image_if_needed()
    _IMAGE_CAPABLE_PATTERNS = (
        "claude-agent/",  # All claude-agent models support vision
        "copilot/claude-",  # All Copilot Claude models
        "copilot/gemini-",  # All Copilot Gemini models
        "copilot/gpt-4.1",  # GPT-4.1 series
        "copilot/gpt-4o",  # All GPT-4o variants (including mini)
        "copilot/gpt-5",  # GPT-5 series (gpt-5, gpt-5-mini, gpt-5.1*, gpt-5.2*)
        "copilot/raptor-",  # GitHub's fine-tuned GPT-5 mini (inherits vision)
    )
    # Note: copilot/gpt-3.5*, copilot/gpt-4 (non-4o/4.1), copilot/grok-* do NOT support vision

    def __init__(self, model_list: list[dict[str, Any]]) -> None:
        """Initialize with model list.

        Args:
            model_list: List of model configurations (same format as Router)
        """
        self.model_list = model_list
        # Map model_name to list of (model_id, weight) tuples for load balancing
        self._model_groups: dict[str, list[tuple[str, float]]] = {}
        for model_config in model_list:
            model_name = model_config.get("model_name", "default")
            litellm_params = model_config.get("litellm_params", {})
            model_id = litellm_params.get("model", "")
            weight = litellm_params.get("weight", 1.0)

            if model_name not in self._model_groups:
                self._model_groups[model_name] = []
            self._model_groups[model_name].append((model_id, weight))

        # Log model groups for debugging
        for name, models in self._model_groups.items():
            if len(models) > 1:
                model_strs = [f"{m}(w={w})" for m, w in models]
                logger.debug(
                    f"[LocalProviderWrapper] Model group '{name}': {', '.join(model_strs)}"
                )

    def _has_images(self, messages: list[Any]) -> bool:
        """Check if messages contain image content.

        Args:
            messages: List of chat messages

        Returns:
            True if any message contains image_url content
        """
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    def _is_image_capable(self, model_id: str) -> bool:
        """Check if a model supports image/vision requests.

        Args:
            model_id: Full model identifier (e.g., "claude-agent/haiku", "copilot/gemini-3-pro")

        Returns:
            True if model has confirmed image support
        """
        return any(model_id.startswith(p) for p in self._IMAGE_CAPABLE_PATTERNS)

    def _select_model(self, model_name: str, has_images: bool = False) -> str:
        """Select a model using weighted random selection (simple-shuffle).

        For image requests, prioritizes models with confirmed image support.

        Args:
            model_name: Logical model name (e.g., "default")
            has_images: Whether the request contains images

        Returns:
            Selected model ID
        """
        models = self._model_groups.get(model_name)
        if not models:
            # If model_name not in groups, assume it's already the model ID
            return model_name

        # For image requests, prefer image-capable models
        if has_images and len(models) > 1:
            image_capable = [(m, w) for m, w in models if self._is_image_capable(m)]
            if image_capable:
                # Use only image-capable models
                if len(image_capable) < len(models):
                    excluded = [m for m, _ in models if not self._is_image_capable(m)]
                    logger.debug(
                        f"[LocalProviderWrapper] Image request: preferring image-capable "
                        f"models, excluding {excluded}"
                    )
                models = image_capable
            # Note: If no image-capable models found, we proceed with available models
            # and let the underlying provider handle any limitations

        if len(models) == 1:
            return models[0][0]

        # Weighted random selection
        total_weight = sum(w for _, w in models)
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for model_id, weight in models:
            cumulative += weight
            if r <= cumulative:
                return model_id

        # Fallback to last model (shouldn't happen)
        return models[-1][0]

    async def acompletion(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Make async completion call using litellm directly.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            **kwargs: Additional parameters (max_tokens, metadata, etc.)

        Returns:
            LiteLLM ModelResponse
        """
        # Check if request contains images
        has_images = self._has_images(messages)

        # Select model (with image-awareness)
        model_id = self._select_model(model, has_images=has_images)

        # Remove metadata since litellm.acompletion doesn't support it
        kwargs.pop("metadata", None)

        # Call litellm directly (which will use custom_provider_map)
        response = await litellm.acompletion(
            model=model_id,
            messages=messages,
            **kwargs,
        )
        return response


def _is_all_local_providers(model_list: list[dict[str, Any]]) -> bool:
    """Check if all models in list use local providers.

    Args:
        model_list: List of model configurations

    Returns:
        True if ALL models use local providers (claude-agent/, copilot/)
    """
    from markitai.providers import is_local_provider_model

    if not model_list:
        return False

    for model_config in model_list:
        model_id = model_config.get("litellm_params", {}).get("model", "")
        if not is_local_provider_model(model_id):
            return False
    return True


class HybridRouter:
    """Router that combines LiteLLM Router with LocalProviderWrapper.

    Handles mixed configurations where both standard models (gemini/*, deepseek/*)
    and local providers (claude-agent/*, copilot/*) are configured.

    Routes requests to the appropriate backend based on the selected model:
    - Local provider models -> LocalProviderWrapper
    - Standard models -> LiteLLM Router
    """

    def __init__(
        self,
        standard_router: Router,
        local_wrapper: LocalProviderWrapper,
    ) -> None:
        """Initialize HybridRouter.

        Args:
            standard_router: LiteLLM Router for standard models
            local_wrapper: LocalProviderWrapper for local provider models
        """
        self.standard_router = standard_router
        self.local_wrapper = local_wrapper

        # Combine model lists for weighted selection
        # Format: (model_id, weight, is_local)
        self._all_models: list[tuple[str, float, bool]] = []

        # Add standard models
        for model_config in standard_router.model_list:
            model_id = model_config.get("litellm_params", {}).get("model", "")
            weight = model_config.get("litellm_params", {}).get("weight", 1.0)
            self._all_models.append((model_id, weight, False))

        # Add local provider models
        for model_config in local_wrapper.model_list:
            model_id = model_config.get("litellm_params", {}).get("model", "")
            weight = model_config.get("litellm_params", {}).get("weight", 1.0)
            self._all_models.append((model_id, weight, True))

        # Build model groups for logical model name resolution
        self._model_groups: dict[str, list[tuple[str, float, bool]]] = {}
        for model_config in standard_router.model_list:
            model_name = model_config.get("model_name", "default")
            model_id = model_config.get("litellm_params", {}).get("model", "")
            weight = model_config.get("litellm_params", {}).get("weight", 1.0)
            if model_name not in self._model_groups:
                self._model_groups[model_name] = []
            self._model_groups[model_name].append((model_id, weight, False))

        for model_config in local_wrapper.model_list:
            model_name = model_config.get("model_name", "default")
            model_id = model_config.get("litellm_params", {}).get("model", "")
            weight = model_config.get("litellm_params", {}).get("weight", 1.0)
            if model_name not in self._model_groups:
                self._model_groups[model_name] = []
            self._model_groups[model_name].append((model_id, weight, True))

        # Log hybrid router configuration
        local_models = [m for m, _, is_local in self._all_models if is_local]
        standard_models = [m for m, _, is_local in self._all_models if not is_local]
        logger.debug(
            f"[HybridRouter] Initialized with {len(standard_models)} standard "
            f"and {len(local_models)} local provider models"
        )

    def _has_images(self, messages: list[Any]) -> bool:
        """Check if messages contain image content."""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    def _is_local_model(self, model_id: str) -> bool:
        """Check if model_id is a local provider model."""
        from markitai.providers import is_local_provider_model

        return is_local_provider_model(model_id)

    def _is_image_capable(self, model_id: str) -> bool:
        """Check if model supports images."""
        # Local provider models - use LocalProviderWrapper's logic
        if self._is_local_model(model_id):
            return self.local_wrapper._is_image_capable(model_id)

        # Standard models - check litellm model info
        info = get_model_info_cached(model_id)
        return info.get("supports_vision", False)

    def _select_model(self, model_name: str, has_images: bool = False) -> str:
        """Select a model using weighted random selection.

        For image requests, prioritizes image-capable models.

        Args:
            model_name: Logical model name (e.g., "default")
            has_images: Whether the request contains images

        Returns:
            Selected model ID
        """
        models = self._model_groups.get(model_name)
        if not models:
            # If model_name not in groups, assume it's already the model ID
            return model_name

        # For image requests, prefer image-capable models
        if has_images and len(models) > 1:
            image_capable = [
                (m, w, is_local)
                for m, w, is_local in models
                if self._is_image_capable(m)
            ]
            if image_capable:
                if len(image_capable) < len(models):
                    excluded = [
                        m for m, _, _ in models if not self._is_image_capable(m)
                    ]
                    logger.debug(
                        f"[HybridRouter] Image request: preferring image-capable "
                        f"models, excluding {excluded}"
                    )
                models = image_capable

        if len(models) == 1:
            return models[0][0]

        # Weighted random selection
        total_weight = sum(w for _, w, _ in models)
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for model_id, weight, _ in models:
            cumulative += weight
            if r <= cumulative:
                return model_id

        return models[-1][0]

    async def acompletion(
        self,
        model: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Route completion request to appropriate backend.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            **kwargs: Additional parameters

        Returns:
            LiteLLM ModelResponse
        """
        has_images = self._has_images(messages)
        selected_model = self._select_model(model, has_images)

        if self._is_local_model(selected_model):
            logger.debug(f"[HybridRouter] Routing to local provider: {selected_model}")
            return await self.local_wrapper.acompletion(
                selected_model, messages, **kwargs
            )
        else:
            logger.debug(f"[HybridRouter] Routing to standard router: {selected_model}")
            return await self.standard_router.acompletion(model, messages, **kwargs)

    @property
    def model_list(self) -> list[dict[str, Any]]:
        """Get combined model list from both routers."""
        return self.standard_router.model_list + self.local_wrapper.model_list


# Enable automatic max_tokens adjustment to model limits
# When user-specified max_tokens exceeds model's max_output_tokens,
# LiteLLM will automatically cap it to the model's limit
litellm.modify_params = True

# Retryable exceptions (kept here as they depend on litellm types)
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
)

# Cache for model info to avoid repeated litellm queries
_model_info_cache: dict[str, dict[str, Any]] = {}


def get_model_info_cached(model: str) -> dict[str, Any]:
    """Get model info from litellm with caching.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-2.5-flash")

    Returns:
        Dict with keys:
            - max_input_tokens: int (context window size)
            - max_output_tokens: int (max output tokens)
            - supports_vision: bool (whether model supports images)
        Returns defaults if litellm info unavailable.
    """
    if model in _model_info_cache:
        return _model_info_cache[model]

    # Check if it's a local provider model first
    from markitai.providers import get_local_provider_model_info

    local_info = get_local_provider_model_info(model)
    if local_info is not None:
        # Local provider models have known specs, no need to query litellm
        result = {
            "max_input_tokens": local_info["max_input_tokens"],
            "max_output_tokens": local_info["max_output_tokens"],
            "supports_vision": local_info["supports_vision"],
        }
        _model_info_cache[model] = result
        return result

    # Defaults for non-local providers
    result = {
        "max_input_tokens": 128000,  # Conservative default
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "supports_vision": False,
    }

    try:
        info = litellm.get_model_info(model)
        if info.get("max_input_tokens"):
            result["max_input_tokens"] = info["max_input_tokens"]
        if info.get("max_output_tokens"):
            result["max_output_tokens"] = info["max_output_tokens"]
        supports_vision = info.get("supports_vision")
        if supports_vision is not None:
            result["supports_vision"] = bool(supports_vision)
    except Exception:
        logger.debug(f"[ModelInfo] Could not get info for {model}, using defaults")

    _model_info_cache[model] = result
    return result


def get_model_max_output_tokens(model: str) -> int:
    """Get max_output_tokens for a model using litellm.get_model_info().

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat", "gemini/gemini-2.5-flash")

    Returns:
        max_output_tokens value, or DEFAULT_MAX_OUTPUT_TOKENS if unavailable
    """
    return get_model_info_cached(model)["max_output_tokens"]


def get_response_cost(response: Any) -> float:
    """Get cost from a LiteLLM ModelResponse.

    For local providers (claude-agent, copilot), the cost is stored in
    _hidden_params["total_cost_usd"] by the provider. For other providers,
    uses litellm.completion_cost().

    Args:
        response: LiteLLM ModelResponse object

    Returns:
        Cost in USD, or 0.0 if unavailable
    """
    # Check for cost from local provider (stored in _hidden_params)
    hidden_params = getattr(response, "_hidden_params", None)
    if hidden_params and isinstance(hidden_params, dict):
        provider_cost = hidden_params.get("total_cost_usd")
        if provider_cost is not None:
            return float(provider_cost)

    # Fall back to litellm's cost calculation
    try:
        return completion_cost(completion_response=response)
    except Exception:
        return 0.0


def _context_display_name(context: str) -> str:
    """Extract display name from context for logging.

    Converts full paths to filenames while preserving suffixes like ':images'.
    Examples:
        'C:/path/to/file.pdf:images' -> 'file.pdf:images'
        'file.pdf' -> 'file.pdf'
        '' -> ''
    """
    from pathlib import PurePosixPath, PureWindowsPath

    if not context:
        return context

    # Check if this is a Windows-style path (with drive letter)
    is_windows_path = len(context) >= 2 and context[1] == ":" and context[0].isalpha()

    # Split context into path part and suffix (e.g., ':images')
    # For Windows paths, find suffix after the drive letter portion
    if is_windows_path:
        # Find the last colon that's not the drive letter colon
        rest = context[2:]  # Skip "C:"
        if ":" in rest:
            last_colon_pos = context.rfind(":")
            path_part = context[:last_colon_pos]
            suffix = context[last_colon_pos + 1 :]
            # Use PureWindowsPath for proper Windows path handling
            return f"{PureWindowsPath(path_part).name}:{suffix}"
        # No suffix, just extract filename
        return PureWindowsPath(context).name
    elif ":" in context:
        # Unix path with suffix
        parts = context.rsplit(":", 1)
        if len(parts) == 2:
            path_part, suffix = parts
            return f"{PurePosixPath(path_part).name}:{suffix}"
    # Simple path without suffix
    return Path(context).name


class MarkitaiLLMLogger(CustomLogger):
    """Custom LiteLLM callback logger for capturing additional call details."""

    def __init__(self) -> None:
        self.last_call_details: dict[str, Any] = {}

    def log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Capture details from successful LLM calls."""
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "response_time": slo.get("response_time"),
            "cache_hit": kwargs.get("cache_hit", False),
            "model_id": slo.get("model_id"),
        }

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async version of success event logging."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Capture details from failed LLM calls."""
        slo = kwargs.get("standard_logging_object", {})
        self.last_call_details = {
            "api_base": slo.get("api_base"),
            "error_code": slo.get("error_code"),
            "error_class": slo.get("error_class"),
        }

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async version of failure event logging."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


# Global callback instance
_markitai_llm_logger = MarkitaiLLMLogger()


class LLMProcessor(VisionMixin, DocumentMixin):
    """LLM processor using LiteLLM Router for load balancing."""

    # Static method proxies to content module (for backward compatibility)
    # These methods were originally defined in this class but have been
    # refactored to markitai.llm.content module with public names (no underscore)
    extract_protected_content = staticmethod(content.extract_protected_content)
    _protect_content = staticmethod(content.protect_content)
    _unprotect_content = staticmethod(content.unprotect_content)
    _fix_malformed_image_refs = staticmethod(content.fix_malformed_image_refs)
    _clean_frontmatter = staticmethod(content.clean_frontmatter)
    _smart_truncate = staticmethod(content.smart_truncate)
    _split_text_by_pages = staticmethod(content.split_text_by_pages)
    restore_protected_content = staticmethod(content.restore_protected_content)

    # Public aliases without underscore prefix
    protect_content = staticmethod(content.protect_content)
    unprotect_content = staticmethod(content.unprotect_content)
    fix_malformed_image_refs = staticmethod(content.fix_malformed_image_refs)
    clean_frontmatter = staticmethod(content.clean_frontmatter)
    smart_truncate = staticmethod(content.smart_truncate)
    split_text_by_pages = staticmethod(content.split_text_by_pages)

    def __init__(
        self,
        config: LLMConfig,
        prompts_config: PromptsConfig | None = None,
        runtime: LLMRuntime | None = None,
        no_cache: bool = False,
        no_cache_patterns: list[str] | None = None,
        cache_global_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize LLM processor.

        Args:
            config: LLM configuration
            prompts_config: Optional prompts configuration
            runtime: Optional shared runtime for concurrency control.
                     If provided, uses runtime's semaphore instead of creating one.
            no_cache: If True, skip reading from cache but still write results.
                      Follows Bun's --no-cache semantics (force fresh, update cache).
            no_cache_patterns: List of glob patterns to skip cache for specific files.
                              Patterns are matched against relative paths from input_dir.
                              E.g., ["*.pdf", "reports/**", "file.docx"]
            cache_global_dir: Global cache directory. If provided, overrides the default
                              ~/.markitai directory. Should be passed from config.cache.global_dir.
        """
        self.config = config
        self._runtime = runtime
        self._router: Router | LocalProviderWrapper | HybridRouter | None = None
        self._vision_router: Router | LocalProviderWrapper | HybridRouter | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._prompt_manager = PromptManager(prompts_config)

        # Usage tracking (global across all contexts)
        # Use defaultdict to avoid check-then-create race conditions
        def _make_usage_dict() -> dict[str, Any]:
            return {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        self._usage: defaultdict[str, dict[str, Any]] = defaultdict(_make_usage_dict)

        # Per-context usage tracking for batch processing
        self._context_usage: defaultdict[str, defaultdict[str, dict[str, Any]]] = (
            defaultdict(lambda: defaultdict(_make_usage_dict))
        )

        # Call counter for each context (file)
        self._call_counter: defaultdict[str, int] = defaultdict(int)

        # Lock for thread-safe access to usage tracking dicts in concurrent contexts
        # Using threading.Lock instead of asyncio.Lock because:
        # 1. Dict operations are CPU-bound and don't need await
        # 2. Works in both sync and async contexts
        # The lock hold time is minimal (only simple dict updates)
        self._usage_lock = threading.Lock()

        # In-memory content cache for session-level deduplication (fast, no I/O)
        self._cache = ContentCache()
        self._cache_hits = 0
        self._cache_misses = 0

        # Persistent cache for cross-session reuse (SQLite-based)
        # no_cache=True: skip reading but still write (Bun semantics)
        # no_cache_patterns: skip reading for specific files matching patterns
        # Resolve cache_global_dir to Path if provided as string
        resolved_cache_dir: Path | None = None
        if cache_global_dir is not None:
            resolved_cache_dir = Path(cache_global_dir).expanduser()
        self._persistent_cache = PersistentCache(
            global_dir=resolved_cache_dir,
            skip_read=no_cache,
            no_cache_patterns=no_cache_patterns,
        )

        # Image cache for avoiding repeated file reads during document processing
        # Key: file path string, Value: (bytes, base64_encoded_string)
        # Uses OrderedDict for LRU eviction when limits are reached
        from collections import OrderedDict

        self._image_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._image_cache_max_size = 200  # Max number of images to cache
        self._image_cache_max_bytes = 500 * 1024 * 1024  # 500MB max total cache size
        self._image_cache_bytes = 0  # Current total bytes in cache

        # Register LiteLLM callback for additional details
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Register LiteLLM callbacks and custom providers."""
        # Add our custom logger to litellm callbacks if not already added
        if _markitai_llm_logger not in (litellm.callbacks or []):
            if litellm.callbacks is None:
                litellm.callbacks = []
            litellm.callbacks.append(_markitai_llm_logger)

        # Register custom providers (claude-agent, copilot, etc.)
        from markitai.providers import register_providers

        register_providers()

    def _get_next_call_index(self, context: str) -> int:
        """Get the next call index for a given context.

        Thread-safe: uses lock for atomic increment.
        """
        with self._usage_lock:
            self._call_counter[context] += 1
            return self._call_counter[context]

    def reset_call_counter(self, context: str = "") -> None:
        """Reset call counter for a context or all contexts.

        Thread-safe: uses lock for safe modification.
        """
        with self._usage_lock:
            if context:
                self._call_counter.pop(context, None)
            else:
                self._call_counter.clear()

    @property
    def router(self) -> Router | LocalProviderWrapper | HybridRouter:
        """Get or create the LiteLLM Router (or LocalProviderWrapper/HybridRouter for local models)."""
        if self._router is None:
            self._router = self._create_router()
        return self._router

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get the LLM concurrency semaphore.

        If a runtime was provided, uses the shared semaphore from runtime.
        Otherwise creates a local semaphore.
        """
        if self._runtime is not None:
            return self._runtime.semaphore
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.concurrency)
        return self._semaphore

    @property
    def io_semaphore(self) -> asyncio.Semaphore:
        """Get the I/O concurrency semaphore for file operations.

        Separate from LLM semaphore to allow higher I/O parallelism.
        """
        if self._runtime is not None:
            return self._runtime.io_semaphore
        # Fallback: use higher limit for local I/O operations
        return asyncio.Semaphore(DEFAULT_IO_CONCURRENCY)

    def _create_router(self) -> Router | LocalProviderWrapper | HybridRouter:
        """Create LiteLLM Router from configuration.

        If all models use local providers (claude-agent/, copilot/), returns
        a LocalProviderWrapper instead. If mixed, returns a HybridRouter.

        Returns:
            Router, LocalProviderWrapper, or HybridRouter instance
        """
        if not self.config.model_list:
            raise ValueError("No models configured in llm.model_list")

        # Import availability check
        from markitai.providers import is_local_provider_available

        # Build model list with resolved API keys and max_tokens
        # Skip models whose SDKs are not available
        model_list = []
        skipped_models = []
        for model_config in self.config.model_list:
            model_id = model_config.litellm_params.model

            # Skip local provider models if their SDK is not available
            if not is_local_provider_available(model_id):
                skipped_models.append(model_id)
                continue

            model_entry = {
                "model_name": model_config.model_name,
                "litellm_params": {
                    "model": model_id,
                },
            }

            # Add optional params
            api_key = model_config.litellm_params.get_resolved_api_key()
            if api_key:
                model_entry["litellm_params"]["api_key"] = api_key

            if model_config.litellm_params.api_base:
                model_entry["litellm_params"]["api_base"] = (
                    model_config.litellm_params.api_base
                )

            if model_config.litellm_params.weight != 1:
                model_entry["litellm_params"]["weight"] = (
                    model_config.litellm_params.weight
                )

            # Note: max_tokens is NOT set at Router level
            # It will be calculated dynamically per-request based on input size
            # This avoids context overflow issues with shared context models

            if model_config.model_info:
                model_entry["model_info"] = model_config.model_info.model_dump()

            model_list.append(model_entry)

        if skipped_models:
            logger.debug(
                f"[Router] Skipped {len(skipped_models)} models (SDK unavailable): "
                f"{', '.join(skipped_models)}"
            )

        if not model_list:
            raise ValueError(
                "No available models after filtering. "
                "Check that required SDKs are installed for configured models."
            )

        # Log router configuration (compact format)
        model_names = [e["litellm_params"]["model"].split("/")[-1] for e in model_list]

        # Check if all models use local providers
        # LiteLLM Router doesn't support custom providers, so we use a wrapper
        if _is_all_local_providers(model_list):
            logger.info(
                f"[Router] Creating LocalProviderWrapper for {len(model_list)} local models"
            )
            logger.debug(f"[Router] Local models: {', '.join(model_names)}")
            return LocalProviderWrapper(model_list=model_list)

        # Separate local providers from standard models
        from markitai.providers import is_local_provider_model

        standard_models = []
        local_models = []
        for entry in model_list:
            model_id = entry["litellm_params"]["model"]
            if is_local_provider_model(model_id):
                local_models.append(entry)
            else:
                standard_models.append(entry)

        if not standard_models:
            raise ValueError(
                "No standard models available after filtering local providers. "
                "Use only local providers (claude-agent/, copilot/) or add standard models."
            )

        # Build router settings
        router_settings = self.config.router_settings.model_dump()

        # Disable internal retries - we handle retries ourselves for better logging
        router_settings["num_retries"] = 0

        standard_names = [
            e["litellm_params"]["model"].split("/")[-1] for e in standard_models
        ]

        # Create standard router
        standard_router = Router(model_list=standard_models, **router_settings)

        # If we have both local and standard models, use HybridRouter
        if local_models:
            local_wrapper = LocalProviderWrapper(model_list=local_models)
            local_names = [e["litellm_params"]["model"] for e in local_models]
            logger.info(
                f"[Router] Creating HybridRouter: {len(standard_models)} standard + "
                f"{len(local_models)} local models"
            )
            logger.debug(
                f"[Router] Standard: {', '.join(standard_names)}, "
                f"Local: {', '.join(local_names)}"
            )
            return HybridRouter(
                standard_router=standard_router,
                local_wrapper=local_wrapper,
            )

        logger.info(
            f"[Router] Creating with strategy={router_settings.get('routing_strategy')}, "
            f"models={len(standard_models)}"
        )
        logger.debug(f"[Router] Models: {', '.join(standard_names)}")

        return standard_router

    def _create_router_from_models(
        self, models: list[Any], router_settings: dict[str, Any] | None = None
    ) -> Router | LocalProviderWrapper | HybridRouter:
        """Create a Router from a subset of model configurations.

        If all models use local providers (claude-agent/, copilot/), returns
        a LocalProviderWrapper. If mixed, returns a HybridRouter.

        Args:
            models: List of ModelConfig objects from self.config.model_list
            router_settings: Optional router settings (uses default if not provided)

        Returns:
            Router, LocalProviderWrapper, or HybridRouter instance
        """
        # Import availability check
        from markitai.providers import is_local_provider_available

        # Build model list with resolved API keys and max_tokens
        # Skip models whose SDKs are not available
        model_list = []
        for model_config in models:
            model_id = model_config.litellm_params.model

            # Skip local provider models if their SDK is not available
            if not is_local_provider_available(model_id):
                continue

            model_entry = {
                "model_name": model_config.model_name,
                "litellm_params": {
                    "model": model_id,
                },
            }

            # Add optional params
            api_key = model_config.litellm_params.get_resolved_api_key()
            if api_key:
                model_entry["litellm_params"]["api_key"] = api_key

            if model_config.litellm_params.api_base:
                model_entry["litellm_params"]["api_base"] = (
                    model_config.litellm_params.api_base
                )

            if model_config.litellm_params.weight != 1:
                model_entry["litellm_params"]["weight"] = (
                    model_config.litellm_params.weight
                )

            # Note: max_tokens calculated dynamically per-request

            if model_config.model_info:
                model_entry["model_info"] = model_config.model_info.model_dump()

            model_list.append(model_entry)

        if not model_list:
            raise ValueError(
                "No available models after filtering. "
                "Check that required SDKs are installed for configured models."
            )

        # Check if all models use local providers
        if _is_all_local_providers(model_list):
            model_names = [
                e["litellm_params"]["model"].split("/")[-1] for e in model_list
            ]
            logger.debug(
                f"[Router] Using LocalProviderWrapper: {', '.join(model_names)}"
            )
            return LocalProviderWrapper(model_list=model_list)

        # Separate local providers from standard models
        from markitai.providers import is_local_provider_model

        local_models = []
        standard_models = []
        for entry in model_list:
            model_id = entry["litellm_params"]["model"]
            if is_local_provider_model(model_id):
                local_models.append(entry)
            else:
                standard_models.append(entry)

        if not standard_models:
            raise ValueError(
                "No standard models available after filtering local providers. "
                "Use only local providers (claude-agent/, copilot/) or add standard models."
            )

        # Use provided settings or default
        settings = router_settings or self.config.router_settings.model_dump()
        settings["num_retries"] = 0  # We handle retries ourselves

        # Create standard router
        standard_router = Router(model_list=standard_models, **settings)

        # If we have both local and standard models, use HybridRouter
        if local_models:
            local_wrapper = LocalProviderWrapper(model_list=local_models)
            local_names = [e["litellm_params"]["model"] for e in local_models]
            standard_names = [e["litellm_params"]["model"] for e in standard_models]
            logger.debug(
                f"[Router] Using HybridRouter: local={local_names}, standard={standard_names}"
            )
            return HybridRouter(
                standard_router=standard_router,
                local_wrapper=local_wrapper,
            )

        return standard_router

    def _is_vision_model(self, model_config: Any) -> bool:
        """Check if a model supports vision.

        Priority:
        1. Config override (model_info.supports_vision) if explicitly set
        2. Local providers (claude-agent/, copilot/) - always support vision
        3. Auto-detect from litellm.get_model_info()

        Args:
            model_config: Model configuration object

        Returns:
            True if model supports vision
        """
        model_id = model_config.litellm_params.model

        # Check config override first
        if (
            model_config.model_info
            and model_config.model_info.supports_vision is not None
        ):
            return model_config.model_info.supports_vision

        # Local providers (claude-agent/, copilot/) support vision via attachments
        # when using non-streaming mode (which is our default)
        from markitai.providers import is_local_provider_model

        if is_local_provider_model(model_id):
            return True

        # Auto-detect from litellm
        info = get_model_info_cached(model_id)
        return info.get("supports_vision", False)

    @property
    def vision_router(self) -> Router | LocalProviderWrapper | HybridRouter:
        """Get or create Router with only vision-capable models (lazy).

        Filters models using auto-detection from litellm or config override.
        Falls back to main router if no vision models found.

        Returns:
            Router or LocalProviderWrapper with vision-capable models only
        """
        if self._vision_router is None:
            vision_models = [
                m for m in self.config.model_list if self._is_vision_model(m)
            ]

            if not vision_models:
                # No dedicated vision models - fall back to main router
                logger.warning(
                    "[Router] No vision-capable models configured, using main router"
                )
                self._vision_router = self.router
            else:
                model_names = [
                    m.litellm_params.model.split("/")[-1] for m in vision_models
                ]
                logger.info(
                    f"[Router] Creating vision router with {len(vision_models)} models"
                )
                logger.debug(f"[Router] Vision models: {', '.join(model_names)}")
                self._vision_router = self._create_router_from_models(vision_models)

        return self._vision_router

    def _message_contains_image(self, messages: list[dict[str, Any]]) -> bool:
        """Detect if messages contain image content.

        Checks for image_url type in message content parts.

        Args:
            messages: List of chat messages

        Returns:
            True if any message contains an image
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        context: str = "",
    ) -> LLMResponse:
        """
        Make an LLM call with rate limiting, retry logic, and detailed logging.

        Smart router selection: automatically uses vision_router when messages
        contain images, otherwise uses the main router.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            context: Context identifier for logging (e.g., filename)

        Returns:
            LLMResponse with content and usage info
        """
        # Generate call ID for logging
        call_index = self._get_next_call_index(context) if context else 0
        call_id = f"{context}:{call_index}" if context else f"call:{call_index}"

        # Smart router selection based on message content
        requires_vision = self._message_contains_image(messages)
        router = self.vision_router if requires_vision else self.router

        max_retries = self.config.router_settings.num_retries
        return await self._call_llm_with_retry(
            model=model,
            messages=messages,
            call_id=call_id,
            context=context,
            max_retries=max_retries,
            router=router,
        )

    def _calculate_dynamic_max_tokens(
        self,
        messages: list[Any],
        target_model_id: str | None = None,
        router: Router | LocalProviderWrapper | HybridRouter | None = None,
    ) -> int | None:
        """Calculate dynamic max_tokens based on input size and target model.

        Uses the target model's limits when available, otherwise returns None
        to let LiteLLM use model defaults.

        When router is provided, uses the minimum max_output_tokens across all
        models in the router to ensure compatibility with any model the router
        might select.

        Args:
            messages: Chat messages to estimate input tokens
            target_model_id: Specific model ID (from router pre-selection)
            router: Optional Router or LocalProviderWrapper for model limit lookup

        Returns:
            Safe max_tokens value, or None to let LiteLLM use model defaults
        """
        import re

        # Estimate input tokens (use gpt-4 tokenizer as reasonable approximation)
        try:
            input_tokens = litellm.token_counter(model="gpt-4", messages=messages)
        except Exception:
            # Fallback: rough estimate based on character count
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            input_tokens = total_chars // 4  # ~4 chars per token

        # Detect table-heavy content (tables require more output tokens for formatting)
        content_str = str(messages)
        table_rows = len(re.findall(r"\|[^|]+\|", content_str))
        is_table_heavy = table_rows > 20  # More than 20 table rows

        # Get model limits - use minimum across all router models if available
        max_context: int | None = None
        max_output: int | None = None

        # If router is provided, get minimum max_output_tokens across all models
        # This ensures compatibility with any model the router might select
        if router:
            all_max_outputs: list[int] = []
            all_max_contexts: list[int] = []
            for model_config in router.model_list:
                model_id = model_config.get("litellm_params", {}).get("model")
                if model_id:
                    info = get_model_info_cached(model_id)
                    if info.get("max_output_tokens"):
                        all_max_outputs.append(info["max_output_tokens"])
                    if info.get("max_input_tokens"):
                        all_max_contexts.append(info["max_input_tokens"])
            if all_max_outputs:
                max_output = min(all_max_outputs)
                logger.debug(
                    f"[DynamicTokens] Using min max_output across router models: {max_output}"
                )
            if all_max_contexts:
                max_context = min(all_max_contexts)
        elif target_model_id:
            info = get_model_info_cached(target_model_id)
            max_context = info.get("max_input_tokens")
            max_output = info.get("max_output_tokens")
            if max_context and max_output:
                logger.debug(
                    f"[DynamicTokens] Using target model {target_model_id}: "
                    f"context={max_context}, max_output={max_output}"
                )

        # If target model info unavailable, return None to let LiteLLM handle it
        if not max_context or not max_output:
            logger.debug(
                f"[DynamicTokens] Could not get limits for model={target_model_id}, "
                f"returning None to use LiteLLM defaults"
            )
            return None

        # Calculate available output space
        # Reserve buffer for safety (tokenizer differences, system overhead)
        buffer = max(500, int(input_tokens * 0.1))  # 10% or 500, whichever is larger
        available_context = max_context - input_tokens - buffer

        # For table-heavy content, ensure output has at least 1.5x input tokens
        # since reformatting tables to Markdown often expands token count
        if is_table_heavy:
            min_required_output = int(input_tokens * 1.5)
            available_context = max(available_context, min_required_output)
            logger.debug(
                f"[DynamicTokens] Table-heavy content detected ({table_rows} rows), "
                f"min_required_output={min_required_output}"
            )

        # max_tokens = min(model's max_output, available context space)
        max_tokens = min(max_output, available_context)

        # Ensure reasonable minimum (higher for table-heavy content)
        min_floor = 4000 if is_table_heavy else 1000
        max_tokens = max(max_tokens, min_floor)

        logger.debug(
            f"[DynamicTokens] input={input_tokens}, model={target_model_id}, "
            f"max_output={max_output}, calculated={max_tokens}"
        )

        return max_tokens

    def _get_router_primary_model(
        self, router: Router | LocalProviderWrapper | HybridRouter
    ) -> str | None:
        """Get the primary model ID from a Router's model_list.

        Args:
            router: LiteLLM Router, LocalProviderWrapper, or HybridRouter instance

        Returns:
            Model ID string (e.g., "deepseek/deepseek-chat"), or None if unavailable
        """
        try:
            model_list = router.model_list
            if model_list and len(model_list) > 0:
                return model_list[0].get("litellm_params", {}).get("model")
        except Exception:
            pass
        return None

    async def _call_llm_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        call_id: str,
        context: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        router: Router | LocalProviderWrapper | HybridRouter | None = None,
    ) -> LLMResponse:
        """
        Make an LLM call with custom retry logic and detailed logging.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            call_id: Unique identifier for this call (for logging)
            context: Context identifier for usage tracking (e.g., filename)
            max_retries: Maximum number of retry attempts
            router: Router or LocalProviderWrapper to use (defaults to self.router)

        Returns:
            LLMResponse with content and usage info
        """
        # Use provided router or default to main router
        active_router = router or self.router
        last_exception: Exception | None = None

        # Calculate dynamic max_tokens based on input size and target model
        target_model_id = self._get_router_primary_model(active_router)
        max_tokens = self._calculate_dynamic_max_tokens(messages, target_model_id)

        for attempt in range(max_retries + 1):
            start_time = time.perf_counter()

            async with self.semaphore:
                try:
                    # Log request start
                    if attempt == 0:
                        logger.debug(f"[LLM:{call_id}] Request to {model}")
                    else:
                        # Log retry attempt
                        error_type = (
                            type(last_exception).__name__
                            if last_exception
                            else "Unknown"
                        )
                        status_code = getattr(last_exception, "status_code", "N/A")
                        logger.warning(
                            f"[LLM:{call_id}] Retry #{attempt}: {error_type} "
                            f"status={status_code}"
                        )

                    response = await active_router.acompletion(
                        model=model,
                        messages=cast(list[AllMessageValues], messages),
                        max_tokens=max_tokens,
                        metadata={"call_id": call_id, "attempt": attempt},
                    )

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    # litellm returns Choices (not StreamingChoices) for non-streaming
                    choice = cast(Choices, response.choices[0])
                    content = choice.message.content or ""
                    actual_model = response.model or model

                    # Calculate cost (uses _hidden_params for local providers)
                    cost = get_response_cost(response)

                    # Track usage (usage attr exists at runtime but not in type stubs)
                    usage = getattr(response, "usage", None)
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0

                    self._track_usage(
                        actual_model, input_tokens, output_tokens, cost, context
                    )

                    # Log result
                    logger.info(
                        f"[LLM:{call_id}] {actual_model} "
                        f"tokens={input_tokens}+{output_tokens} "
                        f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
                    )

                    # Detect empty response (0 output tokens with substantial input)
                    # This usually indicates a model failure that should be retried
                    if output_tokens == 0 and input_tokens > 100:
                        if attempt < max_retries:
                            logger.warning(
                                f"[LLM:{call_id}] Empty response (0 output tokens), "
                                f"retrying with different model..."
                            )
                            # Treat as retryable error
                            await asyncio.sleep(
                                min(
                                    DEFAULT_RETRY_BASE_DELAY * (2**attempt),
                                    DEFAULT_RETRY_MAX_DELAY,
                                )
                            )
                            continue
                        else:
                            logger.error(
                                f"[LLM:{call_id}] Empty response after {max_retries + 1} "
                                f"attempts, returning empty content"
                            )

                    return LLMResponse(
                        content=content,
                        model=actual_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                    )

                except RETRYABLE_ERRORS as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    last_exception = e

                    # Check for quota/billing errors that should NOT be retried
                    # These errors are wrapped by LiteLLM as APIConnectionError but
                    # are actually non-recoverable without user action
                    error_msg_lower = str(e).lower()
                    non_retryable_patterns = (
                        "quota",
                        "billing",
                        "payment",
                        "subscription",
                        "402",
                        "insufficient_quota",
                        "exceeded your current quota",
                    )
                    if any(
                        pattern in error_msg_lower for pattern in non_retryable_patterns
                    ):
                        status_code = getattr(e, "status_code", "N/A")
                        logger.error(
                            f"[LLM:{call_id}] Quota/billing error (not retrying): "
                            f"status={status_code} {format_error_message(e)} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                    if attempt == max_retries:
                        # Final failure after all retries
                        error_type = type(e).__name__
                        status_code = getattr(e, "status_code", "N/A")
                        provider = getattr(e, "llm_provider", "N/A")
                        logger.error(
                            f"[LLM:{call_id}] Failed after {max_retries + 1} attempts: "
                            f"{error_type} status={status_code} provider={provider} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                    # Calculate exponential backoff delay
                    delay = min(
                        DEFAULT_RETRY_BASE_DELAY * (2**attempt), DEFAULT_RETRY_MAX_DELAY
                    )
                    await asyncio.sleep(delay)

                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    status_code = getattr(e, "status_code", "N/A")
                    logger.error(
                        f"[LLM:{call_id}] Failed: status={status_code} "
                        f"{format_error_message(e)} time={elapsed_ms:.0f}ms"
                    )
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(f"[LLM:{call_id}] Unexpected state in retry loop")

    def _track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        context: str = "",
    ) -> None:
        """Track usage statistics per model (and optionally per context).

        Thread-safe: uses lock to protect concurrent access to usage dicts.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            context: Optional context identifier (e.g., filename)
        """
        with self._usage_lock:
            # Track global usage (defaultdict auto-creates entries)
            self._usage[model]["requests"] += 1
            self._usage[model]["input_tokens"] += input_tokens
            self._usage[model]["output_tokens"] += output_tokens
            self._usage[model]["cost_usd"] += cost

            # Track per-context usage if context provided
            if context:
                self._context_usage[context][model]["requests"] += 1
                self._context_usage[context][model]["input_tokens"] += input_tokens
                self._context_usage[context][model]["output_tokens"] += output_tokens
                self._context_usage[context][model]["cost_usd"] += cost

    def get_usage(self) -> dict[str, dict[str, Any]]:
        """Get global usage statistics.

        Thread-safe: uses lock and returns a deep copy.
        """

        with self._usage_lock:
            return copy.deepcopy(self._usage)

    def get_total_cost(self) -> float:
        """Get total cost across all models.

        Thread-safe: uses lock for consistent read.
        """
        with self._usage_lock:
            return sum(u["cost_usd"] for u in self._usage.values())

    def get_context_usage(self, context: str) -> dict[str, dict[str, Any]]:
        """Get usage statistics for a specific context.

        Thread-safe: uses lock and returns a deep copy.

        Args:
            context: Context identifier (e.g., filename)

        Returns:
            Usage statistics for that context, or empty dict if not found
        """

        with self._usage_lock:
            return copy.deepcopy(self._context_usage.get(context, {}))

    def get_context_cost(self, context: str) -> float:
        """Get total cost for a specific context.

        Thread-safe: uses lock for consistent read.

        Args:
            context: Context identifier (e.g., filename)

        Returns:
            Total cost for that context
        """
        with self._usage_lock:
            context_usage = self._context_usage.get(context, {})
            return sum(u["cost_usd"] for u in context_usage.values())

    def clear_context_usage(self, context: str) -> None:
        """Clear usage tracking for a specific context.

        Thread-safe: uses lock for safe modification.

        Args:
            context: Context identifier to clear
        """
        with self._usage_lock:
            self._context_usage.pop(context, None)
            self._call_counter.pop(context, None)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with memory cache stats, persistent cache stats, and combined hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "memory": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": round(hit_rate * 100, 2),
                "size": self._cache.size,
            },
            "persistent": self._persistent_cache.stats(),
        }

    def clear_cache(self, scope: str = "memory") -> dict[str, Any]:
        """Clear the content cache and reset statistics.

        Args:
            scope: "memory" (in-memory only), "global", or "all"

        Returns:
            Dict with counts of cleared entries
        """
        result: dict[str, Any] = {"memory": 0, "global": 0}

        if scope in ("memory", "all"):
            result["memory"] = self._cache.size
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

        if scope in ("global", "all"):
            result["global"] = self._persistent_cache.clear()

        return result

    def clear_image_cache(self) -> None:
        """Clear the image cache to free memory after document processing."""
        self._image_cache.clear()
        self._image_cache_bytes = 0

    def _get_cached_image(self, image_path: Path) -> tuple[bytes, str]:
        """Get image bytes and base64 encoding, using cache if available.

        Uses LRU eviction when cache limits are reached (both count and bytes).
        Also ensures image is under 5MB limit for LLM API compatibility.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (raw bytes, base64 encoded string)
        """
        path_key = str(image_path)

        if path_key in self._image_cache:
            # Move to end for LRU (most recently used)
            self._image_cache.move_to_end(path_key)
            return self._image_cache[path_key]

        # Read and encode image
        image_data = image_path.read_bytes()

        # Check size limit (5MB API limit after base64 encoding)
        # Base64 encoding increases size by ~33%, so 5MB / 1.33 ≈ 3.76MB
        # Using 3.5MB raw bytes to ensure base64 encoded size stays under 5MB
        MAX_IMAGE_SIZE = 3.5 * 1024 * 1024
        if len(image_data) > MAX_IMAGE_SIZE:
            try:
                import io

                from PIL import Image

                with io.BytesIO(image_data) as buffer:
                    img = Image.open(buffer)
                    # Resize logic: iterative downscaling if needed
                    quality = 85
                    max_dim = 2048

                    while True:
                        if max(img.size) > max_dim:
                            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

                        out_buffer = io.BytesIO()
                        # Use JPEG for compression efficiency unless transparency is needed
                        fmt = "JPEG"
                        if img.mode in ("RGBA", "LA") or (
                            img.format and img.format.upper() == "PNG"
                        ):
                            # If PNG is too big, convert to JPEG (losing transparency) or resize more
                            # For document analysis, JPEG is usually fine
                            if len(image_data) > 8 * 1024 * 1024:  # If huge, force JPEG
                                img = img.convert("RGB")
                                fmt = "JPEG"
                            else:
                                fmt = "PNG"

                        if fmt == "JPEG" and img.mode != "RGB":
                            img = img.convert("RGB")

                        if fmt == "JPEG":
                            img.save(out_buffer, format=fmt, quality=quality)
                        else:
                            img.save(out_buffer, format=fmt)
                        new_data = out_buffer.getvalue()

                        if len(new_data) <= MAX_IMAGE_SIZE:
                            image_data = new_data
                            logger.debug(
                                f"Resized large image {image_path.name}: {len(new_data) / 1024 / 1024:.2f}MB"
                            )
                            break

                        # If still too big, reduce quality/size
                        if quality > 50 and fmt == "JPEG":
                            quality -= 15
                        else:
                            max_dim = int(max_dim * 0.75)
                            if max_dim < 512:  # Safety floor
                                logger.warning(
                                    f"Could not compress {image_path.name} below 5MB even at 512px"
                                )
                                break

            except Exception as e:
                logger.warning(f"Failed to resize large image {image_path.name}: {e}")

        base64_image = base64.b64encode(image_data).decode()

        # Calculate entry size: raw bytes + base64 string (roughly 1.33x raw size)
        entry_bytes = len(image_data) + len(base64_image)

        # Evict old entries if adding this would exceed limits
        while self._image_cache and (
            len(self._image_cache) >= self._image_cache_max_size
            or self._image_cache_bytes + entry_bytes > self._image_cache_max_bytes
        ):
            # Remove oldest entry (first item in OrderedDict)
            _, oldest_value = self._image_cache.popitem(last=False)
            old_bytes = len(oldest_value[0]) + len(oldest_value[1])
            self._image_cache_bytes -= old_bytes

        # Cache if entry size is reasonable (skip very large single images)
        if entry_bytes < self._image_cache_max_bytes // 2:
            self._image_cache[path_key] = (image_data, base64_image)
            self._image_cache_bytes += entry_bytes

        return image_data, base64_image
