"""LLM integration module using LiteLLM Router."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import instructor
import litellm
from litellm import completion_cost
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from markit.config import LLMConfig, PromptsConfig

from markit.prompts import PromptManager

# Retryable exceptions
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    Timeout,
    ServiceUnavailableError,
)

# Default retry configuration
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds


class MarkitLLMLogger(CustomLogger):
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
_markit_llm_logger = MarkitLLMLogger()


@dataclass
class LLMRuntime:
    """Global LLM runtime with shared concurrency control.

    This allows multiple LLMProcessor instances to share semaphores
    for rate limiting across the entire application.

    Supports separate concurrency limits for:
    - LLM API calls (rate-limited by provider)
    - I/O operations (disk reads, can be higher)

    Usage:
        runtime = LLMRuntime(concurrency=10, io_concurrency=20)
        processor1 = LLMProcessor(config, runtime=runtime)
        processor2 = LLMProcessor(config, runtime=runtime)
        # Both processors share the same semaphores
    """

    concurrency: int
    io_concurrency: int = 20  # Higher limit for I/O operations
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _io_semaphore: asyncio.Semaphore | None = field(
        default=None, init=False, repr=False
    )

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get or create the shared LLM concurrency semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    @property
    def io_semaphore(self) -> asyncio.Semaphore:
        """Get or create the shared I/O concurrency semaphore."""
        if self._io_semaphore is None:
            self._io_semaphore = asyncio.Semaphore(self.io_concurrency)
        return self._io_semaphore


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class ImageAnalysis:
    """Result of image analysis."""

    caption: str  # Short alt text
    description: str  # Detailed description
    extracted_text: str | None = None  # Text extracted from image
    model: str | None = None  # Model used for analysis


class ImageAnalysisResult(BaseModel):
    """Pydantic model for structured image analysis output."""

    caption: str = Field(description="Short alt text for the image (10-30 characters)")
    description: str = Field(description="Detailed markdown description of the image")
    extracted_text: str | None = Field(
        default=None,
        description="Text extracted from the image, preserving original layout",
    )


class SingleImageResult(BaseModel):
    """Result for a single image in batch analysis."""

    image_index: int = Field(description="Index of the image (1-based)")
    caption: str = Field(description="Short alt text for the image (10-30 characters)")
    description: str = Field(description="Detailed markdown description of the image")
    extracted_text: str | None = Field(
        default=None,
        description="Text extracted from the image, preserving original layout",
    )


class BatchImageAnalysisResult(BaseModel):
    """Result for batch image analysis."""

    images: list[SingleImageResult] = Field(
        description="Analysis results for each image"
    )


class Frontmatter(BaseModel):
    """Pydantic model for document frontmatter."""

    title: str = Field(description="Document title extracted from content")
    description: str = Field(
        description="Brief summary of the document (100 chars max)"
    )
    tags: list[str] = Field(description="Related tags (3-5 items)")


class DocumentProcessResult(BaseModel):
    """Pydantic model for combined cleaner + frontmatter output."""

    cleaned_markdown: str = Field(description="Cleaned and formatted markdown content")
    frontmatter: Frontmatter = Field(description="Document metadata")


class EnhancedDocumentResult(BaseModel):
    """Pydantic model for complete document enhancement output (Vision+LLM combined)."""

    cleaned_markdown: str = Field(description="Enhanced and cleaned markdown content")
    frontmatter: Frontmatter = Field(description="Document metadata")


class ContentCache:
    """LRU cache with TTL for LLM responses based on content hash.

    Uses OrderedDict for O(1) LRU eviction instead of O(n) min() search.
    """

    def __init__(self, maxsize: int = 100, ttl_seconds: int = 300) -> None:
        """
        Initialize content cache.

        Args:
            maxsize: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds (default 5 minutes)
        """
        from collections import OrderedDict

        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def _compute_hash(self, prompt: str, content: str) -> str:
        """Compute hash key from prompt and content.

        Uses full content for accurate cache keys. For very large content,
        the hash computation is still fast due to incremental SHA256.
        """
        import hashlib

        combined = f"{prompt}|{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, prompt: str, content: str) -> Any | None:
        """
        Get cached result if exists and not expired.

        On hit, moves the entry to end (most recently used).

        Args:
            prompt: Prompt template used
            content: Content being processed

        Returns:
            Cached result or None if not found/expired
        """
        key = self._compute_hash(prompt, content)
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        # Move to end on access (LRU behavior)
        self._cache.move_to_end(key)
        return result

    def set(self, prompt: str, content: str, result: Any) -> None:
        """
        Cache a result.

        Uses O(1) LRU eviction via OrderedDict.popitem(last=False).

        Args:
            prompt: Prompt template used
            content: Content being processed
            result: Result to cache
        """
        key = self._compute_hash(prompt, content)

        # If key exists, update and move to end
        if key in self._cache:
            self._cache[key] = (result, time.time())
            self._cache.move_to_end(key)
            return

        # Evict oldest entry if cache is full - O(1) operation
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)

        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)


class LLMProcessor:
    """LLM processor using LiteLLM Router for load balancing."""

    def __init__(
        self,
        config: LLMConfig,
        prompts_config: PromptsConfig | None = None,
        runtime: LLMRuntime | None = None,
    ) -> None:
        """
        Initialize LLM processor.

        Args:
            config: LLM configuration
            prompts_config: Optional prompts configuration
            runtime: Optional shared runtime for concurrency control.
                     If provided, uses runtime's semaphore instead of creating one.
        """
        self.config = config
        self._runtime = runtime
        self._router: Router | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._prompt_manager = PromptManager(prompts_config)

        # Usage tracking (global across all contexts)
        self._usage: dict[str, dict[str, Any]] = {}

        # Per-context usage tracking for batch processing
        self._context_usage: dict[str, dict[str, dict[str, Any]]] = {}

        # Call counter for each context (file)
        self._call_counter: dict[str, int] = {}

        # Content cache for avoiding duplicate LLM calls
        self._cache = ContentCache()
        self._cache_hits = 0
        self._cache_misses = 0

        # Image cache for avoiding repeated file reads during document processing
        # Key: file path string, Value: (bytes, base64_encoded_string)
        self._image_cache: dict[str, tuple[bytes, str]] = {}
        self._image_cache_max_size = 50  # Max number of images to cache

        # Register LiteLLM callback for additional details
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """Register LiteLLM callbacks for detailed logging."""
        # Add our custom logger to litellm callbacks if not already added
        if _markit_llm_logger not in (litellm.callbacks or []):
            if litellm.callbacks is None:
                litellm.callbacks = []
            litellm.callbacks.append(_markit_llm_logger)

    def _get_next_call_index(self, context: str) -> int:
        """Get the next call index for a given context."""
        self._call_counter[context] = self._call_counter.get(context, 0) + 1
        return self._call_counter[context]

    def reset_call_counter(self, context: str = "") -> None:
        """Reset call counter for a context or all contexts."""
        if context:
            self._call_counter.pop(context, None)
        else:
            self._call_counter.clear()

    @property
    def router(self) -> Router:
        """Get or create the LiteLLM Router."""
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
        return asyncio.Semaphore(20)

    def _create_router(self) -> Router:
        """Create LiteLLM Router from configuration."""
        if not self.config.model_list:
            raise ValueError("No models configured in llm.model_list")

        # Build model list with resolved API keys
        model_list = []
        for model_config in self.config.model_list:
            model_entry = {
                "model_name": model_config.model_name,
                "litellm_params": {
                    "model": model_config.litellm_params.model,
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

            if model_config.model_info:
                model_entry["model_info"] = model_config.model_info.model_dump()

            model_list.append(model_entry)

        # Build router settings
        router_settings = self.config.router_settings.model_dump()

        # Disable internal retries - we handle retries ourselves for better logging
        router_settings["num_retries"] = 0

        # Log router configuration for debugging load balancing
        logger.info(
            f"[Router] Creating with strategy={router_settings.get('routing_strategy')}, "
            f"models={len(model_list)}"
        )
        for entry in model_list:
            weight = entry.get("litellm_params", {}).get("weight", 1)
            logger.debug(
                f"[Router] Model: {entry['model_name']} -> {entry['litellm_params']['model']} (weight={weight})"
            )

        return Router(model_list=model_list, **router_settings)

    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, Any]],
        context: str = "",
    ) -> LLMResponse:
        """
        Make an LLM call with rate limiting, retry logic, and detailed logging.

        Args:
            model: Logical model name (e.g., "default", "vision")
            messages: Chat messages
            context: Context identifier for logging (e.g., filename)

        Returns:
            LLMResponse with content and usage info
        """
        # Generate call ID for logging
        call_index = self._get_next_call_index(context) if context else 0
        call_id = f"{context}:{call_index}" if context else f"call:{call_index}"

        max_retries = self.config.router_settings.num_retries
        return await self._call_llm_with_retry(
            model=model,
            messages=messages,
            call_id=call_id,
            context=context,
            max_retries=max_retries,
        )

    async def _call_llm_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        call_id: str,
        context: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> LLMResponse:
        """
        Make an LLM call with custom retry logic and detailed logging.

        Args:
            model: Logical model name (e.g., "default", "vision")
            messages: Chat messages
            call_id: Unique identifier for this call (for logging)
            context: Context identifier for usage tracking (e.g., filename)
            max_retries: Maximum number of retry attempts

        Returns:
            LLMResponse with content and usage info
        """
        last_exception: Exception | None = None

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

                    response = await self.router.acompletion(
                        model=model,
                        messages=messages,  # type: ignore[arg-type]
                        metadata={"call_id": call_id, "attempt": attempt},
                    )

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    content = response.choices[0].message.content or ""
                    actual_model = response.model or model

                    # Calculate cost
                    try:
                        cost = completion_cost(completion_response=response)
                    except Exception:
                        cost = 0.0

                    # Track usage
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0

                    self._track_usage(
                        actual_model, input_tokens, output_tokens, cost, context
                    )

                    # Log success
                    logger.info(
                        f"[LLM:{call_id}] {actual_model} "
                        f"tokens={input_tokens}+{output_tokens} "
                        f"time={elapsed_ms:.0f}ms cost=${cost:.6f}"
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
                    error_type = type(e).__name__
                    status_code = getattr(e, "status_code", "N/A")
                    error_msg = str(e)[:200]  # Truncate long messages
                    logger.error(
                        f"[LLM:{call_id}] Failed: {error_type} "
                        f"status={status_code} msg={error_msg} "
                        f"time={elapsed_ms:.0f}ms"
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

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            context: Optional context identifier (e.g., filename)
        """
        # Track global usage
        if model not in self._usage:
            self._usage[model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }

        self._usage[model]["requests"] += 1
        self._usage[model]["input_tokens"] += input_tokens
        self._usage[model]["output_tokens"] += output_tokens
        self._usage[model]["cost_usd"] += cost

        # Track per-context usage if context provided
        if context:
            if context not in self._context_usage:
                self._context_usage[context] = {}
            if model not in self._context_usage[context]:
                self._context_usage[context][model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }

            self._context_usage[context][model]["requests"] += 1
            self._context_usage[context][model]["input_tokens"] += input_tokens
            self._context_usage[context][model]["output_tokens"] += output_tokens
            self._context_usage[context][model]["cost_usd"] += cost

    def get_usage(self) -> dict[str, dict[str, Any]]:
        """Get global usage statistics."""
        return self._usage.copy()

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(u["cost_usd"] for u in self._usage.values())

    def get_context_usage(self, context: str) -> dict[str, dict[str, Any]]:
        """Get usage statistics for a specific context.

        Args:
            context: Context identifier (e.g., filename)

        Returns:
            Usage statistics for that context, or empty dict if not found
        """
        return self._context_usage.get(context, {}).copy()

    def get_context_cost(self, context: str) -> float:
        """Get total cost for a specific context.

        Args:
            context: Context identifier (e.g., filename)

        Returns:
            Total cost for that context
        """
        context_usage = self._context_usage.get(context, {})
        return sum(u["cost_usd"] for u in context_usage.values())

    def clear_context_usage(self, context: str) -> None:
        """Clear usage tracking for a specific context.

        Args:
            context: Context identifier to clear
        """
        self._context_usage.pop(context, None)
        self._call_counter.pop(context, None)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache hits, misses, hit rate, and size
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": self._cache.size,
        }

    def clear_cache(self) -> None:
        """Clear the content cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def clear_image_cache(self) -> None:
        """Clear the image cache to free memory after document processing."""
        self._image_cache.clear()

    def _get_cached_image(self, image_path: Path) -> tuple[bytes, str]:
        """Get image bytes and base64 encoding, using cache if available.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (raw bytes, base64 encoded string)
        """
        path_key = str(image_path)

        if path_key in self._image_cache:
            return self._image_cache[path_key]

        # Read and encode image
        image_data = image_path.read_bytes()
        base64_image = base64.b64encode(image_data).decode()

        # Cache if not at max capacity
        if len(self._image_cache) < self._image_cache_max_size:
            self._image_cache[path_key] = (image_data, base64_image)

        return image_data, base64_image

    @staticmethod
    def _smart_truncate(text: str, max_chars: int, preserve_end: bool = False) -> str:
        """Truncate text at sentence/paragraph boundary to preserve readability.

        Instead of cutting at arbitrary positions, finds the nearest sentence
        or paragraph ending before the limit.

        Args:
            text: Text to truncate
            max_chars: Maximum character limit
            preserve_end: If True, preserve the end instead of the beginning

        Returns:
            Truncated text at a natural boundary
        """
        if len(text) <= max_chars:
            return text

        if preserve_end:
            # Find a good starting point from the end
            search_start = len(text) - max_chars
            search_text = text[search_start : search_start + 500]

            # Look for paragraph or sentence boundary
            for marker in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
                idx = search_text.find(marker)
                if idx != -1:
                    return text[search_start + idx + len(marker) :]

            return text[-max_chars:]

        # Default: preserve beginning, find a good ending point
        search_text = (
            text[max_chars - 500 : max_chars + 200]
            if max_chars > 500
            else text[: max_chars + 200]
        )
        search_offset = max(0, max_chars - 500)

        # Priority: paragraph > sentence > any break
        for marker in ["\n\n", "。\n", ".\n", "。", ".", "！", "!", "？", "?"]:
            idx = search_text.rfind(marker)
            if idx != -1:
                end_pos = search_offset + idx + len(marker)
                if (
                    end_pos <= max_chars + 100
                ):  # Allow slight overflow for better breaks
                    return text[:end_pos].rstrip()

        # Fall back to simple truncation
        return text[:max_chars]

    @staticmethod
    def _extract_protected_content(content: str) -> dict[str, list[str]]:
        """Extract content that must be preserved through LLM processing.

        Extracts:
        - Image links: ![...](...)
        - Slide comments: <!-- Slide X -->
        - Page image comments: <!-- ![Page X](...) --> and <!-- Page images... -->

        Args:
            content: Original markdown content

        Returns:
            Dict with 'images', 'slides', 'page_comments' lists
        """
        import re

        protected: dict[str, list[str]] = {
            "images": [],
            "slides": [],
            "page_comments": [],
        }

        # Extract image links
        protected["images"] = re.findall(r"!\[[^\]]*\]\([^)]+\)", content)

        # Extract slide comments
        protected["slides"] = re.findall(r"<!--\s*Slide\s+\d+\s*-->", content)

        # Extract page image comments
        # Pattern 1: <!-- Page images for reference -->
        # Pattern 2: <!-- ![Page X](screenshots/...) -->
        page_header_pattern = r"<!--\s*Page images for reference\s*-->"
        page_img_pattern = r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->"
        protected["page_comments"] = re.findall(
            page_header_pattern, content
        ) + re.findall(page_img_pattern, content)

        return protected

    @staticmethod
    def _protect_content(content: str) -> tuple[str, dict[str, str]]:
        """Replace protected content with placeholders before LLM processing.

        This preserves the position of images, slides, and page comments
        by replacing them with unique placeholders that the LLM is unlikely
        to modify.

        Args:
            content: Original markdown content

        Returns:
            Tuple of (content with placeholders, mapping of placeholder -> original)
        """
        import re

        mapping: dict[str, str] = {}
        result = content

        # Protect images: ![...](...)
        img_pattern = r"!\[[^\]]*\]\([^)]+\)"
        for i, match in enumerate(re.finditer(img_pattern, content)):
            placeholder = f"__MARKIT_IMG_{i}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # Protect slide comments: <!-- Slide X -->
        slide_pattern = r"<!--\s*Slide\s+\d+\s*-->"
        for i, match in enumerate(re.finditer(slide_pattern, result)):
            placeholder = f"__MARKIT_SLIDE_{i}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)

        # Protect page image comments: <!-- ![Page X](...) --> and <!-- Page images... -->
        # Use separate patterns for header and individual page image comments
        page_header_pattern = r"<!--\s*Page images for reference\s*-->"
        page_img_pattern = r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->"
        page_idx = 0
        for match in re.finditer(page_header_pattern, result):
            placeholder = f"__MARKIT_PAGE_{page_idx}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)
            page_idx += 1
        for match in re.finditer(page_img_pattern, result):
            placeholder = f"__MARKIT_PAGE_{page_idx}__"
            mapping[placeholder] = match.group(0)
            result = result.replace(match.group(0), placeholder, 1)
            page_idx += 1

        return result, mapping

    @staticmethod
    def _unprotect_content(
        content: str,
        mapping: dict[str, str],
        protected: dict[str, list[str]] | None = None,
    ) -> str:
        """Restore protected content from placeholders after LLM processing.

        Also handles cases where the LLM removed placeholders by appending
        missing content at the end.

        Args:
            content: LLM output with placeholders
            mapping: Mapping of placeholder -> original content
            protected: Optional dict of protected content for fallback restoration

        Returns:
            Content with placeholders replaced by original content
        """
        result = content

        # First pass: replace placeholders with original content
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)

        # Second pass: if protected content was provided, restore any missing items
        # This handles cases where the LLM removed placeholders entirely
        if protected:
            import re

            # Helper to check if an image is already in result (by filename)
            def image_exists_in_result(img_syntax: str, text: str) -> bool:
                """Check if image already exists in result by filename."""
                match = re.search(r"\]\(([^)]+)\)", img_syntax)
                if match:
                    img_path = match.group(1)
                    img_name = img_path.split("/")[-1]
                    # Check if same filename exists in any image reference
                    return bool(
                        re.search(rf"!\[[^\]]*\]\([^)]*{re.escape(img_name)}\)", text)
                    )
                return False

            # Restore missing images at end (fallback)
            # Only restore if the image filename doesn't already exist
            for img in protected.get("images", []):
                if img not in result and not image_exists_in_result(img, result):
                    match = re.search(r"\]\(([^)]+)\)", img)
                    if match:
                        img_name = match.group(1).split("/")[-1]
                        logger.debug(f"Restoring missing image at end: {img_name}")
                    result = result.rstrip() + "\n\n" + img

            # Restore missing slide comments at heading boundaries
            # Key fix: Match slides to H1/H2 headings more intelligently
            missing_slides = [s for s in protected.get("slides", []) if s not in result]
            if missing_slides:
                slide_info = []
                for slide in missing_slides:
                    match = re.search(r"Slide\s+(\d+)", slide)
                    if match:
                        slide_info.append((int(match.group(1)), slide))
                slide_info.sort()

                lines = result.split("\n")
                # Find H1 and H2 headings as potential slide boundaries
                heading_positions = [
                    i
                    for i, line in enumerate(lines)
                    if line.startswith("# ") or line.startswith("## ")
                ]

                # Only insert if we have matching heading positions
                # Don't append orphan slide comments to the end
                inserted_count = 0
                for idx, (slide_num, slide) in enumerate(slide_info):
                    if idx < len(heading_positions):
                        insert_pos = heading_positions[idx] + inserted_count * 2
                        lines.insert(insert_pos, slide)
                        lines.insert(insert_pos + 1, "")
                        inserted_count += 1
                        logger.debug(
                            f"Restored slide {slide_num} before heading at line {insert_pos}"
                        )
                    # Don't append orphan slides to the end - they look wrong
                result = "\n".join(lines)

            # Restore missing page comments at end
            # Only restore if not already present (avoid duplicates)
            page_header = "<!-- Page images for reference -->"
            has_page_header = page_header in result

            for comment in protected.get("page_comments", []):
                if comment not in result:
                    # For page header, only add if not present
                    if comment == page_header:
                        if not has_page_header:
                            result = result.rstrip() + "\n\n" + comment
                            has_page_header = True
                    # For individual page image comments, check if already exists
                    else:
                        # Extract page number to check for duplicates
                        page_match = re.search(r"!\[Page\s+(\d+)\]", comment)
                        if page_match:
                            page_num = page_match.group(1)
                            # Check if this page is already referenced (commented or not)
                            page_pattern = rf"!\[Page\s+{page_num}\]"
                            if not re.search(page_pattern, result):
                                result = result.rstrip() + "\n" + comment

        return result

    @staticmethod
    def _restore_protected_content(result: str, protected: dict[str, list[str]]) -> str:
        """Restore any protected content that was lost during LLM processing.

        Legacy method - use _unprotect_content for new code.

        Args:
            result: LLM output
            protected: Dict of protected content from _extract_protected_content

        Returns:
            Result with missing protected content restored
        """
        return LLMProcessor._unprotect_content(result, {}, protected)

    async def clean_markdown(self, content: str, context: str = "") -> str:
        """
        Clean and optimize markdown content.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Args:
            content: Raw markdown content
            context: Context identifier for logging (e.g., filename)

        Returns:
            Cleaned markdown content
        """
        # Check cache first
        cached = self._cache.get("cleaner", content)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"[{context}] Cache hit for clean_markdown")
            return cached

        self._cache_misses += 1

        # Extract and protect content before LLM processing
        protected = self._extract_protected_content(content)
        protected_content, mapping = self._protect_content(content)

        prompt = self._prompt_manager.get_prompt("cleaner", content=protected_content)

        response = await self._call_llm(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            context=context,
        )

        # Restore protected content from placeholders, with fallback for removed items
        result = self._unprotect_content(response.content, mapping, protected)

        # Cache the result
        self._cache.set("cleaner", content, result)
        return result

    async def generate_frontmatter(
        self,
        content: str,
        source: str,
    ) -> str:
        """
        Generate YAML frontmatter for markdown content.

        Args:
            content: Markdown content
            source: Source file name

        Returns:
            YAML frontmatter string (without --- markers)
        """
        # Detect document language
        language = self._detect_language(content)

        prompt = self._prompt_manager.get_prompt(
            "frontmatter",
            content=self._smart_truncate(content, 4000),
            source=source,
            language=language,
        )

        response = await self._call_llm(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            context=source,
        )

        return response.content

    async def analyze_image(
        self, image_path: Path, language: str = "en"
    ) -> ImageAnalysis:
        """
        Analyze an image using vision model.

        Uses Instructor for structured output with fallback mechanisms:
        1. Try Instructor with structured output
        2. Fallback to JSON mode + manual parsing
        3. Fallback to original two-call method

        Args:
            image_path: Path to the image file
            language: Language for output (e.g., "en", "zh")

        Returns:
            ImageAnalysis with caption and description
        """
        # Get cached image data and base64 encoding
        _, base64_image = self._get_cached_image(image_path)

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        # Language instruction
        lang_instruction = (
            "Output in English." if language == "en" else "使用中文输出。"
        )

        # Get combined prompt
        prompt = self._prompt_manager.get_prompt("image_analysis")
        prompt = prompt.replace(
            "**输出语言必须与源文档保持一致** - 英文文档用英文，中文文档用中文",
            lang_instruction,
        )

        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            }
        ]

        # Use logical model name for router load balancing
        # Router will select actual model based on routing strategy
        vision_model = "vision"

        # Try structured output methods with fallbacks
        result = await self._analyze_image_with_fallback(
            messages, vision_model, image_path.name
        )

        logger.debug(f"Analyzed image: {image_path.name}")
        return result

    async def analyze_images_batch(
        self,
        image_paths: list[Path],
        language: str = "en",
        max_images_per_batch: int = 10,
    ) -> list[ImageAnalysis]:
        """
        Analyze multiple images in batches to reduce LLM calls.

        Args:
            image_paths: List of image paths to analyze
            language: Language for output ("en" or "zh")
            max_images_per_batch: Max images per LLM call (default 10)

        Returns:
            List of ImageAnalysis results in same order as input
        """
        if not image_paths:
            return []

        # Process in batches
        all_results: list[ImageAnalysis] = []
        num_batches = (
            len(image_paths) + max_images_per_batch - 1
        ) // max_images_per_batch

        for batch_num in range(num_batches):
            batch_start = batch_num * max_images_per_batch
            batch_end = min(batch_start + max_images_per_batch, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            logger.info(
                f"Analyzing images batch {batch_num + 1}/{num_batches}: "
                f"{len(batch_paths)} images"
            )

            batch_results = await self._analyze_batch_impl(batch_paths, language)
            all_results.extend(batch_results)

        return all_results

    async def _analyze_batch_impl(
        self,
        image_paths: list[Path],
        language: str,
    ) -> list[ImageAnalysis]:
        """Implementation of batch image analysis using Instructor."""
        # Language instruction
        lang_instruction = (
            "Output in English." if language == "en" else "使用中文输出。"
        )

        # Build batch prompt
        prompt = f"""Analyze each of the following {len(image_paths)} images.

For each image, provide:
1. caption: Short alt text (10-30 characters)
2. description: Detailed markdown description
3. extracted_text: Any text visible in the image (null if none)

{lang_instruction}

Return a JSON object with an "images" array containing results for each image in order."""

        # Build content parts with all images
        content_parts: list[dict] = [{"type": "text", "text": prompt}]

        for i, image_path in enumerate(image_paths, 1):
            _, base64_image = self._get_cached_image(image_path)
            suffix = image_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/jpeg")

            content_parts.append(
                {"type": "text", "text": f"\n## Image {i}: {image_path.name}"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        try:
            async with self.semaphore:
                client = instructor.from_litellm(
                    self.router.acompletion, mode=instructor.Mode.JSON
                )
                (
                    response,
                    raw_response,
                ) = await client.chat.completions.create_with_completion(
                    model="vision",
                    messages=[{"role": "user", "content": content_parts}],  # type: ignore[arg-type]
                    response_model=BatchImageAnalysisResult,
                    max_retries=0,
                )

                # Track usage
                actual_model = getattr(raw_response, "model", None) or "vision"
                if hasattr(raw_response, "usage") and raw_response.usage is not None:
                    input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                    output_tokens = (
                        getattr(raw_response.usage, "completion_tokens", 0) or 0
                    )
                    try:
                        cost = completion_cost(completion_response=raw_response)
                    except Exception:
                        cost = 0.0
                    self._track_usage(
                        actual_model, input_tokens, output_tokens, cost, "batch_images"
                    )

                # Convert to ImageAnalysis list
                results: list[ImageAnalysis] = []
                for img_result in response.images:
                    results.append(
                        ImageAnalysis(
                            caption=img_result.caption,
                            description=img_result.description,
                            extracted_text=img_result.extracted_text,
                        )
                    )

                # Ensure we have results for all images
                while len(results) < len(image_paths):
                    results.append(
                        ImageAnalysis(
                            caption="Image",
                            description="Image analysis failed",
                            extracted_text=None,
                        )
                    )

                return results

        except Exception as e:
            logger.warning(
                f"Batch image analysis failed: {e}, falling back to individual analysis"
            )
            # Fallback: analyze each image individually
            results = []
            for image_path in image_paths:
                try:
                    result = await self.analyze_image(image_path, language)
                    results.append(result)
                except Exception:
                    results.append(
                        ImageAnalysis(
                            caption="Image",
                            description="Image analysis failed",
                            extracted_text=None,
                        )
                    )
            return results

    def _get_actual_model_name(self, logical_name: str) -> str:
        """Get actual model name from router configuration."""
        for model_config in self.config.model_list:
            if model_config.model_name == logical_name:
                return model_config.litellm_params.model
        # Fallback to first model if logical name not found
        if self.config.model_list:
            return self.config.model_list[0].litellm_params.model
        return "gpt-4o-mini"  # Ultimate fallback

    async def _analyze_image_with_fallback(
        self,
        messages: list[dict],
        model: str,
        image_name: str,
    ) -> ImageAnalysis:
        """
        Analyze image with multiple fallback strategies.

        Strategy 1: Instructor structured output (most precise)
        Strategy 2: JSON mode + manual parsing
        Strategy 3: Original two-call method (most compatible)
        """
        # Strategy 1: Try Instructor
        try:
            # Deep copy to prevent Instructor from modifying original messages
            result = await self._analyze_with_instructor(copy.deepcopy(messages), model)
            logger.debug(f"[{image_name}] Used Instructor structured output")
            return result
        except Exception as e:
            logger.debug(f"[{image_name}] Instructor failed: {e}, trying JSON mode")

        # Strategy 2: Try JSON mode
        try:
            result = await self._analyze_with_json_mode(copy.deepcopy(messages), model)
            logger.debug(f"[{image_name}] Used JSON mode fallback")
            return result
        except Exception as e:
            logger.debug(
                f"[{image_name}] JSON mode failed: {e}, using two-call fallback"
            )

        # Strategy 3: Original two-call method
        return await self._analyze_with_two_calls(
            copy.deepcopy(messages), model, context=image_name
        )

    async def _analyze_with_instructor(
        self,
        messages: list[dict],
        model: str,
    ) -> ImageAnalysis:
        """Analyze using Instructor for structured output."""
        async with self.semaphore:
            # Create instructor client from router for load balancing
            client = instructor.from_litellm(
                self.router.acompletion, mode=instructor.Mode.JSON
            )

            # Use create_with_completion to get both the model and the raw response
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=ImageAnalysisResult,
                max_retries=0,
            )

            # Track usage from raw API response
            # Get actual model from response for accurate tracking
            actual_model = getattr(raw_response, "model", None) or model
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(actual_model, input_tokens, output_tokens, cost)

            return ImageAnalysis(
                caption=response.caption.strip(),
                description=response.description,
                extracted_text=response.extracted_text,
                model=actual_model,
            )

    async def _analyze_with_json_mode(
        self,
        messages: list[dict],
        model: str,
    ) -> ImageAnalysis:
        """Analyze using JSON mode with manual parsing."""
        # Add JSON instruction to the prompt
        json_messages = messages.copy()
        json_messages[0] = {
            **messages[0],
            "content": [
                {
                    "type": "text",
                    "text": messages[0]["content"][0]["text"]
                    + "\n\nReturn a JSON object with 'caption' and 'description' fields.",
                },
                messages[0]["content"][1],  # image
            ],
        }

        async with self.semaphore:
            # Use router for load balancing
            response = await self.router.acompletion(
                model=model,
                messages=json_messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"  # type: ignore[union-attr]
            actual_model = response.model or model

            # Track usage
            usage = response.usage  # type: ignore[union-attr]
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            try:
                cost = completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            self._track_usage(actual_model, input_tokens, output_tokens, cost)

            # Parse JSON
            data = json.loads(content)
            return ImageAnalysis(
                caption=data.get("caption", "").strip(),
                description=data.get("description", ""),
                extracted_text=data.get("extracted_text"),
                model=actual_model,
            )

    async def _analyze_with_two_calls(
        self,
        messages: list[dict],
        model: str,  # noqa: ARG002
        context: str = "",
    ) -> ImageAnalysis:
        """Original two-call method as final fallback."""
        # Extract original prompt and image from messages
        original_content = messages[0]["content"]
        image_content = original_content[1]  # The image part

        # Language instruction (extract from original prompt)
        lang_instruction = "Output in English."
        if "使用中文输出" in original_content[0]["text"]:
            lang_instruction = "使用中文输出。"

        # Generate caption
        caption_prompt = self._prompt_manager.get_prompt("image_caption")
        caption_prompt = caption_prompt.replace(
            "**输出语言必须与源文档保持一致** - 英文文档用英文，中文文档用中文",
            lang_instruction,
        )
        caption_response = await self._call_llm(
            model="vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption_prompt},
                        image_content,
                    ],
                }
            ],
            context=context,
        )

        # Generate description
        desc_prompt = self._prompt_manager.get_prompt("image_description")
        desc_response = await self._call_llm(
            model="vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": desc_prompt},
                        image_content,
                    ],
                }
            ],
            context=context,
        )

        return ImageAnalysis(
            caption=caption_response.content.strip(),
            description=desc_response.content,
            model=caption_response.model,
        )

    async def extract_page_content(self, image_path: Path, context: str = "") -> str:
        """
        Extract text content from a document page image.

        Used for OCR+LLM mode and PPTX+LLM mode where pages are rendered
        as images and we want to extract structured text content.

        Args:
            image_path: Path to the page image file
            context: Context identifier for logging (e.g., parent document name)

        Returns:
            Extracted markdown content from the page
        """
        # Get cached image data and base64 encoding
        _, base64_image = self._get_cached_image(image_path)

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        # Get page content extraction prompt
        prompt = self._prompt_manager.get_prompt("page_content")

        # Use image_path.name as context if not provided
        call_context = context or image_path.name

        response = await self._call_llm(
            model="vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            context=call_context,
        )

        return response.content

    @staticmethod
    def _protect_image_positions(text: str) -> tuple[str, dict[str, str]]:
        """Replace image references with position markers to prevent LLM from moving them.

        Args:
            text: Markdown text with image references

        Returns:
            Tuple of (text with markers, mapping of marker -> original image reference)
        """
        import re

        mapping: dict[str, str] = {}
        result = text

        # Match image references: ![...](assets/...) - only assets folder images
        # This excludes page screenshots which have their own protection
        img_pattern = r"!\[[^\]]*\]\(assets/[^)]+\)"
        for i, match in enumerate(re.finditer(img_pattern, text)):
            marker = f"<!-- IMG_MARKER: {i} -->"
            mapping[marker] = match.group(0)
            result = result.replace(match.group(0), marker, 1)

        return result, mapping

    @staticmethod
    def _restore_image_positions(text: str, mapping: dict[str, str]) -> str:
        """Restore original image references from position markers.

        Args:
            text: Text with position markers
            mapping: Mapping of marker -> original image reference

        Returns:
            Text with original image references restored
        """
        result = text
        for marker, original in mapping.items():
            result = result.replace(marker, original)
        return result

    async def enhance_document_with_vision(
        self,
        extracted_text: str,
        page_images: list[Path],
        context: str = "",
    ) -> str:
        """
        Clean document format using extracted text and page images as reference.

        This method only cleans formatting issues (removes residuals, fixes structure).
        It does NOT restructure or rewrite content.

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            context: Context identifier for logging (e.g., document name)

        Returns:
            Cleaned markdown content (same content, cleaner format)
        """
        if not page_images:
            return extracted_text

        # Build message with text + images
        prompt = self._prompt_manager.get_prompt("document_enhance")

        # Prepare content parts
        content_parts: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\n## Extracted Text:\n\n{extracted_text}",
            },
        ]

        # Add page images (using cache to avoid repeated reads)
        for i, image_path in enumerate(page_images, 1):
            _, base64_image = self._get_cached_image(image_path)

            suffix = image_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/jpeg")

            content_parts.append(
                {
                    "type": "text",
                    "text": f"\n## Page {i} Image:",
                }
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        response = await self._call_llm(
            model="vision",
            messages=[{"role": "user", "content": content_parts}],
            context=context,
        )

        return response.content

    async def enhance_document_complete(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str = "",
        max_pages_per_batch: int = 10,
    ) -> tuple[str, str]:
        """
        Complete document enhancement: clean format + generate frontmatter.

        Architecture (simplified - vision only cleans, frontmatter separate):
        - Vision calls only do format cleaning (no Instructor, pure markdown output)
        - Frontmatter generated separately after all cleaning is done
        - More stable output, avoids JSON mode content truncation

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of paths to page/slide images
            source: Source file name
            max_pages_per_batch: Max pages per batch (default 10)

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        if not page_images:
            # No images, fall back to regular process_document
            return await self.process_document(extracted_text, source)

        # Step 1: Clean format with vision (single or batched)
        if len(page_images) <= max_pages_per_batch:
            logger.info(
                f"[{source}] Processing {len(page_images)} pages in single call"
            )
            cleaned = await self.enhance_document_with_vision(
                extracted_text, page_images, context=source
            )
        else:
            logger.info(
                f"[{source}] Processing {len(page_images)} pages in batches of {max_pages_per_batch}"
            )
            cleaned = await self._enhance_document_batched_simple(
                extracted_text, page_images, max_pages_per_batch, source
            )

        # Step 2: Generate frontmatter from cleaned content
        frontmatter = await self.generate_frontmatter(cleaned, source)

        return cleaned, frontmatter

    async def _enhance_with_frontmatter(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str,
    ) -> tuple[str, str]:
        """Enhance document with vision and generate frontmatter in one call.

        Uses Instructor for structured output.

        Args:
            extracted_text: Text to clean
            page_images: Page images for visual reference
            source: Source file name

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        import yaml

        # Get combined prompt
        prompt = self._prompt_manager.get_prompt(
            "document_enhance_complete",
            source=source,
        )

        # Build content parts
        content_parts: list[dict] = [
            {
                "type": "text",
                "text": f"{prompt}\n\n## Extracted Text:\n\n{extracted_text}",
            },
        ]

        # Add page images
        for i, image_path in enumerate(page_images, 1):
            _, base64_image = self._get_cached_image(image_path)
            suffix = image_path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(suffix, "image/jpeg")
            content_parts.append({"type": "text", "text": f"\n## Page {i} Image:"})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

        async with self.semaphore:
            client = instructor.from_litellm(
                self.router.acompletion, mode=instructor.Mode.JSON
            )
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model="vision",
                messages=[{"role": "user", "content": content_parts}],  # type: ignore[arg-type]
                response_model=EnhancedDocumentResult,
                max_retries=0,
            )

            # Track usage
            actual_model = getattr(raw_response, "model", None) or "vision"
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(
                    actual_model, input_tokens, output_tokens, cost, source
                )

            # Build frontmatter YAML
            frontmatter_dict = {
                "title": response.frontmatter.title,
                "description": response.frontmatter.description,
                "tags": response.frontmatter.tags,
                "source": source,
            }
            frontmatter_yaml = yaml.dump(
                frontmatter_dict, allow_unicode=True, default_flow_style=False
            ).strip()

            return response.cleaned_markdown, frontmatter_yaml

    @staticmethod
    def _split_text_by_pages(text: str, num_pages: int) -> list[str]:
        """Split text into chunks corresponding to page ranges.

        Uses page markers (<!-- ![Page N]... -->) if present, otherwise
        splits by paragraph count proportionally.

        Args:
            text: Full document text
            num_pages: Number of pages/images

        Returns:
            List of text chunks, one per page
        """
        import re

        # Try to find page markers
        page_pattern = r"<!--\s*!\[Page\s+(\d+)\]"
        markers = list(re.finditer(page_pattern, text))

        if len(markers) >= num_pages - 1:
            # Use page markers to split
            chunks = []
            for i in range(num_pages):
                if i == 0:
                    start = 0
                else:
                    start = (
                        markers[i - 1].start() if i - 1 < len(markers) else len(text)
                    )

                if i < len(markers):
                    end = markers[i].start()
                else:
                    end = len(text)

                chunks.append(text[start:end].strip())
            return chunks

        # Fallback: split by paragraphs proportionally
        paragraphs = text.split("\n\n")
        if len(paragraphs) < num_pages:
            # Very short text, just return whole text for each page
            return [text] * num_pages

        paragraphs_per_page = len(paragraphs) // num_pages
        chunks = []
        for i in range(num_pages):
            start_idx = i * paragraphs_per_page
            if i == num_pages - 1:
                # Last chunk gets remaining paragraphs
                end_idx = len(paragraphs)
            else:
                end_idx = start_idx + paragraphs_per_page
            chunks.append("\n\n".join(paragraphs[start_idx:end_idx]))

        return chunks

    async def _enhance_document_batched_simple(
        self,
        extracted_text: str,
        page_images: list[Path],
        batch_size: int,
        source: str = "",
    ) -> str:
        """Process long documents in batches - vision cleaning only.

        All batches use the same method for consistent output format.

        Args:
            extracted_text: Full document text
            page_images: All page images
            batch_size: Pages per batch
            source: Source file name

        Returns:
            Merged cleaned content
        """
        num_pages = len(page_images)
        num_batches = (num_pages + batch_size - 1) // batch_size

        # Split text by pages
        page_texts = self._split_text_by_pages(extracted_text, num_pages)

        cleaned_parts = []

        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, num_pages)

            # Get text and images for this batch
            batch_texts = page_texts[batch_start:batch_end]
            batch_images = page_images[batch_start:batch_end]
            batch_text = "\n\n".join(batch_texts)

            logger.info(
                f"[{source}] Batch {batch_num + 1}/{num_batches}: "
                f"pages {batch_start + 1}-{batch_end}"
            )

            # All batches: clean only (no frontmatter)
            batch_cleaned = await self.enhance_document_with_vision(
                batch_text, batch_images, context=f"{source}:batch{batch_num + 1}"
            )

            cleaned_parts.append(batch_cleaned)

        # Merge all batches
        return "\n\n".join(cleaned_parts)

    async def process_document(
        self,
        markdown: str,
        source: str,
    ) -> tuple[str, str]:
        """
        Process a document with LLM: clean and generate frontmatter.

        Uses placeholder-based protection to preserve images, slides, and
        page comments in their original positions during LLM processing.

        Uses a combined prompt with Instructor for structured output,
        falling back to parallel separate calls if structured output fails.

        Args:
            markdown: Raw markdown content
            source: Source file name

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml)
        """
        # Extract and protect content before LLM processing
        protected = self._extract_protected_content(markdown)
        protected_content, mapping = self._protect_content(markdown)

        # Try combined approach with Instructor first
        try:
            result = await self._process_document_combined(protected_content, source)

            # Restore protected content from placeholders, with fallback
            cleaned = self._unprotect_content(
                result.cleaned_markdown, mapping, protected
            )

            # Convert Frontmatter to YAML string
            import yaml

            frontmatter_dict = {
                "title": result.frontmatter.title,
                "description": result.frontmatter.description,
                "tags": result.frontmatter.tags,
                "source": source,
            }
            frontmatter_yaml = yaml.dump(
                frontmatter_dict, allow_unicode=True, default_flow_style=False
            ).strip()
            logger.debug(f"[{source}] Used combined document processing")
            return cleaned, frontmatter_yaml
        except Exception as e:
            logger.debug(
                f"[{source}] Combined processing failed: {e}, using parallel fallback"
            )

        # Fallback: Run cleaning and frontmatter generation in parallel
        # clean_markdown uses its own protection mechanism
        clean_task = asyncio.create_task(self.clean_markdown(markdown, context=source))
        frontmatter_task = asyncio.create_task(
            self.generate_frontmatter(markdown, source)
        )

        cleaned_result, frontmatter_result = await asyncio.gather(
            clean_task, frontmatter_task, return_exceptions=True
        )

        cleaned: str = (
            markdown if isinstance(cleaned_result, BaseException) else cleaned_result
        )
        if isinstance(cleaned_result, BaseException):
            logger.warning(f"Markdown cleaning failed: {cleaned_result}")

        frontmatter: str = (
            f"title: {source}\nsource: {source}"
            if isinstance(frontmatter_result, BaseException)
            else frontmatter_result
        )
        if isinstance(frontmatter_result, BaseException):
            logger.warning(f"Frontmatter generation failed: {frontmatter_result}")

        return cleaned, frontmatter

    def _detect_language(self, content: str) -> str:
        """Detect the primary language of the content.

        Uses a simple heuristic: if more than 10% of characters are CJK,
        consider it Chinese.

        Args:
            content: Text content to analyze

        Returns:
            Language string: "English" or "Chinese"
        """
        if not content:
            return "English"

        cjk_count = 0
        total_count = 0

        for char in content:
            if char.isalpha():
                total_count += 1
                if "\u4e00" <= char <= "\u9fff":
                    cjk_count += 1

        if total_count == 0:
            return "English"

        if cjk_count / total_count > 0.1:
            return "Chinese"

        return "English"

    async def _process_document_combined(
        self,
        markdown: str,
        source: str,
    ) -> DocumentProcessResult:
        """
        Process document with combined cleaner + frontmatter using Instructor.

        Args:
            markdown: Raw markdown content
            source: Source file name

        Returns:
            DocumentProcessResult with cleaned markdown and frontmatter
        """
        # Detect document language
        language = self._detect_language(markdown)

        # Get combined prompt with language
        prompt = self._prompt_manager.get_prompt(
            "document_process",
            content=self._smart_truncate(markdown, 8000),
            source=source,
            language=language,
        )

        async with self.semaphore:
            # Create instructor client from router for load balancing
            client = instructor.from_litellm(
                self.router.acompletion, mode=instructor.Mode.JSON
            )

            # Use create_with_completion to get both the model and the raw response
            # Use logical model name for router load balancing
            (
                response,
                raw_response,
            ) = await client.chat.completions.create_with_completion(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                response_model=DocumentProcessResult,
                max_retries=0,
            )

            # Track usage from raw API response
            # Get actual model from response for accurate tracking
            actual_model = getattr(raw_response, "model", None) or "default"
            input_tokens = 0
            output_tokens = 0
            cost = 0.0
            if hasattr(raw_response, "usage") and raw_response.usage is not None:
                input_tokens = getattr(raw_response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(raw_response.usage, "completion_tokens", 0) or 0
                try:
                    cost = completion_cost(completion_response=raw_response)
                except Exception:
                    cost = 0.0
                self._track_usage(
                    actual_model, input_tokens, output_tokens, cost, source
                )

            return response

    def format_llm_output(
        self,
        markdown: str,
        frontmatter: str,
    ) -> str:
        """
        Format final output with frontmatter.

        Args:
            markdown: Cleaned markdown content
            frontmatter: YAML frontmatter (without --- markers)

        Returns:
            Complete markdown with frontmatter
        """
        frontmatter = self._clean_frontmatter(frontmatter)

        # Remove non-commented screenshot references that shouldn't be in content
        # These are page screenshots that should only appear as comments at the end
        # Pattern: ![Page N](screenshots/...) or ![Page N](path/screenshots/...)
        # But NOT: <!-- ![Page N](screenshots/...) --> (already commented)
        markdown = self._remove_uncommented_screenshots(markdown)

        markdown = self._normalize_whitespace(markdown)
        return f"---\n{frontmatter}\n---\n\n{markdown}"

    @staticmethod
    def _remove_uncommented_screenshots(content: str) -> str:
        """Remove non-commented page screenshot references from content.

        Page screenshots should only appear as HTML comments at the end of the document.
        If LLM accidentally outputs them as regular image references, remove them.

        Also ensures that any screenshot references in the "Page images for reference"
        section are properly commented.

        Args:
            content: Markdown content

        Returns:
            Content with uncommented screenshots removed/fixed
        """
        import re

        # Find the position of "<!-- Page images for reference -->" if it exists
        page_images_header = "<!-- Page images for reference -->"
        header_pos = content.find(page_images_header)

        if header_pos == -1:
            # No page images section, just remove any stray screenshot references
            # Patterns to remove:
            # 1. ![Page N](screenshots/...) - standard pattern
            # 2. ![...](screenshots/page_N...) - LLM-generated variants
            patterns = [
                r"^!\[Page\s+\d+\]\([^)]*screenshots/[^)]+\)\s*$",
                r"^!\[[^\]]*\]\(screenshots/page_\d+[^)]*\)\s*$",
            ]
            for pattern in patterns:
                content = re.sub(pattern, "", content, flags=re.MULTILINE)

            # Also remove any section headers that preceded these removed images
            # Pattern: ### Page N Image: followed by empty line
            content = re.sub(
                r"^###\s+Page\s+\d+\s+Image:\s*\n\s*\n",
                "",
                content,
                flags=re.MULTILINE,
            )

            # Clean up any resulting empty lines
            content = re.sub(r"\n{3,}", "\n\n", content)
        else:
            # Split at the page images section
            before = content[:header_pos]
            after = content[header_pos:]

            # Remove screenshot references from BEFORE the page images header
            patterns = [
                r"^!\[Page\s+\d+\]\([^)]*screenshots/[^)]+\)\s*$",
                r"^!\[[^\]]*\]\(screenshots/page_\d+[^)]*\)\s*$",
            ]
            for pattern in patterns:
                before = re.sub(pattern, "", before, flags=re.MULTILINE)

            # Also remove any section headers that preceded these removed images
            before = re.sub(
                r"^###\s+Page\s+\d+\s+Image:\s*\n\s*\n",
                "",
                before,
                flags=re.MULTILINE,
            )
            before = re.sub(r"\n{3,}", "\n\n", before)

            # Fix the AFTER section: convert any non-commented page images to comments
            # Match lines with page image references that are not already commented
            # This handles: ![Page N](screenshots/...)
            after_lines = after.split("\n")
            fixed_lines = []
            for line in after_lines:
                stripped = line.strip()
                # Check if it's an uncommented page image reference
                if (
                    stripped.startswith("![Page")
                    and "screenshots/" in stripped
                    and not stripped.startswith("<!--")
                ):
                    fixed_lines.append(f"<!-- {stripped} -->")
                else:
                    fixed_lines.append(line)
            after = "\n".join(fixed_lines)

            content = before + after

        # Clean up screenshot comments section: remove blank lines between comments
        # Pattern: <!-- Page images for reference --> followed by page image comments
        page_section_pattern = (
            r"(<!-- Page images for reference -->)"
            r"((?:\s*<!-- !\[Page \d+\]\([^)]+\) -->)+)"
        )

        def clean_page_section(match: re.Match) -> str:
            header = match.group(1)
            comments_section = match.group(2)
            # Extract individual comments and rejoin without blank lines
            comments = re.findall(r"<!-- !\[Page \d+\]\([^)]+\) -->", comments_section)
            return header + "\n" + "\n".join(comments)

        content = re.sub(page_section_pattern, clean_page_section, content)

        return content

    @staticmethod
    def _normalize_whitespace(content: str) -> str:
        """Normalize whitespace in markdown content.

        - Merge 3+ consecutive blank lines into 2 blank lines
        - Ensure consistent line endings
        - Strip trailing whitespace from lines

        Args:
            content: Markdown content to normalize

        Returns:
            Normalized content
        """
        import re

        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in content.split("\n")]
        content = "\n".join(lines)

        # Merge 3+ consecutive blank lines into 2 (keep one blank line between blocks)
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Ensure single newline at end
        return content.strip() + "\n"

    def _clean_frontmatter(self, frontmatter: str) -> str:
        """
        Clean frontmatter by removing code block markers and --- markers.

        Args:
            frontmatter: Raw frontmatter from LLM

        Returns:
            Clean YAML frontmatter
        """
        import re

        frontmatter = frontmatter.strip()

        # Remove code block markers (```yaml, ```yml, ```)
        # Pattern: ```yaml or ```yml at start, ``` at end
        code_block_pattern = r"^```(?:ya?ml)?\s*\n?(.*?)\n?```$"
        match = re.match(code_block_pattern, frontmatter, re.DOTALL | re.IGNORECASE)
        if match:
            frontmatter = match.group(1).strip()

        # Remove --- markers
        if frontmatter.startswith("---"):
            frontmatter = frontmatter[3:].strip()
        if frontmatter.endswith("---"):
            frontmatter = frontmatter[:-3].strip()

        return frontmatter
