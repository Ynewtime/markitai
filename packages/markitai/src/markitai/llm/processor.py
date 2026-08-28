"""LLM integration module using MarkitaiRouter (LiteLLM Router + local providers)."""

from __future__ import annotations

import asyncio
import base64
import copy
import threading
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Suppress LiteLLM's async logging warnings (coroutine never awaited)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'Logging.async_success_handler' was never awaited",
    category=RuntimeWarning,
)

import litellm

# Suppress LiteLLM's "Provider List" debug messages for custom providers
litellm.suppress_debug_info = True
from loguru import logger

if TYPE_CHECKING:
    from markitai.config import LLMConfig, ModelConfig, PromptsConfig

# Optional: cairosvg for SVG → PNG rasterization (LLM vision)
try:
    import cairosvg
except (ImportError, OSError):
    cairosvg = None  # type: ignore[assignment]

from markitai.constants import (
    DEFAULT_IO_CONCURRENCY,
    DEFAULT_MAX_IMAGES_PER_BATCH,
    DEFAULT_MAX_OUTPUT_TOKENS_HARD_CAP,
    DEFAULT_MAX_PAGES_PER_BATCH,
    DEFAULT_VISION_MAX_DIMENSION,
)
from markitai.llm import content
from markitai.llm.cache import ContentCache, PersistentCache
from markitai.llm.document import DocumentEnhancer

# Canonical definition lives in markitai.llm.engine (processor imports engine,
# not vice versa); re-exported here for backwards compatibility.
from markitai.llm.engine import (
    RETRYABLE_ERRORS as RETRYABLE_ERRORS,
)
from markitai.llm.engine import LLMEngine, RequestBudget
from markitai.llm.models import (
    MarkitaiLLMLogger,
    get_model_info_cached,
    model_list_fingerprint,
)
from markitai.llm.router import MarkitaiRouter
from markitai.llm.types import (
    ImageAnalysis,
    LLMResponse,
    LLMRuntime,
)
from markitai.llm.vision import VisionAnalyzer
from markitai.prompts import PromptManager
from markitai.providers.common import has_images
from markitai.utils.text import preview_items_for_log

# Enable automatic max_tokens adjustment to model limits
# When user-specified max_tokens exceeds model's max_output_tokens,
# LiteLLM will automatically cap it to the model's limit
litellm.modify_params = True


# Global callback instance (uses MarkitaiLLMLogger from models.py)
_markitai_llm_logger = MarkitaiLLMLogger()

# Pseudo-model key marking a tripped request budget in the per-context usage
# report. All-zero numbers: totals are unaffected, only the marker shows up
# under the document's llm_usage models.
REQUEST_BUDGET_EXCEEDED_MARKER = "markitai:request-budget-exceeded"


class LLMProcessor:
    """LLM processor using MarkitaiRouter for load balancing.

    Facade over two composed services (Phase 2.3, ex-mixins):

    - ``documents`` (:class:`DocumentEnhancer`): document cleaning,
      frontmatter generation, vision-enhanced document processing
    - ``vision`` (:class:`VisionAnalyzer`): image analysis and page
      content extraction

    All historical public methods remain available on the processor as
    thin delegates, so callers and tests keep working unchanged.
    """

    # Static method proxies to content module
    extract_protected_content = staticmethod(content.extract_protected_content)
    protect_content = staticmethod(content.protect_content)
    unprotect_content = staticmethod(content.unprotect_content)
    fix_malformed_image_refs = staticmethod(content.fix_malformed_image_refs)
    strip_prompt_echo = staticmethod(content.strip_prompt_echo)
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
        extra_cleaning_rules: str = "",
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
            extra_cleaning_rules: Extra rules appended to the text-cleaning
                              prompts (e.g. the rag profile's table-consistency
                              constraint). Empty keeps prompts and cache keys
                              byte-identical to the default.
        """
        self.config = config
        self._runtime = runtime
        self._extra_cleaning_rules = extra_cleaning_rules
        self._router: MarkitaiRouter | None = None
        self._vision_router: MarkitaiRouter | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._io_semaphore: asyncio.Semaphore | None = None
        self._engine: LLMEngine | None = None
        self._documents: DocumentEnhancer | None = None
        self._vision: VisionAnalyzer | None = None
        self._cache_model_scope: str | None = None
        self._vision_cache_model_scope: str | None = None
        self._prompt_manager = PromptManager(prompts_config)

        # Usage tracking (global across all contexts)
        # Use defaultdict to avoid check-then-create race conditions
        def _make_usage_dict() -> dict[str, Any]:
            return {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
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

        # Per-document circuit breaker: bounds the total LLM requests one
        # document context may issue (retry storms included). Shared with
        # the engine; cleared per context by clear_context_usage.
        self._request_budget = RequestBudget(
            limit=config.max_requests_per_document,
            cost_limit=config.max_cost_per_document_usd,
            on_exceeded=self._record_budget_exceeded,
        )

        # In-memory content cache for session-level deduplication (fast, no I/O)
        # (hit/miss counters live on the LLMEngine since Phase 2.3)
        self._cache = ContentCache()

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
        self._image_cache_lock = threading.Lock()
        self._image_cache_max_size = 200  # Max number of images to cache
        self._image_cache_max_bytes = 500 * 1024 * 1024  # 500MB max total cache size
        self._image_cache_bytes = 0  # Current total bytes in cache

        # Register LiteLLM callback for additional details
        self._setup_callbacks()

        # Eagerly initialize vision router to avoid first-call latency
        if config.model_list:
            _ = self.vision_router

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
    def router(self) -> MarkitaiRouter:
        """Get or create the router for the full configured model pool."""
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
        if self._io_semaphore is None:
            self._io_semaphore = asyncio.Semaphore(DEFAULT_IO_CONCURRENCY)
        return self._io_semaphore

    @property
    def engine(self) -> LLMEngine:
        """Get or create the unified structured LLM call engine (lazy).

        Lazy like ``router``/``semaphore``, and built with ``get_router``
        (a provider, not the router instance): router creation raises when
        no models are configured, and that error must surface at call time
        inside the callers' fallback handling — not when the engine or the
        ``documents``/``vision`` services are constructed. Late binding
        also keeps ``_router`` test doubles injectable after construction.
        """
        if self._engine is None:
            self._engine = LLMEngine(
                get_router=lambda: self.router,
                semaphore=self.semaphore,
                memory_cache=self._cache,
                persistent_cache=self._persistent_cache,
                track_usage=self._track_usage,
                calculate_max_tokens=self._calculate_dynamic_max_tokens,
                get_primary_model=self._get_router_primary_model,
                max_retries=self.config.router_settings.num_retries,
                request_budget=self._request_budget,
            )
        return self._engine

    @property
    def cache_model_scope(self) -> str:
        """Persistent-cache scope for calls routed through the main router.

        Fingerprint of the configured model pool (lazily computed, see
        ``model_list_fingerprint``): results cached under one model
        configuration are not served for a different one.
        """
        if self._cache_model_scope is None:
            self._cache_model_scope = model_list_fingerprint(self.config.model_list)
        return self._cache_model_scope

    @property
    def vision_cache_model_scope(self) -> str:
        """Persistent-cache scope for calls routed through the vision router.

        Mirrors the ``vision_router`` pool selection: fingerprint of the
        enabled vision-capable subset of ``model_list``, falling back to the
        main pool scope when the vision router falls back to the main router.
        """
        if self._vision_cache_model_scope is None:
            vision_models = [
                m for m in self.config.model_list if self._is_vision_model(m)
            ]
            enabled_vision = [m for m in vision_models if m.litellm_params.weight > 0]
            if enabled_vision:
                self._vision_cache_model_scope = model_list_fingerprint(enabled_vision)
            else:
                self._vision_cache_model_scope = self.cache_model_scope
        return self._vision_cache_model_scope

    @property
    def documents(self) -> DocumentEnhancer:
        """Document enhancement service (lazy, engine-backed).

        Lazy for the same reason as ``engine``: eager creation would force
        router creation before tests can inject router doubles. Callbacks
        are late-bound lambdas so instance-level patches on the processor
        (``_get_cached_image``, ``_get_next_call_index``, ``_vision_router``)
        keep working after the service exists.
        """
        if self._documents is None:
            self._documents = DocumentEnhancer(
                engine=self.engine,
                prompt_manager=self._prompt_manager,
                config=self.config,
                cache_model_scope=self.cache_model_scope,
                vision_cache_model_scope=self.vision_cache_model_scope,
                get_vision_router=lambda: self.vision_router,
                get_cached_image=lambda image_path: self._get_cached_image(image_path),
                get_next_call_index=lambda context: self._get_next_call_index(context),
                extra_cleaning_rules=self._extra_cleaning_rules,
            )
        return self._documents

    @property
    def vision(self) -> VisionAnalyzer:
        """Vision analysis service (lazy, engine-backed).

        See ``documents`` for the laziness and late-binding rationale.
        """
        if self._vision is None:
            self._vision = VisionAnalyzer(
                engine=self.engine,
                prompt_manager=self._prompt_manager,
                config=self.config,
                vision_cache_model_scope=self.vision_cache_model_scope,
                get_vision_router=lambda: self.vision_router,
                get_cached_image=lambda image_path: self._get_cached_image(image_path),
                get_next_call_index=lambda context: self._get_next_call_index(context),
            )
        return self._vision

    # =========================================================================
    # Document facade — thin delegates to DocumentEnhancer
    # (signatures preserved verbatim for callers and tests)
    # =========================================================================

    async def clean_markdown(self, content: str, context: str = "") -> str:
        """Clean and optimize markdown content (see DocumentEnhancer)."""
        return await self.documents.clean_markdown(content, context)

    async def clean_document_pure(self, markdown: str, source: str) -> str:
        """Pure cleaning: raw markdown in, LLM response out (see DocumentEnhancer)."""
        return await self.documents.clean_document_pure(markdown, source)

    async def process_document(
        self,
        markdown: str,
        source: str,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> tuple[str, str]:
        """Process a document: clean + frontmatter (see DocumentEnhancer)."""
        return await self.documents.process_document(
            markdown,
            source,
            fetch_strategy=fetch_strategy,
            extra_meta=extra_meta,
            title=title,
        )

    async def enhance_document_complete(
        self,
        extracted_text: str,
        page_images: list[Path],
        source: str = "",
        max_pages_per_batch: int = DEFAULT_MAX_PAGES_PER_BATCH,
        original_title: str | None = None,
    ) -> tuple[str, str]:
        """Vision enhancement + frontmatter (see DocumentEnhancer)."""
        return await self.documents.enhance_document_complete(
            extracted_text,
            page_images,
            source=source,
            max_pages_per_batch=max_pages_per_batch,
            original_title=original_title,
        )

    async def enhance_document_with_vision(
        self,
        extracted_text: str,
        page_images: list[Path],
        context: str = "",
    ) -> str:
        """Vision-referenced format cleaning (see DocumentEnhancer)."""
        return await self.documents.enhance_document_with_vision(
            extracted_text, page_images, context=context
        )

    async def enhance_url_with_vision(
        self,
        content: str,
        screenshot_path: Path,
        context: str = "",
        original_title: str | None = None,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Enhance URL content with screenshot reference (see DocumentEnhancer)."""
        return await self.documents.enhance_url_with_vision(
            content,
            screenshot_path,
            context=context,
            original_title=original_title,
            fetch_strategy=fetch_strategy,
            extra_meta=extra_meta,
        )

    async def extract_from_screenshot(
        self,
        screenshot_path: Path,
        context: str = "",
        original_title: str | None = None,
    ) -> tuple[str, str]:
        """Screenshot-only content extraction (see DocumentEnhancer)."""
        return await self.documents.extract_from_screenshot(
            screenshot_path, context=context, original_title=original_title
        )

    def format_llm_output(self, markdown: str, frontmatter: str) -> str:
        """Format final output with frontmatter (see DocumentEnhancer)."""
        return self.documents.format_llm_output(markdown, frontmatter)

    def _stabilize_paged_markdown(
        self,
        original_markdown: str,
        cleaned_markdown: str,
        source: str,
    ) -> str:
        """Protect page-marked documents from LLM structural drift.

        Kept on the facade (not only on the service) because
        ``workflow.helpers.maybe_stabilize_markdown`` duck-types it on the
        processor.
        """
        return self.documents._stabilize_paged_markdown(
            original_markdown, cleaned_markdown, source
        )

    # =========================================================================
    # Vision facade — thin delegates to VisionAnalyzer
    # (signatures preserved verbatim for callers and tests)
    # =========================================================================

    async def analyze_image(
        self,
        image_path: Path,
        context: str = "",
        document_context: str = "",
    ) -> ImageAnalysis:
        """Analyze an image using a vision model (see VisionAnalyzer)."""
        return await self.vision.analyze_image(
            image_path, context=context, document_context=document_context
        )

    async def analyze_images_batch(
        self,
        image_paths: list[Path],
        max_images_per_batch: int = DEFAULT_MAX_IMAGES_PER_BATCH,
        context: str = "",
        document_context: str = "",
    ) -> list[ImageAnalysis]:
        """Analyze multiple images in parallel batches (see VisionAnalyzer)."""
        return await self.vision.analyze_images_batch(
            image_paths,
            max_images_per_batch=max_images_per_batch,
            context=context,
            document_context=document_context,
        )

    async def analyze_batch(
        self,
        image_paths: list[Path],
        context: str = "",
        document_context: str = "",
    ) -> list[ImageAnalysis]:
        """Batch image analysis via Instructor (see VisionAnalyzer)."""
        return await self.vision.analyze_batch(
            image_paths, context=context, document_context=document_context
        )

    def _create_router(self, models: list[ModelConfig] | None = None) -> MarkitaiRouter:
        """Create a MarkitaiRouter from model configurations.

        The main router (``models is None``) honors configured
        ``router_settings.fallbacks`` by preserving model groups. Subset
        routers (the vision pool) always balance over all their models:
        fallbacks are stripped and groups normalized, so a vision pool
        missing the "default" entry group keeps working.

        Args:
            models: Optional subset of ``self.config.model_list`` (used by
                the vision router). Defaults to the full configured list.

        Returns:
            MarkitaiRouter over the usable models (local providers
            dispatched directly, standard models delegated to an inner
            LiteLLM Router).

        Raises:
            ValueError: If no usable models remain after filtering.
        """
        source_models = self.config.model_list if models is None else models
        if not source_models:
            raise ValueError("No models configured in llm.model_list")

        settings = self.config.router_settings.model_dump()
        honor_fallbacks = models is None and bool(settings.get("fallbacks"))
        if not honor_fallbacks:
            settings["fallbacks"] = []

        model_list = self._build_router_entries(
            source_models, preserve_groups=honor_fallbacks
        )

        model_names = [e["litellm_params"]["model"].split("/")[-1] for e in model_list]
        logger.info(
            f"[Router] Model pool ({len(model_list)}): "
            f"{preview_items_for_log(model_names)}"
        )

        return MarkitaiRouter(model_list=model_list, router_settings=settings)

    def _build_router_entries(
        self, models: list[ModelConfig], *, preserve_groups: bool = False
    ) -> list[dict[str, Any]]:
        """Build LiteLLM-Router-format entries from model configurations.

        Resolves ``env:`` API keys/bases and filters out unusable models
        (SDK unavailable, ``weight <= 0``, missing environment variables).
        By default all entries are normalized into the single ``"default"``
        balancing pool; with ``preserve_groups`` (fallbacks configured),
        standard models keep their configured ``model_name`` so LiteLLM
        group fallbacks can route between them. Local provider models are
        always pooled into "default" (they cannot be LiteLLM fallback
        targets).

        Args:
            models: ModelConfig objects (full list or a subset).
            preserve_groups: Keep standard models' configured group names.

        Returns:
            Non-empty list of model entry dicts.

        Raises:
            ValueError: If no usable models remain, with a message naming
                the dominant cause (all disabled / missing env vars / no
                SDKs).
        """
        from markitai.config import EnvVarNotFoundError
        from markitai.providers import is_local_provider_available

        model_list: list[dict[str, Any]] = []
        skipped_models: list[str] = []
        disabled_models: list[str] = []
        skipped_env_vars: list[str] = []
        for model_config in models:
            model_id = model_config.litellm_params.model

            # Skip local provider models if their SDK is not available
            if not is_local_provider_available(model_id):
                skipped_models.append(model_id)
                continue

            # weight <= 0 means model is disabled (e.g. no API quota)
            if model_config.litellm_params.weight <= 0:
                disabled_models.append(model_id)
                continue

            model_entry: dict[str, Any] = {
                "model_name": model_config.model_name,
                "litellm_params": {
                    "model": model_id,
                },
            }

            # Add optional params — skip model if env var is missing
            try:
                api_key = model_config.litellm_params.get_resolved_api_key()
            except EnvVarNotFoundError as e:
                skipped_env_vars.append(e.var_name)
                logger.warning(
                    f"[Router] Skipping model {model_id}: "
                    f"environment variable {e.var_name} not set"
                )
                continue

            if api_key:
                model_entry["litellm_params"]["api_key"] = api_key

            try:
                api_base = model_config.litellm_params.get_resolved_api_base()
            except EnvVarNotFoundError as e:
                skipped_env_vars.append(e.var_name)
                logger.warning(
                    f"[Router] Skipping model {model_id}: "
                    f"environment variable {e.var_name} not set"
                )
                continue

            if api_base:
                model_entry["litellm_params"]["api_base"] = api_base

            # Azure deployments are addressed by version; without it the
            # request goes out against whatever default litellm infers.
            if model_config.litellm_params.api_version:
                model_entry["litellm_params"]["api_version"] = (
                    model_config.litellm_params.api_version
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
                f"{preview_items_for_log(skipped_models)}"
            )
        if disabled_models:
            logger.debug(
                f"[Router] Skipped {len(disabled_models)} disabled models (weight=0): "
                f"{preview_items_for_log(disabled_models)}"
            )

        if not model_list:
            disabled_count = sum(1 for m in models if m.litellm_params.weight <= 0)
            if disabled_count == len(models):
                raise ValueError(
                    f"All {disabled_count} configured models have weight=0 (disabled). "
                    "Set weight > 0 on at least one model to enable it."
                )
            if skipped_env_vars:
                vars_str = ", ".join(skipped_env_vars)
                raise ValueError(
                    f"No available models: missing environment variable(s): {vars_str}. "
                    "Set them via 'export VAR=value' or add to .env file."
                )
            raise ValueError(
                "No available models after filtering. "
                "Check that required SDKs are installed for configured models."
            )

        if preserve_groups:
            self._normalize_fallback_groups(model_list)
        else:
            # Normalize all model_name to "default" for unified load balancing pool
            for entry in model_list:
                entry["model_name"] = "default"

        return model_list

    @staticmethod
    def _normalize_fallback_groups(model_list: list[dict[str, Any]]) -> None:
        """Adjust entry groups for fallback routing (in place).

        Standard models keep their configured groups; local provider models
        are pooled into "default" (LiteLLM fallbacks cannot target them).
        Requests always enter at group "default", so that group must exist.

        Raises:
            ValueError: If no entry belongs to the "default" group.
        """
        from markitai.providers import is_local_provider_model

        for entry in model_list:
            model_id = entry["litellm_params"]["model"]
            if is_local_provider_model(model_id) and entry["model_name"] != "default":
                logger.warning(
                    f"[Router] Local provider model {model_id} cannot be a "
                    f"fallback target; pooling it into the 'default' group"
                )
                entry["model_name"] = "default"

        if not any(e["model_name"] == "default" for e in model_list):
            raise ValueError(
                "router_settings.fallbacks requires a 'default' model group as "
                "the entry point. Name at least one model_list entry 'default'."
            )

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
    def vision_router(self) -> MarkitaiRouter:
        """Get or create the router restricted to vision-capable models (lazy).

        Filters models using auto-detection from litellm or config override.
        Falls back to main router if no vision models found.

        Returns:
            MarkitaiRouter with vision-capable models only
        """
        if self._vision_router is None:
            vision_models = [
                m for m in self.config.model_list if self._is_vision_model(m)
            ]

            # Also check that at least one vision model is enabled (weight > 0)
            enabled_vision = [m for m in vision_models if m.litellm_params.weight > 0]

            if not enabled_vision:
                # No enabled vision models - fall back to main router
                reason = (
                    "no vision-capable models configured"
                    if not vision_models
                    else "all vision models disabled (weight=0)"
                )
                logger.warning(
                    f"[Router] {reason}, falling back to main router for vision"
                )
                self._vision_router = self.router
            else:
                model_names = [
                    m.litellm_params.model.split("/")[-1] for m in enabled_vision
                ]
                logger.info(
                    f"[Router] Vision router ({len(enabled_vision)}): "
                    f"{preview_items_for_log(model_names)}"
                )
                self._vision_router = self._create_router(enabled_vision)

        return self._vision_router

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
        requires_vision = has_images(messages)
        router = self.vision_router if requires_vision else self.router

        return await self._call_llm_with_retry(
            model=model,
            messages=messages,
            call_id=call_id,
            context=context,
            router=router,
        )

    def _calculate_dynamic_max_tokens(
        self,
        messages: list[Any],
        target_model_id: str | None = None,
        router: MarkitaiRouter | None = None,
    ) -> int | None:
        """Calculate dynamic max_tokens based on input size and target model.

        Uses the target model's limits when available, otherwise returns None
        to let LiteLLM use model defaults.

        When router is provided, uses the minimum max_output_tokens across all
        models in the router to ensure compatibility with any model the router
        might select.

        Also detects table-heavy content (>20 rows) and applies a higher token floor.

        Args:
            messages: Chat messages to estimate input tokens
            target_model_id: Specific model ID (from router pre-selection)
            router: Optional MarkitaiRouter for model limit lookup

        Returns:
            Safe max_tokens value, or None to let LiteLLM use model defaults
        """
        import re

        # Estimate input tokens (use gpt-4 tokenizer as reasonable approximation)
        try:
            # Note: gpt-4 tokenizer does not account for images
            input_tokens = litellm.token_counter(model="gpt-4", messages=messages)

            # Manual correction for images (token_counter often ignores them)
            # Standard high-res image is ~1105-1445 tokens for most models
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and (
                            item.get("type") == "image_url" or "image" in item
                        ):
                            # Add 1600 tokens per image as a safe buffer
                            input_tokens += 1600
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
            if all_max_contexts:
                max_context = min(all_max_contexts)
        elif target_model_id:
            info = get_model_info_cached(target_model_id)
            max_context = info.get("max_input_tokens")
            max_output = info.get("max_output_tokens")

        # If target model info unavailable, return None to let LiteLLM handle it
        if not max_context or not max_output:
            logger.trace(
                f"[DynamicTokens] Could not get limits for model={target_model_id}, "
                "returning None to use LiteLLM defaults"
            )
            return None

        # Calculate available output space
        # Reserve buffer for safety (tokenizer differences, system overhead)
        # Use a larger buffer (2000) to avoid edge cases with large context models
        buffer = max(2000, int(input_tokens * 0.2))
        available_context = max_context - input_tokens - buffer

        # max_tokens = min(model's max_output, available context space)
        max_tokens = min(max_output, available_context)

        # Apply a reasonable cap for standard document conversion
        # 128k output is plenty for even the largest single page/chunk
        # This prevents requesting "too many" tokens which some providers dislike
        max_tokens = min(max_tokens, DEFAULT_MAX_OUTPUT_TOKENS_HARD_CAP)

        # Ensure reasonable minimum (higher for table-heavy content)
        min_floor = 4000 if is_table_heavy else 1000
        max_tokens = max(max_tokens, min_floor)

        logger.trace(
            f"[DynamicTokens] input_est={input_tokens}, model={target_model_id}, "
            f"max_output_lim={max_output}, calculated={max_tokens}"
        )

        return max_tokens

    def _get_router_primary_model(self, router: MarkitaiRouter) -> str | None:
        """Get the highest-weight model ID from a Router's model_list.

        Returns the model with the greatest weight, which is most likely
        to be selected during routing. Used for dynamic max_tokens calculation
        and error reporting.

        Args:
            router: MarkitaiRouter instance

        Returns:
            Model ID string (e.g., "chatgpt/gpt-5.3"), or None if unavailable
        """
        try:
            model_list = router.model_list
            if not model_list:
                return None

            best_model = None
            best_weight = -1.0
            for config in model_list:
                params = config.get("litellm_params", {})
                model_id = params.get("model", "")
                weight = params.get("weight", 1.0)
                if weight > best_weight:
                    best_weight = weight
                    best_model = model_id

            return best_model
        except Exception as e:
            logger.debug("[LLM] Failed to select highest-weight model: {}", e)
        return None

    async def _call_llm_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        call_id: str,
        context: str = "",
        max_retries: int | None = None,
        router: MarkitaiRouter | None = None,
    ) -> LLMResponse:
        """
        Make an LLM call with custom retry logic and detailed logging.

        Thin delegate: the transport retry loop lives in
        ``LLMEngine.complete_text`` (Phase 2.3). This wrapper keeps the
        historical signature for existing callers and tests.

        Args:
            model: Logical model name (e.g., "default")
            messages: Chat messages
            call_id: Unique identifier for this call (for logging)
            context: Context identifier for usage tracking (e.g., filename)
            max_retries: Retry-attempt override (None -> the engine-wide
                default from ``router_settings.num_retries``)
            router: Router to use (defaults to self.router)

        Returns:
            LLMResponse with content and usage info
        """
        return await self.engine.complete_text(
            model=model,
            messages=messages,
            call_id=call_id,
            context=context,
            max_retries=max_retries,
            router=router,
        )

    def _track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        context: str = "",
        cached_tokens: int = 0,
    ) -> None:
        """Track usage statistics per model (and optionally per context).

        Thread-safe: uses lock to protect concurrent access to usage dicts.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            context: Optional context identifier (e.g., filename)
            cached_tokens: Cache-read input tokens (prompt caching hits);
                a subset of input_tokens billed at the provider's cache rate
        """
        with self._usage_lock:
            # Track global usage (defaultdict auto-creates entries; the
            # cached key uses .get so hand-built dicts without it survive)
            self._usage[model]["requests"] += 1
            self._usage[model]["input_tokens"] += input_tokens
            self._usage[model]["output_tokens"] += output_tokens
            self._usage[model]["cached_input_tokens"] = (
                self._usage[model].get("cached_input_tokens", 0) + cached_tokens
            )
            self._usage[model]["cost_usd"] += cost

            # Track per-context usage if context provided
            if context:
                ctx = self._context_usage[context][model]
                ctx["requests"] += 1
                ctx["input_tokens"] += input_tokens
                ctx["output_tokens"] += output_tokens
                ctx["cached_input_tokens"] = (
                    ctx.get("cached_input_tokens", 0) + cached_tokens
                )
                ctx["cost_usd"] += cost

        # Outside the usage lock: the breaker keeps its own, and this is the
        # one place every answer's price passes through.
        if context:
            self._request_budget.charge(context, cost)

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
        """Clear usage tracking (and request budget) for a specific context.

        Thread-safe: uses lock for safe modification.

        Args:
            context: Context identifier to clear
        """
        with self._usage_lock:
            self._context_usage.pop(context, None)
            self._call_counter.pop(context, None)
        self._request_budget.clear(context)

    def _record_budget_exceeded(self, context: str) -> None:
        """Mark a tripped request budget in the context's usage report.

        Creates an all-zero pseudo-model entry so the per-file
        ``llm_usage.models`` shows the trip without touching any totals.
        """
        with self._usage_lock:
            # Touching the defaultdict creates the all-zero entry
            _ = self._context_usage[context][REQUEST_BUDGET_EXCEEDED_MARKER]

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics (delegates to the engine's counters).

        Returns:
            Dict with memory cache stats, persistent cache stats, and combined hit rate
        """
        if self._engine is not None:
            return self._engine.get_cache_stats()
        # Engine not created yet: no LLM calls have happened, counters are 0.
        # (Avoids forcing router creation just to read stats.)
        return {
            "memory": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
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
            if self._engine is not None:
                self._engine.reset_cache_counters()

        if scope in ("global", "all"):
            result["global"] = self._persistent_cache.clear()

        return result

    def clear_image_cache(self) -> None:
        """Clear the image cache to free memory after document processing."""
        with self._image_cache_lock:
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

        with self._image_cache_lock:
            if path_key in self._image_cache:
                # Move to end for LRU (most recently used)
                self._image_cache.move_to_end(path_key)
                return self._image_cache[path_key]

        # Read and encode image
        image_data = image_path.read_bytes()

        # Convert LLM-unsupported formats (BMP, TIFF) to PNG
        from markitai.utils.mime import LLM_CONVERTIBLE_EXTENSIONS

        if image_path.suffix.lower() in LLM_CONVERTIBLE_EXTENSIONS:
            try:
                import io

                from PIL import Image

                with io.BytesIO(image_data) as buffer:
                    img = Image.open(buffer)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    out = io.BytesIO()
                    img.save(out, format="PNG")
                    image_data = out.getvalue()
                    out.close()
                    logger.debug(
                        "[LLM] Converted {} to PNG for LLM compatibility",
                        image_path.name,
                    )
            except Exception as e:
                logger.warning(
                    "[LLM] Failed to convert {} to PNG: {}", image_path.name, e
                )

        # Rasterize SVG to PNG via cairosvg (optional dependency)
        if image_path.suffix.lower() == ".svg" and cairosvg is not None:
            try:
                svg_result = cairosvg.svg2png(
                    bytestring=image_data,
                    output_width=DEFAULT_VISION_MAX_DIMENSION,
                )
                assert isinstance(svg_result, bytes)
                image_data = svg_result
                logger.debug(
                    "[LLM] Rasterized {} to PNG for LLM compatibility",
                    image_path.name,
                )
            except Exception as e:
                logger.warning(
                    "[LLM] Failed to rasterize SVG {}: {}", image_path.name, e
                )

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
                    max_dim = DEFAULT_VISION_MAX_DIMENSION

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
        with self._image_cache_lock:
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
