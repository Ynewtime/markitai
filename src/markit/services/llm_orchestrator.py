"""LLM orchestration service for managing providers and creating tasks.

This service centralizes all LLM-related functionality extracted from
ConversionPipeline, providing a clean interface for:
- Provider management and initialization
- Markdown enhancement
- Image analysis
- Task creation for batch processing
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markit.config.constants import DEFAULT_LLM_MODELS
from markit.utils.logging import get_logger

if TYPE_CHECKING:
    from markit.config.settings import LLMConfig, PromptConfig
    from markit.image.analyzer import ImageAnalysis, ImageAnalyzer
    from markit.image.compressor import CompressedImage
    from markit.llm.base import LLMTaskResultWithStats
    from markit.llm.enhancer import MarkdownEnhancer
    from markit.llm.manager import ProviderManager

log = get_logger(__name__)


class LLMOrchestrator:
    """Orchestrates LLM operations: provider management, enhancer, analyzer.

    This service centralizes all LLM-related functionality extracted from
    ConversionPipeline, providing a clean interface for:
    - Provider management and initialization
    - Markdown enhancement
    - Image analysis
    - Task creation for batch processing
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        enhancement_chunk_size: int = 32000,
        use_concurrent_fallback: bool = False,
        prompt_config: PromptConfig | None = None,
        output_dir: Path | None = None,
        chunk_concurrency: int = 4,
    ) -> None:
        """Initialize the LLM orchestrator.

        Args:
            llm_config: LLM configuration from settings (can be pre-resolved
                        using LLMConfigResolver, or raw config with CLI overrides)
            llm_provider: Override LLM provider (from CLI) - deprecated, prefer
                          using LLMConfigResolver before instantiation
            llm_model: Override LLM model (from CLI) - deprecated, prefer
                       using LLMConfigResolver before instantiation
            enhancement_chunk_size: Chunk size for markdown enhancement
            use_concurrent_fallback: Enable concurrent fallback for LLM calls
            prompt_config: Optional prompt configuration for customizing prompts.
                          If not provided, uses builtin prompts.
            output_dir: If provided, image analysis results will be immediately
                       written to output_dir/assets/<filename>.md as they complete.
            chunk_concurrency: Max concurrent LLM calls within a single document's
                              chunks. Prevents API overload from large documents.
        """
        # Resolve CLI overrides using centralized resolver
        from markit.config.settings import LLMConfigResolver

        self.llm_config = LLMConfigResolver.resolve(
            base_config=llm_config,
            cli_provider=llm_provider,
            cli_model=llm_model,
        )
        self.llm_provider = llm_provider  # Keep for backward compatibility
        self.llm_model = llm_model
        self.enhancement_chunk_size = enhancement_chunk_size
        self.use_concurrent_fallback = use_concurrent_fallback
        self.prompt_config = prompt_config
        self.output_dir = output_dir
        self.chunk_concurrency = chunk_concurrency

        # Lazy-loaded components
        self._provider_manager: ProviderManager | None = None
        self._provider_manager_initialized = False
        self._enhancer: MarkdownEnhancer | None = None
        self._image_analyzer: ImageAnalyzer | None = None

        # Note: We no longer use a global _chunk_semaphore here.
        # Each enhancement task creates its own per-file semaphore to ensure
        # fair chunk concurrency across multiple documents being processed in parallel.

        # Locks for thread-safe lazy initialization
        self._provider_lock = asyncio.Lock()
        self._enhancer_lock = asyncio.Lock()
        self._analyzer_lock = asyncio.Lock()

    def _create_provider_manager(self) -> ProviderManager:
        """Create a new ProviderManager instance (not initialized)."""
        from markit.llm.manager import ProviderManager

        # Use pre-resolved config (CLI overrides already applied in __init__)
        return ProviderManager(llm_config=self.llm_config)

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider.

        Args:
            provider: LLM provider name

        Returns:
            Default model name for the provider

        Raises:
            ValueError: If provider is not supported
        """
        if provider not in DEFAULT_LLM_MODELS:
            raise ValueError(
                f"No default model configured for provider '{provider}'. "
                f"Please specify a model explicitly using --llm-model. "
                f"Supported providers: {', '.join(DEFAULT_LLM_MODELS.keys())}"
            )
        return DEFAULT_LLM_MODELS[provider]

    def _get_required_capabilities(
        self,
        llm_enabled: bool = False,
        analyze_image: bool = False,
    ) -> list[str]:
        """Determine required LLM capabilities based on configuration.

        Args:
            llm_enabled: Whether text enhancement is enabled
            analyze_image: Whether image analysis is enabled

        Returns:
            List of required capabilities (e.g., ["text"], ["text", "vision"])
        """
        capabilities = []
        if llm_enabled:
            capabilities.append("text")
        if analyze_image:
            if "text" not in capabilities:
                capabilities.append("text")
            capabilities.append("vision")
        return capabilities if capabilities else ["text"]

    async def get_provider_manager(
        self,
        required_capabilities: list[str] | None = None,
        lazy: bool = True,
    ) -> ProviderManager:
        """Get or create the LLM provider manager (async).

        Creates the manager on first call and initializes it with required capabilities.
        Subsequent calls return the cached instance.

        Args:
            required_capabilities: Override auto-detected capabilities. If None,
                                   capabilities are inferred from configuration.
            lazy: If True (default), only load configs without network validation.
                  Providers will be validated on-demand when first used.
                  Set to False in batch mode to validate upfront (fail fast).

        Returns:
            Initialized ProviderManager instance
        """
        # Double-checked locking pattern
        if self._provider_manager is not None and self._provider_manager_initialized:
            return self._provider_manager

        async with self._provider_lock:
            if self._provider_manager is None:
                self._provider_manager = self._create_provider_manager()

            if not self._provider_manager_initialized:
                # Use provided capabilities or default to text
                caps = required_capabilities or ["text"]
                await self._provider_manager.initialize(required_capabilities=caps, lazy=lazy)
                self._provider_manager_initialized = True

        return self._provider_manager

    def _get_provider_manager_sync(self) -> ProviderManager:
        """Get or create the LLM provider manager (sync).

        Note: This does NOT initialize the provider. Use get_provider_manager()
        in async contexts to ensure proper initialization.

        Returns:
            ProviderManager instance (may not be initialized)
        """
        if self._provider_manager is None:
            self._provider_manager = self._create_provider_manager()
        return self._provider_manager

    async def warmup(
        self,
        llm_enabled: bool = False,
        analyze_image: bool = False,
    ) -> None:
        """Warmup LLM providers by forcing network validation.

        This method should be called before batch processing to:
        1. Fail fast if providers are misconfigured
        2. Avoid concurrent validation races during parallel processing

        Only performs warmup if LLM features are enabled.

        Args:
            llm_enabled: Whether text enhancement is enabled
            analyze_image: Whether image analysis is enabled
        """
        if not (llm_enabled or analyze_image):
            log.debug("LLM features not enabled, skipping warmup")
            return

        log.debug("Warming up LLM providers...")

        # Force non-lazy initialization to validate providers upfront
        manager = self._get_provider_manager_sync()
        caps = self._get_required_capabilities(llm_enabled, analyze_image)
        await manager.initialize(required_capabilities=caps, lazy=False)
        self._provider_manager_initialized = True

        log.debug(
            "LLM providers warmed up",
            valid_providers=manager.available_providers,
            capabilities=caps,
        )

    async def get_enhancer(self) -> MarkdownEnhancer:
        """Get or create the Markdown enhancer (async, thread-safe).

        Returns:
            MarkdownEnhancer instance
        """
        if self._enhancer is not None:
            return self._enhancer

        async with self._enhancer_lock:
            if self._enhancer is None:
                from markit.llm.enhancer import EnhancementConfig, MarkdownEnhancer

                provider_manager = await self.get_provider_manager()
                self._enhancer = MarkdownEnhancer(
                    provider_manager=provider_manager,
                    config=EnhancementConfig(
                        chunk_size=self.enhancement_chunk_size,
                    ),
                    use_concurrent_fallback=self.use_concurrent_fallback,
                    prompt_config=self.prompt_config,
                )
        return self._enhancer

    async def get_image_analyzer(self) -> ImageAnalyzer:
        """Get or create the image analyzer (async, thread-safe).

        Returns:
            ImageAnalyzer instance
        """
        if self._image_analyzer is not None:
            return self._image_analyzer

        async with self._analyzer_lock:
            if self._image_analyzer is None:
                from markit.image.analyzer import ImageAnalyzer

                provider_manager = await self.get_provider_manager()
                self._image_analyzer = ImageAnalyzer(
                    provider_manager=provider_manager,
                )
        return self._image_analyzer

    def has_capability(self, capability: str) -> bool:
        """Check if any configured provider has the capability.

        Args:
            capability: Capability to check (e.g., "text", "vision")

        Returns:
            True if capability is available
        """
        if self._provider_manager:
            return self._provider_manager.has_capability(capability)
        return False

    async def create_image_analysis_task(
        self,
        image: CompressedImage,
        return_stats: bool = True,
        output_dir: Path | None = None,
    ) -> ImageAnalysis | LLMTaskResultWithStats:
        """Create and execute an image analysis task.

        Args:
            image: Compressed image to analyze
            return_stats: If True, return LLMTaskResultWithStats with statistics
            output_dir: If provided (or self.output_dir is set), immediately write
                       analysis result to output_dir/assets/<filename>.md

        Returns:
            ImageAnalysis or LLMTaskResultWithStats containing the analysis
        """
        from markit.image.analyzer import ImageAnalysis
        from markit.llm.base import LLMTaskResultWithStats

        # Use provided output_dir or fall back to instance's output_dir
        effective_output_dir = output_dir or self.output_dir

        analyzer = await self.get_image_analyzer()
        try:
            return await analyzer.analyze(
                image, return_stats=return_stats, output_dir=effective_output_dir
            )
        except Exception as e:
            log.warning("Image analysis failed", filename=image.filename, error=str(e))
            fallback = ImageAnalysis(
                alt_text=f"Image: {image.filename}",
                detailed_description="Image analysis failed.",
                detected_text=None,
                image_type="other",
            )
            if return_stats:
                return LLMTaskResultWithStats(result=fallback)
            return fallback

    async def create_enhancement_task(
        self,
        markdown: str,
        source_file: Path,
        return_stats: bool = True,
    ) -> str | LLMTaskResultWithStats:
        """Create and execute a markdown enhancement task.

        Each task creates its own per-file chunk semaphore to ensure fair
        concurrency across multiple documents being processed in parallel.

        Args:
            markdown: Markdown content to enhance
            source_file: Source file path for context
            return_stats: If True, return LLMTaskResultWithStats with statistics

        Returns:
            Enhanced markdown string or LLMTaskResultWithStats containing the content
        """
        from markit.llm.base import LLMTaskResultWithStats
        from markit.llm.enhancer import EnhancedMarkdown
        from markit.markdown.formatter import FormatterConfig, format_markdown

        enhancer = await self.get_enhancer()
        try:
            # Create per-file chunk semaphore for fair concurrency distribution
            # This ensures each file gets its own chunk_concurrency slots,
            # preventing a single large document from starving others
            per_file_semaphore = asyncio.Semaphore(self.chunk_concurrency)

            result = await enhancer.enhance(
                markdown, source_file, semaphore=per_file_semaphore, return_stats=return_stats
            )

            # Post-processing: format markdown for consistent output
            # Note: Only format, don't clean (cleaning is LLM's responsibility)
            def _post_process(content: str) -> str:
                # Normalize CRLF to LF first
                content = content.replace("\r\n", "\n").replace("\r", "\n")
                # Format with ensure_h2_start=False to preserve LLM's heading structure
                return format_markdown(content, FormatterConfig(ensure_h2_start=False))

            if return_stats:
                # enhancer returns LLMTaskResultWithStats when return_stats=True
                # but we need to extract content for the final result
                if isinstance(result, LLMTaskResultWithStats):
                    # Replace the EnhancedMarkdown result with just the content string
                    return LLMTaskResultWithStats(
                        result=_post_process(result.result.content),
                        model=result.model,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        estimated_cost=result.estimated_cost,
                    )
                # If not LLMTaskResultWithStats, wrap the result
                if isinstance(result, EnhancedMarkdown):
                    return LLMTaskResultWithStats(result=_post_process(result.content))
                return LLMTaskResultWithStats(result=_post_process(str(result)))
            # When not returning stats, extract content from EnhancedMarkdown
            if isinstance(result, EnhancedMarkdown):
                return _post_process(result.content)
            return _post_process(str(result))
        except Exception as e:
            log.warning("LLM enhancement failed", error=str(e))
            from markit.llm.enhancer import SimpleMarkdownCleaner

            cleaner = SimpleMarkdownCleaner()
            cleaned = cleaner.clean(markdown)
            if return_stats:
                return LLMTaskResultWithStats(result=cleaned)
            return cleaned

    async def create_llm_tasks(
        self,
        images_for_analysis: list[CompressedImage],
        markdown_content: str,
        input_file: Path,
        llm_enabled: bool,
        analyze_image: bool,
    ) -> list[Coroutine[Any, Any, Any]]:
        """Create LLM task coroutines without executing them.

        This creates coroutines for image analysis and markdown enhancement
        that can be submitted to a global LLM queue for rate-limited execution.

        Args:
            images_for_analysis: Images prepared for analysis
            markdown_content: Markdown content for enhancement
            input_file: Original input file
            llm_enabled: Whether to create enhancement task
            analyze_image: Whether to create analysis tasks

        Returns:
            List of coroutines (not awaited) for LLM tasks
        """
        tasks: list[Coroutine[Any, Any, Any]] = []

        # Image analysis tasks
        if analyze_image and images_for_analysis:
            # Check vision capability
            manager = await self.get_provider_manager()
            if not manager.has_capability("vision"):
                log.warning(
                    "Image analysis enabled but no vision-capable model configured. "
                    "Skipping image analysis tasks."
                )
            else:
                for img in images_for_analysis:
                    # Create analysis coroutine (will be awaited by caller)
                    tasks.append(self.create_image_analysis_task(img))

        # Markdown enhancement task
        if llm_enabled:
            tasks.append(self.create_enhancement_task(markdown_content, input_file))

        log.debug(
            "Created LLM tasks",
            file=str(input_file.name),
            image_tasks=len(images_for_analysis) if analyze_image else 0,
            enhancement_task=1 if llm_enabled else 0,
        )

        return tasks

    async def enhance_markdown(self, markdown: str, source_file: Path) -> str:
        """Enhance markdown content using LLM.

        A convenience method that wraps create_enhancement_task for simple usage.

        Args:
            markdown: Raw markdown content
            source_file: Original source file path

        Returns:
            Enhanced markdown content
        """
        from markit.llm.enhancer import EnhancedMarkdown

        enhancer = await self.get_enhancer()

        try:
            enhanced = await enhancer.enhance(markdown, source_file)
            if isinstance(enhanced, EnhancedMarkdown):
                return enhanced.content
            return str(enhanced)
        except Exception as e:
            log.warning(
                "LLM enhancement failed, returning original",
                error=str(e),
            )
            # Fall back to simple cleaning
            from markit.llm.enhancer import SimpleMarkdownCleaner

            cleaner = SimpleMarkdownCleaner()
            return cleaner.clean(markdown)
