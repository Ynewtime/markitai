"""Core conversion pipeline.

This module provides the main ConversionPipeline class that orchestrates
document conversion. It delegates to specialized services:
- ImageProcessingService: Image format conversion, compression, deduplication
- LLMOrchestrator: LLM provider management, enhancement, analysis
- OutputManager: File writing, conflict resolution
"""

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markit.config.settings import MarkitSettings
from markit.converters.base import ConversionPlan, ConversionResult
from markit.core.router import FormatRouter
from markit.exceptions import ConversionError
from markit.services.image_processor import (
    ImageProcessingConfig,
    ImageProcessingService,
    ProcessedImageInfo,
    _generate_image_filename,
    _sanitize_filename,
)
from markit.services.llm_orchestrator import LLMOrchestrator
from markit.services.output_manager import OutputManager
from markit.utils.logging import get_logger

if TYPE_CHECKING:
    from markit.converters.base import ExtractedImage
    from markit.image.analyzer import ImageAnalysis, ImageAnalyzer
    from markit.image.compressor import CompressedImage, ImageCompressor
    from markit.llm.base import LLMTaskResultWithStats
    from markit.llm.enhancer import MarkdownEnhancer
    from markit.llm.manager import ProviderManager

log = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "ConversionPipeline",
    "DocumentConversionResult",
    "PipelineResult",
    "ProcessedImageInfo",
    "_generate_image_filename",
    "_sanitize_filename",
]


@dataclass
class PipelineResult:
    """Result of a pipeline conversion."""

    output_path: Path | None = None
    markdown_content: str = ""
    images_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DocumentConversionResult:
    """Intermediate result from Phase 1 document conversion.

    This holds the results of document conversion before LLM processing,
    allowing the pipeline to be split into phases for better parallelism.
    """

    input_file: Path
    output_dir: Path
    conversion_result: ConversionResult
    plan: ConversionPlan
    processed_images: list[Any] = field(default_factory=list)  # ExtractedImage
    images_for_analysis: list["CompressedImage"] = field(default_factory=list)
    markdown_content: str = ""  # Current markdown (may have image refs updated)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if conversion was successful."""
        return self.error is None and self.conversion_result.success


class ConversionPipeline:
    """Main conversion pipeline orchestrator.

    This class orchestrates the document conversion flow by delegating to
    specialized services:
    - ImageProcessingService: Image format conversion, compression, deduplication
    - LLMOrchestrator: LLM provider management, enhancement, analysis
    - OutputManager: File writing, conflict resolution

    The conversion flow:
    1. Route file to appropriate converter
    2. Pre-process if needed (e.g., Office conversion)
    3. Convert to Markdown
    4. Process images (extract, convert, compress) via ImageProcessingService
    5. Optionally enhance with LLM via LLMOrchestrator
    6. Write output via OutputManager
    """

    def __init__(
        self,
        settings: MarkitSettings,
        llm_enabled: bool = False,
        analyze_image: bool = False,
        analyze_image_with_md: bool = False,
        compress_images: bool = True,
        pdf_engine: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        use_concurrent_fallback: bool = False,
        # Dependency injection (optional, for testing)
        image_processor: ImageProcessingService | None = None,
        llm_orchestrator: LLMOrchestrator | None = None,
        output_manager: OutputManager | None = None,
    ) -> None:
        """Initialize the conversion pipeline.

        Args:
            settings: Application settings
            llm_enabled: Enable LLM Markdown enhancement
            analyze_image: Enable LLM image analysis for alt text
            analyze_image_with_md: Enable detailed image description .md file generation
            compress_images: Enable image compression
            pdf_engine: Override PDF engine from settings
            llm_provider: Override LLM provider
            llm_model: Override LLM model
            use_concurrent_fallback: Enable concurrent fallback for LLM calls
                                     (starts backup model if primary exceeds timeout)
            image_processor: Optional ImageProcessingService instance (for DI/testing)
            llm_orchestrator: Optional LLMOrchestrator instance (for DI/testing)
            output_manager: Optional OutputManager instance (for DI/testing)
        """
        self.settings = settings
        self.llm_enabled = llm_enabled
        self.analyze_image = analyze_image
        self.analyze_image_with_md = analyze_image_with_md
        self.compress_images = compress_images
        self.pdf_engine = pdf_engine or settings.pdf.engine
        self.llm_provider = llm_provider
        self.use_concurrent_fallback = use_concurrent_fallback
        self.llm_model = llm_model

        # Initialize router with image filtering settings
        self.router = FormatRouter(
            pdf_engine=self.pdf_engine,
            filter_small_images=settings.image.filter_small_images,
            min_image_dimension=settings.image.min_dimension,
            min_image_area=settings.image.min_area,
            min_image_size=settings.image.min_file_size,
        )

        # Initialize services (with DI support)
        self._image_processor = image_processor or ImageProcessingService(
            config=ImageProcessingConfig(
                compress_images=compress_images,
                png_optimization_level=settings.image.png_optimization_level,
                jpeg_quality=settings.image.jpeg_quality,
                max_dimension=settings.image.max_dimension,
            ),
        )

        self._llm_orchestrator = llm_orchestrator or LLMOrchestrator(
            llm_config=settings.llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            enhancement_chunk_size=settings.enhancement.chunk_size,
            use_concurrent_fallback=use_concurrent_fallback,
        )

        self._output_manager = output_manager or OutputManager(
            on_conflict=settings.output.on_conflict,
            create_assets_subdir=settings.output.create_assets_subdir,
            generate_image_descriptions=analyze_image_with_md,
        )

        # Legacy lazy-loaded components (for backward compatibility during transition)
        self._provider_manager: ProviderManager | None = None
        self._provider_manager_initialized = False
        self._enhancer: MarkdownEnhancer | None = None
        self._image_compressor: ImageCompressor | None = None
        self._image_analyzer: ImageAnalyzer | None = None

        # Locks for thread-safe lazy initialization
        self._provider_lock = asyncio.Lock()
        self._enhancer_lock = asyncio.Lock()
        self._analyzer_lock = asyncio.Lock()

    def _create_provider_manager(self) -> "ProviderManager":
        """Create a new ProviderManager instance (not initialized)."""
        from markit.config.settings import LLMProviderConfig
        from markit.llm.manager import ProviderManager

        # Start with the base configuration from settings
        llm_config = self.settings.llm.model_copy()

        # If preferred provider/model specified via CLI, prepend it to legacy providers list
        # This ensures it gets priority
        if self.llm_provider:
            preferred_config = LLMProviderConfig(
                provider=self.llm_provider,  # type: ignore[arg-type]
                model=self.llm_model or self._get_default_model(self.llm_provider),
            )
            # Insert at beginning of providers list
            new_providers = [preferred_config] + list(llm_config.providers)
            llm_config.providers = new_providers

        return ProviderManager(llm_config=llm_config)

    def _get_required_capabilities(self) -> list[str]:
        """Determine required LLM capabilities based on pipeline configuration.

        Returns:
            List of required capabilities (e.g., ["text"], ["text", "vision"])
        """
        capabilities = []
        if self.llm_enabled:
            capabilities.append("text")
        if self.analyze_image or self.analyze_image_with_md:
            if "text" not in capabilities:
                capabilities.append("text")
            capabilities.append("vision")
        return capabilities if capabilities else ["text"]

    async def _get_provider_manager_async(
        self,
        required_capabilities: list[str] | None = None,
        lazy: bool = True,
    ) -> "ProviderManager":
        """Get or create the LLM provider manager (async).

        Creates the manager on first call and initializes it with required capabilities.
        Subsequent calls return the cached instance.

        Args:
            required_capabilities: Override auto-detected capabilities. If None,
                                   capabilities are inferred from pipeline settings.
            lazy: If True (default), only load configs without network validation.
                  Providers will be validated on-demand when first used.
                  Set to False in batch mode to validate upfront (fail fast).
        """
        # Double-checked locking pattern
        if self._provider_manager is not None and self._provider_manager_initialized:
            return self._provider_manager

        async with self._provider_lock:
            if self._provider_manager is None:
                self._provider_manager = self._create_provider_manager()

            if not self._provider_manager_initialized:
                # Use provided capabilities or infer from pipeline settings
                caps = required_capabilities or self._get_required_capabilities()
                await self._provider_manager.initialize(required_capabilities=caps, lazy=lazy)
                self._provider_manager_initialized = True

        return self._provider_manager

    def _get_provider_manager(self) -> "ProviderManager":
        """Get or create the LLM provider manager (sync).

        Note: This does NOT initialize the provider. Use _get_provider_manager_async()
        in async contexts to ensure proper initialization.
        """
        if self._provider_manager is None:
            self._provider_manager = self._create_provider_manager()
        return self._provider_manager

    async def warmup(self) -> None:
        """Warmup LLM providers by forcing network validation.

        This method should be called before batch processing to:
        1. Fail fast if providers are misconfigured
        2. Avoid concurrent validation races during parallel processing

        Only performs warmup if LLM features are enabled (llm_enabled or analyze_image).
        """
        # Delegate to LLMOrchestrator
        await self._llm_orchestrator.warmup(
            llm_enabled=self.llm_enabled,
            analyze_image=self.analyze_image or self.analyze_image_with_md,
        )

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider.

        Delegates to LLMOrchestrator for consistency.

        Args:
            provider: LLM provider name

        Returns:
            Default model name for the provider

        Raises:
            ValueError: If provider is not supported
        """
        return self._llm_orchestrator._get_default_model(provider)

    async def _get_enhancer_async(self) -> "MarkdownEnhancer":
        """Get or create the Markdown enhancer (async, thread-safe).

        Delegates to LLMOrchestrator.
        """
        return await self._llm_orchestrator.get_enhancer()

    def _get_image_compressor(self) -> "ImageCompressor":
        """Get or create the image compressor.

        Delegates to ImageProcessingService.
        """
        return self._image_processor._get_image_compressor()

    async def _get_image_analyzer_async(self) -> "ImageAnalyzer":
        """Get or create the image analyzer (async, thread-safe).

        Delegates to LLMOrchestrator.
        """
        return await self._llm_orchestrator.get_image_analyzer()

    def convert_file(self, input_file: Path, output_dir: Path) -> PipelineResult:
        """Convert a single file synchronously.

        Args:
            input_file: Path to input file
            output_dir: Directory for output files

        Returns:
            PipelineResult with conversion details
        """
        return asyncio.run(self._convert_file_async(input_file, output_dir))

    async def convert_file_async(self, input_file: Path, output_dir: Path) -> PipelineResult:
        """Convert a single file asynchronously.

        Args:
            input_file: Path to input file
            output_dir: Directory for output files

        Returns:
            PipelineResult with conversion details
        """
        return await self._convert_file_async(input_file, output_dir)

    # ========================================================================
    # Phased Pipeline Methods (for batch processing parallelism)
    # ========================================================================

    async def convert_document_only(
        self, input_file: Path, output_dir: Path
    ) -> DocumentConversionResult:
        """Phase 1: Convert document without LLM processing.

        This method performs only the CPU-bound document conversion,
        allowing the file semaphore to be released early. LLM tasks
        are deferred to create_llm_tasks().

        Args:
            input_file: Path to input file
            output_dir: Directory for output files

        Returns:
            DocumentConversionResult with conversion data for further processing
        """
        from markit.image.compressor import CompressedImage

        log.info(
            "Phase 1: Document conversion",
            file=str(input_file),
        )

        try:
            # 1. Get conversion plan from router
            plan = self.router.route(input_file)
            log.debug(
                "Conversion plan",
                primary=plan.primary_converter.name,
                fallback=plan.fallback_converter.name if plan.fallback_converter else None,
                file=str(input_file),
            )

            # 2. Run pre-processors if any
            current_file = input_file
            converted_dir = output_dir / "converted"
            for processor in plan.pre_processors:
                if hasattr(processor, "set_converted_dir"):
                    processor.set_converted_dir(converted_dir)  # type: ignore[attr-defined]
                log.debug("Running pre-processor", processor=processor.name, file=str(input_file))
                current_file = await processor.process(current_file)

            # 3. Convert to Markdown
            conversion_result = await self._convert_with_fallback(current_file, plan)

            # 4. Process images via ImageProcessingService
            processed_images = []
            images_for_analysis: list[CompressedImage] = []
            markdown = conversion_result.markdown

            if conversion_result.images:
                log.info(
                    "Processing images (format/compress)",
                    count=len(conversion_result.images),
                    file=str(input_file),
                )

                # Delegate to ImageProcessingService
                (
                    processed_images,
                    filename_map,
                ) = await self._image_processor.optimize_images_parallel(
                    conversion_result.images, input_file
                )

                # Update markdown references via service helper
                markdown = self._image_processor.update_markdown_references(markdown, filename_map)

                # Prepare for LLM analysis if enabled
                if self.analyze_image:
                    images_for_analysis = self._image_processor.prepare_for_analysis(
                        processed_images
                    )

            # Update conversion result with processed images and markdown
            updated_result = ConversionResult(
                markdown=markdown,
                images=processed_images,
                metadata=conversion_result.metadata,
            )

            return DocumentConversionResult(
                input_file=input_file,
                output_dir=output_dir,
                conversion_result=updated_result,
                plan=plan,
                processed_images=processed_images,
                images_for_analysis=images_for_analysis,
                markdown_content=markdown,
            )

        except ConversionError as e:
            log.error("Document conversion failed", file=str(input_file), error=str(e))
            # Return error result
            empty_plan = self.router.route(input_file)  # Get plan for structure
            return DocumentConversionResult(
                input_file=input_file,
                output_dir=output_dir,
                conversion_result=ConversionResult(markdown="", success=False, error=str(e)),
                plan=empty_plan,
                error=str(e),
            )
        except Exception as e:
            log.error(
                "Unexpected error in document conversion",
                file=str(input_file),
                error=str(e),
                exc_info=True,
            )
            empty_plan = self.router.route(input_file)
            return DocumentConversionResult(
                input_file=input_file,
                output_dir=output_dir,
                conversion_result=ConversionResult(
                    markdown="", success=False, error=f"Unexpected error: {e}"
                ),
                plan=empty_plan,
                error=f"Unexpected error: {e}",
            )

    async def create_llm_tasks(
        self, doc_result: DocumentConversionResult
    ) -> list[Coroutine[Any, Any, Any]]:
        """Phase 2: Create LLM task coroutines without executing them.

        This creates coroutines for image analysis and markdown enhancement
        that can be submitted to a global LLM queue for rate-limited execution.

        Args:
            doc_result: Result from convert_document_only()

        Returns:
            List of coroutines (not awaited) for LLM tasks
        """
        if not doc_result.success:
            return []

        # Delegate to LLMOrchestrator
        return await self._llm_orchestrator.create_llm_tasks(
            images_for_analysis=doc_result.images_for_analysis,
            markdown_content=doc_result.markdown_content,
            input_file=doc_result.input_file,
            llm_enabled=self.llm_enabled,
            analyze_image=self.analyze_image,
        )

    async def _create_image_analysis_task(
        self, image: "CompressedImage", return_stats: bool = True
    ) -> "ImageAnalysis | LLMTaskResultWithStats":
        """Create and execute an image analysis task.

        Delegates to LLMOrchestrator.

        Args:
            image: Compressed image to analyze
            return_stats: If True, return LLMTaskResultWithStats with statistics

        Returns:
            ImageAnalysis or LLMTaskResultWithStats containing the analysis
        """
        return await self._llm_orchestrator.create_image_analysis_task(image, return_stats)

    async def _create_enhancement_task(
        self, markdown: str, source_file: Path, return_stats: bool = True
    ) -> "str | LLMTaskResultWithStats":
        """Create and execute a markdown enhancement task.

        Delegates to LLMOrchestrator.

        Args:
            markdown: Markdown content to enhance
            source_file: Source file path for context
            return_stats: If True, return LLMTaskResultWithStats with statistics

        Returns:
            Enhanced markdown string or LLMTaskResultWithStats containing the content
        """
        return await self._llm_orchestrator.create_enhancement_task(
            markdown, source_file, return_stats
        )

    async def finalize_output(
        self,
        doc_result: DocumentConversionResult,
        image_analyses: list["ImageAnalysis"] | None = None,
        enhanced_markdown: str | None = None,
    ) -> PipelineResult:
        """Phase 3: Merge LLM results and write final output.

        Args:
            doc_result: Result from convert_document_only()
            image_analyses: Results from image analysis tasks (in order)
            enhanced_markdown: Result from enhancement task (if enabled)

        Returns:
            Final PipelineResult
        """
        if not doc_result.success:
            return PipelineResult(
                output_path=doc_result.output_dir / f"{doc_result.input_file.name}.md",
                markdown_content="",
                success=False,
                error=doc_result.error,
            )

        log.info("Phase 3: Finalizing output", file=str(doc_result.input_file.name))

        # Use enhanced markdown if available, otherwise use original
        markdown = enhanced_markdown if enhanced_markdown else doc_result.markdown_content

        # Build image info list and update markdown with analysis results
        image_info_list: list[ProcessedImageInfo] = []

        if image_analyses and doc_result.processed_images:
            for i, img in enumerate(doc_result.processed_images):
                analysis = image_analyses[i] if i < len(image_analyses) else None
                image_info_list.append(ProcessedImageInfo(filename=img.filename, analysis=analysis))

                # Update markdown with alt text from analysis
                if analysis:
                    old_ref = f"![]({img.filename})"
                    new_ref = f"![{analysis.alt_text}]({img.filename})"
                    markdown = markdown.replace(old_ref, new_ref)

                    old_ref_assets = f"![](assets/{img.filename})"
                    new_ref_assets = f"![{analysis.alt_text}](assets/{img.filename})"
                    markdown = markdown.replace(old_ref_assets, new_ref_assets)
        else:
            # No analysis, just create info without analysis
            for img in doc_result.processed_images:
                image_info_list.append(ProcessedImageInfo(filename=img.filename, analysis=None))

        # Update conversion result with final markdown
        final_result = ConversionResult(
            markdown=markdown,
            images=doc_result.processed_images,
            metadata=doc_result.conversion_result.metadata,
        )

        # Write output via OutputManager
        output_path = await self._output_manager.write_output(
            doc_result.input_file, doc_result.output_dir, final_result, image_info_list
        )

        return PipelineResult(
            output_path=output_path,
            markdown_content=markdown,
            images_count=len(doc_result.processed_images),
            metadata=doc_result.conversion_result.metadata,
            success=True,
        )

    # ========================================================================
    # Original Pipeline Methods (backward compatible)
    # ========================================================================

    async def _convert_file_async(self, input_file: Path, output_dir: Path) -> PipelineResult:
        """Internal async conversion implementation."""
        log.info(
            "Starting conversion pipeline",
            file=str(input_file),
            output_dir=str(output_dir),
        )

        try:
            # 1. Get conversion plan from router
            plan = self.router.route(input_file)
            log.debug(
                "Conversion plan",
                primary=plan.primary_converter.name,
                fallback=plan.fallback_converter.name if plan.fallback_converter else None,
                file=str(input_file),
            )

            # 2. Run pre-processors if any
            current_file = input_file
            converted_dir = (
                output_dir / "converted"
            )  # Store intermediate files in output/converted/
            for processor in plan.pre_processors:
                # Set converted directory for Office preprocessor
                if hasattr(processor, "set_converted_dir"):
                    processor.set_converted_dir(converted_dir)  # type: ignore[attr-defined]
                log.debug("Running pre-processor", processor=processor.name, file=str(input_file))
                current_file = await processor.process(current_file)

            # 3. Convert to Markdown
            conversion_result = await self._convert_with_fallback(current_file, plan)

            # 4. Process images (compress and optionally analyze)
            image_info_list: list[ProcessedImageInfo] = []
            if conversion_result.images:
                log.info(
                    "Processing images",
                    count=len(conversion_result.images),
                    file=str(input_file),
                )
                conversion_result, image_info_list = await self._process_images(
                    conversion_result, input_file, output_dir
                )

            # 5. LLM Enhancement via LLMOrchestrator
            markdown_content = conversion_result.markdown
            if self.llm_enabled:
                log.info("Applying LLM enhancement", file=input_file.name)
                markdown_content = await self._llm_orchestrator.enhance_markdown(
                    markdown_content, input_file
                )
                # Update the result with enhanced content
                conversion_result = ConversionResult(
                    markdown=markdown_content,
                    images=conversion_result.images,
                    metadata=conversion_result.metadata,
                )

            # 6. Write output via OutputManager
            # Skip images if they were already written during analysis (either mode)
            images_written_immediately = self.analyze_image or self.analyze_image_with_md
            output_path = await self._output_manager.write_output(
                input_file,
                output_dir,
                conversion_result,
                image_info_list,
                skip_images=images_written_immediately,
            )

            return PipelineResult(
                output_path=output_path,
                markdown_content=conversion_result.markdown,
                images_count=conversion_result.images_count,
                metadata=conversion_result.metadata,
                success=True,
            )

        except ConversionError as e:
            log.error("Conversion failed", file=str(input_file), error=str(e))
            return PipelineResult(
                output_path=output_dir / f"{input_file.name}.md",
                markdown_content="",
                success=False,
                error=str(e),
            )
        except Exception as e:
            log.error("Unexpected error", file=str(input_file), error=str(e), exc_info=True)
            return PipelineResult(
                output_path=output_dir / f"{input_file.name}.md",
                markdown_content="",
                success=False,
                error=f"Unexpected error: {e}",
            )

    async def _optimize_images_parallel(
        self, images: list["ExtractedImage"], input_file: Path
    ) -> tuple[list["ExtractedImage"], dict[str, str | None]]:
        """Process images in parallel: format convert, deduplicate, compress.

        Delegates to ImageProcessingService.

        Returns:
            Tuple of (processed_unique_images, filename_map)
        """
        return await self._image_processor.optimize_images_parallel(images, input_file)

    async def _process_images(
        self, result: ConversionResult, input_file: Path, output_dir: Path
    ) -> tuple[ConversionResult, list[ProcessedImageInfo]]:
        """Process images: convert format if needed, compress, and optionally analyze.

        Uses parallel processing via ImageProcessingService and LLM analysis
        via LLMOrchestrator. When output_dir is provided, images and their
        analysis results are written immediately after analysis completes.

        Args:
            result: Conversion result with extracted images
            input_file: Original input file path (for generating standardized filenames)
            output_dir: Output directory for writing images immediately

        Returns:
            Tuple of (updated conversion result, list of processed image info)
        """
        if not result.images:
            return result, []

        # Phase 1: Format conversion, deduplication, and compression via ImageProcessingService
        processed_images, filename_map = await self._image_processor.optimize_images_parallel(
            result.images, input_file
        )

        # Update markdown references via service helper
        markdown = self._image_processor.update_markdown_references(result.markdown, filename_map)

        # Prepare for analysis
        # analyze_image: only generate alt text in markdown
        # analyze_image_with_md: also generate description .md files
        needs_analysis = self.analyze_image or self.analyze_image_with_md
        images_for_analysis = []
        if needs_analysis:
            images_for_analysis = self._image_processor.prepare_for_analysis(processed_images)

        # Phase 2: Parallel LLM analysis via LLMOrchestrator
        analyses: list[ImageAnalysis | None] = [None] * len(processed_images)
        if needs_analysis and images_for_analysis:
            analyzer = await self._llm_orchestrator.get_image_analyzer()
            log.info(
                "Analyzing images in parallel",
                file=input_file.name,
                count=len(images_for_analysis),
            )

            try:
                analysis_results = await analyzer.batch_analyze(
                    images_for_analysis, output_dir=output_dir
                )

                # Map results back and update markdown
                for i, analysis in enumerate(analysis_results):
                    analyses[i] = analysis
                    processed_image = processed_images[i]

                    log.debug(
                        "Image analyzed",
                        file=input_file.name,
                        filename=processed_image.filename,
                        type=analysis.image_type,
                    )

                    # Update markdown with alt text
                    old_ref = f"![]({processed_image.filename})"
                    new_ref = f"![{analysis.alt_text}]({processed_image.filename})"
                    markdown = markdown.replace(old_ref, new_ref)

                    old_ref_assets = f"![](assets/{processed_image.filename})"
                    new_ref_assets = f"![{analysis.alt_text}](assets/{processed_image.filename})"
                    markdown = markdown.replace(old_ref_assets, new_ref_assets)

            except Exception as e:
                log.warning(
                    "Batch image analysis failed",
                    file=str(input_file),
                    error=str(e),
                )

        # Build processed info list
        processed_info = [
            ProcessedImageInfo(filename=img.filename, analysis=analyses[i])
            for i, img in enumerate(processed_images)
        ]

        return (
            ConversionResult(
                markdown=markdown,
                images=processed_images,
                metadata=result.metadata,
            ),
            processed_info,
        )

    async def _enhance_markdown(self, markdown: str, source_file: Path) -> str:
        """Enhance markdown content using LLM.

        Delegates to LLMOrchestrator.

        Args:
            markdown: Raw markdown content
            source_file: Original source file path

        Returns:
            Enhanced markdown content
        """
        return await self._llm_orchestrator.enhance_markdown(markdown, source_file)

    async def _convert_with_fallback(self, file_path: Path, plan) -> ConversionResult:
        """Attempt conversion with fallback support."""
        errors = []

        # Try primary converter
        try:
            log.debug(
                "Trying primary converter",
                converter=plan.primary_converter.name,
                file=str(file_path),
            )
            return await plan.primary_converter.convert(file_path)
        except ConversionError as e:
            log.warning(
                "Primary converter failed",
                converter=plan.primary_converter.name,
                error=str(e),
                file=str(file_path),
            )
            errors.append(e)

        # Try fallback converter
        if plan.fallback_converter:
            try:
                log.debug(
                    "Trying fallback converter",
                    converter=plan.fallback_converter.name,
                    file=str(file_path),
                )
                return await plan.fallback_converter.convert(file_path)
            except ConversionError as e:
                log.warning(
                    "Fallback converter failed",
                    converter=plan.fallback_converter.name,
                    error=str(e),
                    file=str(file_path),
                )
                errors.append(e)

        # All converters failed
        from markit.exceptions import FallbackExhaustedError

        raise FallbackExhaustedError(file_path, errors)

    async def _write_output(
        self,
        input_file: Path,
        output_dir: Path,
        result: ConversionResult,
        image_info_list: list[ProcessedImageInfo] | None = None,
    ) -> Path:
        """Write conversion result to output directory.

        Delegates to OutputManager.

        Args:
            input_file: Original input file
            output_dir: Output directory
            result: Conversion result
            image_info_list: List of processed image info (with analysis results)

        Returns:
            Path to the output markdown file
        """
        return await self._output_manager.write_output(
            input_file, output_dir, result, image_info_list
        )

    def _generate_image_description_md(
        self,
        filename: str,
        analysis: "ImageAnalysis",
        generated_at: Any,  # datetime.datetime
    ) -> str:
        """Generate markdown content for image description file.

        Delegates to OutputManager.

        Args:
            filename: Image filename
            analysis: Image analysis result
            generated_at: Timestamp when the description was generated

        Returns:
            Markdown content for the image description file
        """
        return self._output_manager.generate_image_description_md(filename, analysis, generated_at)

    def _resolve_conflict(self, output_path: Path) -> Path:
        """Resolve output file conflicts based on settings.

        Delegates to OutputManager.
        """
        return self._output_manager.resolve_conflict(output_path)
