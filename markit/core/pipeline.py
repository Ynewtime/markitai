"""Core conversion pipeline."""

import asyncio
import re
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markit.config.settings import MarkitSettings
from markit.converters.base import ConversionPlan, ConversionResult
from markit.core.router import FormatRouter
from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

if TYPE_CHECKING:
    from markit.image.analyzer import ImageAnalysis, ImageAnalyzer
    from markit.image.compressor import CompressedImage, ImageCompressor
    from markit.llm.enhancer import MarkdownEnhancer
    from markit.llm.manager import ProviderManager

log = get_logger(__name__)


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename by removing/replacing problematic characters.

    Args:
        name: Original filename

    Returns:
        Sanitized filename safe for filesystem use
    """
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Replace multiple underscores/spaces with single underscore
    sanitized = re.sub(r"[_\s]+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def _generate_image_filename(original_file: Path, image_index: int, image_format: str) -> str:
    """Generate standardized image filename.

    Format: <original_filename>.<original_extension>.<index>.<image_format>
    Example: file-sample_100kB.doc.001.jpeg

    Args:
        original_file: Original input file path
        image_index: 1-based index of the image
        image_format: Image format extension (png, jpeg, etc.)

    Returns:
        Standardized image filename
    """
    # Get the full original filename (name + extension)
    original_name = original_file.name
    # Sanitize the filename
    safe_name = _sanitize_filename(original_name)
    # Generate filename: <name>.<ext>.<index>.<format>
    return f"{safe_name}.{image_index:03d}.{image_format}"


@dataclass
class ProcessedImageInfo:
    """Information about a processed image including analysis results."""

    filename: str
    analysis: "ImageAnalysis | None" = None


@dataclass
class PipelineResult:
    """Result of a pipeline conversion."""

    output_path: Path
    markdown_content: str
    images_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


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

    Handles the complete conversion flow:
    1. Route file to appropriate converter
    2. Pre-process if needed (e.g., Office conversion)
    3. Convert to Markdown
    4. Process images (extract, convert, compress)
    5. Optionally enhance with LLM
    6. Write output
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
        """
        self.settings = settings
        self.llm_enabled = llm_enabled
        self.analyze_image = analyze_image
        self.analyze_image_with_md = analyze_image_with_md
        self.compress_images = compress_images
        self.pdf_engine = pdf_engine or settings.pdf.engine
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize router with image filtering settings
        self.router = FormatRouter(
            pdf_engine=self.pdf_engine,
            filter_small_images=settings.image.filter_small_images,
            min_image_dimension=settings.image.min_dimension,
            min_image_area=settings.image.min_area,
            min_image_size=settings.image.min_file_size,
        )

        # Lazy-loaded components
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
        from markit.llm.manager import ProviderManager

        # Build configs list, optionally overriding with preferred provider/model
        configs = list(self.settings.llm.providers)

        # If preferred provider/model specified, prepend it to the list
        if self.llm_provider:
            from markit.config.settings import LLMProviderConfig

            preferred_config = LLMProviderConfig(
                provider=self.llm_provider,  # type: ignore[arg-type]
                model=self.llm_model or self._get_default_model(self.llm_provider),
            )
            configs.insert(0, preferred_config)

        return ProviderManager(configs=configs)

    async def _get_provider_manager_async(self) -> "ProviderManager":
        """Get or create the LLM provider manager (async).

        Creates the manager on first call and initializes it.
        Subsequent calls return the cached instance.
        """
        # Double-checked locking pattern
        if self._provider_manager is not None and self._provider_manager_initialized:
            return self._provider_manager

        async with self._provider_lock:
            if self._provider_manager is None:
                self._provider_manager = self._create_provider_manager()

            if not self._provider_manager_initialized:
                await self._provider_manager.initialize()
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

    def _get_default_model(self, provider: str) -> str:
        """Get default model for a provider.

        Args:
            provider: LLM provider name

        Returns:
            Default model name for the provider

        Raises:
            ValueError: If provider is not supported
        """
        defaults = {
            "openai": "gpt-5.2",
            "anthropic": "claude-sonnet-4-5",
            "gemini": "gemini-3-flash-preview",
            "ollama": "llama3.2",
            "openrouter": "google/gemini-3-flash-preview",
        }
        if provider not in defaults:
            raise ValueError(
                f"No default model configured for provider '{provider}'. "
                f"Please specify a model explicitly using --llm-model. "
                f"Supported providers: {', '.join(defaults.keys())}"
            )
        return defaults[provider]

    async def _get_enhancer_async(self) -> "MarkdownEnhancer":
        """Get or create the Markdown enhancer (async, thread-safe)."""
        if self._enhancer is not None:
            return self._enhancer

        async with self._enhancer_lock:
            if self._enhancer is None:
                from markit.llm.enhancer import EnhancementConfig, MarkdownEnhancer

                provider_manager = await self._get_provider_manager_async()
                self._enhancer = MarkdownEnhancer(
                    provider_manager=provider_manager,
                    config=EnhancementConfig(
                        chunk_size=self.settings.enhancement.chunk_size,
                    ),
                )
        return self._enhancer

    def _get_image_compressor(self) -> "ImageCompressor":
        """Get or create the image compressor."""
        if self._image_compressor is None:
            from markit.image.compressor import CompressionConfig, ImageCompressor

            self._image_compressor = ImageCompressor(
                config=CompressionConfig(
                    png_optimization_level=self.settings.image.png_optimization_level,
                    jpeg_quality=self.settings.image.jpeg_quality,
                    max_dimension=self.settings.image.max_dimension,
                ),
            )
        return self._image_compressor

    async def _get_image_analyzer_async(self) -> "ImageAnalyzer":
        """Get or create the image analyzer (async, thread-safe)."""
        if self._image_analyzer is not None:
            return self._image_analyzer

        async with self._analyzer_lock:
            if self._image_analyzer is None:
                from markit.image.analyzer import ImageAnalyzer

                provider_manager = await self._get_provider_manager_async()
                self._image_analyzer = ImageAnalyzer(
                    provider_manager=provider_manager,
                )
        return self._image_analyzer

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
        from markit.converters.base import ExtractedImage
        from markit.image.compressor import CompressedImage
        from markit.image.converter import ImageFormatConverter

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
            )

            # 2. Run pre-processors if any
            current_file = input_file
            converted_dir = output_dir / "converted"
            for processor in plan.pre_processors:
                if hasattr(processor, "set_converted_dir"):
                    processor.set_converted_dir(converted_dir)
                log.debug("Running pre-processor", processor=processor.name)
                current_file = await processor.process(current_file)

            # 3. Convert to Markdown
            conversion_result = await self._convert_with_fallback(current_file, plan)

            # 4. Process images (format conversion and compression only, no LLM)
            processed_images: list[ExtractedImage] = []
            images_for_analysis: list[CompressedImage] = []
            markdown = conversion_result.markdown

            if conversion_result.images:
                log.info("Processing images (format/compress)", count=len(conversion_result.images))

                format_converter = ImageFormatConverter()
                compressor = self._get_image_compressor()
                image_index = 0  # Counter for generating new filenames

                for image in conversion_result.images:
                    current_image = image
                    old_filename = image.filename
                    original_size = len(image.data)

                    # Step 1: Convert format if needed (WMF -> PNG, etc.)
                    if format_converter.needs_conversion(image.format):
                        try:
                            converted = format_converter.convert(image)
                            if converted is None:
                                log.info(
                                    "Image conversion not possible, removing",
                                    filename=image.filename,
                                )
                                markdown = markdown.replace(f"![](assets/{old_filename})", "")
                                markdown = markdown.replace(
                                    f"![{old_filename}](assets/{old_filename})", ""
                                )
                                continue

                            current_image = ExtractedImage(
                                data=converted.data,
                                format=converted.format,
                                filename=converted.filename,
                                source_document=image.source_document,
                                position=image.position,
                                width=converted.width,
                                height=converted.height,
                            )
                        except Exception as e:
                            log.warning(
                                "Image format conversion failed",
                                filename=image.filename,
                                error=str(e),
                            )
                            markdown = markdown.replace(f"![](assets/{old_filename})", "")
                            markdown = markdown.replace(
                                f"![{old_filename}](assets/{old_filename})", ""
                            )
                            continue

                    # Step 2: Compress image if enabled
                    if self.compress_images:
                        try:
                            compressed = compressor.compress(current_image)
                            current_image = ExtractedImage(
                                data=compressed.data,
                                format=compressed.format,
                                filename=compressed.filename,
                                source_document=current_image.source_document,
                                position=current_image.position,
                                width=compressed.width,
                                height=compressed.height,
                            )
                        except Exception as e:
                            log.warning(
                                "Image compression failed",
                                filename=current_image.filename,
                                error=str(e),
                            )

                    # Step 3: Rename image using original input filename
                    image_index += 1
                    new_filename = _generate_image_filename(
                        input_file, image_index, current_image.format
                    )
                    # Update markdown references
                    markdown = markdown.replace(
                        f"assets/{old_filename}",
                        f"assets/{new_filename}",
                    )
                    # Also try to replace converted filename if different
                    if current_image.filename != old_filename:
                        markdown = markdown.replace(
                            f"assets/{current_image.filename}",
                            f"assets/{new_filename}",
                        )

                    processed_image = ExtractedImage(
                        data=current_image.data,
                        format=current_image.format,
                        filename=new_filename,
                        source_document=current_image.source_document,
                        position=current_image.position,
                        width=current_image.width,
                        height=current_image.height,
                    )

                    processed_images.append(processed_image)

                    # Prepare for LLM analysis if enabled
                    if self.analyze_image:
                        images_for_analysis.append(
                            CompressedImage(
                                data=processed_image.data,
                                format=processed_image.format,
                                filename=processed_image.filename,
                                original_size=original_size,
                                compressed_size=len(processed_image.data),
                                width=processed_image.width or 0,
                                height=processed_image.height or 0,
                            )
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
            log.error("Document conversion failed", error=str(e))
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
            log.error("Unexpected error in document conversion", error=str(e), exc_info=True)
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

    def create_llm_tasks(
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
        tasks: list[Coroutine[Any, Any, Any]] = []

        if not doc_result.success:
            return tasks

        # Image analysis tasks
        if self.analyze_image and doc_result.images_for_analysis:
            for img in doc_result.images_for_analysis:
                # Create analysis coroutine (will be awaited by caller)
                tasks.append(self._create_image_analysis_task(img))

        # Markdown enhancement task
        if self.llm_enabled:
            tasks.append(
                self._create_enhancement_task(doc_result.markdown_content, doc_result.input_file)
            )

        log.debug(
            "Created LLM tasks",
            file=str(doc_result.input_file.name),
            image_tasks=len(doc_result.images_for_analysis) if self.analyze_image else 0,
            enhancement_task=1 if self.llm_enabled else 0,
        )

        return tasks

    async def _create_image_analysis_task(self, image: "CompressedImage") -> "ImageAnalysis":
        """Create and execute an image analysis task."""
        from markit.image.analyzer import ImageAnalysis

        analyzer = await self._get_image_analyzer_async()
        try:
            return await analyzer.analyze(image)
        except Exception as e:
            log.warning("Image analysis failed", filename=image.filename, error=str(e))
            return ImageAnalysis(
                alt_text=f"Image: {image.filename}",
                detailed_description="Image analysis failed.",
                detected_text=None,
                image_type="other",
            )

    async def _create_enhancement_task(self, markdown: str, source_file: Path) -> str:
        """Create and execute a markdown enhancement task."""
        enhancer = await self._get_enhancer_async()
        try:
            enhanced = await enhancer.enhance(markdown, source_file)
            return enhanced.content
        except Exception as e:
            log.warning("LLM enhancement failed", error=str(e))
            from markit.llm.enhancer import SimpleMarkdownCleaner

            cleaner = SimpleMarkdownCleaner()
            return cleaner.clean(markdown)

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

        # Write output
        output_path = await self._write_output(
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
            )

            # 2. Run pre-processors if any
            current_file = input_file
            converted_dir = (
                output_dir / "converted"
            )  # Store intermediate files in output/converted/
            for processor in plan.pre_processors:
                # Set converted directory for Office preprocessor
                if hasattr(processor, "set_converted_dir"):
                    processor.set_converted_dir(converted_dir)
                log.debug("Running pre-processor", processor=processor.name)
                current_file = await processor.process(current_file)

            # 3. Convert to Markdown
            conversion_result = await self._convert_with_fallback(current_file, plan)

            # 4. Process images (compress and optionally analyze)
            image_info_list: list[ProcessedImageInfo] = []
            if conversion_result.images:
                log.info(
                    "Processing images",
                    count=len(conversion_result.images),
                )
                conversion_result, image_info_list = await self._process_images(
                    conversion_result, input_file
                )

            # 5. LLM Enhancement
            markdown_content = conversion_result.markdown
            if self.llm_enabled:
                log.info("Applying LLM enhancement")
                markdown_content = await self._enhance_markdown(markdown_content, input_file)
                # Update the result with enhanced content
                conversion_result = ConversionResult(
                    markdown=markdown_content,
                    images=conversion_result.images,
                    metadata=conversion_result.metadata,
                )

            # 6. Write output (including image description .md files if enabled)
            output_path = await self._write_output(
                input_file, output_dir, conversion_result, image_info_list
            )

            return PipelineResult(
                output_path=output_path,
                markdown_content=conversion_result.markdown,
                images_count=conversion_result.images_count,
                metadata=conversion_result.metadata,
                success=True,
            )

        except ConversionError as e:
            log.error("Conversion failed", error=str(e))
            return PipelineResult(
                output_path=output_dir / f"{input_file.name}.md",
                markdown_content="",
                success=False,
                error=str(e),
            )
        except Exception as e:
            log.error("Unexpected error", error=str(e), exc_info=True)
            return PipelineResult(
                output_path=output_dir / f"{input_file.name}.md",
                markdown_content="",
                success=False,
                error=f"Unexpected error: {e}",
            )

    async def _process_images(
        self, result: ConversionResult, input_file: Path
    ) -> tuple[ConversionResult, list[ProcessedImageInfo]]:
        """Process images: convert format if needed, compress, and optionally analyze.

        Uses parallel LLM analysis for better performance.

        Args:
            result: Conversion result with extracted images
            input_file: Original input file path (for generating standardized filenames)

        Returns:
            Tuple of (updated conversion result, list of processed image info)
        """
        if not result.images:
            return result, []

        from markit.converters.base import ExtractedImage
        from markit.image.compressor import CompressedImage
        from markit.image.converter import ImageFormatConverter

        processed_images: list[ExtractedImage] = []
        images_for_analysis: list[CompressedImage] = []
        original_sizes: list[int] = []  # Track original sizes for CompressedImage
        format_converter = ImageFormatConverter()
        compressor = self._get_image_compressor()
        markdown = result.markdown
        image_index = 0  # Counter for generating new filenames

        # Phase 1: Format conversion and compression (CPU-bound, fast)
        for image in result.images:
            current_image = image
            old_filename = image.filename
            original_size = len(image.data)

            # Step 1: Convert format if needed (WMF -> PNG, etc.)
            if format_converter.needs_conversion(image.format):
                try:
                    converted = format_converter.convert(image)
                    # If conversion returned None, skip this image
                    if converted is None:
                        log.info(
                            "Image conversion not possible, removing from output",
                            filename=image.filename,
                        )
                        # Remove image reference from markdown
                        markdown = markdown.replace(f"![](assets/{old_filename})", "")
                        markdown = markdown.replace(f"![{old_filename}](assets/{old_filename})", "")
                        continue

                    current_image = ExtractedImage(
                        data=converted.data,
                        format=converted.format,
                        filename=converted.filename,
                        source_document=image.source_document,
                        position=image.position,
                        width=converted.width,
                        height=converted.height,
                    )
                    log.debug(
                        "Image format converted",
                        filename=image.filename,
                        from_format=image.format,
                        to_format=converted.format,
                    )
                except Exception as e:
                    log.warning(
                        "Image format conversion failed, skipping image",
                        filename=image.filename,
                        error=str(e),
                    )
                    # Remove image reference from markdown
                    markdown = markdown.replace(f"![](assets/{old_filename})", "")
                    markdown = markdown.replace(f"![{old_filename}](assets/{old_filename})", "")
                    continue

            # Step 2: Compress image if enabled
            if self.compress_images:
                try:
                    compressed = compressor.compress(current_image)
                    current_image = ExtractedImage(
                        data=compressed.data,
                        format=compressed.format,
                        filename=compressed.filename,
                        source_document=current_image.source_document,
                        position=current_image.position,
                        width=compressed.width,
                        height=compressed.height,
                    )
                    log.debug(
                        "Image compressed",
                        filename=current_image.filename,
                        savings=f"{compressed.savings_percent:.1f}%",
                    )
                except Exception as e:
                    log.warning(
                        "Image compression failed, using original",
                        filename=current_image.filename,
                        error=str(e),
                    )

            # Step 3: Rename image using original input filename
            image_index += 1
            new_filename = _generate_image_filename(input_file, image_index, current_image.format)
            # Update markdown references
            markdown = markdown.replace(
                f"assets/{old_filename}",
                f"assets/{new_filename}",
            )
            # Also try to replace converted filename if different
            if current_image.filename != old_filename:
                markdown = markdown.replace(
                    f"assets/{current_image.filename}",
                    f"assets/{new_filename}",
                )

            processed_image = ExtractedImage(
                data=current_image.data,
                format=current_image.format,
                filename=new_filename,
                source_document=current_image.source_document,
                position=current_image.position,
                width=current_image.width,
                height=current_image.height,
            )

            processed_images.append(processed_image)
            original_sizes.append(original_size)

            # Prepare for analysis if enabled
            if self.analyze_image:
                images_for_analysis.append(
                    CompressedImage(
                        data=processed_image.data,
                        format=processed_image.format,
                        filename=processed_image.filename,
                        original_size=original_size,
                        compressed_size=len(processed_image.data),
                        width=processed_image.width or 0,
                        height=processed_image.height or 0,
                    )
                )

        # Phase 2: Parallel LLM analysis (I/O-bound, main bottleneck)
        analyses: list[ImageAnalysis | None] = [None] * len(processed_images)
        if self.analyze_image and images_for_analysis:
            analyzer = await self._get_image_analyzer_async()
            log.info(
                "Analyzing images in parallel",
                count=len(images_for_analysis),
            )

            try:
                analysis_results = await analyzer.batch_analyze(images_for_analysis)

                # Map results back and update markdown
                for i, analysis in enumerate(analysis_results):
                    analyses[i] = analysis
                    processed_image = processed_images[i]

                    log.debug(
                        "Image analyzed",
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

        Args:
            markdown: Raw markdown content
            source_file: Original source file path

        Returns:
            Enhanced markdown content
        """
        enhancer = await self._get_enhancer_async()

        try:
            enhanced = await enhancer.enhance(markdown, source_file)
            return enhanced.content
        except Exception as e:
            log.warning(
                "LLM enhancement failed, returning original",
                error=str(e),
            )
            # Fall back to simple cleaning
            from markit.llm.enhancer import SimpleMarkdownCleaner

            cleaner = SimpleMarkdownCleaner()
            return cleaner.clean(markdown)

    async def _convert_with_fallback(self, file_path: Path, plan) -> ConversionResult:
        """Attempt conversion with fallback support."""
        errors = []

        # Try primary converter
        try:
            log.debug("Trying primary converter", converter=plan.primary_converter.name)
            return await plan.primary_converter.convert(file_path)
        except ConversionError as e:
            log.warning(
                "Primary converter failed",
                converter=plan.primary_converter.name,
                error=str(e),
            )
            errors.append(e)

        # Try fallback converter
        if plan.fallback_converter:
            try:
                log.debug(
                    "Trying fallback converter",
                    converter=plan.fallback_converter.name,
                )
                return await plan.fallback_converter.convert(file_path)
            except ConversionError as e:
                log.warning(
                    "Fallback converter failed",
                    converter=plan.fallback_converter.name,
                    error=str(e),
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

        Args:
            input_file: Original input file
            output_dir: Output directory
            result: Conversion result
            image_info_list: List of processed image info (with analysis results)

        Returns:
            Path to the output markdown file
        """
        from datetime import datetime

        import anyio

        # Determine output file path (preserve original extension for clarity)
        output_file = output_dir / f"{input_file.name}.md"

        # Handle conflicts
        output_file = self._resolve_conflict(output_file)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write markdown content
        async with await anyio.open_file(output_file, "w", encoding="utf-8") as f:
            await f.write(result.markdown)

        log.info("Output written", path=str(output_file))

        # Write images if any
        if result.images and self.settings.output.create_assets_subdir:
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Build a lookup for image analysis
            analysis_lookup: dict[str, ImageAnalysis | None] = {}
            if image_info_list:
                for info in image_info_list:
                    analysis_lookup[info.filename] = info.analysis

            for image in result.images:
                image_path = assets_dir / image.filename
                async with await anyio.open_file(image_path, "wb") as f:
                    await f.write(image.data)
                log.debug("Image written", path=str(image_path))

                # Write image description .md file if analyze_image_with_md is enabled
                if self.analyze_image_with_md:
                    analysis = analysis_lookup.get(image.filename)
                    if analysis:
                        md_content = self._generate_image_description_md(
                            image.filename, analysis, datetime.now(UTC)
                        )
                        md_path = assets_dir / f"{image.filename}.md"
                        async with await anyio.open_file(md_path, "w", encoding="utf-8") as f:
                            await f.write(md_content)
                        log.debug("Image description written", path=str(md_path))

        return output_file

    def _generate_image_description_md(
        self,
        filename: str,
        analysis: "ImageAnalysis",
        generated_at: Any,  # datetime.datetime
    ) -> str:
        """Generate markdown content for image description file.

        Args:
            filename: Image filename
            analysis: Image analysis result
            generated_at: Timestamp when the description was generated

        Returns:
            Markdown content for the image description file
        """

        lines = [
            "---",
            f"source_image: {filename}",
            f"image_type: {analysis.image_type}",
            f"generated_at: {generated_at.isoformat()}",
            "---",
            "",
            "# Image Description",
            "",
            "## Alt Text",
            "",
            analysis.alt_text,
            "",
            "## Detailed Description",
            "",
            analysis.detailed_description,
        ]

        # Add detected text if available
        if analysis.detected_text:
            lines.extend(
                [
                    "",
                    "## Detected Text",
                    "",
                    analysis.detected_text,
                ]
            )

        return "\n".join(lines) + "\n"

    def _resolve_conflict(self, output_path: Path) -> Path:
        """Resolve output file conflicts based on settings."""
        if not output_path.exists():
            return output_path

        strategy = self.settings.output.on_conflict

        if strategy == "overwrite":
            return output_path
        elif strategy == "skip":
            raise ConversionError(
                output_path,
                f"Output file already exists: {output_path}",
            )
        elif strategy == "rename":
            counter = 1
            stem = output_path.stem
            suffix = output_path.suffix
            parent = output_path.parent

            while True:
                new_path = parent / f"{stem}_{counter}{suffix}"
                if not new_path.exists():
                    return new_path
                counter += 1
        else:
            return output_path
