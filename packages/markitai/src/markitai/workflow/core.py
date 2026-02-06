"""Core document conversion logic.

This module provides the unified core conversion flow shared between
single-file and batch processing modes.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.constants import IMAGE_EXTENSIONS
from markitai.converter.base import FileFormat, detect_format, get_converter
from markitai.image import ImageProcessor
from markitai.security import (
    atomic_write_text,
    check_symlink_safety,
    escape_glob_pattern,
    validate_file_size,
)
from markitai.utils.paths import ensure_dir
from markitai.utils.text import format_error_message
from markitai.workflow.helpers import add_basic_frontmatter, merge_llm_usage

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.converter.base import ConvertResult
    from markitai.llm import LLMProcessor
    from markitai.workflow.single import ImageAnalysisResult


@dataclass
class ConversionContext:
    """Context for a document conversion operation.

    This dataclass holds all the input parameters and intermediate state
    for a single document conversion.
    """

    # Required inputs
    input_path: Path
    output_dir: Path
    config: MarkitaiConfig

    # Optional inputs
    actual_file: Path | None = None  # For pre-converted files (batch COM)
    shared_processor: LLMProcessor | None = None

    # Processing flags
    use_multiprocess_images: bool = False

    # Intermediate state (set during processing)
    converter: Any = None
    conversion_result: ConvertResult | None = None
    output_file: Path | None = None
    embedded_images_count: int = 0
    screenshots_count: int = 0

    # LLM tracking
    llm_cost: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    image_analysis: ImageAnalysisResult | None = None

    # Additional tracking (for caller use)
    duration: float = 0.0
    cache_hit: bool = False
    input_base_dir: Path | None = None  # For batch relative path calculation

    # Optional callback for stage completion (stage_name, duration)
    on_stage_complete: Callable[[str, float], None] | None = None

    @property
    def effective_input(self) -> Path:
        """Return actual file to process (handles pre-conversion)."""
        return self.actual_file if self.actual_file else self.input_path

    @property
    def is_preconverted(self) -> bool:
        """Check if this is a pre-converted file."""
        return self.actual_file is not None and self.actual_file != self.input_path


@dataclass
class ConversionStepResult:
    """Result of a conversion step."""

    success: bool
    error: str | None = None
    skip_reason: str | None = None


class DocumentConversionError(Exception):
    """Error during document conversion."""

    pass


class UnsupportedFormatError(DocumentConversionError):
    """Unsupported file format."""

    pass


class FileSizeError(DocumentConversionError):
    """File size exceeds limit."""

    pass


async def run_in_converter_thread(func, *args, **kwargs):
    """Run a converter function in the shared thread pool.

    Uses the shared ThreadPoolExecutor from utils.executor to avoid
    creating a new executor for each conversion.
    """
    from markitai.utils.executor import run_in_converter_thread as _run_in_thread

    return await _run_in_thread(func, *args, **kwargs)


def validate_and_detect_format(
    ctx: ConversionContext, max_size: int
) -> ConversionStepResult:
    """Validate file size and detect format.

    Args:
        ctx: Conversion context
        max_size: Maximum file size in bytes

    Returns:
        ConversionStepResult indicating success or failure
    """
    try:
        validate_file_size(ctx.input_path, max_size)
    except ValueError as e:
        return ConversionStepResult(success=False, error=str(e))

    fmt = detect_format(ctx.effective_input)
    if fmt == FileFormat.UNKNOWN:
        return ConversionStepResult(
            success=False, error=f"Unsupported file format: {ctx.input_path.suffix}"
        )

    ctx.converter = get_converter(ctx.effective_input, config=ctx.config)
    if ctx.converter is None:
        return ConversionStepResult(
            success=False, error=f"No converter available for format: {fmt.value}"
        )

    return ConversionStepResult(success=True)


def prepare_output_directory(ctx: ConversionContext) -> ConversionStepResult:
    """Create output directory with symlink safety check.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    try:
        check_symlink_safety(
            ctx.output_dir, allow_symlinks=ctx.config.output.allow_symlinks
        )
        ensure_dir(ctx.output_dir)
        return ConversionStepResult(success=True)
    except Exception as e:
        return ConversionStepResult(success=False, error=str(e))


async def convert_document(ctx: ConversionContext) -> ConversionStepResult:
    """Execute document conversion.

    Uses heavy task semaphore for memory-intensive formats (PPT, PDF with screenshots)
    to prevent OOM in concurrent environments.
    """
    from markitai.utils.executor import get_heavy_task_semaphore

    try:
        # Determine if this is a heavy conversion task
        heavy_extensions = {".ppt", ".pptx", ".pdf", ".doc", ".docx"}
        is_heavy = (
            ctx.input_path.suffix.lower() in heavy_extensions
            and ctx.config.screenshot.enabled
        ) or (ctx.input_path.suffix.lower() in {".ppt", ".doc", ".xls"})

        logger.debug(
            f"Converting {ctx.input_path.name}..." + (" [HEAVY]" if is_heavy else "")
        )

        if is_heavy:
            async with get_heavy_task_semaphore(ctx.config.batch.heavy_task_limit):
                ctx.conversion_result = await run_in_converter_thread(
                    ctx.converter.convert,
                    ctx.effective_input,
                    output_dir=ctx.output_dir,
                )
        else:
            ctx.conversion_result = await run_in_converter_thread(
                ctx.converter.convert,
                ctx.effective_input,
                output_dir=ctx.output_dir,
            )
        return ConversionStepResult(success=True)
    except Exception as e:
        return ConversionStepResult(success=False, error=f"Conversion failed: {e}")


def resolve_output_file(ctx: ConversionContext) -> ConversionStepResult:
    """Resolve output file path with conflict handling.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult - may have skip_reason if file exists
    """
    from markitai.utils.output import resolve_output_path

    base_output_file = ctx.output_dir / f"{ctx.input_path.name}.md"
    ctx.output_file = resolve_output_path(
        base_output_file, ctx.config.output.on_conflict
    )

    if ctx.output_file is None:
        logger.debug(f"[SKIP] Output exists: {base_output_file}")
        return ConversionStepResult(success=True, skip_reason="exists")

    return ConversionStepResult(success=True)


async def process_embedded_images(ctx: ConversionContext) -> ConversionStepResult:
    """Extract and process embedded images from markdown.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None:
        return ConversionStepResult(success=False, error="No conversion result")

    conversion_result = ctx.conversion_result
    image_processor = ImageProcessor(config=ctx.config.image)
    base64_images = await asyncio.to_thread(
        image_processor.extract_base64_images,
        conversion_result.markdown,
    )

    # Count screenshots from page images
    page_images = conversion_result.metadata.get("page_images", [])
    ctx.screenshots_count = len(page_images)

    # Count embedded images from two sources:
    # 1. Base64 images in markdown (will be processed below)
    # 2. Images already extracted by converter (e.g., PDF converter saves directly to assets)
    converter_images = len(conversion_result.images)
    ctx.embedded_images_count = len(base64_images) + converter_images

    if base64_images:
        logger.debug(f"Processing {len(base64_images)} embedded images...")

        # Use multiprocess for large batches if enabled
        from markitai.constants import DEFAULT_IMAGE_MULTIPROCESS_THRESHOLD

        if (
            ctx.use_multiprocess_images
            and len(base64_images) > DEFAULT_IMAGE_MULTIPROCESS_THRESHOLD
        ):
            image_result = await image_processor.process_and_save_multiprocess(
                base64_images,
                output_dir=ctx.output_dir,
                base_name=ctx.input_path.name,
            )
        else:
            image_result = await asyncio.to_thread(
                lambda: image_processor.process_and_save(
                    base64_images,
                    output_dir=ctx.output_dir,
                    base_name=ctx.input_path.name,
                )
            )

        # Update markdown with image paths using index mapping for correct replacement
        conversion_result.markdown = await asyncio.to_thread(
            lambda: image_processor.replace_base64_with_paths(
                conversion_result.markdown,
                image_result.saved_images,
                index_mapping=image_result.index_mapping,
            )
        )

        # Also update extracted_text in metadata if present (for PPTX+LLM mode)
        if "extracted_text" in conversion_result.metadata:
            conversion_result.metadata["extracted_text"] = await asyncio.to_thread(
                lambda: image_processor.replace_base64_with_paths(
                    conversion_result.metadata["extracted_text"],
                    image_result.saved_images,
                    index_mapping=image_result.index_mapping,
                )
            )

        # Update count: saved base64 images + converter-extracted images
        ctx.embedded_images_count = len(image_result.saved_images) + converter_images

    return ConversionStepResult(success=True)


def write_base_markdown(ctx: ConversionContext) -> ConversionStepResult:
    """Write base markdown file with basic frontmatter.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(
            success=False, error="Missing conversion result or output file"
        )

    base_md_content = add_basic_frontmatter(
        ctx.conversion_result.markdown, ctx.input_path.name
    )
    atomic_write_text(ctx.output_file, base_md_content)
    logger.debug(f"Written output: {ctx.output_file}")

    return ConversionStepResult(success=True)


def get_saved_images(ctx: ConversionContext) -> list[Path]:
    """Get list of saved images for this file from assets directory.

    Args:
        ctx: Conversion context

    Returns:
        List of image file paths
    """
    assets_dir = ctx.output_dir / "assets"
    if not assets_dir.exists():
        return []

    escaped_name = escape_glob_pattern(ctx.input_path.name)
    saved_images = list(assets_dir.glob(f"{escaped_name}*"))
    return [p for p in saved_images if p.suffix.lower() in IMAGE_EXTENSIONS]


def apply_alt_text_updates(
    llm_file: Path,
    image_analysis: Any,
) -> bool:
    """Apply alt text updates from image analysis to .llm.md file.

    This is called after document processing completes to update alt text
    in the .llm.md file with results from parallel image analysis.

    Args:
        llm_file: Path to the .llm.md file
        image_analysis: ImageAnalysisResult with analyzed images

    Returns:
        True if updates were applied, False otherwise
    """
    if not llm_file.exists() or image_analysis is None:
        return False

    try:
        llm_content = llm_file.read_text(encoding="utf-8")

        # Build combined pattern and replacement map for a single-pass substitution
        replacements: dict[str, str] = {}
        patterns: list[str] = []
        for asset in image_analysis.assets:
            asset_path = Path(asset.get("asset", ""))
            alt_text = asset.get("alt", "")
            if not alt_text or not asset_path.name:
                continue

            pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(asset_path.name)}\)"
            new_ref = f"![{alt_text}](assets/{asset_path.name})"
            patterns.append(pattern)
            replacements[asset_path.name] = new_ref

        if patterns:
            combined = re.compile("|".join(patterns))

            def replace_match(m: re.Match[str]) -> str:
                text = m.group(0)
                for name, ref in replacements.items():
                    if name in text:
                        return ref
                return text

            new_content = combined.sub(replace_match, llm_content)
            if new_content != llm_content:
                atomic_write_text(llm_file, new_content)
                logger.debug(f"Applied alt text updates to {llm_file}")
                return True

    except Exception as e:
        logger.warning(f"Failed to apply alt text updates: {e}")

    return False


async def process_with_vision_llm(
    ctx: ConversionContext,
) -> ConversionStepResult:
    """Process document with Vision LLM (screenshot mode).

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    page_images = ctx.conversion_result.metadata.get("page_images", [])
    if not page_images:
        return ConversionStepResult(success=True)

    logger.info(f"[LLM] {ctx.input_path.name}: Starting Screenshot+LLM processing")

    # Use shared processor or create new one
    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(
        ctx.config,
        processor=processor,
    )

    # Check for screenshot-only mode
    use_screenshot_only = ctx.config.screenshot.screenshot_only

    if use_screenshot_only:
        # Screenshot-only mode: extract content purely from screenshots
        (
            cleaned_content,
            frontmatter,
            enhance_cost,
            enhance_usage,
        ) = await workflow.extract_from_screenshots(
            page_images,
            source=ctx.input_path.name,
        )
    else:
        # Standard mode: use extracted text + screenshots for enhancement
        # Get extracted text (use markdown which has base64 replaced)
        extracted_text = ctx.conversion_result.markdown

        # Enhance with vision
        (
            cleaned_content,
            frontmatter,
            enhance_cost,
            enhance_usage,
        ) = await workflow.enhance_with_vision(
            extracted_text,
            page_images,
            source=ctx.input_path.name,
        )
    ctx.llm_cost += enhance_cost
    merge_llm_usage(ctx.llm_usage, enhance_usage)

    # Build final content with page image comments
    commented_images_str = ""
    if page_images:
        commented_images = [
            f"<!-- ![Page {img['page']}](screenshots/{img['name']}) -->"
            for img in sorted(page_images, key=lambda x: x.get("page", 0))
        ]
        commented_images_str = "\n\n<!-- Page images for reference -->\n" + "\n".join(
            commented_images
        )

    ctx.conversion_result.markdown = cleaned_content + commented_images_str

    # Strip any hallucinated base64 images
    image_processor = ImageProcessor(config=ctx.config.image)
    ctx.conversion_result.markdown = image_processor.strip_base64_images(
        ctx.conversion_result.markdown
    )

    # Validate image references
    assets_dir = ctx.output_dir / "assets"
    if assets_dir.exists():
        ctx.conversion_result.markdown = ImageProcessor.remove_nonexistent_images(
            ctx.conversion_result.markdown, assets_dir
        )

    # Write LLM version
    llm_output = ctx.output_file.with_suffix(".llm.md")
    llm_content = processor.format_llm_output(
        ctx.conversion_result.markdown, frontmatter, source=ctx.input_path.name
    )
    atomic_write_text(llm_output, llm_content)
    logger.info(f"Written LLM version: {llm_output}")

    return ConversionStepResult(success=True)


async def process_with_standard_llm(
    ctx: ConversionContext,
) -> ConversionStepResult:
    """Process document with standard LLM (no screenshots).

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    # Check if standalone image
    is_standalone_image = ctx.input_path.suffix.lower() in IMAGE_EXTENSIONS
    saved_images = get_saved_images(ctx)

    # Use shared processor or create new one
    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(
        ctx.config,
        processor=processor,
    )

    if is_standalone_image and saved_images:
        # Standalone image: only run image analysis
        logger.info(f"[LLM] {ctx.input_path.name}: Processing standalone image")
        (
            _,
            image_cost,
            image_usage,
            ctx.image_analysis,
        ) = await workflow.analyze_images(
            saved_images,
            ctx.conversion_result.markdown,
            ctx.output_file,
            ctx.input_path,
            concurrency_limit=ctx.config.llm.concurrency,
        )
        ctx.llm_cost += image_cost
        merge_llm_usage(ctx.llm_usage, image_usage)
    else:
        # Standard LLM processing
        logger.info(f"[LLM] {ctx.input_path.name}: Starting standard LLM processing")

        # Save original markdown for base .md file
        original_markdown = ctx.conversion_result.markdown

        # Check if image analysis should run
        should_analyze_images = (
            ctx.config.image.alt_enabled or ctx.config.image.desc_enabled
        ) and saved_images

        # Run document processing and image analysis in parallel
        # These are independent: doc processing writes .llm.md, image analysis generates descriptions
        if should_analyze_images:
            doc_task = workflow.process_document_with_llm(
                ctx.conversion_result.markdown,
                ctx.input_path.name,
                ctx.output_file,
            )
            img_task = workflow.analyze_images(
                saved_images,
                ctx.conversion_result.markdown,
                ctx.output_file,
                ctx.input_path,
                concurrency_limit=ctx.config.llm.concurrency,
            )

            # Execute in parallel
            doc_result, img_result = await asyncio.gather(doc_task, img_task)

            # Unpack results
            ctx.conversion_result.markdown, doc_cost, doc_usage = doc_result
            _, image_cost, image_usage, ctx.image_analysis = img_result

            ctx.llm_cost += doc_cost + image_cost
            merge_llm_usage(ctx.llm_usage, doc_usage)
            merge_llm_usage(ctx.llm_usage, image_usage)

            # Apply alt text updates to .llm.md after document processing completes
            # This ensures no race condition - .llm.md is guaranteed to exist
            if ctx.config.image.alt_enabled and ctx.image_analysis:
                llm_output = ctx.output_file.with_suffix(".llm.md")
                apply_alt_text_updates(llm_output, ctx.image_analysis)
        else:
            # Only document processing
            (
                ctx.conversion_result.markdown,
                doc_cost,
                doc_usage,
            ) = await workflow.process_document_with_llm(
                ctx.conversion_result.markdown,
                ctx.input_path.name,
                ctx.output_file,
            )
            ctx.llm_cost += doc_cost
            merge_llm_usage(ctx.llm_usage, doc_usage)

        # Re-write base .md with original markdown (without LLM alt text)
        base_md_content = add_basic_frontmatter(original_markdown, ctx.input_path.name)
        atomic_write_text(ctx.output_file, base_md_content)

    return ConversionStepResult(success=True)


async def analyze_embedded_images(ctx: ConversionContext) -> ConversionStepResult:
    """Analyze embedded images with LLM after Vision processing.

    Used in screenshot+LLM mode to also analyze embedded document images.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=True)

    if not (ctx.config.image.alt_enabled or ctx.config.image.desc_enabled):
        return ConversionStepResult(success=True)

    saved_images = get_saved_images(ctx)
    if not saved_images:
        return ConversionStepResult(success=True)

    # Filter out page/slide screenshots, only analyze embedded images
    import re

    page_pattern = re.compile(r"\.page\d+\.|\.slide\d+\.", re.IGNORECASE)
    embedded_images = [p for p in saved_images if not page_pattern.search(p.name)]

    if not embedded_images:
        return ConversionStepResult(success=True)

    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.single import SingleFileWorkflow

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    workflow = SingleFileWorkflow(
        ctx.config,
        processor=processor,
    )

    logger.info(
        f"[LLM] {ctx.input_path.name}: Analyzing {len(embedded_images)} embedded images"
    )

    (
        ctx.conversion_result.markdown,
        image_cost,
        image_usage,
        ctx.image_analysis,
    ) = await workflow.analyze_images(
        embedded_images,
        ctx.conversion_result.markdown,
        ctx.output_file,
        ctx.input_path,
        concurrency_limit=ctx.config.llm.concurrency,
    )
    ctx.llm_cost += image_cost
    merge_llm_usage(ctx.llm_usage, image_usage)

    return ConversionStepResult(success=True)


async def convert_document_core(
    ctx: ConversionContext,
    max_document_size: int,
) -> ConversionStepResult:
    """Core document conversion pipeline.

    This function implements the unified conversion logic shared between
    single-file and batch processing modes.

    The pipeline:
    1. Validate file size and detect format
    2. Prepare output directory
    3. Execute document conversion
    4. Resolve output file path (with conflict handling)
    5. Process embedded images
    6. Write base markdown file
    7. LLM processing (if enabled):
       - Vision mode (with page screenshots)
       - Standard mode (no screenshots)
       - Embedded image analysis

    Args:
        ctx: Conversion context with all inputs and state
        max_document_size: Maximum allowed document size in bytes

    Returns:
        ConversionStepResult indicating overall success or failure
    """
    # Step 1: Validate and detect format
    result = validate_and_detect_format(ctx, max_document_size)
    if not result.success:
        return result

    # Step 2: Prepare output directory
    result = prepare_output_directory(ctx)
    if not result.success:
        return result

    # Step 3: Execute conversion
    result = await convert_document(ctx)
    if not result.success:
        return result

    # Step 4: Resolve output file
    result = resolve_output_file(ctx)
    if not result.success or result.skip_reason:
        return result

    # Step 5: Process embedded images
    result = await process_embedded_images(ctx)
    if not result.success:
        return result

    # Step 6: Write base markdown
    result = write_base_markdown(ctx)
    if not result.success:
        return result

    # Step 7: LLM processing (if enabled)
    if ctx.config.llm.enabled and ctx.conversion_result is not None:
        # Ensure shared processor exists for all LLM operations
        # This is critical for:
        # 1. Sharing semaphore (concurrency control)
        # 2. Sharing Router instances (avoid duplicate creation)
        # 3. Sharing cache connections
        if ctx.shared_processor is None:
            from markitai.workflow.helpers import create_llm_processor

            ctx.shared_processor = create_llm_processor(ctx.config)

        page_images = ctx.conversion_result.metadata.get("page_images", [])
        has_page_images = len(page_images) > 0

        if has_page_images:
            # Vision mode with screenshots - run vision LLM and embedded image
            # analysis in parallel for better performance
            vision_task = asyncio.create_task(process_with_vision_llm(ctx))
            embed_task = asyncio.create_task(analyze_embedded_images(ctx))

            results = await asyncio.gather(
                vision_task, embed_task, return_exceptions=True
            )
            vision_result_raw, embed_result_raw = results

            # Check vision result (critical)
            if isinstance(vision_result_raw, BaseException):
                return ConversionStepResult(
                    success=False,
                    error=f"Vision LLM failed: {format_error_message(vision_result_raw)}",
                )
            vision_result: ConversionStepResult = vision_result_raw
            if not vision_result.success:
                return vision_result

            # Check embed result (non-critical, log warning)
            if isinstance(embed_result_raw, BaseException):
                logger.warning(
                    f"Embedded image analysis failed: "
                    f"{format_error_message(embed_result_raw)}"
                )
            else:
                embed_result: ConversionStepResult = embed_result_raw
                if not embed_result.success:
                    logger.warning(
                        f"Embedded image analysis failed: {embed_result.error}"
                    )

            # Apply alt text updates to .llm.md after both tasks complete
            # This fixes the missing alt text replacement in Vision mode
            if ctx.config.image.alt_enabled and ctx.image_analysis and ctx.output_file:
                llm_output = ctx.output_file.with_suffix(".llm.md")
                apply_alt_text_updates(llm_output, ctx.image_analysis)
        else:
            # Standard LLM mode
            result = await process_with_standard_llm(ctx)
            if not result.success:
                return result

    return ConversionStepResult(success=True)
