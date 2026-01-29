"""LLM processing helpers for CLI.

This module contains functions for LLM-based document and image processing.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.config import MarkitaiConfig
from markitai.constants import DEFAULT_MAX_IMAGES_PER_BATCH, IMAGE_EXTENSIONS
from markitai.image import ImageProcessor
from markitai.security import atomic_write_text
from markitai.utils.text import format_error_message
from markitai.workflow.helpers import (
    create_llm_processor,
)
from markitai.workflow.helpers import (
    detect_language as _detect_language,
)
from markitai.workflow.single import ImageAnalysisResult

if TYPE_CHECKING:
    from markitai.llm import ImageAnalysis, LLMProcessor


async def process_with_llm(
    markdown: str,
    source: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    page_images: list[dict] | None = None,
    processor: LLMProcessor | None = None,
    original_markdown: str | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """Process markdown with LLM and write enhanced version to .llm.md file.

    The LLM-enhanced content is written to output_file with .llm.md suffix.
    Returns the original markdown unchanged for use in base .md file.

    Args:
        markdown: Markdown content to process
        source: Source file name (used as LLM context identifier)
        cfg: Configuration with LLM and prompt settings
        output_file: Base output file path (.llm.md suffix added automatically)
        page_images: Optional page image info for adding commented references
        processor: Optional shared LLMProcessor (created if not provided)
        original_markdown: Original markdown for detecting hallucinated images

    Returns:
        Tuple of (original_markdown, cost_usd, llm_usage):
        - original_markdown: Input markdown unchanged (for base .md file)
        - cost_usd: LLM API cost for this file
        - llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}

    Side Effects:
        Writes LLM-enhanced content to {output_file}.llm.md
    """
    try:
        if processor is None:
            processor = create_llm_processor(cfg)

        cleaned, frontmatter = await processor.process_document(markdown, source)

        # Remove hallucinated image URLs (URLs that don't exist in original)
        original_for_comparison = original_markdown if original_markdown else markdown
        cleaned = ImageProcessor.remove_hallucinated_images(
            cleaned, original_for_comparison
        )

        # Validate local image references - remove non-existent assets
        assets_dir = output_file.parent / "assets"
        if assets_dir.exists():
            cleaned = ImageProcessor.remove_nonexistent_images(cleaned, assets_dir)

        # Write LLM version
        llm_output = output_file.with_suffix(".llm.md")
        llm_content = processor.format_llm_output(cleaned, frontmatter, source=source)

        # Check if page_images comments already exist in content
        # process_document's placeholder protection should preserve them
        # Append missing page image comments
        if page_images:
            page_header = "<!-- Page images for reference -->"
            has_page_images_header = page_header in llm_content

            # Build the complete page images section
            commented_images = [
                f"<!-- ![Page {img['page']}](screenshots/{img['name']}) -->"
                for img in sorted(page_images, key=lambda x: x.get("page", 0))
            ]

            if not has_page_images_header:
                # No header exists, add complete section
                llm_content += "\n\n" + page_header + "\n" + "\n".join(commented_images)
            else:
                # Header exists, check for missing page comments
                for comment in commented_images:
                    # Check if this specific page is already referenced
                    page_match = re.search(r"!\[Page\s+(\d+)\]", comment)
                    if page_match:
                        page_num = page_match.group(1)
                        # Look for this page number in any form (commented or not)
                        if not re.search(rf"!\[Page\s+{page_num}\]", llm_content):
                            # Append missing page comment
                            llm_content = llm_content.rstrip() + "\n" + comment

        atomic_write_text(llm_output, llm_content)
        logger.info(f"Written LLM version: {llm_output}")

        # Get usage for THIS file only, not global cumulative usage
        cost = processor.get_context_cost(source)
        usage = processor.get_context_usage(source)
        return markdown, cost, usage  # Return original for base .md file

    except Exception as e:
        logger.error(f"LLM processing failed: {format_error_message(e)}")
        return markdown, 0.0, {}


def format_standalone_image_markdown(
    input_path: Path,
    analysis: ImageAnalysis,
    image_ref_path: str,
    include_frontmatter: bool = False,
) -> str:
    """Format analysis results for a standalone image file.

    This is a wrapper that delegates to workflow/helpers.format_standalone_image_markdown.

    Args:
        input_path: Original image file path
        analysis: ImageAnalysis result with caption, description, extracted_text
        image_ref_path: Relative path for image reference
        include_frontmatter: Whether to include YAML frontmatter

    Returns:
        Formatted markdown string
    """
    from markitai.workflow.helpers import (
        format_standalone_image_markdown as _format_standalone_image_markdown,
    )

    return _format_standalone_image_markdown(
        input_path, analysis, image_ref_path, include_frontmatter
    )


async def analyze_images_with_llm(
    image_paths: list[Path],
    markdown: str,
    output_file: Path,
    cfg: MarkitaiConfig,
    input_path: Path | None = None,
    concurrency_limit: int | None = None,  # noqa: ARG001 - kept for API compat
    processor: LLMProcessor | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]], ImageAnalysisResult | None]:
    """Analyze images with LLM Vision using batch processing.

    Uses batch analysis to reduce LLM calls (10 images per call instead of 1).

    Behavior controlled by config:
    - alt_enabled: Update alt text in markdown
    - desc_enabled: Collect asset descriptions (caller writes JSON)

    Args:
        image_paths: List of image file paths
        markdown: Original markdown content
        output_file: Output markdown file path
        cfg: Configuration
        input_path: Source input file path (for absolute path in JSON)
        concurrency_limit: Deprecated - concurrency controlled by processor.semaphore
        processor: Optional shared LLMProcessor (created if not provided)

    Returns:
        Tuple of (updated_markdown, cost_usd, llm_usage, image_analysis_result):
        - updated_markdown: Markdown with updated alt text (if alt_enabled)
        - cost_usd: LLM API cost for image analysis
        - llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}
        - image_analysis_result: Analysis data for JSON output (None if desc_enabled=False)
    """
    from datetime import datetime

    alt_enabled = cfg.image.alt_enabled
    desc_enabled = cfg.image.desc_enabled

    try:
        if processor is None:
            processor = create_llm_processor(cfg)

        # Use unique context for image analysis to track usage separately from doc processing
        # Format: "full_path:images" ensures isolation even for files with same name in different dirs
        # This prevents usage from concurrent files being mixed together
        source_path = (
            str(input_path.resolve()) if input_path else str(output_file.resolve())
        )
        context = f"{source_path}:images"

        # Detect document language from markdown content
        language = _detect_language(markdown)

        # Use batch analysis
        logger.info(f"Analyzing {len(image_paths)} images in batches...")
        analyses = await processor.analyze_images_batch(
            image_paths,
            language=language,
            max_images_per_batch=DEFAULT_MAX_IMAGES_PER_BATCH,
            context=context,
        )

        timestamp = datetime.now().astimezone().isoformat()

        # Collect asset descriptions for JSON output
        asset_descriptions: list[dict[str, Any]] = []

        # Check if this is a standalone image file
        is_standalone_image = (
            input_path is not None
            and input_path.suffix.lower() in IMAGE_EXTENSIONS
            and len(image_paths) == 1
        )

        # Process results (analyses is in same order as image_paths)
        results: list[tuple[Path, ImageAnalysis | None, str]] = []
        for image_path, analysis in zip(image_paths, analyses):
            results.append((image_path, analysis, timestamp))

            # Collect for JSON output (if desc_enabled)
            if desc_enabled:
                asset_descriptions.append(
                    {
                        "asset": str(image_path.resolve()),
                        "alt": analysis.caption,
                        "desc": analysis.description,
                        "text": analysis.extracted_text or "",
                        "llm_usage": analysis.llm_usage or {},
                        "created": timestamp,
                    }
                )

            # Update alt text in markdown (if alt_enabled)
            if alt_enabled and not is_standalone_image:
                old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                markdown = re.sub(old_pattern, new_ref, markdown)

        # Update .llm.md file
        llm_output = output_file.with_suffix(".llm.md")
        if is_standalone_image and results and results[0][1] is not None:
            # For standalone images, write the rich formatted content with frontmatter
            assert input_path is not None
            _, analysis, _ = results[0]
            if analysis:
                rich_content = format_standalone_image_markdown(
                    input_path,
                    analysis,
                    f"assets/{input_path.name}",
                    include_frontmatter=True,
                )
                # Normalize whitespace (ensure headers have blank lines before/after)
                from markitai.utils.text import normalize_markdown_whitespace

                rich_content = normalize_markdown_whitespace(rich_content)
                atomic_write_text(llm_output, rich_content)
        elif alt_enabled:
            # For other files, update alt text in .llm.md
            # Wait for .llm.md file to exist (it's written by parallel doc processing)
            max_wait_seconds = 120  # Max wait time
            poll_interval = 0.5  # Check every 0.5 seconds
            waited = 0.0
            while not llm_output.exists() and waited < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                waited += poll_interval

            if llm_output.exists():
                llm_content = llm_output.read_text(encoding="utf-8")
                for image_path, analysis, _ in results:
                    if analysis is None:
                        continue
                    old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                    new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                    llm_content = re.sub(old_pattern, new_ref, llm_content)
                atomic_write_text(llm_output, llm_content)
            else:
                logger.warning(
                    f"Skipped alt text update: {llm_output} not created within {max_wait_seconds}s"
                )

        # Build analysis result for caller to aggregate
        analysis_result: ImageAnalysisResult | None = None
        if desc_enabled and asset_descriptions:
            source_path = str(input_path.resolve()) if input_path else output_file.stem
            analysis_result = ImageAnalysisResult(
                source_file=source_path,
                assets=asset_descriptions,
            )

        # Get usage for THIS file only using context-based tracking
        # This is concurrency-safe: only includes LLM calls tagged with this context
        incremental_usage = processor.get_context_usage(context)
        incremental_cost = processor.get_context_cost(context)

        return (
            markdown,
            incremental_cost,
            incremental_usage,
            analysis_result,
        )

    except Exception as e:
        logger.error(f"Image analysis failed: {format_error_message(e)}")
        return markdown, 0.0, {}, None


async def enhance_document_with_vision(
    extracted_text: str,
    page_images: list[dict],
    cfg: MarkitaiConfig,
    source: str = "document",
    processor: LLMProcessor | None = None,
) -> tuple[str, str, float, dict[str, dict[str, Any]]]:
    """Enhance document by combining extracted text with page images.

    This is used for OCR+LLM mode where we have:
    1. Text extracted programmatically (pymupdf4llm/markitdown) - accurate content
    2. Page images - visual reference for layout/structure

    The LLM uses both to produce optimized markdown + frontmatter.

    Args:
        extracted_text: Text extracted by pymupdf4llm/markitdown
        page_images: List of page image info dicts with 'path' key
        cfg: Configuration
        source: Source file name for logging context
        processor: Optional shared LLMProcessor (created if not provided)

    Returns:
        Tuple of (cleaned_markdown, frontmatter_yaml, cost_usd, llm_usage)
    """
    try:
        if processor is None:
            processor = create_llm_processor(cfg)

        # Sort images by page number
        def get_page_num(img_info: dict) -> int:
            return img_info.get("page", 0)

        sorted_images = sorted(page_images, key=get_page_num)

        # Convert to Path list
        image_paths = [Path(img["path"]) for img in sorted_images]

        logger.info(
            f"[START] {source}: Enhancing with {len(image_paths)} page images..."
        )

        # Call the combined enhancement method (clean + frontmatter)
        cleaned_content, frontmatter = await processor.enhance_document_complete(
            extracted_text, image_paths, source=source
        )

        # Get usage for THIS file only, not global cumulative usage
        return (
            cleaned_content,
            frontmatter,
            processor.get_context_cost(source),
            processor.get_context_usage(source),
        )

    except Exception as e:
        logger.error(f"Document enhancement failed: {format_error_message(e)}")
        # Return original text with basic frontmatter as fallback
        basic_frontmatter = f"title: {source}\nsource: {source}"
        return extracted_text, basic_frontmatter, 0.0, {}


# Backward compatibility alias
_format_standalone_image_markdown = format_standalone_image_markdown
