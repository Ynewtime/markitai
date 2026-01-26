"""Single file workflow processing."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.security import atomic_write_text

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.llm import LLMProcessor


@dataclass
class ImageAnalysisResult:
    """Result of image analysis for a single source file."""

    source_file: str
    assets: list[dict[str, Any]]


@dataclass
class WorkflowResult:
    """Result of processing a file through the workflow."""

    markdown: str
    llm_cost: float = 0.0
    llm_usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    image_analysis: ImageAnalysisResult | None = None


class SingleFileWorkflow:
    """Workflow for processing a single file with LLM enhancement.

    This class encapsulates the LLM processing logic extracted from cli.py,
    including document processing, image analysis, and vision enhancement.
    """

    def __init__(
        self,
        config: MarkitaiConfig,
        processor: LLMProcessor | None = None,
        project_dir: Path | None = None,
        no_cache: bool = False,
        no_cache_patterns: list[str] | None = None,
    ) -> None:
        """Initialize workflow.

        Args:
            config: Markitai configuration
            processor: Optional shared LLMProcessor (created if not provided)
            project_dir: Optional project directory for project-level cache
            no_cache: If True, skip reading from cache but still write results
            no_cache_patterns: List of glob patterns to skip cache for specific files
        """
        self.config = config
        self._processor = processor
        self._project_dir = project_dir
        self._no_cache = no_cache
        self._no_cache_patterns = no_cache_patterns
        self._llm_cost = 0.0
        self._llm_usage: dict[str, dict[str, Any]] = {}

    @property
    def processor(self) -> LLMProcessor:
        """Get or create LLM processor."""
        if self._processor is None:
            from markitai.workflow.helpers import create_llm_processor

            # Create a temporary config with the no_cache settings
            # This is needed because SingleFileWorkflow stores these separately
            temp_config = self.config.model_copy()
            temp_config.cache.no_cache = self._no_cache
            temp_config.cache.no_cache_patterns = self._no_cache_patterns or []

            self._processor = create_llm_processor(
                temp_config, project_dir=self._project_dir
            )
        return self._processor

    def _merge_usage(self, usage: dict[str, dict[str, Any]]) -> None:
        """Merge usage statistics into workflow totals."""
        from markitai.workflow.helpers import merge_llm_usage

        merge_llm_usage(self._llm_usage, usage)

    async def process_document_with_llm(
        self,
        markdown: str,
        source: str,
        output_file: Path,
        page_images: list[dict] | None = None,
    ) -> tuple[str, float, dict[str, dict[str, Any]]]:
        """Process markdown with LLM (clean + frontmatter).

        Args:
            markdown: Markdown content to process
            source: Source file name
            output_file: Output file path for .llm.md
            page_images: Optional list of page image info dicts

        Returns:
            Tuple of (markdown, cost_usd, llm_usage)
        """
        try:
            cleaned, frontmatter = await self.processor.process_document(
                markdown, source
            )

            # Write LLM version
            llm_output = output_file.with_suffix(".llm.md")
            llm_content = self.processor.format_llm_output(cleaned, frontmatter)

            # Append commented image links if provided
            if page_images:
                commented_images = [
                    f"<!-- ![Page {img['page']}](screenshots/{img['name']}) -->"
                    for img in sorted(page_images, key=lambda x: x.get("page", 0))
                ]
                llm_content += "\n\n<!-- Page images for reference -->\n" + "\n".join(
                    commented_images
                )

            atomic_write_text(llm_output, llm_content)
            logger.info(f"Written LLM version: {llm_output}")

            # Use context-based tracking for accurate per-file usage in concurrent scenarios
            cost = self.processor.get_context_cost(source)
            usage = self.processor.get_context_usage(source)
            return markdown, cost, usage

        except Exception as e:
            logger.warning(f"LLM processing failed: {e}")
            return markdown, 0.0, {}

    async def analyze_images(
        self,
        image_paths: list[Path],
        markdown: str,
        output_file: Path,
        input_path: Path | None = None,
        concurrency_limit: int | None = None,
    ) -> tuple[str, float, dict[str, dict[str, Any]], ImageAnalysisResult | None]:
        """Analyze images with LLM Vision.

        Args:
            image_paths: List of image file paths
            markdown: Original markdown content
            output_file: Output markdown file path
            input_path: Source input file path
            concurrency_limit: Max concurrent requests

        Returns:
            Tuple of (updated markdown, cost_usd, llm_usage, image_analysis_result)
        """
        from markitai.llm import ImageAnalysis
        from markitai.workflow.helpers import detect_language

        alt_enabled = self.config.image.alt_enabled
        desc_enabled = self.config.image.desc_enabled

        # Use unique context for accurate per-file usage tracking in concurrent scenarios
        source_path = (
            str(input_path.resolve()) if input_path else str(output_file.resolve())
        )
        context = f"{source_path}:images"

        try:
            # Detect document language from markdown content
            language = detect_language(markdown)

            async def analyze_single_image(
                image_path: Path,
            ) -> tuple[Path, ImageAnalysis | None, str]:
                """Analyze a single image."""
                timestamp = datetime.now().astimezone().isoformat()
                try:
                    analysis = await self.processor.analyze_image(
                        image_path, language=language, context=context
                    )
                    return image_path, analysis, timestamp
                except Exception as e:
                    logger.warning(f"Failed to analyze image {image_path.name}: {e}")
                    return image_path, None, timestamp

            # Queue-based analysis with concurrency limit
            logger.info(f"Analyzing {len(image_paths)} images...")
            limit = (
                concurrency_limit
                if concurrency_limit is not None
                else self.config.llm.concurrency
            )
            worker_count = min(len(image_paths), max(1, limit))
            queue: asyncio.Queue[Path] = asyncio.Queue()
            for image_path in image_paths:
                queue.put_nowait(image_path)

            results_map: dict[Path, tuple[Path, ImageAnalysis | None, str]] = {}

            async def worker() -> None:
                while True:
                    try:
                        image_path = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    result = await analyze_single_image(image_path)
                    results_map[image_path] = result
                    queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
            await queue.join()
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

            results = [results_map[p] for p in image_paths if p in results_map]

            # Collect asset descriptions for JSON output
            asset_descriptions: list[dict[str, Any]] = []

            # Process results
            for image_path, analysis, timestamp in results:
                # Use default values if analysis failed
                # This ensures the image is still recorded in images.json
                if analysis is None:
                    analysis_caption = "Image"
                    analysis_desc = "Image analysis failed"
                    analysis_text = ""
                    analysis_usage: dict[str, Any] = {}
                else:
                    analysis_caption = analysis.caption
                    analysis_desc = analysis.description
                    analysis_text = analysis.extracted_text or ""
                    analysis_usage = analysis.llm_usage or {}

                # Collect for JSON output (if desc_enabled)
                if desc_enabled:
                    asset_descriptions.append(
                        {
                            "asset": str(image_path.resolve()),
                            "alt": analysis_caption,
                            "desc": analysis_desc,
                            "text": analysis_text,
                            "llm_usage": analysis_usage,
                            "created": timestamp,
                        }
                    )

                # Update alt text in markdown (if alt_enabled)
                if alt_enabled:
                    old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                    new_ref = f"![{analysis_caption}](assets/{image_path.name})"
                    markdown = re.sub(old_pattern, new_ref, markdown)

            # Check if this is a standalone image file
            from markitai.constants import IMAGE_EXTENSIONS

            is_standalone_image = (
                input_path is not None
                and input_path.suffix.lower() in IMAGE_EXTENSIONS
                and len(image_paths) == 1
            )

            # Update/create .llm.md file
            llm_output = output_file.with_suffix(".llm.md")
            if is_standalone_image and results and results[0][1] is not None:
                # For standalone images, create rich formatted content with frontmatter
                from markitai.utils.text import normalize_markdown_whitespace
                from markitai.workflow.helpers import format_standalone_image_markdown

                # input_path is guaranteed non-None by is_standalone_image check
                assert input_path is not None
                _, analysis, _ = results[0]
                if analysis:
                    rich_content = format_standalone_image_markdown(
                        input_path,
                        analysis,
                        f"assets/{input_path.name}",
                        include_frontmatter=True,
                    )
                    rich_content = normalize_markdown_whitespace(rich_content)
                    atomic_write_text(llm_output, rich_content)
                    logger.info(f"Written LLM version: {llm_output}")
            elif alt_enabled:
                # NOTE: Alt text update moved to caller (workflow/core.py) to avoid race condition.
                # The caller will apply alt text updates after document processing completes.
                # See P0-4 fix: image analysis no longer waits for .llm.md file.
                pass

            # Build analysis result for caller to aggregate
            analysis_result: ImageAnalysisResult | None = None
            if desc_enabled and asset_descriptions:
                source_path = (
                    str(input_path.resolve()) if input_path else output_file.stem
                )
                analysis_result = ImageAnalysisResult(
                    source_file=source_path,
                    assets=asset_descriptions,
                )

            # Use context-based tracking for accurate per-file usage in concurrent scenarios
            return (
                markdown,
                self.processor.get_context_cost(context),
                self.processor.get_context_usage(context),
                analysis_result,
            )

        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return markdown, 0.0, {}, None

    async def enhance_with_vision(
        self,
        extracted_text: str,
        page_images: list[dict],
        source: str = "document",
    ) -> tuple[str, str, float, dict[str, dict[str, Any]]]:
        """Enhance document by combining extracted text with page images.

        Args:
            extracted_text: Text extracted by pymupdf4llm/markitdown
            page_images: List of page image info dicts with 'path' key
            source: Source file name for logging context

        Returns:
            Tuple of (cleaned_markdown, frontmatter_yaml, cost_usd, llm_usage)
        """
        try:
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
            (
                cleaned_content,
                frontmatter,
            ) = await self.processor.enhance_document_complete(
                extracted_text, image_paths, source=source
            )

            # Use context-based tracking for accurate per-file usage in concurrent scenarios
            return (
                cleaned_content,
                frontmatter,
                self.processor.get_context_cost(source),
                self.processor.get_context_usage(source),
            )

        except Exception as e:
            logger.warning(f"Document enhancement failed: {e}")
            basic_frontmatter = f"title: {source}\nsource: {source}"
            return extracted_text, basic_frontmatter, 0.0, {}
