"""Single file workflow processing."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markit.security import atomic_write_text

if TYPE_CHECKING:
    from markit.config import MarkitConfig
    from markit.llm import LLMProcessor


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
        config: MarkitConfig,
        processor: LLMProcessor | None = None,
    ) -> None:
        """Initialize workflow.

        Args:
            config: Markit configuration
            processor: Optional shared LLMProcessor (created if not provided)
        """
        self.config = config
        self._processor = processor
        self._llm_cost = 0.0
        self._llm_usage: dict[str, dict[str, Any]] = {}

    @property
    def processor(self) -> LLMProcessor:
        """Get or create LLM processor."""
        if self._processor is None:
            from markit.llm import LLMProcessor

            self._processor = LLMProcessor(self.config.llm, self.config.prompts)
        return self._processor

    def _merge_usage(self, usage: dict[str, dict[str, Any]]) -> None:
        """Merge usage statistics into workflow totals."""
        from markit.workflow.helpers import merge_llm_usage

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

            cost = self.processor.get_total_cost()
            usage = self.processor.get_usage()
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
        from markit.llm import ImageAnalysis
        from markit.workflow.helpers import detect_language

        alt_enabled = self.config.image.alt_enabled
        desc_enabled = self.config.image.desc_enabled

        try:
            # Detect document language from markdown content
            language = detect_language(markdown)

            async def analyze_single_image(
                image_path: Path,
            ) -> tuple[Path, ImageAnalysis | None, str]:
                """Analyze a single image."""
                timestamp = datetime.now(UTC).isoformat()
                try:
                    logger.debug(f"Analyzing image: {image_path.name}")
                    analysis = await self.processor.analyze_image(
                        image_path, language=language
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
                if analysis is None:
                    continue

                # Collect for JSON output (if desc_enabled)
                if desc_enabled:
                    asset_descriptions.append(
                        {
                            "asset": str(image_path.resolve()),
                            "alt": analysis.caption,
                            "desc": analysis.description,
                            "text": analysis.extracted_text or "",
                            "model": analysis.model or "",
                            "created": timestamp,
                        }
                    )

                # Update alt text in markdown (if alt_enabled)
                if alt_enabled:
                    old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                    new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                    markdown = re.sub(old_pattern, new_ref, markdown)

            # Update .llm.md file with alt text changes (if alt_enabled)
            if alt_enabled:
                llm_output = output_file.with_suffix(".llm.md")
                if llm_output.exists():
                    llm_content = llm_output.read_text(encoding="utf-8")
                    for image_path, analysis, _ in results:
                        if analysis is None:
                            continue
                        old_pattern = (
                            rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                        )
                        new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                        llm_content = re.sub(old_pattern, new_ref, llm_content)
                    atomic_write_text(llm_output, llm_content)

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

            return (
                markdown,
                self.processor.get_total_cost(),
                self.processor.get_usage(),
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

            return (
                cleaned_content,
                frontmatter,
                self.processor.get_total_cost(),
                self.processor.get_usage(),
            )

        except Exception as e:
            logger.warning(f"Document enhancement failed: {e}")
            basic_frontmatter = f"title: {source}\nsource: {source}"
            return extracted_text, basic_frontmatter, 0.0, {}
