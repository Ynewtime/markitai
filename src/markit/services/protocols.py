"""Protocol definitions for markit services.

These protocols define the interfaces that service classes must implement,
enabling dependency injection and easier testing through duck typing.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from datetime import datetime

    from markit.converters.base import ConversionResult, ExtractedImage
    from markit.image.analyzer import ImageAnalysis, ImageAnalyzer
    from markit.image.compressor import CompressedImage
    from markit.llm.base import LLMTaskResultWithStats
    from markit.llm.enhancer import MarkdownEnhancer
    from markit.llm.manager import ProviderManager


class ImageProcessingServiceProtocol(Protocol):
    """Protocol for image processing service."""

    async def optimize_images_parallel(
        self,
        images: list["ExtractedImage"],
        input_file: Path,
    ) -> tuple[list["ExtractedImage"], dict[str, str | None]]:
        """Process images in parallel: format convert, deduplicate, compress.

        Args:
            images: List of extracted images to process
            input_file: Original input file path for filename generation

        Returns:
            Tuple of (processed_unique_images, filename_map: old -> new, None if failed)
        """
        ...

    async def process_images(
        self,
        images: list["ExtractedImage"],
        input_file: Path,
        image_analyzer: "ImageAnalyzer | None" = None,
    ) -> tuple[list["ExtractedImage"], list[Any], str]:
        """Full image processing pipeline including optional LLM analysis.

        Args:
            images: List of extracted images to process
            input_file: Original input file path
            image_analyzer: Optional analyzer for LLM-based image analysis

        Returns:
            Tuple of (processed_images, image_info_list, updated_markdown)
        """
        ...

    def prepare_for_analysis(
        self,
        processed_images: list["ExtractedImage"],
    ) -> list["CompressedImage"]:
        """Prepare processed images for LLM analysis.

        Args:
            processed_images: List of processed images

        Returns:
            List of CompressedImage objects ready for analysis
        """
        ...


class LLMOrchestratorProtocol(Protocol):
    """Protocol for LLM orchestration service."""

    async def initialize(
        self,
        required_capabilities: list[str] | None = None,
        lazy: bool = True,
    ) -> None:
        """Initialize the LLM providers.

        Args:
            required_capabilities: List of required capabilities (e.g., ["text", "vision"])
            lazy: If True, defer network validation until first use
        """
        ...

    async def warmup(
        self,
        llm_enabled: bool = False,
        analyze_image: bool = False,
    ) -> None:
        """Warmup LLM providers by forcing network validation.

        Args:
            llm_enabled: Whether text enhancement is enabled
            analyze_image: Whether image analysis is enabled
        """
        ...

    async def get_enhancer(self) -> "MarkdownEnhancer":
        """Get or create the Markdown enhancer.

        Returns:
            MarkdownEnhancer instance
        """
        ...

    async def get_image_analyzer(self) -> "ImageAnalyzer":
        """Get or create the image analyzer.

        Returns:
            ImageAnalyzer instance
        """
        ...

    async def get_provider_manager(
        self,
        required_capabilities: list[str] | None = None,
        lazy: bool = True,
    ) -> "ProviderManager":
        """Get or create the LLM provider manager.

        Args:
            required_capabilities: Override auto-detected capabilities
            lazy: If True, only load configs without network validation

        Returns:
            ProviderManager instance
        """
        ...

    def has_capability(self, capability: str) -> bool:
        """Check if any provider has the specified capability.

        Args:
            capability: Capability to check (e.g., "text", "vision")

        Returns:
            True if capability is available
        """
        ...

    async def create_image_analysis_task(
        self,
        image: "CompressedImage",
        return_stats: bool = True,
    ) -> "ImageAnalysis | LLMTaskResultWithStats":
        """Create and execute an image analysis task.

        Args:
            image: Compressed image to analyze
            return_stats: If True, return statistics with result

        Returns:
            ImageAnalysis or LLMTaskResultWithStats containing the analysis
        """
        ...

    async def create_enhancement_task(
        self,
        markdown: str,
        source_file: Path,
        return_stats: bool = True,
    ) -> "str | LLMTaskResultWithStats":
        """Create and execute a markdown enhancement task.

        Args:
            markdown: Markdown content to enhance
            source_file: Source file path for context
            return_stats: If True, return statistics with result

        Returns:
            Enhanced markdown string or LLMTaskResultWithStats
        """
        ...

    async def create_llm_tasks(
        self,
        images_for_analysis: list["CompressedImage"],
        markdown_content: str,
        input_file: Path,
        llm_enabled: bool,
        analyze_image: bool,
    ) -> list["Coroutine[Any, Any, Any]"]:
        """Create LLM task coroutines without executing them.

        Args:
            images_for_analysis: Images prepared for analysis
            markdown_content: Markdown content for enhancement
            input_file: Original input file
            llm_enabled: Whether to create enhancement task
            analyze_image: Whether to create analysis tasks

        Returns:
            List of coroutines (not awaited) for LLM tasks
        """
        ...


class OutputManagerProtocol(Protocol):
    """Protocol for output management service."""

    async def write_output(
        self,
        input_file: Path,
        output_dir: Path,
        result: "ConversionResult",
        image_info_list: list[Any] | None = None,
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
        ...

    def resolve_conflict(self, output_path: Path) -> Path:
        """Resolve output file conflicts based on settings.

        Args:
            output_path: Desired output path

        Returns:
            Resolved output path (may be renamed)
        """
        ...

    def generate_image_description_md(
        self,
        filename: str,
        analysis: "ImageAnalysis",
        generated_at: "datetime",
    ) -> str:
        """Generate markdown content for image description file.

        Args:
            filename: Image filename
            analysis: Image analysis result
            generated_at: Generation timestamp

        Returns:
            Markdown content for the image description file
        """
        ...
