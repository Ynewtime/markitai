"""Image processing service for format conversion, compression, and deduplication.

This service handles all image-related operations extracted from ConversionPipeline,
including parallel processing with intelligent executor selection (thread pool vs process pool).
"""

import asyncio
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from markit.config.constants import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_IMAGE_DIMENSION,
    DEFAULT_PNG_OPTIMIZATION_LEVEL,
    DEFAULT_PROCESS_POOL_MAX_WORKERS,
    DEFAULT_PROCESS_POOL_THRESHOLD,
)
from markit.utils.logging import get_logger

if TYPE_CHECKING:
    from markit.converters.base import ExtractedImage
    from markit.image.compressor import CompressedImage, ImageCompressor

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


# Module-level function for process pool compatibility (must be picklable)
def _compress_single_image_process(
    image_data: bytes,
    image_format: str,
    png_optimization_level: int,
    jpeg_quality: int,
    max_dimension: int,
) -> tuple[bytes, str, int, int]:
    """Compress a single image (process pool compatible).

    This function is defined at module level to be picklable for ProcessPoolExecutor.
    It delegates to the unified `compress_image_data` function with oxipng disabled
    to avoid subprocess issues in process pool workers.

    Args:
        image_data: Raw image bytes
        image_format: Image format (png, jpeg, etc.)
        png_optimization_level: PNG optimization level (0-6)
        jpeg_quality: JPEG quality (0-100)
        max_dimension: Maximum dimension for resizing

    Returns:
        Tuple of (compressed_data, format, width, height)
    """
    from markit.image.compressor import compress_image_data

    # Use unified compression function with oxipng disabled for process pool
    return compress_image_data(
        data=image_data,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        png_optimization_level=png_optimization_level,
        max_dimension=max_dimension,
        use_oxipng=False,  # Disable oxipng in process pool workers
    )


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing service."""

    compress_images: bool = True
    png_optimization_level: int = DEFAULT_PNG_OPTIMIZATION_LEVEL
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION
    use_process_pool: bool = True
    process_pool_threshold: int = DEFAULT_PROCESS_POOL_THRESHOLD
    max_workers: int = DEFAULT_PROCESS_POOL_MAX_WORKERS


@dataclass
class ProcessedImageInfo:
    """Information about a processed image including analysis results."""

    filename: str
    analysis: "ImageAnalysis | None" = None


# Import ImageAnalysis type for the dataclass
if TYPE_CHECKING:
    from markit.image.analyzer import ImageAnalysis


class ImageProcessingService:
    """Service for processing images: format conversion, deduplication, compression.

    This service provides:
    - Parallel image processing with intelligent executor selection
    - Image deduplication using MD5 hashing
    - Format conversion (EMF, WMF, TIFF -> PNG/JPEG)
    - Compression with configurable quality settings
    - Process pool support for CPU-intensive operations
    """

    def __init__(
        self,
        config: ImageProcessingConfig | None = None,
        image_compressor: "ImageCompressor | None" = None,
    ) -> None:
        """Initialize the image processing service.

        Args:
            config: Image processing configuration
            image_compressor: Optional pre-configured compressor (for DI)
        """
        self.config = config or ImageProcessingConfig()
        self._image_compressor = image_compressor
        self._process_pool: ProcessPoolExecutor | None = None

    def _get_image_compressor(self) -> "ImageCompressor":
        """Get or create the image compressor (lazy initialization)."""
        if self._image_compressor is None:
            from markit.image.compressor import CompressionConfig, ImageCompressor

            self._image_compressor = ImageCompressor(
                config=CompressionConfig(
                    png_optimization_level=self.config.png_optimization_level,
                    jpeg_quality=self.config.jpeg_quality,
                    max_dimension=self.config.max_dimension,
                ),
            )
        return self._image_compressor

    def _should_use_process_pool(self, image_count: int) -> bool:
        """Determine whether to use process pool based on image count.

        Args:
            image_count: Number of images to process

        Returns:
            True if process pool should be used
        """
        if not self.config.use_process_pool:
            return False
        return image_count >= self.config.process_pool_threshold

    def _get_process_pool(self) -> ProcessPoolExecutor:
        """Get or create the process pool executor."""
        if self._process_pool is None:
            import multiprocessing

            max_workers = min(multiprocessing.cpu_count(), self.config.max_workers)
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers)
            log.debug("Created process pool", max_workers=max_workers)
        return self._process_pool

    def shutdown(self) -> None:
        """Shutdown the process pool if it was created."""
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
            log.debug("Process pool shutdown")

    async def optimize_images_parallel(
        self,
        images: list["ExtractedImage"],
        input_file: Path,
    ) -> tuple[list["ExtractedImage"], dict[str, str | None]]:
        """Process images in parallel: format convert, deduplicate, compress.

        This method performs:
        1. Deduplication using MD5 hashing
        2. Format conversion for unsupported formats
        3. Compression with configurable quality
        4. Standardized filename generation

        Args:
            images: List of extracted images to process
            input_file: Original input file path for filename generation

        Returns:
            Tuple of (processed_unique_images, filename_map: old -> new)
        """
        from markit.converters.base import ExtractedImage
        from markit.image.converter import ImageFormatConverter

        if not images:
            return [], {}

        format_converter = ImageFormatConverter()
        compressor = self._get_image_compressor()

        # 1. Identification & Deduplication
        unique_images_map: dict[str, ExtractedImage] = {}  # hash -> image
        unique_hashes_order: list[str] = []  # to preserve order of first appearance

        for img in images:
            img_hash = hashlib.md5(img.data).hexdigest()

            if img_hash not in unique_images_map:
                unique_images_map[img_hash] = img
                unique_hashes_order.append(img_hash)

        log.debug(
            "Image deduplication",
            total=len(images),
            unique=len(unique_hashes_order),
            duplicates_removed=len(images) - len(unique_hashes_order),
        )

        # 2. Process Unique Images in Parallel
        def process_single(img: ExtractedImage, index: int) -> ExtractedImage | None:
            """Process a single image (thread pool compatible)."""
            # Format conversion
            current_img = img
            if format_converter.needs_conversion(img.format):
                try:
                    converted = format_converter.convert(img)
                    if converted is None:
                        return None
                    current_img = ExtractedImage(
                        data=converted.data,
                        format=converted.format,
                        filename=converted.filename,
                        source_document=img.source_document,
                        position=img.position,
                        width=converted.width,
                        height=converted.height,
                    )
                except Exception as e:
                    log.warning("Format conversion failed", filename=img.filename, error=str(e))
                    return None

            # Compression
            if self.config.compress_images:
                try:
                    compressed = compressor.compress(current_img)
                    current_img = ExtractedImage(
                        data=compressed.data,
                        format=compressed.format,
                        filename=compressed.filename,
                        source_document=current_img.source_document,
                        position=current_img.position,
                        width=compressed.width,
                        height=compressed.height,
                    )
                except Exception as e:
                    log.warning("Compression failed", filename=current_img.filename, error=str(e))

            # Generate final filename
            new_filename = _generate_image_filename(input_file, index, current_img.format)

            return ExtractedImage(
                data=current_img.data,
                format=current_img.format,
                filename=new_filename,
                source_document=current_img.source_document,
                position=current_img.position,
                width=current_img.width,
                height=current_img.height,
            )

        # Determine executor type based on image count
        use_process_pool = self._should_use_process_pool(len(unique_hashes_order))

        # Create tasks
        tasks = []
        for i, h in enumerate(unique_hashes_order):
            img = unique_images_map[h]
            # index is i+1 (1-based)
            tasks.append(asyncio.to_thread(process_single, img, i + 1))

        # Run tasks
        results: list[ExtractedImage | None] = []
        if tasks:
            executor_type = "process_pool" if use_process_pool else "thread_pool"
            log.info(
                "Processing unique images in parallel",
                count=len(tasks),
                total_extracted=len(images),
                executor=executor_type,
            )
            results = await asyncio.gather(*tasks)

        # 3. Rebuild Maps and Results
        final_images: list[ExtractedImage] = []
        hash_to_filename: dict[str, str | None] = {}

        for h, result in zip(unique_hashes_order, results, strict=True):
            if result is not None:
                final_images.append(result)
                hash_to_filename[h] = result.filename
            else:
                hash_to_filename[h] = None

        # Build filename map (old -> new)
        filename_map: dict[str, str | None] = {}
        for img in images:
            img_hash = hashlib.md5(img.data).hexdigest()
            if img_hash in hash_to_filename:
                filename_map[img.filename] = hash_to_filename[img_hash]

        return final_images, filename_map

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
        from markit.image.compressor import CompressedImage

        return [
            CompressedImage(
                data=img.data,
                format=img.format,
                filename=img.filename,
                original_size=len(img.data),
                compressed_size=len(img.data),
                width=img.width or 0,
                height=img.height or 0,
            )
            for img in processed_images
        ]

    def update_markdown_references(
        self,
        markdown: str,
        filename_map: dict[str, str | None],
    ) -> str:
        """Update markdown image references based on filename mapping.

        Args:
            markdown: Original markdown content
            filename_map: Mapping from old filenames to new filenames

        Returns:
            Updated markdown content
        """
        import re

        for old_filename, new_filename in filename_map.items():
            if new_filename:
                # Update all references (both standard and assets/)
                markdown = markdown.replace(f"assets/{old_filename}", f"assets/{new_filename}")
                # Handle cases where markdown might reference old filename without assets/
                markdown = markdown.replace(f"({old_filename})", f"({new_filename})")
            else:
                # Image processing failed, remove references with any alt text
                # Pattern: ![any alt text](assets/filename) or ![any alt text](filename)
                escaped_filename = re.escape(old_filename)
                # Remove references with assets/ prefix
                markdown = re.sub(
                    rf"!\[[^\]]*\]\(assets/{escaped_filename}\)\n?",
                    "",
                    markdown,
                )
                # Remove references without assets/ prefix
                markdown = re.sub(
                    rf"!\[[^\]]*\]\({escaped_filename}\)\n?",
                    "",
                    markdown,
                )

        return markdown
