"""LLM-powered image analysis."""

import asyncio
import io
from dataclasses import dataclass

from PIL import Image

from markit.image.compressor import CompressedImage
from markit.image.converter import is_llm_supported_format
from markit.llm.base import LLMResponse
from markit.llm.manager import ProviderManager
from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ImageAnalysis:
    """Result of LLM image analysis."""

    alt_text: str  # Short description for Markdown alt text
    detailed_description: str  # Detailed description for .md file
    detected_text: str | None  # OCR-detected text in image
    image_type: str  # Type: diagram, photo, screenshot, chart, etc.


# Prompt for image analysis
IMAGE_ANALYSIS_PROMPT = """Analyze this image and provide a JSON response with the following fields:

1. "alt_text": A brief, descriptive alt text (1 sentence, max 100 characters) suitable for Markdown image syntax.
2. "detailed_description": A comprehensive description of the image content (2-5 sentences).
3. "detected_text": Any text visible in the image. If no text is visible, use null.
4. "image_type": The type of image. Choose one of: "diagram", "photo", "screenshot", "chart", "graph", "table", "logo", "icon", "illustration", "other".

Respond ONLY with valid JSON, no other text. Example:
{
    "alt_text": "Architecture diagram showing microservices",
    "detailed_description": "This diagram illustrates a microservices architecture with three main components...",
    "detected_text": "API Gateway, User Service, Database",
    "image_type": "diagram"
}"""


class ImageAnalyzer:
    """Analyzes images using LLM vision capabilities."""

    def __init__(self, provider_manager: ProviderManager) -> None:
        """Initialize the image analyzer.

        Args:
            provider_manager: LLM provider manager for making API calls
        """
        self.provider_manager = provider_manager

    def _convert_to_supported_format(
        self,
        image: CompressedImage,
    ) -> tuple[bytes, str]:
        """Convert image to LLM-supported format if needed.

        Args:
            image: Image to potentially convert

        Returns:
            Tuple of (image_data, format) - converted if needed
        """
        if is_llm_supported_format(image.format):
            return image.data, image.format

        # Convert unsupported format (like GIF) to PNG
        log.info(
            "Converting image to PNG for LLM analysis",
            filename=image.filename,
            original_format=image.format,
        )

        try:
            img = Image.open(io.BytesIO(image.data))

            # Handle animated GIF - use first frame
            if hasattr(img, "n_frames") and img.n_frames > 1:
                img.seek(0)  # Get first frame

            # Convert to RGB/RGBA for PNG
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGBA")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue(), "png"

        except Exception as e:
            log.warning(
                "Failed to convert image format, using original",
                filename=image.filename,
                error=str(e),
            )
            return image.data, image.format

    async def analyze(
        self,
        image: CompressedImage,
        context: str | None = None,
    ) -> ImageAnalysis:
        """Analyze an image using LLM vision.

        Args:
            image: Compressed image to analyze
            context: Optional context about the image (e.g., from surrounding document)

        Returns:
            ImageAnalysis with descriptions and metadata
        """
        log.info("Analyzing image with LLM", filename=image.filename)

        # Convert to LLM-supported format if needed (e.g., GIF -> PNG)
        image_data, image_format = self._convert_to_supported_format(image)

        # Build prompt with optional context
        prompt = IMAGE_ANALYSIS_PROMPT
        if context:
            prompt += f"\n\nContext from document: {context}"

        try:
            # Call LLM with image (using converted format if applicable)
            response = await self.provider_manager.analyze_image_with_fallback(
                image_data=image_data,
                prompt=prompt,
                image_format=image_format,
            )

            # Parse JSON response
            analysis = self._parse_response(response)

            log.debug(
                "Image analysis complete",
                filename=image.filename,
                image_type=analysis.image_type,
            )

            return analysis

        except Exception as e:
            log.error("Image analysis failed", filename=image.filename, error=str(e))
            # Return fallback analysis
            return ImageAnalysis(
                alt_text=f"Image: {image.filename}",
                detailed_description="Image analysis failed.",
                detected_text=None,
                image_type="other",
            )

    def _parse_response(self, response: LLMResponse) -> ImageAnalysis:
        """Parse the LLM response into ImageAnalysis."""
        import json

        content = response.content.strip()

        # Try to extract JSON from the response
        # Sometimes LLMs wrap JSON in markdown code blocks
        if content.startswith("```"):
            # Extract content between code blocks
            lines = content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        try:
            data = json.loads(content)
            return ImageAnalysis(
                alt_text=data.get("alt_text", "Image"),
                detailed_description=data.get("detailed_description", ""),
                detected_text=data.get("detected_text"),
                image_type=data.get("image_type", "other"),
            )
        except json.JSONDecodeError:
            log.warning("Failed to parse image analysis JSON, using raw content")
            # Fallback: use the raw response as description
            return ImageAnalysis(
                alt_text="Image",
                detailed_description=content[:500] if content else "",
                detected_text=None,
                image_type="other",
            )

    async def generate_alt_text(
        self,
        image: CompressedImage,
    ) -> str:
        """Generate just the alt text for an image.

        Args:
            image: Compressed image

        Returns:
            Alt text string
        """
        analysis = await self.analyze(image)
        return analysis.alt_text

    async def batch_analyze(
        self,
        images: list[CompressedImage],
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ImageAnalysis]:
        """Analyze multiple images.

        Args:
            images: List of compressed images
            semaphore: Optional semaphore for rate limiting LLM calls

        Returns:
            List of ImageAnalysis results
        """

        async def analyze_with_limit(img: CompressedImage) -> ImageAnalysis:
            """Analyze with optional rate limiting."""
            if semaphore:
                async with semaphore:
                    return await self.analyze(img)
            return await self.analyze(img)

        results = await asyncio.gather(
            *[analyze_with_limit(img) for img in images],
            return_exceptions=True,
        )

        # Convert exceptions to fallback analyses
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(
                    "Batch image analysis failed",
                    filename=images[i].filename,
                    error=str(result),
                )
                final_results.append(
                    ImageAnalysis(
                        alt_text=f"Image: {images[i].filename}",
                        detailed_description="Image analysis failed.",
                        detected_text=None,
                        image_type="other",
                    )
                )
            else:
                final_results.append(result)

        return final_results
