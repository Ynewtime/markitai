"""LLM-powered image analysis."""

import asyncio
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from markit.image.compressor import CompressedImage
from markit.image.converter import is_llm_supported_format
from markit.llm.base import (
    LLMResponse,
    LLMTaskResultWithStats,
    ResponseFormat,
)
from markit.llm.manager import ProviderManager
from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class KnowledgeGraphMeta:
    """Knowledge graph metadata extracted from image analysis."""

    entities: list[str]  # Named entities (people, organizations, products, terms)
    relationships: list[str]  # Entity relationships (e.g., "A -> uses -> B")
    topics: list[str]  # Topic tags
    domain: str | None  # Domain classification


@dataclass
class ImageAnalysis:
    """Result of LLM image analysis."""

    alt_text: str  # Short description for Markdown alt text
    detailed_description: str  # Detailed description for .md file
    detected_text: str | None  # OCR-detected text in image
    image_type: str  # Type: diagram, photo, screenshot, chart, etc.
    knowledge_meta: KnowledgeGraphMeta | None = None  # Optional knowledge graph metadata


# Chinese image analysis prompt with knowledge graph support
# IMPORTANT: Prompt explicitly instructs to output raw JSON without code blocks
IMAGE_ANALYSIS_PROMPT_ZH = """分析此图像并返回 JSON 格式的响应。**alt_text 和 detailed_description 必须使用中文**。

## 返回字段说明

1. **alt_text** (中文): 简短的图像描述（1句话，最多50个汉字）
2. **detailed_description** (中文): 图像内容的详细描述（2-5句话）
3. **detected_text**: 图像中可见的文本内容
   - 提取有意义的文本，按阅读顺序排列
   - **忽略**: OCR乱码、装饰性水印、无意义单字符
   - 如果没有有意义的文本，返回 null
4. **image_type**: 图像类型分类
   - diagram | chart | graph | table | screenshot | photo | illustration | logo | icon | formula | code | other
5. **knowledge_meta** (可选): 知识图谱元数据
   - **entities**: 实体列表（人名、组织、产品、技术术语）
   - **relationships**: 实体间关系（格式: "实体A -> 关系 -> 实体B"）
   - **topics**: 主题标签
   - **domain**: 领域分类（技术、商业、学术、医疗等）

## 响应格式要求
- 直接输出有效 JSON 对象，以 { 开头，以 } 结尾
- 不要使用 markdown 代码块（不要使用 ```json 或 ```）
- 不要在 JSON 前后添加任何说明文字

示例输出:
{"alt_text": "展示微服务架构的系统设计图", "detailed_description": "该图展示了一个典型的微服务架构，包含API网关、用户服务、订单服务三个核心组件。", "detected_text": "API Gateway, User Service", "image_type": "diagram", "knowledge_meta": {"entities": ["API网关", "用户服务"], "relationships": ["用户服务 -> 调用 -> API网关"], "topics": ["微服务", "系统架构"], "domain": "技术"}}"""

# English image analysis prompt
# IMPORTANT: Prompt explicitly instructs to output raw JSON without code blocks
IMAGE_ANALYSIS_PROMPT_EN = """Analyze this image and provide a JSON response with the following fields:

1. "alt_text": A brief, descriptive alt text (1 sentence, max 100 characters) suitable for Markdown image syntax.
2. "detailed_description": A comprehensive description of the image content (2-5 sentences).
3. "detected_text": Any text visible in the image. Extract meaningful text in reading order. Ignore OCR artifacts, decorative watermarks, and meaningless single characters. If no meaningful text is visible, use null.
4. "image_type": The type of image. Choose one of: "diagram", "chart", "graph", "table", "screenshot", "photo", "illustration", "logo", "icon", "formula", "code", "other".
5. "knowledge_meta" (optional): Knowledge graph metadata
   - "entities": List of named entities (people, organizations, products, technical terms)
   - "relationships": Entity relationships (format: "Entity A -> relation -> Entity B")
   - "topics": Topic tags
   - "domain": Domain classification (technology, business, academic, medical, etc.)

## Response Format Requirements
- Output a valid JSON object directly, starting with { and ending with }
- Do NOT use markdown code blocks (no ```json or ```)
- Do NOT add any explanatory text before or after the JSON

Example output:
{"alt_text": "Architecture diagram showing microservices", "detailed_description": "This diagram illustrates a microservices architecture with three main components.", "detected_text": "API Gateway, User Service", "image_type": "diagram", "knowledge_meta": {"entities": ["API Gateway", "User Service"], "relationships": ["User Service -> calls -> API Gateway"], "topics": ["microservices", "architecture"], "domain": "technology"}}"""

# Default prompt (uses Chinese)
IMAGE_ANALYSIS_PROMPT = IMAGE_ANALYSIS_PROMPT_ZH


def get_image_analysis_prompt(language: str = "zh") -> str:
    """Get the image analysis prompt for the specified language.

    Args:
        language: Language code ("zh" for Chinese, "en" for English)

    Returns:
        Image analysis prompt string (always includes knowledge_meta since it's optional)
    """
    return IMAGE_ANALYSIS_PROMPT_ZH if language == "zh" else IMAGE_ANALYSIS_PROMPT_EN


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
        return_stats: bool = False,
        output_dir: Path | None = None,
    ) -> ImageAnalysis | LLMTaskResultWithStats:
        """Analyze an image using LLM vision.

        Args:
            image: Compressed image to analyze
            context: Optional context about the image (e.g., from surrounding document)
            return_stats: If True, return LLMTaskResultWithStats containing both
                         the ImageAnalysis and LLM statistics
            output_dir: If provided, immediately write the analysis result to
                       output_dir/assets/<filename>.md. This ensures results are
                       persisted as soon as they're available.

        Returns:
            ImageAnalysis with descriptions and metadata, or LLMTaskResultWithStats
            if return_stats=True
        """
        # Changed to debug to reduce log noise when processing many images
        log.debug("Analyzing image with LLM", filename=image.filename)

        # Convert to LLM-supported format if needed (e.g., GIF -> PNG)
        image_data, image_format = self._convert_to_supported_format(image)

        # Build prompt with optional context
        prompt = IMAGE_ANALYSIS_PROMPT
        if context:
            prompt += f"\n\nContext from document: {context}"

        try:
            # Call LLM with image (using converted format if applicable)
            # Use JSON mode for providers that support it (OpenAI, Gemini, Ollama)
            # Anthropic will fall back to prompt-based JSON extraction
            response = await self.provider_manager.analyze_image_with_fallback(
                image_data=image_data,
                prompt=prompt,
                image_format=image_format,
                response_format=ResponseFormat(
                    type="json_object",
                    # Note: json_schema is available but not all providers support it
                    # Using simple json_object mode for maximum compatibility
                ),
            )

            # Parse JSON response
            analysis = self._parse_response(response)

            log.debug(
                "Image analysis complete",
                filename=image.filename,
                image_type=analysis.image_type,
            )

            # Immediately write result to disk if output_dir is provided
            if output_dir:
                await self._write_analysis_immediately(
                    image.filename, image.data, analysis, output_dir
                )

            if return_stats:
                return LLMTaskResultWithStats.from_response(analysis, response)
            return analysis

        except Exception as e:
            log.error("Image analysis failed", filename=image.filename, error=str(e))
            # Return fallback analysis
            fallback = ImageAnalysis(
                alt_text=f"Image: {image.filename}",
                detailed_description="Image analysis failed.",
                detected_text=None,
                image_type="other",
            )
            if return_stats:
                return LLMTaskResultWithStats(result=fallback)
            return fallback

    async def _write_analysis_immediately(
        self,
        filename: str,
        image_data: bytes,
        analysis: ImageAnalysis,
        output_dir: Path,
    ) -> None:
        """Write image file and analysis result to disk immediately.

        This ensures that images and their analysis results are persisted as soon
        as they're available, rather than waiting for the entire batch to complete.

        Args:
            filename: Image filename
            image_data: Image binary data
            analysis: Analysis result
            output_dir: Output directory (will write to output_dir/assets/)
        """
        import anyio

        try:
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            # Write image file
            image_path = assets_dir / filename
            async with await anyio.open_file(image_path, "wb") as f:
                await f.write(image_data)

            # Write analysis description file
            md_content = self._generate_description_md(filename, analysis)
            md_path = assets_dir / f"{filename}.md"
            async with await anyio.open_file(md_path, "w", encoding="utf-8") as f:
                await f.write(md_content)

            log.debug(
                "Image and analysis written immediately",
                image_path=str(image_path),
                md_path=str(md_path),
            )

        except Exception as e:
            # Don't fail the analysis if write fails, just log warning
            log.warning(
                "Failed to write image/analysis immediately",
                filename=filename,
                error=str(e),
            )

    def _generate_description_md(
        self,
        filename: str,
        analysis: ImageAnalysis,
    ) -> str:
        """Generate markdown content for image description file.

        Args:
            filename: Image filename
            analysis: Image analysis result

        Returns:
            Markdown content for the image description file
        """
        generated_at = datetime.now(UTC)
        lines = [
            "---",
            f"source_image: {filename}",
            f"image_type: {analysis.image_type}",
            f"generated_at: {generated_at.isoformat()}",
        ]

        # Add knowledge graph metadata if available
        if analysis.knowledge_meta:
            km = analysis.knowledge_meta
            if km.entities:
                entities_str = ", ".join(km.entities)
                lines.append(f"entities: [{entities_str}]")
            if km.topics:
                topics_str = ", ".join(km.topics)
                lines.append(f"topics: [{topics_str}]")
            if km.domain:
                lines.append(f"domain: {km.domain}")

        lines.extend(
            [
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
        )

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

        # Add knowledge graph section if available
        if analysis.knowledge_meta and analysis.knowledge_meta.relationships:
            lines.extend(
                [
                    "",
                    "## Relationships",
                    "",
                ]
            )
            for rel in analysis.knowledge_meta.relationships:
                lines.append(f"- {rel}")

        return "\n".join(lines) + "\n"

    def _parse_response(self, response: LLMResponse) -> ImageAnalysis:
        """Parse the LLM response into ImageAnalysis."""
        import json
        import re

        content = response.content.strip()
        json_str = content

        # 1. Try to extract from code blocks first (standard format)
        # Use greedy match for JSON object inside code blocks
        code_block_pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", re.IGNORECASE)
        match = code_block_pattern.search(content)

        if match:
            json_str = match.group(1)
        else:
            # 2. Try to find the outermost JSON object
            # This regex matches a brace, followed by anything (non-greedy), ending with a brace
            # It's simple and relies on the LLM outputting a single main object
            json_pattern = re.compile(r"(\{[\s\S]*\})")
            match = json_pattern.search(content)
            if match:
                json_str = match.group(1)

        # 3. Clean up common JSON formatting issues from LLMs
        json_str = self._fix_json_string(json_str)

        try:
            data = json.loads(json_str)

            # Parse knowledge_meta if present
            knowledge_meta = None
            if "knowledge_meta" in data and data["knowledge_meta"]:
                km_data = data["knowledge_meta"]
                knowledge_meta = KnowledgeGraphMeta(
                    entities=self._ensure_string_list(km_data.get("entities", [])),
                    relationships=self._ensure_string_list(km_data.get("relationships", [])),
                    topics=self._ensure_string_list(km_data.get("topics", [])),
                    domain=self._ensure_string(km_data.get("domain")),
                )

            return ImageAnalysis(
                alt_text=self._ensure_string(data.get("alt_text", "Image")) or "Image",
                detailed_description=self._ensure_string(data.get("detailed_description", ""))
                or "",
                detected_text=self._ensure_string(data.get("detected_text")),
                image_type=self._ensure_string(data.get("image_type", "other")) or "other",
                knowledge_meta=knowledge_meta,
            )
        except json.JSONDecodeError as e:
            log.warning(
                "Failed to parse image analysis JSON, using raw content",
                error=str(e),
                json_preview=json_str[:200] if json_str else "",
            )
            # Fallback: use the raw response as description
            return ImageAnalysis(
                alt_text="Image",
                detailed_description=content[:500] if content else "",
                detected_text=None,
                image_type="other",
            )

    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses.

        Handles:
        - Trailing commas before closing brackets/braces
        - Incomplete JSON (missing closing brackets/braces)
        """
        import re

        if not json_str:
            return json_str

        # Remove trailing commas before closing brackets/braces
        # Pattern: comma followed by optional whitespace and closing bracket/brace
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # Try to fix incomplete JSON by counting brackets
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")

        # Add missing closing brackets/braces
        if open_braces > close_braces or open_brackets > close_brackets:
            # Remove any trailing incomplete content after the last complete value
            # Look for patterns like: trailing comma, incomplete key, etc.
            json_str = re.sub(r',\s*"[^"]*$', "", json_str)  # Incomplete key at end
            json_str = re.sub(r",\s*$", "", json_str)  # Trailing comma

            # Recalculate after cleanup
            open_braces = json_str.count("{")
            close_braces = json_str.count("}")
            open_brackets = json_str.count("[")
            close_brackets = json_str.count("]")

            # Add missing closures
            missing_brackets = open_brackets - close_brackets
            missing_braces = open_braces - close_braces

            if missing_brackets > 0 or missing_braces > 0:
                # Append missing closures in reverse order of expected nesting
                # Typically arrays close before objects in our schema
                json_str += "]" * missing_brackets + "}" * missing_braces
                log.debug(
                    "Fixed incomplete JSON",
                    added_brackets=missing_brackets,
                    added_braces=missing_braces,
                )

        return json_str

    def _ensure_string(self, value: Any) -> str | None:
        """Ensure value is a string, converting if necessary.

        Handles cases where LLM returns a list instead of a string.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # Join list items with newlines
            return "\n".join(str(item) for item in value)
        # Convert other types to string
        return str(value)

    def _ensure_string_list(self, value: Any) -> list[str]:
        """Ensure value is a list of strings."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return [value]
        return [str(value)]

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
        output_dir: Path | None = None,
    ) -> list[ImageAnalysis]:
        """Analyze multiple images.

        Args:
            images: List of compressed images
            semaphore: Optional semaphore for rate limiting LLM calls
            output_dir: If provided, immediately write each analysis result to
                       output_dir/assets/<filename>.md as it completes

        Returns:
            List of ImageAnalysis results
        """

        async def analyze_with_limit(img: CompressedImage) -> ImageAnalysis:
            """Analyze with optional rate limiting."""
            if semaphore:
                async with semaphore:
                    return await self.analyze(img, output_dir=output_dir)
            return await self.analyze(img, output_dir=output_dir)

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
