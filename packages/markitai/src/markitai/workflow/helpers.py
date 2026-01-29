"""Helper utilities for workflow processing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from markitai.json_order import order_images
from markitai.security import atomic_write_json
from markitai.utils.paths import ensure_dir

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.llm import ImageAnalysis, LLMProcessor, LLMRuntime
    from markitai.workflow.single import ImageAnalysisResult

# Canonical frontmatter field order
FRONTMATTER_FIELD_ORDER = [
    "title",
    "source",
    "description",
    "tags",
    "markitai_processed",
]

# Patterns to detect prompt leakage in frontmatter keys
# These are LLM hallucinations where prompt text appears as YAML keys
PROMPT_LEAKAGE_KEY_PATTERNS = [
    r"根据.*生成",  # "根据文档内容生成 YAML frontmatter"
    r"请.*生成",  # "请生成元数据"
    r"以下是",  # "以下是生成的 frontmatter"
    r"YAML.*frontmatter",  # "YAML frontmatter"
    r"元数据",  # "元数据"
    r"任务\s*\d",  # "任务 1" or "任务1"
    r"Task\s*\d",  # "Task 1" or "Task1"
]


def normalize_frontmatter(frontmatter: str | dict[str, Any]) -> str:
    """Normalize frontmatter to ensure consistent field order.

    Parses the frontmatter (if string), reorders fields according to
    FRONTMATTER_FIELD_ORDER, and outputs clean YAML without markers.

    Args:
        frontmatter: YAML string (with or without --- markers) or dict

    Returns:
        Normalized YAML string without --- markers
    """
    if isinstance(frontmatter, str):
        # Remove --- markers and code block markers
        cleaned = frontmatter.strip()
        # Remove ```yaml ... ``` wrapper
        code_block_pattern = r"^```(?:ya?ml)?\s*\n?(.*?)\n?```$"
        match = re.match(code_block_pattern, cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        # Remove --- markers
        if cleaned.startswith("---"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("---"):
            cleaned = cleaned[:-3].strip()

        try:
            data = yaml.safe_load(cleaned) or {}
        except yaml.YAMLError:
            # If parsing fails, return as-is
            return cleaned
    else:
        data = frontmatter

    if not isinstance(data, dict):
        return str(data)

    # Build ordered output
    ordered_lines = []

    def format_field(field: str, value: Any) -> str:
        """Format a single field as valid YAML."""
        # Use yaml.dump for proper escaping of special characters
        # default_flow_style=False ensures block style (key: value, not {key: value})
        formatted = yaml.dump(
            {field: value},
            allow_unicode=True,
            default_flow_style=False,
            width=1000,  # Prevent line wrapping
        ).strip()
        return formatted

    # First, add fields in canonical order
    for field in FRONTMATTER_FIELD_ORDER:
        if field in data:
            ordered_lines.append(format_field(field, data[field]))

    # Then, add any remaining fields not in the canonical order
    # But filter out prompt leakage keys (LLM hallucinations)
    for field, value in data.items():
        if field not in FRONTMATTER_FIELD_ORDER:
            # Check if field name matches prompt leakage patterns
            is_leakage = False
            for pattern in PROMPT_LEAKAGE_KEY_PATTERNS:
                if re.search(pattern, field, re.IGNORECASE):
                    is_leakage = True
                    logger.debug(
                        f"Filtered prompt leakage key from frontmatter: {field}"
                    )
                    break
            if not is_leakage:
                ordered_lines.append(format_field(field, value))

    return "\n".join(ordered_lines)


# Language code to full name mapping for LLM prompts
LANGUAGE_NAMES: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
}


def detect_language(content: str) -> str:
    """Detect the primary language of the content.

    Uses a simple heuristic: if more than 10% of characters are CJK,
    consider it Chinese.

    Args:
        content: Text content to analyze

    Returns:
        Language code: "zh" for Chinese, "en" for English/other
    """
    if not content:
        return "en"

    # Count CJK characters (Chinese, Japanese, Korean)
    cjk_count = 0
    total_count = 0

    for char in content:
        if char.isalpha():
            total_count += 1
            # CJK Unified Ideographs range
            if "\u4e00" <= char <= "\u9fff":
                cjk_count += 1

    if total_count == 0:
        return "en"

    # If more than 10% CJK characters, consider it Chinese
    if cjk_count / total_count > 0.1:
        return "zh"

    return "en"


def get_language_name(language_code: str) -> str:
    """Get full language name from language code.

    Args:
        language_code: Language code ("zh" or "en")

    Returns:
        Full language name ("Chinese" or "English")
    """
    return LANGUAGE_NAMES.get(language_code, "English")


def add_basic_frontmatter(
    content: str,
    source: str,
    fetch_strategy: str | None = None,
    screenshot_path: Path | None = None,
    output_dir: Path | None = None,
    dedupe: bool = True,
) -> str:
    """Add basic frontmatter (title, source, markitai_processed) to markdown content.

    Used for .md files that don't go through full LLM processing.

    Args:
        content: Markdown content
        source: Source file name or URL
        fetch_strategy: Optional fetch strategy used (e.g., "static", "browser")
        screenshot_path: Optional path to page screenshot
        output_dir: Optional output directory (for relative screenshot path)
        dedupe: Whether to deduplicate paragraphs (default True)

    Returns:
        Content with basic frontmatter prepended
    """
    from markitai.utils.text import dedupe_long_text_blocks, dedupe_paragraphs

    # Apply deduplication for browser-fetched content
    if dedupe:
        # First pass: paragraph-level deduplication (for standard markdown)
        content = dedupe_paragraphs(content)
        # Second pass: long text block deduplication (for social media content)
        content = dedupe_long_text_blocks(content)

    # Extract title from first heading or use source name
    title = source
    lines = content.strip().split("\n")
    for line in lines:
        if line.startswith("#"):
            # Remove # and ** markers, strip whitespace
            title = line.lstrip("#").strip()
            title = title.replace("**", "").strip()
            if title:
                break

    timestamp = datetime.now().astimezone().isoformat()

    frontmatter_dict: dict[str, Any] = {
        "title": title,
        "source": source,
        "markitai_processed": timestamp,
    }

    # Add fetch_strategy if provided
    if fetch_strategy:
        frontmatter_dict["fetch_strategy"] = fetch_strategy

    frontmatter_yaml = normalize_frontmatter(frontmatter_dict)

    result = f"---\n{frontmatter_yaml}\n---\n\n{content}"

    # Add screenshot reference as HTML comment at the end
    if screenshot_path and screenshot_path.exists():
        if output_dir:
            # Calculate relative path from output file to screenshot
            try:
                rel_path = screenshot_path.relative_to(output_dir)
            except ValueError:
                rel_path = screenshot_path
        else:
            rel_path = screenshot_path.name

        # Add screenshot reference at the end
        result = (
            result.rstrip()
            + f"\n\n<!-- Screenshot for reference -->\n<!-- ![Screenshot]({rel_path}) -->\n"
        )

    return result


def merge_llm_usage(
    target: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> None:
    """Merge LLM usage statistics from source into target.

    Args:
        target: Target dict to merge into (modified in place)
        source: Source dict to merge from
    """
    for model, usage in source.items():
        if model not in target:
            target[model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        # Use .get() for robustness in case target has incomplete fields
        target[model]["requests"] = target[model].get("requests", 0) + usage.get(
            "requests", 0
        )
        target[model]["input_tokens"] = target[model].get(
            "input_tokens", 0
        ) + usage.get("input_tokens", 0)
        target[model]["output_tokens"] = target[model].get(
            "output_tokens", 0
        ) + usage.get("output_tokens", 0)
        target[model]["cost_usd"] = target[model].get("cost_usd", 0.0) + usage.get(
            "cost_usd", 0.0
        )


class LLMUsageAccumulator:
    """Accumulator for LLM usage statistics and cost.

    This class centralizes the common pattern of tracking LLM cost and usage,
    reducing the boilerplate of `llm_cost += cost` and `merge_llm_usage(usage, new_usage)`.

    Example:
        >>> acc = LLMUsageAccumulator()
        >>> # After each LLM call
        >>> acc.add(cost=0.05, usage={"gpt-4": {"requests": 1, ...}})
        >>> acc.add(cost=0.03, usage={"gpt-4": {"requests": 1, ...}})
        >>> # Get totals
        >>> print(f"Total cost: ${acc.total_cost:.4f}")
        >>> print(f"Usage by model: {acc.usage}")
    """

    def __init__(self) -> None:
        """Initialize accumulator with zero cost and empty usage."""
        self.total_cost: float = 0.0
        self.usage: dict[str, dict[str, Any]] = {}

    def add(
        self, cost: float = 0.0, usage: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Add cost and usage statistics.

        Args:
            cost: Cost in USD to add
            usage: Usage statistics by model to merge
        """
        self.total_cost += cost
        if usage:
            merge_llm_usage(self.usage, usage)

    def reset(self) -> None:
        """Reset accumulator to initial state."""
        self.total_cost = 0.0
        self.usage = {}


def write_images_json(
    output_dir: Path,
    analysis_results: list[ImageAnalysisResult],
) -> list[Path]:
    """Write or merge image descriptions to JSON files in each assets directory.

    Each assets directory (e.g., output/assets/, output/sub_dir/assets/) gets
    its own images.json file containing only the images from that directory.

    Args:
        output_dir: Output directory
        analysis_results: List of ImageAnalysisResult objects

    Returns:
        List of paths to created/updated JSON files
    """
    if not analysis_results:
        return []

    # Group images by their containing assets directory
    # Key: assets_dir path, Value: list of (source_file, image_dict) tuples
    images_by_dir: dict[Path, list[tuple[str, dict[str, Any]]]] = {}

    for result in analysis_results:
        if not result.assets:
            continue

        for asset in result.assets:
            # Determine assets directory from the image path
            # Note: asset dict uses "asset" key internally, will be renamed to "path" in output
            image_path = Path(asset.get("asset", ""))
            if image_path.parent.name == "assets":
                assets_dir = image_path.parent
            else:
                # Fallback to default assets directory
                assets_dir = output_dir / "assets"

            if assets_dir not in images_by_dir:
                images_by_dir[assets_dir] = []
            images_by_dir[assets_dir].append((result.source_file, asset))

    # Write an images.json file for each assets directory
    created_files: list[Path] = []
    local_now = datetime.now().astimezone().isoformat()

    for assets_dir, image_entries in images_by_dir.items():
        # Check for both old (assets.json) and new (images.json) filenames
        json_file = assets_dir / "images.json"
        old_json_file = assets_dir / "assets.json"

        # Load existing data if file exists (prefer new name, fallback to old)
        existing_data: dict[str, Any] = {}
        if json_file.exists():
            try:
                existing_data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_data = {}
        elif old_json_file.exists():
            try:
                existing_data = json.loads(old_json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_data = {}

        # Build images map keyed by path (merge with existing)
        # Support both old (assets/asset) and new (images/path) field names
        images_map: dict[str, dict[str, Any]] = {}
        existing_images = existing_data.get("images") or existing_data.get("assets", [])
        for existing_image in existing_images:
            if isinstance(existing_image, dict):
                # Get path from either "path" (new) or "asset" (old)
                img_path = existing_image.get("path") or existing_image.get("asset", "")
                if img_path:
                    images_map[img_path] = existing_image

        # Add/update images from this batch
        for source_file, asset in image_entries:
            # Convert internal "asset" key to "path" for output
            # Filter out llm_usage (internal tracking, not needed in output)
            image_entry = {k: v for k, v in asset.items() if k != "llm_usage"}
            image_entry["source"] = source_file
            if "asset" in image_entry:
                image_entry["path"] = image_entry.pop("asset")
            images_map[image_entry.get("path", "")] = image_entry

        # Build final JSON structure
        images_json = {
            "version": "1.0",
            "created": existing_data.get("created", local_now),
            "updated": local_now,
            "images": list(images_map.values()),
        }

        ensure_dir(assets_dir)
        atomic_write_json(json_file, images_json, order_func=order_images)
        created_files.append(json_file)

    # Log summary of created files
    if created_files:
        if len(created_files) == 1:
            logger.info(f"Image descriptions saved: {created_files[0]}")
        else:
            logger.info(f"Asset descriptions saved: {len(created_files)} files")

    return created_files


def format_standalone_image_markdown(
    input_path: Path,
    analysis: ImageAnalysis,
    image_ref_path: str,
    include_frontmatter: bool = False,
) -> str:
    """Format analysis results for a standalone image file.

    Creates a rich markdown document with:
    - Optional frontmatter (for .llm.md files)
    - Title (image filename)
    - Image preview
    - Image description section
    - Extracted text section (if any text was found)

    Args:
        input_path: Original image file path
        analysis: ImageAnalysis result with caption, description, extracted_text
        image_ref_path: Relative path for image reference
        include_frontmatter: Whether to include YAML frontmatter

    Returns:
        Formatted markdown string
    """
    sections = []

    # Frontmatter (for .llm.md files)
    if include_frontmatter:
        timestamp = datetime.now().astimezone().isoformat()
        frontmatter_lines = [
            "---",
            f"title: {input_path.stem}",
            f"description: {analysis.caption}",
            f"source: {input_path.name}",
            "tags:",
            "- image",
            "- analysis",
            f"markitai_processed: {timestamp}",
            "---",
            "",
        ]
        sections.append("\n".join(frontmatter_lines))

    # Title
    sections.append(f"# {input_path.stem}\n")

    # Image preview with alt text
    sections.append(f"![{analysis.caption}]({image_ref_path})\n")

    # Image description section
    if analysis.description:
        desc = analysis.description.strip()
        # Only add section header if description doesn't already start with a header
        if not desc.startswith("#"):
            sections.append("## Image Description\n")
        sections.append(f"{desc}\n")

    # Extracted text section (only if text was found)
    if analysis.extracted_text and analysis.extracted_text.strip():
        sections.append("## Extracted Text\n")
        sections.append(f"```\n{analysis.extracted_text}\n```\n")

    return "\n".join(sections)


def create_llm_processor(
    config: MarkitaiConfig,
    runtime: LLMRuntime | None = None,
) -> LLMProcessor:
    """Create an LLMProcessor instance from configuration.

    This is a factory function to centralize LLMProcessor instantiation,
    reducing code duplication across CLI and workflow modules.

    Args:
        config: Markitai configuration object
        runtime: Optional shared runtime for concurrency control.
                 If provided, uses runtime's semaphore instead of creating one.

    Returns:
        Configured LLMProcessor instance

    Example:
        >>> processor = create_llm_processor(cfg)
        >>> result = await processor.process_document(content)
    """
    from markitai.llm import LLMProcessor

    return LLMProcessor(
        config.llm,
        config.prompts,
        runtime=runtime,
        no_cache=config.cache.no_cache,
        no_cache_patterns=config.cache.no_cache_patterns,
    )
