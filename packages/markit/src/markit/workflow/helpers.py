"""Helper utilities for workflow processing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from markit.security import atomic_write_json

if TYPE_CHECKING:
    from markit.workflow.single import ImageAnalysisResult

# Canonical frontmatter field order
FRONTMATTER_FIELD_ORDER = [
    "title",
    "source",
    "description",
    "tags",
    "markit_processed",
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

    # First, add fields in canonical order
    for field in FRONTMATTER_FIELD_ORDER:
        if field in data:
            value = data[field]
            if isinstance(value, list):
                # Format list as YAML flow style for tags
                formatted = yaml.dump(
                    {field: value}, allow_unicode=True, default_flow_style=None
                ).strip()
                ordered_lines.append(formatted)
            else:
                ordered_lines.append(f"{field}: {value}")

    # Then, add any remaining fields not in the canonical order
    for field, value in data.items():
        if field not in FRONTMATTER_FIELD_ORDER:
            if isinstance(value, list):
                formatted = yaml.dump(
                    {field: value}, allow_unicode=True, default_flow_style=None
                ).strip()
                ordered_lines.append(formatted)
            else:
                ordered_lines.append(f"{field}: {value}")

    return "\n".join(ordered_lines)


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


def add_basic_frontmatter(content: str, source: str) -> str:
    """Add basic frontmatter (title, source, markit_processed) to markdown content.

    Used for .md files that don't go through full LLM processing.

    Args:
        content: Markdown content
        source: Source file name

    Returns:
        Content with basic frontmatter prepended
    """
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

    frontmatter_dict = {
        "title": title,
        "source": source,
        "markit_processed": timestamp,
    }
    frontmatter_yaml = normalize_frontmatter(frontmatter_dict)

    return f"---\n{frontmatter_yaml}\n---\n\n{content}"


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


def write_assets_json(
    output_dir: Path,
    analysis_results: list[ImageAnalysisResult],
) -> list[Path]:
    """Write or merge asset descriptions to JSON files in each assets directory.

    Each assets directory (e.g., output/assets/, output/sub_dir/assets/) gets
    its own assets.json file containing only the assets from that directory.

    Args:
        output_dir: Output directory
        analysis_results: List of ImageAnalysisResult objects

    Returns:
        List of paths to created/updated JSON files
    """
    if not analysis_results:
        return []

    # Group assets by their containing assets directory
    # Key: assets_dir path, Value: list of (source_file, asset_dict) tuples
    assets_by_dir: dict[Path, list[tuple[str, dict[str, Any]]]] = {}

    for result in analysis_results:
        if not result.assets:
            continue

        for asset in result.assets:
            # Determine assets directory from the asset path
            asset_path = Path(asset.get("asset", ""))
            if asset_path.parent.name == "assets":
                assets_dir = asset_path.parent
            else:
                # Fallback to default assets directory
                assets_dir = output_dir / "assets"

            if assets_dir not in assets_by_dir:
                assets_by_dir[assets_dir] = []
            assets_by_dir[assets_dir].append((result.source_file, asset))

    # Write an assets.json file for each assets directory
    created_files: list[Path] = []
    local_now = datetime.now().astimezone().isoformat()

    for assets_dir, asset_entries in assets_by_dir.items():
        json_file = assets_dir / "assets.json"

        # Load existing data if file exists
        existing_data: dict[str, Any] = {}
        if json_file.exists():
            try:
                existing_data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing_data = {}

        # Build assets map keyed by asset path (merge with existing)
        assets_map: dict[str, dict[str, Any]] = {}
        for existing_asset in existing_data.get("assets", []):
            if isinstance(existing_asset, dict) and "asset" in existing_asset:
                assets_map[existing_asset["asset"]] = existing_asset

        # Add/update assets from this batch
        for source_file, asset in asset_entries:
            asset_with_source = {**asset, "source": source_file}
            assets_map[asset.get("asset", "")] = asset_with_source

        # Build final JSON structure (flat assets array)
        assets_json = {
            "version": "1.0",
            "created": existing_data.get("created", local_now),
            "updated": local_now,
            "assets": list(assets_map.values()),
        }

        assets_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(json_file, assets_json)
        logger.info(f"Asset description saved: {json_file}")
        created_files.append(json_file)

    return created_files
