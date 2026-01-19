"""Helper utilities for workflow processing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markit.security import atomic_write_json

if TYPE_CHECKING:
    from markit.workflow.single import ImageAnalysisResult


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

    timestamp = datetime.now(UTC).isoformat()

    frontmatter = f"""---
title: {title}
source: {source}
markit_processed: {timestamp}
---

"""
    return frontmatter + content


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


def write_assets_desc_json(
    output_dir: Path,
    analysis_results: list[ImageAnalysisResult],
) -> Path | None:
    """Write or merge asset descriptions to a single JSON file.

    Args:
        output_dir: Output directory
        analysis_results: List of ImageAnalysisResult objects

    Returns:
        Path to the created file, or None if no results
    """
    if not analysis_results:
        return None

    desc_file = output_dir / "assets.desc.json"

    existing_data: dict[str, Any] = {}
    if desc_file.exists():
        try:
            existing_data = json.loads(desc_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_data = {}

    sources_map: dict[str, dict[str, Any]] = {}
    for source in existing_data.get("sources", []):
        if isinstance(source, dict) and "file" in source:
            sources_map[source["file"]] = source

    for result in analysis_results:
        sources_map[result.source_file] = {
            "file": result.source_file,
            "assets": result.assets,
        }

    desc_json = {
        "version": "1.0",
        "created": existing_data.get("created", datetime.now(UTC).isoformat()),
        "updated": datetime.now(UTC).isoformat(),
        "sources": list(sources_map.values()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(desc_file, desc_json)
    logger.info(f"Written asset descriptions: {desc_file}")

    return desc_file
