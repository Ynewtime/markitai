"""Programmatic frontmatter field generation utilities."""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def extract_title_from_content(content: str, fallback: str = "") -> str:
    """Extract title from markdown content.

    Priority:
    1. First H1 heading (# Title)
    2. First H2 heading (## Title)
    3. First non-empty line (skip frontmatter and comments)
    4. Fallback value

    Args:
        content: Markdown content
        fallback: Fallback title if extraction fails

    Returns:
        Extracted title string (max 100 chars)
    """
    if not content or not content.strip():
        return fallback

    # Remove YAML frontmatter block first
    content_without_frontmatter = _strip_frontmatter(content)

    # Try to find H1 heading first (priority 1)
    h1_match = _find_heading(content_without_frontmatter, level=1)
    if h1_match:
        return _truncate_title(h1_match)

    # Try to find H2 heading (priority 2)
    h2_match = _find_heading(content_without_frontmatter, level=2)
    if h2_match:
        return _truncate_title(h2_match)

    # Fall back to first non-empty, non-comment line (priority 3)
    first_line = _find_first_content_line(content_without_frontmatter)
    if first_line:
        return _truncate_title(first_line)

    return fallback


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content.

    Args:
        content: Markdown content potentially with frontmatter

    Returns:
        Content without frontmatter block
    """
    # Match frontmatter at the start of the document
    # Pattern: starts with ---, ends with ---, may have leading whitespace
    frontmatter_pattern = r"^\s*---\s*\n.*?\n---\s*\n?"
    return re.sub(frontmatter_pattern, "", content, count=1, flags=re.DOTALL)


def _find_heading(content: str, level: int) -> str | None:
    """Find heading of specified level, respecting code blocks.

    Args:
        content: Markdown content
        level: Heading level (1 for H1, 2 for H2)

    Returns:
        Heading text or None if not found
    """
    lines = content.split("\n")
    in_code_block = False
    code_fence_pattern = re.compile(r"^(`{3,}|~{3,})")

    for line in lines:
        # Track code blocks to skip headings inside them
        fence_match = code_fence_pattern.match(line)
        if fence_match:
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        # Match exact heading level: # for H1, ## for H2
        # Pattern ensures we match exactly N hashes followed by space or end
        heading_pattern = rf"^#{{{level}}}(?!#)\s+(.+)$"
        match = re.match(heading_pattern, line)
        if match:
            return match.group(1).strip()

    return None


def _find_first_content_line(content: str) -> str | None:
    """Find first non-empty, non-comment line.

    Args:
        content: Markdown content

    Returns:
        First content line or None
    """
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip HTML comments
        if stripped.startswith("<!--"):
            # Check if comment ends on same line
            if "-->" in stripped:
                continue
            # Multi-line comment, skip until end
            # For simplicity, just skip this line
            continue

        # Skip lines that are just comment endings
        if stripped.startswith("-->"):
            continue

        # Found a content line
        return stripped

    return None


def _truncate_title(title: str, max_length: int = 100) -> str:
    """Truncate title to max length.

    Args:
        title: Title string
        max_length: Maximum length

    Returns:
        Truncated title
    """
    if len(title) <= max_length:
        return title
    return title[:max_length]


def build_frontmatter_dict(
    *,
    source: str,
    description: str = "",
    tags: list[str] | None = None,
    title: str | None = None,
    content: str = "",
) -> dict[str, Any]:
    """Build complete frontmatter dict with programmatic fields.

    Args:
        source: Source filename (required)
        description: LLM-generated description
        tags: LLM-generated tags
        title: Optional explicit title, or extracted from content
        content: Content for title extraction if title not provided

    Returns:
        Complete frontmatter dict with fields:
        - title (extracted or provided)
        - source
        - description
        - tags (if provided and non-empty)
        - markitai_processed (ISO 8601 timestamp)
    """
    # Determine title
    if title:
        final_title = title
    elif content:
        # Try to extract from content
        extracted = extract_title_from_content(content)
        if extracted:
            final_title = extracted
        else:
            # Fallback to source filename without extension
            final_title = _filename_to_title(source)
    else:
        # No content, use source filename
        final_title = _filename_to_title(source)

    # Generate timestamp
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Build ordered dict to preserve field order
    result: dict[str, Any] = OrderedDict()
    result["title"] = final_title
    result["source"] = source
    result["description"] = description

    # Only include tags if non-empty
    if tags:
        result["tags"] = tags

    result["markitai_processed"] = timestamp

    return result


def _filename_to_title(filename: str) -> str:
    """Convert filename to title by removing extension.

    Args:
        filename: Source filename

    Returns:
        Filename without extension
    """
    return Path(filename).stem


def frontmatter_to_yaml(frontmatter: dict[str, Any]) -> str:
    """Convert frontmatter dict to YAML string.

    Args:
        frontmatter: Frontmatter dict

    Returns:
        YAML string (without --- markers)
    """
    if not frontmatter:
        return ""

    # Use safe_dump with options for clean output
    return yaml.safe_dump(
        dict(frontmatter),  # Convert OrderedDict to regular dict for yaml
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,  # Preserve order
        width=1000,  # Prevent line wrapping
    )
