"""Programmatic frontmatter field generation utilities."""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from loguru import logger

LOW_CONFIDENCE_TITLE_SUFFIXES = {".csv", ".tsv", ".xml"}
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_TITLE_HTML_TAG_RE = re.compile(r"</?(?:strong|em|b|i|code)>", re.IGNORECASE)


def _normalize_title_text(title: str) -> str:
    """Normalize lightweight markdown formatting from a title candidate."""
    cleaned = " ".join(title.split())
    if not cleaned:
        return ""

    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned)
    cleaned = _MARKDOWN_IMAGE_RE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _TITLE_HTML_TAG_RE.sub("", cleaned)

    wrappers = ("**", "__", "~~", "`", "*", "_")
    changed = True
    while changed and cleaned:
        changed = False
        for wrapper in wrappers:
            if (
                cleaned.startswith(wrapper)
                and cleaned.endswith(wrapper)
                and len(cleaned) > len(wrapper) * 2
            ):
                cleaned = cleaned[len(wrapper) : -len(wrapper)].strip()
                changed = True
                break

    return " ".join(cleaned.split())


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
        return _truncate_title(_normalize_title_text(h1_match))

    # Try to find H2 heading (priority 2)
    h2_match = _find_heading(content_without_frontmatter, level=2)
    if h2_match:
        return _truncate_title(_normalize_title_text(h2_match))

    # Fall back to first non-empty, non-comment line (priority 3)
    first_line = _find_first_content_line(content_without_frontmatter)
    if first_line:
        return _truncate_title(_normalize_title_text(first_line))

    return fallback


def extract_frontmatter_title(content: str) -> str | None:
    """Extract title from existing YAML frontmatter.

    Args:
        content: Markdown content with potential frontmatter

    Returns:
        Title string if found in frontmatter, None otherwise
    """
    if not content or not content.strip():
        return None

    # Match frontmatter block
    frontmatter_pattern = r"^\s*---\s*\n(.*?)\n---"
    match = re.match(frontmatter_pattern, content, flags=re.DOTALL)
    if not match:
        return None

    try:
        frontmatter_yaml = match.group(1)
        data = yaml.safe_load(frontmatter_yaml)
        if isinstance(data, dict) and "title" in data:
            return str(data["title"])
    except yaml.YAMLError:
        pass

    return None


def source_supports_content_title(source: str) -> bool:
    """Return whether content-derived titles are trusted for this source."""
    return _source_suffix(source) not in LOW_CONFIDENCE_TITLE_SUFFIXES


def fallback_title_from_source(source: str) -> str:
    """Build a stable fallback title from the source identifier."""
    source_name = _source_name(source)
    if _source_suffix(source) in LOW_CONFIDENCE_TITLE_SUFFIXES:
        return source_name or source

    stem = Path(source_name).stem
    return stem or source_name or source


def resolve_document_title(
    *,
    source: str,
    explicit_title: str | None = None,
    content: str = "",
    extractor: Any | None = None,
) -> str:
    """Resolve a stable document title from explicit, content, or source fallback."""
    if explicit_title:
        return _normalize_title_text(explicit_title)

    if content and source_supports_content_title(source):
        title_extractor = extractor or extract_title_from_content
        extracted = title_extractor(content)
        if extracted:
            return extracted

    return fallback_title_from_source(source)


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
    """Find first non-empty, non-comment line suitable as title.

    Skips lines that are not suitable for titles:
    - Empty lines
    - HTML comments
    - Image-only lines (![...] or [![...])
    - Horizontal rules (---, ***, ___)

    Args:
        content: Markdown content

    Returns:
        First suitable content line or None
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

        # Skip image-only lines (not suitable as titles)
        if stripped.startswith("![") or stripped.startswith("[!["):
            continue

        # Skip horizontal rules
        if stripped in ("---", "***", "___"):
            continue

        # Found a suitable content line
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


def _source_name(source: str) -> str:
    """Extract a display-friendly source name from path or URL."""
    parsed = urlparse(source)
    if parsed.scheme and parsed.netloc:
        path_name = Path(parsed.path).name
        return path_name or parsed.netloc

    path_name = Path(source).name
    return path_name or source


def _source_suffix(source: str) -> str:
    """Extract lowercase suffix from path or URL source."""
    return Path(_source_name(source)).suffix.lower()


def frontmatter_timestamp() -> str:
    """Generate consistent ISO 8601 timestamp for frontmatter.

    Format: 2026-03-06T14:20:21.123+08:00 (milliseconds precision, with timezone)

    Returns:
        ISO 8601 timestamp string
    """
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def build_frontmatter_dict(
    *,
    source: str,
    description: str = "",
    tags: list[str] | None = None,
    title: str | None = None,
    content: str = "",
    fetch_strategy: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build complete frontmatter dict with programmatic fields.

    Args:
        source: Source filename (required)
        description: LLM-generated description
        tags: LLM-generated tags
        title: Optional explicit title, or extracted from content
        content: Content for title extraction if title not provided
        fetch_strategy: Optional fetch strategy used (e.g., "defuddle")
        extra_meta: Optional extra metadata from external strategies
            (will not override canonical fields)

    Returns:
        Complete frontmatter dict
    """
    # Determine title
    final_title = resolve_document_title(
        source=source,
        explicit_title=title,
        content=content,
    )

    # Normalize title: replace newlines with spaces, collapse whitespace, limit length
    if final_title:
        final_title = _normalize_title_text(final_title)
        # Truncate to avoid YAML line-wrapping issues in parsers like Obsidian
        if len(final_title) > 200:
            final_title = final_title[:197] + "..."

    # Normalize description: single line, limited length
    normalized_desc = ""
    if description:
        normalized_desc = " ".join(description.split())
        if len(normalized_desc) > 150:
            normalized_desc = normalized_desc[:147] + "..."
    else:
        logger.debug(f"[{source}] Empty description in frontmatter")

    # Normalize tags: replace spaces with hyphens, remove special chars
    normalized_tags: list[str] = []
    if tags:
        for tag in tags:
            # Collapse whitespace and replace with hyphen
            tag = "-".join(tag.split())
            # Remove any remaining problematic characters for YAML
            tag = tag.replace('"', "").replace("'", "").replace(":", "-")
            # Limit tag length
            if len(tag) > 30:
                tag = tag[:30]
            if tag:  # Only add non-empty tags
                normalized_tags.append(tag)

    # Generate timestamp (consistent format across .md and .llm.md)
    timestamp = frontmatter_timestamp()

    # Fields that extra_meta must not override or that are unreliable from
    # external sources (e.g. HTML <html lang="..."> often doesn't match
    # the actual content language).
    excluded_keys = {
        "title",
        "source",
        "description",
        "tags",
        "markitai_processed",
        "fetch_strategy",
        "language",
    }

    # Build ordered dict to preserve field order
    result: dict[str, Any] = OrderedDict()
    result["title"] = final_title
    result["source"] = source
    result["description"] = normalized_desc

    # Only include tags if non-empty
    if normalized_tags:
        result["tags"] = normalized_tags
    else:
        logger.debug(f"[{source}] Empty tags in frontmatter")

    result["markitai_processed"] = timestamp

    # Add fetch_strategy if provided
    if fetch_strategy:
        result["fetch_strategy"] = fetch_strategy

    # Merge extra metadata from external strategies (after canonical fields)
    if extra_meta:
        for key, value in extra_meta.items():
            if key not in excluded_keys and value is not None:
                result[key] = value

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
