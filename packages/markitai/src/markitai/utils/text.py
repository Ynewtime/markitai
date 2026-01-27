"""Text processing utilities for Markitai."""

from __future__ import annotations

import re


def clean_residual_placeholders(content: str) -> str:
    """Remove residual MARKITAI placeholders from content.

    Some placeholders may leak into the output, especially in image references
    like `![](__MARKITAI_FILE_ASSET__)`. This function cleans them up.

    Args:
        content: Markdown content with potential residual placeholders

    Returns:
        Cleaned content
    """
    # Remove standalone placeholder lines
    content = re.sub(r"^__MARKITAI_[A-Z_]+_?\d*__\s*$", "", content, flags=re.MULTILINE)

    # Remove image references with placeholder URLs
    content = re.sub(r"!\[[^\]]*\]\(__MARKITAI_[A-Z_]+_?\d*__\)\s*\n?", "", content)

    # Remove any other inline placeholders
    content = re.sub(r"__MARKITAI_[A-Z_]+_?\d*__", "", content)

    return content


def normalize_markdown_whitespace(content: str) -> str:
    """Normalize whitespace in markdown content.

    - Fix malformed image references (double alt text, empty paths)
    - Ensure headers (#) have one blank line before and after
    - Merge 3+ consecutive blank lines into 2 blank lines
    - Ensure consistent line endings
    - Strip trailing whitespace from lines

    Note: Header normalization is markdown-aware and correctly handles
    nested code blocks (e.g., ```` containing ```).

    Args:
        content: Markdown content to normalize

    Returns:
        Normalized content
    """
    # Fix malformed image references first
    content = fix_malformed_image_refs(content)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in content.split("\n")]

    # Normalize header spacing (markdown-aware, skip code blocks)
    result_lines: list[str] = []
    code_block_char: str | None = None  # '`' or '~'
    code_block_count: int = 0  # Number of fence chars that opened the block

    for i, line in enumerate(lines):
        # Check for code fence (``` or ~~~, possibly more)
        fence_match = re.match(r"^(`{3,}|~{3,})", line)
        if fence_match:
            fence = fence_match.group(1)
            fence_char = fence[0]
            fence_count = len(fence)

            if code_block_char is None:
                # Start of code block
                code_block_char = fence_char
                code_block_count = fence_count
            elif fence_char == code_block_char and fence_count >= code_block_count:
                # End of code block (same char, count >= opening)
                code_block_char = None
                code_block_count = 0
            # else: fence inside code block, ignore

        # Only process headers and slide comments outside code blocks
        in_code_block = code_block_char is not None

        # ATX headers: 1-6 # followed by space or end of line
        # Excludes: #hashtag, #123, #! (shebang)
        is_atx_header = bool(re.match(r"^#{1,6}(\s|$)", line))

        # Slide comments: <!-- Slide number: X -->
        is_slide_comment = bool(
            re.match(r"^<!--\s*Slide\s+(number:\s*)?\d+\s*-->", line)
        )

        needs_spacing = is_atx_header or is_slide_comment

        if not in_code_block and needs_spacing:
            # Add blank line before if needed
            if result_lines and result_lines[-1] != "":
                result_lines.append("")
            result_lines.append(line)
            # Add blank line after if next line is not empty
            if i + 1 < len(lines) and lines[i + 1] != "":
                result_lines.append("")
        else:
            result_lines.append(line)

    content = "\n".join(result_lines)

    # Merge 3+ consecutive blank lines into 2 (keep one blank line between blocks)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Ensure single newline at end
    return content.strip() + "\n"


def fix_malformed_image_refs(content: str) -> str:
    """Fix malformed image references in markdown.

    Handles common LLM output errors:
    1. Double alt text: `![alt1]![alt2](path)` -> `![alt2](path)`
    2. Empty path: `![alt](assets/)` or `![alt]()` -> removed
    3. Extra closing parenthesis: `![alt](path))` -> `![alt](path)`

    Args:
        content: Markdown content with potential malformed image refs

    Returns:
        Content with fixed image references
    """
    # Fix double alt text: ![alt1]![alt2](path) -> ![alt2](path)
    # This happens when LLM generates two consecutive alt texts
    content = re.sub(
        r"!\[[^\]]*\](!\[[^\]]*\]\([^)]+\))",
        r"\1",
        content,
    )

    # Remove images with empty paths: ![alt](assets/) or ![alt]()
    content = re.sub(r"!\[[^\]]*\]\((?:assets/)?\)\s*\n?", "", content)

    # Fix extra closing parenthesis: ![alt](path)) -> ![alt](path)
    content = re.sub(r"(!\[[^\]]*\]\([^)]+\))\)+", r"\1", content)

    return content


def fix_broken_markdown_links(content: str) -> str:
    """Fix broken markdown links where text and URL are split by newlines.

    Common pattern from web scraping:
    [Title text

    Description text](/url)

    Should become: [Title text](/url)

    Args:
        content: Markdown content with potentially broken links

    Returns:
        Content with fixed links
    """
    # Pattern: [text with newlines inside](url)
    # Captures: [anything with newlines](url) and keeps only first line + url
    pattern = r"\[([^\]]*?)\n+([^\]]*?)\]\(([^)]+)\)"

    def fix_link(match: re.Match[str]) -> str:
        first_part = match.group(1).strip()
        url = match.group(3)
        return f"[{first_part}]({url})"

    # Apply fix iteratively until no more changes
    prev_content = ""
    while prev_content != content:
        prev_content = content
        content = re.sub(pattern, fix_link, content)

    return content


def clean_ppt_headers_footers(content: str) -> str:
    """Clean PPT/PDF headers and footers that appear at the end of each page/slide.

    Pattern: Short lines (< 30 chars each) at the end of page blocks,
    appearing repeatedly across multiple pages.

    Example pattern to remove:
    FTD
    FREE TEST DATA
    2

    Args:
        content: Markdown content with potential headers/footers

    Returns:
        Cleaned content
    """
    # Split by page/slide markers
    page_pattern = r"(<!-- (?:Page|Slide) (?:number: ?)?\d+ -->)"
    parts = re.split(page_pattern, content)

    if len(parts) < 3:
        # Not enough pages to detect pattern
        return content

    # Analyze ending patterns for each page block
    page_endings: list[list[str]] = []

    for i, part in enumerate(parts):
        if re.match(r"<!-- (?:Page|Slide)", part):
            continue
        # Get the content block after a page marker
        if i > 0 and re.match(r"<!-- (?:Page|Slide)", parts[i - 1]):
            # Extract last few lines (potential footer)
            lines = [ln.strip() for ln in part.strip().split("\n") if ln.strip()]
            if len(lines) >= 2:
                # Take last 4 lines as potential footer
                ending = lines[-4:] if len(lines) >= 4 else lines[-len(lines) :]
                # Filter to short lines only (< 30 chars, not starting with # or !)
                short_lines = [
                    ln
                    for ln in ending
                    if len(ln) < 30 and not ln.startswith(("#", "!", "[", "-", "*"))
                ]
                if short_lines:
                    page_endings.append(short_lines)

    if len(page_endings) < 3:
        return content

    # Find common ending pattern (appears in >= 50% of pages)
    from collections import Counter

    # Count each unique ending line
    all_ending_lines: list[str] = []
    for ending in page_endings:
        all_ending_lines.extend(ending)

    line_counts = Counter(all_ending_lines)
    threshold = len(page_endings) * 0.5

    # Lines that appear frequently (excluding pure numbers which are page numbers)
    common_lines = {
        line
        for line, count in line_counts.items()
        if count >= threshold and not line.isdigit()
    }

    if not common_lines:
        return content

    # Remove common footer lines from content
    # Also remove adjacent page numbers (single digit or 2-digit numbers)
    result_lines = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if this line and nearby lines form a footer pattern
        if line in common_lines:
            # Skip this line and check for adjacent page number
            if i + 1 < len(lines) and lines[i + 1].strip().isdigit():
                i += 2  # Skip both
            else:
                i += 1
        elif line.isdigit() and i > 0:
            # Check if previous line was a common footer line
            prev_line = lines[i - 1].strip() if i > 0 else ""
            if prev_line in common_lines or (
                i >= 2 and lines[i - 2].strip() in common_lines
            ):
                i += 1  # Skip page number
            else:
                result_lines.append(lines[i])
                i += 1
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines)


def dedupe_paragraphs(content: str, min_length: int = 100) -> str:
    """Remove duplicate paragraphs from content.

    Useful for cleaning browser-fetched content where the same text
    may appear multiple times (e.g., Twitter/X pages with repeated
    content in og:title, aria-label, and main content).

    Args:
        content: Markdown content with potential duplicate paragraphs
        min_length: Minimum paragraph length to consider for deduplication.
            Shorter paragraphs are always kept (to avoid removing headers, etc.)

    Returns:
        Content with duplicate paragraphs removed (first occurrence kept)
    """
    # Split by double newlines (paragraph separator)
    paragraphs = re.split(r"\n\n+", content)

    seen_paragraphs: set[str] = set()
    result_paragraphs: list[str] = []

    for para in paragraphs:
        para_stripped = para.strip()

        # Skip empty paragraphs
        if not para_stripped:
            continue

        # Normalize for comparison: collapse whitespace
        normalized = re.sub(r"\s+", " ", para_stripped)

        # Short paragraphs: always keep (headers, short lines, etc.)
        if len(normalized) < min_length:
            result_paragraphs.append(para)
            continue

        # Long paragraphs: dedupe
        if normalized not in seen_paragraphs:
            seen_paragraphs.add(normalized)
            result_paragraphs.append(para)
        # else: skip duplicate

    return "\n\n".join(result_paragraphs)


def dedupe_long_text_blocks(content: str, min_length: int = 50) -> str:
    """Remove duplicate long text blocks from content.

    More aggressive deduplication for social media content where the same
    long text appears multiple times in different formatting contexts
    (e.g., Twitter aria-label, og:title, and main content).

    This function:
    1. Finds all "long text blocks" (continuous text >= min_length chars)
    2. Removes duplicate occurrences, keeping the first one
    3. Also handles cases where text is prefixed with usernames/metadata

    Args:
        content: Markdown content with potential duplicate text blocks
        min_length: Minimum text length to consider for deduplication

    Returns:
        Content with duplicate text blocks removed
    """
    lines = content.split("\n")
    result_lines: list[str] = []
    seen_texts: list[str] = []  # Use list to preserve order for substring matching

    for line in lines:
        line_stripped = line.strip()

        # Skip short lines, markdown syntax, or empty lines
        if not line_stripped or len(line_stripped) < min_length:
            result_lines.append(line)
            continue

        # Skip lines that are primarily markdown syntax
        if line_stripped.startswith(("#", "!", "[", "|", "-", "*", ">", "```", "<!--")):
            result_lines.append(line)
            continue

        # Normalize text for comparison
        # Remove leading username patterns (Twitter/X format)
        normalized = re.sub(r"^[A-Za-z\s]+@\w+\s+", "", line_stripped)
        # Remove @ mentions at start
        normalized = re.sub(r"^@\w+\s*", "", normalized)
        # Remove timestamps and metrics
        normalized = re.sub(r"\d+:\d+\s*(AM|PM|am|pm)?\s*·?\s*", "", normalized)
        normalized = re.sub(
            r"\d+\s*(replies|reposts|likes|views|bookmarks)[,\s]*",
            "",
            normalized,
            flags=re.I,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # If still long enough after normalization
        if len(normalized) >= min_length:
            # Check if this text is a duplicate or substring of seen text
            is_duplicate = False
            for seen in seen_texts:
                # Check both directions: new is in seen, or seen is in new
                if normalized in seen or seen in normalized:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_texts.append(normalized)
                result_lines.append(line)
            # else: skip duplicate
        else:
            result_lines.append(line)

    return "\n".join(result_lines)
