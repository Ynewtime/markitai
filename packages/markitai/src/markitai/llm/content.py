"""Content protection and formatting utilities.

This module provides utilities for protecting and restoring content that must
be preserved through LLM processing, such as images, slide markers, and
page number comments.

Usage:
    from markitai.llm.content import protect_content, unprotect_content
"""

from __future__ import annotations

import re

# Pre-compiled regex patterns for hot path functions
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_SLIDE_COMMENT_RE = re.compile(r"<!--\s*Slide\s+(?:number:\s*)?\d+\s*-->")
_PAGE_NUMBER_COMMENT_RE = re.compile(r"<!--\s*Page number:\s*\d+\s*-->")
_PAGE_HEADER_COMMENT_RE = re.compile(r"<!--\s*Page images for reference\s*-->")
_PAGE_IMG_COMMENT_RE = re.compile(r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->")
_PAGE_NUM_MARKER_RE = re.compile(r"<!--\s*Page number:\s*\d+\s*-->")
_SLIDE_NUM_MARKER_RE = re.compile(r"<!--\s*Slide number:\s*\d+\s*-->")
_HALLUCINATED_SLIDE_RE = re.compile(r"<!--\s*Slide\s+number:\s*\d+\s*-->\s*\n?")
_HALLUCINATED_PAGE_RE = re.compile(r"<!--\s*Page\s+number:\s*\d+\s*-->\s*\n?")
_RESIDUAL_PLACEHOLDER_RE = re.compile(r"__MARKITAI_[A-Z]+_\d+__\s*\n?")
_IMG_PATH_RE = re.compile(r"\]\(([^)]+)\)")
_SLIDE_NUM_EXTRACT_RE = re.compile(r"Slide\s+(?:number:\s*)?(\d+)")
_PAGE_NUM_EXTRACT_RE = re.compile(r"Page number:\s*(\d+)")
_PAGE_REF_RE = re.compile(r"!\[Page\s+(\d+)\]")
_UNCOMMENTED_SCREENSHOT_RE = re.compile(r"\n*!\[Page\s+\d+\]\(screenshots/[^)]+\)\s*$")
_CODE_BLOCK_RE = re.compile(
    r"^```(?:ya?ml)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE
)
_PROMPT_LEAKAGE_PATTERNS = [
    re.compile(r"^根据.*生成.*frontmatter.*:.*$", re.IGNORECASE),
    re.compile(r"^请.*生成.*:.*$", re.IGNORECASE),
    re.compile(r"^以下是.*:.*$", re.IGNORECASE),
    re.compile(r"^YAML.*frontmatter.*:.*$", re.IGNORECASE),
    re.compile(r"^元数据.*:.*$", re.IGNORECASE),
]


def smart_truncate(text: str, max_chars: int, preserve_end: bool = False) -> str:
    """Truncate text at sentence/paragraph boundary to preserve readability.

    Instead of cutting at arbitrary positions, finds the nearest sentence
    or paragraph ending before the limit.

    Args:
        text: Text to truncate
        max_chars: Maximum character limit
        preserve_end: If True, preserve the end instead of the beginning

    Returns:
        Truncated text at a natural boundary
    """
    if len(text) <= max_chars:
        return text

    if preserve_end:
        # Find a good starting point from the end
        search_start = len(text) - max_chars
        search_text = text[search_start : search_start + 500]

        # Look for paragraph or sentence boundary
        for marker in ["\n\n", "\n", "。", ".", "！", "!", "？", "?"]:
            idx = search_text.find(marker)
            if idx != -1:
                return text[search_start + idx + len(marker) :]

        return text[-max_chars:]

    # Default: preserve beginning, find a good ending point
    search_text = (
        text[max_chars - 500 : max_chars + 200]
        if max_chars > 500
        else text[: max_chars + 200]
    )
    search_offset = max(0, max_chars - 500)

    # Priority: paragraph > sentence > any break
    for marker in ["\n\n", "。\n", ".\n", "。", ".", "！", "!", "？", "?"]:
        idx = search_text.rfind(marker)
        if idx != -1:
            end_pos = search_offset + idx + len(marker)
            if end_pos <= max_chars + 100:  # Allow slight overflow for better breaks
                return text[:end_pos].rstrip()

    # Fall back to simple truncation
    return text[:max_chars]


def extract_protected_content(content: str) -> dict[str, list[str]]:
    """Extract content that must be preserved through LLM processing.

    Extracts:
    - Image links: ![...](...)
    - Slide comments: <!-- Slide X --> or <!-- Slide number: X -->
    - Page number comments: <!-- Page number: X -->
    - Page image comments: <!-- ![Page X](...) --> and <!-- Page images... -->

    Args:
        content: Original markdown content

    Returns:
        Dict with 'images', 'slides', 'page_numbers', 'page_comments' lists
    """
    protected: dict[str, list[str]] = {
        "images": [],
        "slides": [],
        "page_numbers": [],
        "page_comments": [],
    }

    # Extract image links
    protected["images"] = _IMAGE_LINK_RE.findall(content)

    # Extract slide comments: <!-- Slide X --> or <!-- Slide number: X -->
    protected["slides"] = _SLIDE_COMMENT_RE.findall(content)

    # Extract page number comments: <!-- Page number: X -->
    protected["page_numbers"] = _PAGE_NUMBER_COMMENT_RE.findall(content)

    # Extract page image comments
    # Pattern 1: <!-- Page images for reference -->
    # Pattern 2: <!-- ![Page X](screenshots/...) -->
    protected["page_comments"] = _PAGE_HEADER_COMMENT_RE.findall(
        content
    ) + _PAGE_IMG_COMMENT_RE.findall(content)

    return protected


def protect_content(content: str) -> tuple[str, dict[str, str]]:
    """Replace protected content with placeholders before LLM processing.

    This preserves the position of images, slides, and page comments
    by replacing them with unique placeholders that the LLM is unlikely
    to modify.

    Args:
        content: Original markdown content

    Returns:
        Tuple of (content with placeholders, mapping of placeholder -> original)
    """
    mapping: dict[str, str] = {}
    result = content

    # Note: Images are NOT protected anymore.
    # The prompt instructs LLM to preserve image positions and only add alt text.
    # Protecting images with placeholders caused issues where LLM would delete
    # the placeholders, and then images would be appended to the end of the file.

    # 1. Protect Page number markers (PDF): <!-- Page number: X -->
    # These must stay at the beginning of each page's content
    for page_num_idx, match in enumerate(_PAGE_NUM_MARKER_RE.finditer(result)):
        placeholder = f"__MARKITAI_PAGENUM_{page_num_idx}__"
        mapping[placeholder] = match.group(0)
        result = result.replace(match.group(0), placeholder, 1)

    # 2. Protect Slide number markers (PPTX/PPT): <!-- Slide number: X -->
    # These must stay at the beginning of each slide's content
    for slide_num_idx, match in enumerate(_SLIDE_NUM_MARKER_RE.finditer(result)):
        placeholder = f"__MARKITAI_SLIDENUM_{slide_num_idx}__"
        mapping[placeholder] = match.group(0)
        result = result.replace(match.group(0), placeholder, 1)

    # 3. Protect page image comments: <!-- ![Page X](...) --> and <!-- Page images... -->
    # Use separate patterns for header and individual page image comments
    page_idx = 0
    for match in _PAGE_HEADER_COMMENT_RE.finditer(result):
        placeholder = f"__MARKITAI_PAGE_{page_idx}__"
        mapping[placeholder] = match.group(0)
        result = result.replace(match.group(0), placeholder, 1)
        page_idx += 1
    for match in _PAGE_IMG_COMMENT_RE.finditer(result):
        placeholder = f"__MARKITAI_PAGE_{page_idx}__"
        mapping[placeholder] = match.group(0)
        result = result.replace(match.group(0), placeholder, 1)
        page_idx += 1

    return result, mapping


def unprotect_content(
    content: str,
    mapping: dict[str, str],
    protected: dict[str, list[str]] | None = None,
) -> str:
    """Restore protected content from placeholders after LLM processing.

    Also handles cases where the LLM removed placeholders by appending
    missing content at the end, and detects garbage content replacement.

    Args:
        content: LLM output with placeholders
        mapping: Mapping of placeholder -> original content
        protected: Optional dict of protected content for fallback restoration

    Returns:
        Content with placeholders replaced by original content
    """
    from loguru import logger

    result = content

    # Remove any slide/page number comments that LLM hallucinated
    # These are NOT from our placeholders and should be removed
    # Pattern: <!-- Slide number: X --> or <!-- Page number: X -->

    # Remove hallucinated markers BEFORE replacing placeholders:
    # - If original had markers (placeholders exist): ALL raw markers are hallucinated
    #   because the real ones are protected as __MARKITAI_SLIDENUM_X__ placeholders
    # - If original had NO markers (placeholders empty): ALL raw markers are hallucinated
    # Either way, we should remove all raw slide/page markers at this point
    result = _HALLUCINATED_SLIDE_RE.sub("", result)
    result = _HALLUCINATED_PAGE_RE.sub("", result)

    # First pass: replace placeholders with original content
    # Ensure page/slide number markers have proper blank lines around them
    for placeholder, original in mapping.items():
        # Check if this is a page or slide number marker
        is_page_slide_marker = "PAGENUM" in placeholder or "SLIDENUM" in placeholder
        if is_page_slide_marker:
            # Find the placeholder and ensure blank lines around it
            # Pattern: optional whitespace/newlines before placeholder
            pattern = rf"(\n*)\s*{re.escape(placeholder)}\s*(\n*)"
            match = re.search(pattern, result)
            if match:
                # Replace with proper spacing: \n\n before, \n\n after
                result = re.sub(pattern, f"\n\n{original}\n\n", result, count=1)
            else:
                result = result.replace(placeholder, original)
        else:
            result = result.replace(placeholder, original)

    # Clean up any residual placeholders that LLM might have duplicated or misplaced
    # Pattern: __MARKITAI_*__ (any of our placeholder formats)
    residual_count = len(_RESIDUAL_PLACEHOLDER_RE.findall(result))
    if residual_count > 0:
        logger.debug(f"Removing {residual_count} residual placeholders from LLM output")
        result = _RESIDUAL_PLACEHOLDER_RE.sub("", result)

    # NOTE: Removed heuristic logic that auto-inserted images into short slide sections.
    # This caused false positives where legitimate short slides like "Agenda", "Thanks",
    # "Q&A" were incorrectly replaced with images. The LLM should preserve slide content
    # as-is, and missing images will be handled by the fallback restoration below.

    # Second pass: if protected content was provided, restore any missing items
    # This handles cases where the LLM removed placeholders entirely
    if protected:
        # Helper to check if an image is already in result (by filename)
        def image_exists_in_result(img_syntax: str, text: str) -> bool:
            """Check if image already exists in result by filename."""
            match = _IMG_PATH_RE.search(img_syntax)
            if match:
                img_path = match.group(1)
                img_name = img_path.split("/")[-1]
                # Check if same filename exists in any image reference
                return bool(
                    re.search(rf"!\[[^\]]*\]\([^)]*{re.escape(img_name)}\)", text)
                )
            return False

        # Restore missing images at end (fallback)
        # Only restore if the image filename doesn't already exist
        for img in protected.get("images", []):
            if img not in result and not image_exists_in_result(img, result):
                match = _IMG_PATH_RE.search(img)
                if match:
                    img_name = match.group(1).split("/")[-1]
                    logger.debug(f"Restoring missing image at end: {img_name}")
                result = result.rstrip() + "\n\n" + img

        # Restore missing slide comments at heading boundaries
        # Key fix: Match slides to H1/H2 headings more intelligently
        missing_slides = [s for s in protected.get("slides", []) if s not in result]
        if missing_slides:
            slide_info = []
            for slide in missing_slides:
                # Support both "Slide X" and "Slide number: X" formats
                match = _SLIDE_NUM_EXTRACT_RE.search(slide)
                if match:
                    slide_info.append((int(match.group(1)), slide))
            slide_info.sort()

            lines = result.split("\n")
            # Find H1 and H2 headings as potential slide boundaries
            heading_positions = [
                i
                for i, line in enumerate(lines)
                if line.startswith("# ") or line.startswith("## ")
            ]

            # Only insert if we have matching heading positions
            # Don't append orphan slide comments to the end
            inserted_count = 0
            for idx, (slide_num, slide) in enumerate(slide_info):
                if idx < len(heading_positions):
                    insert_pos = heading_positions[idx] + inserted_count * 2
                    lines.insert(insert_pos, slide)
                    lines.insert(insert_pos + 1, "")
                    inserted_count += 1
                    logger.debug(
                        f"Restored slide {slide_num} before heading at line {insert_pos}"
                    )
                # Don't append orphan slides to the end - they look wrong
            result = "\n".join(lines)

        # Log missing page number markers but do NOT restore them
        # Reason: If LLM removed a page marker, we should respect that decision
        # Programmatic restoration often inserts markers at wrong positions
        missing_page_nums = [
            p for p in protected.get("page_numbers", []) if p not in result
        ]
        if missing_page_nums:
            # Extract page numbers for logging
            missing_nums = []
            for page_marker in missing_page_nums:
                match = _PAGE_NUM_EXTRACT_RE.search(page_marker)
                if match:
                    missing_nums.append(match.group(1))
            if missing_nums:
                logger.debug(
                    f"Page markers not restored (respecting LLM output): {missing_nums}"
                )

        # Restore missing page comments at end
        # Only restore if not already present (avoid duplicates)
        page_header = "<!-- Page images for reference -->"
        has_page_header = page_header in result

        for comment in protected.get("page_comments", []):
            if comment not in result:
                # For page header, only add if not present
                if comment == page_header:
                    if not has_page_header:
                        result = result.rstrip() + "\n\n" + comment
                        has_page_header = True
                # For individual page image comments, check if already exists
                else:
                    # Extract page number to check for duplicates
                    page_match = _PAGE_REF_RE.search(comment)
                    if page_match:
                        page_num = page_match.group(1)
                        # Check if this page is already referenced (commented or not)
                        page_pattern = rf"!\[Page\s+{page_num}\]"
                        if not re.search(page_pattern, result):
                            result = result.rstrip() + "\n" + comment

    return result


def restore_protected_content(result: str, protected: dict[str, list[str]]) -> str:
    """Restore any protected content that was lost during LLM processing.

    Legacy method - use unprotect_content for new code.

    Args:
        result: LLM output
        protected: Dict of protected content from extract_protected_content

    Returns:
        Result with missing protected content restored
    """
    return unprotect_content(result, {}, protected)


def fix_malformed_image_refs(text: str) -> str:
    """Fix malformed image references with extra closing parentheses.

    Fixes cases like: ![alt](path.jpg)) -> ![alt](path.jpg)

    This handles a common LLM output error where extra ) are added
    after image references. Uses context-aware parsing to avoid
    breaking legitimate nested structures like:
    - [![alt](img)](link) - clickable image
    - (text: ![alt](img)) - image inside parentheses

    Args:
        text: Content that may contain malformed image refs

    Returns:
        Content with fixed image references
    """
    result = []
    i = 0
    while i < len(text):
        # Check for image reference start: ![
        if text[i : i + 2] == "![":
            # Find the ]( delimiter
            bracket_end = text.find("](", i + 2)
            if bracket_end != -1:
                # Find the matching ) for the image path
                # Handle nested parens in path like: ![alt](path(1).jpg)
                paren_start = bracket_end + 2
                paren_count = 1
                j = paren_start
                while j < len(text) and paren_count > 0:
                    if text[j] == "(":
                        paren_count += 1
                    elif text[j] == ")":
                        paren_count -= 1
                    j += 1

                # j now points to position after the closing )
                img_ref = text[i:j]
                result.append(img_ref)

                # Count extra ) immediately after the image ref
                extra_parens = 0
                while j + extra_parens < len(text) and text[j + extra_parens] == ")":
                    extra_parens += 1

                if extra_parens > 0:
                    # Check if these ) are legitimate closers for outer parens
                    # by counting unmatched ( in the content before this image
                    prefix = "".join(result[:-1])  # Exclude the image ref just added
                    open_parens = prefix.count("(") - prefix.count(")")

                    # Only keep ) that match unclosed (
                    keep_parens = min(extra_parens, max(0, open_parens))
                    result.append(")" * keep_parens)
                    i = j + extra_parens
                else:
                    i = j
                continue

        result.append(text[i])
        i += 1

    return "".join(result)


def protect_image_positions(
    text: str, exclude_screenshots: bool = False
) -> tuple[str, dict[str, str]]:
    """Protect image positions with unique placeholders.

    This is an alternative protection method that protects images
    in addition to slide/page markers.

    Args:
        text: Original markdown content
        exclude_screenshots: If True, skip screenshot references (screenshots/ paths)

    Returns:
        Tuple of (content with placeholders, mapping of placeholder -> original)
    """
    mapping: dict[str, str] = {}
    result = text

    # Find all image references
    for img_idx, match in enumerate(_IMAGE_LINK_RE.finditer(result)):
        img_ref = match.group(0)
        # Skip screenshot placeholders if requested (handled separately in document processing)
        if exclude_screenshots and "screenshots/" in img_ref:
            continue
        placeholder = f"__MARKITAI_IMG_{img_idx}__"
        mapping[placeholder] = img_ref
        result = result.replace(img_ref, placeholder, 1)

    return result, mapping


def restore_image_positions(text: str, mapping: dict[str, str]) -> str:
    """Restore images from placeholders.

    Args:
        text: Content with image placeholders
        mapping: Mapping of placeholder -> original image markdown

    Returns:
        Content with images restored
    """
    result = text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    return result


def remove_uncommented_screenshots(content: str) -> str:
    """Remove uncommented screenshot references that should be in comments.

    Some LLM outputs incorrectly leave screenshot references as regular
    markdown images instead of HTML comments. This function identifies
    and removes them.

    Args:
        content: Markdown content

    Returns:
        Content with orphan screenshot references removed
    """
    # Pattern: ![Page X](screenshots/...) not inside HTML comment
    # This is a heuristic - we look for screenshot references at the end
    return _UNCOMMENTED_SCREENSHOT_RE.sub("", content)


def clean_frontmatter(frontmatter: str) -> str:
    """Clean frontmatter by removing code block markers, --- markers, and prompt leakage.

    Args:
        frontmatter: Raw frontmatter from LLM

    Returns:
        Clean YAML frontmatter
    """
    from loguru import logger

    frontmatter = frontmatter.strip()

    # Remove code block markers (```yaml, ```yml, ```)
    # Pattern: ```yaml or ```yml at start, ``` at end
    match = _CODE_BLOCK_RE.match(frontmatter)
    if match:
        frontmatter = match.group(1).strip()

    # Remove --- markers
    if frontmatter.startswith("---"):
        frontmatter = frontmatter[3:].strip()
    if frontmatter.endswith("---"):
        frontmatter = frontmatter[:-3].strip()

    # Detect and remove prompt leakage lines (Chinese prompt instructions)
    # These are LLM hallucinations where the prompt text appears in output
    lines = frontmatter.split("\n")
    cleaned_lines = []
    removed_count = 0

    for line in lines:
        is_leakage = False
        for pattern in _PROMPT_LEAKAGE_PATTERNS:
            if pattern.match(line.strip()):
                is_leakage = True
                removed_count += 1
                break
        if not is_leakage:
            cleaned_lines.append(line)

    if removed_count > 0:
        logger.debug(f"Removed {removed_count} prompt leakage line(s) from frontmatter")
        frontmatter = "\n".join(cleaned_lines).strip()

    return frontmatter


def split_text_by_pages(text: str, num_pages: int) -> list[str]:
    """Split text into chunks corresponding to page ranges.

    Split strategy (in priority order):
    1. Remove trailing page image reference section first
    2. Use <!-- Slide number: N --> markers (PPTX/PPT)
    3. Use <!-- Page number: N --> markers (PDF)
    4. Fallback: split by paragraphs proportionally

    Args:
        text: Full document text
        num_pages: Number of pages/images

    Returns:
        List of text chunks, one per page
    """
    # Step 1: Remove trailing page image reference section
    # These are screenshot references at the end, not content separators
    ref_marker = "<!-- Page images for reference -->"
    ref_idx = text.find(ref_marker)
    if ref_idx != -1:
        main_content = text[:ref_idx].rstrip()
    else:
        main_content = text

    # Step 2: Try slide markers (PPTX/PPT)
    slide_pattern = r"<!-- Slide number: (\d+) -->"
    slide_markers = list(re.finditer(slide_pattern, main_content))

    if len(slide_markers) >= num_pages:
        # Use slide markers to split - each chunk starts with its slide marker
        chunks = []
        for i in range(num_pages):
            start = slide_markers[i].start()
            if i + 1 < len(slide_markers):
                end = slide_markers[i + 1].start()
            else:
                end = len(main_content)
            chunks.append(main_content[start:end].strip())
        return chunks

    # Step 3: Try page markers (PDF)
    page_pattern = r"<!-- Page number: (\d+) -->"
    page_markers = list(re.finditer(page_pattern, main_content))

    if len(page_markers) >= num_pages:
        # Use page markers to split - each chunk starts with its page marker
        chunks = []
        for i in range(num_pages):
            start = page_markers[i].start()
            if i + 1 < len(page_markers):
                end = page_markers[i + 1].start()
            else:
                end = len(main_content)
            chunks.append(main_content[start:end].strip())
        return chunks

    # Step 4: Fallback - split by paragraphs proportionally
    paragraphs = main_content.split("\n\n")
    if len(paragraphs) < num_pages:
        # Very short text, just return whole text for each page
        return [main_content] * num_pages

    paragraphs_per_page = len(paragraphs) // num_pages
    chunks = []
    for i in range(num_pages):
        start_idx = i * paragraphs_per_page
        if i == num_pages - 1:
            # Last chunk gets remaining paragraphs
            end_idx = len(paragraphs)
        else:
            end_idx = start_idx + paragraphs_per_page
        chunks.append("\n\n".join(paragraphs[start_idx:end_idx]))

    return chunks


__all__ = [
    # Primary protection functions
    "extract_protected_content",
    "protect_content",
    "unprotect_content",
    # Image-specific protection
    "protect_image_positions",
    "restore_image_positions",
    # Content fixing
    "fix_malformed_image_refs",
    "remove_uncommented_screenshots",
    # Frontmatter utilities
    "clean_frontmatter",
    # Text utilities
    "smart_truncate",
    "split_text_by_pages",
    # Legacy alias
    "restore_protected_content",
]
