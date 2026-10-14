from __future__ import annotations

"""Shared markdown cleanup helpers used before output writing."""


def normalize_markdown(content: str) -> str:
    """Apply shared non-LLM markdown cleanup rules.

    Args:
        content: Markdown content to normalize

    Returns:
        Cleaned markdown content with normalized trailing newline
    """
    from markitai.utils.page_chrome import strip_repeated_marker_lines
    from markitai.utils.text import (
        clean_residual_placeholders,
        drop_stray_table_delimiters,
        fix_broken_markdown_links,
        normalize_markdown_whitespace,
    )

    content = fix_broken_markdown_links(content)
    content = strip_repeated_marker_lines(content)
    content = clean_residual_placeholders(content)
    content = drop_stray_table_delimiters(content)
    content = normalize_markdown_whitespace(content)
    return content.rstrip() + "\n"
