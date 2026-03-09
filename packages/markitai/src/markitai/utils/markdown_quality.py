from __future__ import annotations

"""Shared markdown cleanup helpers used before output writing."""


def normalize_markdown(content: str) -> str:
    """Apply shared non-LLM markdown cleanup rules.

    Args:
        content: Markdown content to normalize

    Returns:
        Cleaned markdown content with normalized trailing newline
    """
    from markitai.utils.text import (
        clean_ppt_headers_footers,
        clean_residual_placeholders,
        fix_broken_markdown_links,
        normalize_markdown_whitespace,
    )

    content = fix_broken_markdown_links(content)
    content = clean_ppt_headers_footers(content)
    content = clean_residual_placeholders(content)
    content = normalize_markdown_whitespace(content)
    return content.rstrip() + "\n"
