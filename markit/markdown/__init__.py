"""Markdown processing module for MarkIt."""

from markit.markdown.chunker import ChunkConfig, MarkdownChunker, SimpleChunker
from markit.markdown.formatter import (
    FormatterConfig,
    MarkdownCleaner,
    MarkdownFormatter,
    clean_markdown,
    format_markdown,
)
from markit.markdown.frontmatter import (
    Frontmatter,
    FrontmatterHandler,
    ImageDescriptionFrontmatter,
    create_frontmatter,
    inject_frontmatter,
)

__all__ = [
    # Chunker
    "ChunkConfig",
    "MarkdownChunker",
    "SimpleChunker",
    # Formatter
    "FormatterConfig",
    "MarkdownFormatter",
    "MarkdownCleaner",
    "format_markdown",
    "clean_markdown",
    # Frontmatter
    "Frontmatter",
    "FrontmatterHandler",
    "ImageDescriptionFrontmatter",
    "create_frontmatter",
    "inject_frontmatter",
]
