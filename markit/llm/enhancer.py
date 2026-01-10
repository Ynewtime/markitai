"""LLM-powered Markdown enhancement."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from markit.llm.base import LLMMessage, LLMTaskResultWithStats
from markit.llm.manager import ProviderManager
from markit.markdown.chunker import ChunkConfig, MarkdownChunker
from markit.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for Markdown enhancement."""

    remove_headers_footers: bool = True
    fix_heading_levels: bool = True
    normalize_blank_lines: bool = True
    follow_gfm: bool = True
    add_frontmatter: bool = True
    generate_summary: bool = True
    # Optimized for large context models (Gemini 3 Flash: 1.05M context)
    chunk_size: int = 32000


@dataclass
class EnhancedMarkdown:
    """Result of Markdown enhancement."""

    content: str
    summary: str
    # LLM statistics aggregated from all chunk processing
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_estimated_cost: float = 0.0
    models_used: list[str] | None = None


# Enhancement prompt
ENHANCEMENT_PROMPT = """Please optimize the following Markdown document's format according to these rules:

1. **Clean up junk content**:
   - Remove headers, footers, watermarks, and meaningless repeated characters
   - For PowerPoint slides: remove repetitive footer text (company names, dates, slide numbers, copyright notices) that appear on every slide
   - Remove standalone numbers that are likely page/slide numbers (e.g., just "1", "2", etc. on their own lines)
2. **Heading levels**: Ensure headings start from ## (h2), avoid multiple # (h1)
3. **Blank line normalization**:
   - One blank line above headings
   - One blank line below headings before content
   - One blank line between paragraphs
4. **Follow GFM specification**:
   - Use `-` for unordered list markers
   - Use ``` with language identifier for code blocks
   - Properly align tables
5. **Keep content complete**: Do NOT delete or modify actual content, only optimize formatting

Original Markdown:
```markdown
{content}
```

Output the optimized Markdown (without ```markdown markers):"""

SUMMARY_PROMPT = """Summarize the following document in one sentence (maximum 100 characters):

{content}

Summary:"""


class MarkdownEnhancer:
    """Enhances Markdown content using LLM."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        config: EnhancementConfig | None = None,
        use_concurrent_fallback: bool = False,
    ) -> None:
        """Initialize the enhancer.

        Args:
            provider_manager: LLM provider manager
            config: Enhancement configuration
            use_concurrent_fallback: If True, use concurrent fallback for LLM calls
                                     (starts backup model if primary exceeds timeout)
        """
        self.provider_manager = provider_manager
        self.config = config or EnhancementConfig()
        self.use_concurrent_fallback = use_concurrent_fallback
        self.chunker = MarkdownChunker(ChunkConfig(max_tokens=self.config.chunk_size))

    async def enhance(
        self,
        markdown: str,
        source_file: Path,
        semaphore: asyncio.Semaphore | None = None,
        return_stats: bool = False,
    ) -> EnhancedMarkdown | LLMTaskResultWithStats:
        """Enhance a Markdown document.

        Enhancement flow:
        1. Check document size, chunk if necessary
        2. Call LLM to clean and standardize
        3. Merge results
        4. Inject Frontmatter

        Args:
            markdown: Raw Markdown content
            source_file: Original source file path
            semaphore: Optional semaphore for rate limiting LLM calls
            return_stats: If True, return LLMTaskResultWithStats wrapping the result

        Returns:
            Enhanced Markdown with metadata, or LLMTaskResultWithStats if return_stats=True
        """
        log.info("Enhancing Markdown", file=str(source_file))

        # Chunk if needed
        chunks = self.chunker.chunk(markdown)
        log.debug("Document split into chunks", count=len(chunks))

        # Track statistics
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        models_used: set[str] = set()

        # Process chunks concurrently (with optional rate limiting)
        async def process_with_limit(chunk: str) -> tuple[str, dict]:
            if semaphore:
                async with semaphore:
                    return await self._process_chunk_with_stats(chunk)
            return await self._process_chunk_with_stats(chunk)

        chunk_results = await asyncio.gather(*[process_with_limit(chunk) for chunk in chunks])

        # Collect results and stats
        enhanced_chunks = []
        for content, stats in chunk_results:
            enhanced_chunks.append(content)
            total_prompt_tokens += stats.get("prompt_tokens", 0)
            total_completion_tokens += stats.get("completion_tokens", 0)
            total_cost += stats.get("estimated_cost", 0.0)
            if stats.get("model"):
                models_used.add(stats["model"])

        # Merge chunks
        enhanced_markdown = self.chunker.merge(enhanced_chunks)

        # Generate summary
        summary = ""
        if self.config.generate_summary:
            summary, summary_stats = await self._generate_summary_with_stats(enhanced_markdown)
            total_prompt_tokens += summary_stats.get("prompt_tokens", 0)
            total_completion_tokens += summary_stats.get("completion_tokens", 0)
            total_cost += summary_stats.get("estimated_cost", 0.0)
            if summary_stats.get("model"):
                models_used.add(summary_stats["model"])

        # Inject frontmatter
        if self.config.add_frontmatter:
            enhanced_markdown = self._inject_frontmatter(enhanced_markdown, source_file, summary)

        log.info("Markdown enhancement complete", file=str(source_file))

        result = EnhancedMarkdown(
            content=enhanced_markdown,
            summary=summary,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_estimated_cost=total_cost,
            models_used=list(models_used) if models_used else None,
        )

        if return_stats:
            return LLMTaskResultWithStats(
                result=result,
                model=list(models_used)[0] if models_used else None,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                estimated_cost=total_cost,
            )
        return result

    async def _process_chunk_with_stats(self, chunk: str) -> tuple[str, dict]:
        """Process a single chunk with LLM and return stats.

        Args:
            chunk: Markdown chunk to process

        Returns:
            Tuple of (enhanced_chunk, stats_dict)
        """
        prompt = ENHANCEMENT_PROMPT.format(content=chunk)
        stats: dict = {}

        try:
            # Use concurrent fallback for potentially long-running chunk processing
            if self.use_concurrent_fallback:
                response = await self.provider_manager.complete_with_concurrent_fallback(
                    messages=[LLMMessage.user(prompt)],
                    temperature=0.3,  # Lower temperature for more consistent formatting
                    required_capability="text",  # Text-only task
                )
            else:
                response = await self.provider_manager.complete_with_fallback(
                    messages=[LLMMessage.user(prompt)],
                    temperature=0.3,  # Lower temperature for more consistent formatting
                    required_capability="text",  # Text-only task
                )
            stats = {
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "estimated_cost": response.estimated_cost or 0.0,
            }
            return response.content.strip(), stats
        except Exception as e:
            log.warning("Chunk enhancement failed, returning original", error=str(e))
            return chunk, stats

    async def _process_chunk(self, chunk: str) -> str:
        """Process a single chunk with LLM (legacy method for backward compatibility).

        Args:
            chunk: Markdown chunk to process

        Returns:
            Enhanced chunk
        """
        content, _ = await self._process_chunk_with_stats(chunk)
        return content

    async def _generate_summary_with_stats(self, markdown: str) -> tuple[str, dict]:
        """Generate a one-sentence summary with stats.

        Args:
            markdown: Full Markdown content

        Returns:
            Tuple of (summary_string, stats_dict)
        """
        # Use first 2000 characters for summary
        preview = markdown[:2000]
        prompt = SUMMARY_PROMPT.format(content=preview)
        stats: dict = {}

        try:
            # Use concurrent fallback if enabled
            if self.use_concurrent_fallback:
                response = await self.provider_manager.complete_with_concurrent_fallback(
                    messages=[LLMMessage.user(prompt)],
                    temperature=0.5,
                    max_tokens=150,
                    required_capability="text",  # Text-only task
                )
            else:
                response = await self.provider_manager.complete_with_fallback(
                    messages=[LLMMessage.user(prompt)],
                    temperature=0.5,
                    max_tokens=150,
                    required_capability="text",  # Text-only task
                )
            stats = {
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "estimated_cost": response.estimated_cost or 0.0,
            }
            return response.content.strip()[:100], stats
        except Exception as e:
            log.warning("Summary generation failed", error=str(e))
            return "", stats

    async def _generate_summary(self, markdown: str) -> str:
        """Generate a one-sentence summary (legacy method for backward compatibility).

        Args:
            markdown: Full Markdown content

        Returns:
            Summary string
        """
        summary, _ = await self._generate_summary_with_stats(markdown)
        return summary

    def _inject_frontmatter(
        self,
        markdown: str,
        source_file: Path,
        summary: str,
    ) -> str:
        """Inject YAML frontmatter into the document.

        Args:
            markdown: Enhanced Markdown content
            source_file: Original source file
            summary: Generated summary

        Returns:
            Markdown with frontmatter
        """
        # Escape quotes in summary
        safe_summary = summary.replace('"', '\\"')
        safe_title = source_file.stem.replace('"', '\\"')

        frontmatter = f'''---
title: "{safe_title}"
processed: "{datetime.now().isoformat()}"
description: "{safe_summary}"
source: "{source_file.name}"
---

'''
        return frontmatter + markdown


class SimpleMarkdownCleaner:
    """Simple Markdown cleanup without LLM."""

    def clean(self, markdown: str) -> str:
        """Apply basic cleanup rules.

        Args:
            markdown: Raw Markdown

        Returns:
            Cleaned Markdown
        """
        import re

        result = markdown

        # Normalize line endings
        result = result.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (more than 2 consecutive)
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Ensure blank line before headings
        result = re.sub(r"([^\n])\n(#{1,6} )", r"\1\n\n\2", result)

        # Ensure blank line after headings
        result = re.sub(r"(#{1,6} [^\n]+)\n([^\n#])", r"\1\n\n\2", result)

        # Standardize list markers to -
        result = re.sub(r"^(\s*)[*+] ", r"\1- ", result, flags=re.MULTILINE)

        # Remove trailing whitespace
        result = "\n".join(line.rstrip() for line in result.split("\n"))

        # Ensure file ends with single newline
        result = result.strip() + "\n"

        return result
