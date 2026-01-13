"""LLM-powered Markdown enhancement."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from markit.llm.base import LLMMessage, LLMTaskResultWithStats
from markit.llm.manager import ProviderManager
from markit.markdown.chunker import ChunkConfig, MarkdownChunker
from markit.markdown.frontmatter import FrontmatterHandler, create_frontmatter
from markit.utils.logging import get_logger, set_request_context

if TYPE_CHECKING:
    from markit.config.settings import PromptConfig

# Type alias for supported languages
LanguageCode = Literal["zh", "en", "auto"]

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


def get_enhancement_prompt(language: str = "zh") -> str:
    """Get the enhancement prompt for the specified language.

    Loads prompt from package files in markit.config.prompts.

    Args:
        language: Language code ("zh" for Chinese, "en" for English)

    Returns:
        Enhancement prompt string

    Raises:
        RuntimeError: If prompt file cannot be loaded
    """
    from markit.config.settings import PromptConfig

    # Cast to LanguageCode - valid values are "zh", "en", "auto"
    config = PromptConfig(output_language=cast(LanguageCode, language))
    prompt = config.get_prompt("enhancement")
    if not prompt:
        raise RuntimeError(f"Failed to load enhancement_{language}.md prompt file")
    return prompt


def get_summary_prompt(language: str = "zh") -> str:
    """Get the summary prompt for the specified language.

    Loads prompt from package files in markit.config.prompts.

    Args:
        language: Language code ("zh" for Chinese, "en" for English)

    Returns:
        Summary prompt string

    Raises:
        RuntimeError: If prompt file cannot be loaded
    """
    from markit.config.settings import PromptConfig

    # Cast to LanguageCode - valid values are "zh", "en", "auto"
    config = PromptConfig(output_language=cast(LanguageCode, language))
    prompt = config.get_prompt("summary")
    if not prompt:
        raise RuntimeError(f"Failed to load summary_{language}.md prompt file")
    return prompt


class MarkdownEnhancer:
    """Enhances Markdown content using LLM."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        config: EnhancementConfig | None = None,
        use_concurrent_fallback: bool = False,
        prompt_config: "PromptConfig | None" = None,
    ) -> None:
        """Initialize the enhancer.

        Args:
            provider_manager: LLM provider manager
            config: Enhancement configuration
            use_concurrent_fallback: If True, use concurrent fallback for LLM calls
                                     (starts backup model if primary exceeds timeout)
            prompt_config: Optional prompt configuration for customizing prompts.
                          If not provided, uses builtin prompts.
        """
        from markit.config.settings import PromptConfig

        self.provider_manager = provider_manager
        self.config = config or EnhancementConfig()
        self.use_concurrent_fallback = use_concurrent_fallback
        # Ensure prompt_config is always available for loading prompts from files
        self.prompt_config = prompt_config or PromptConfig()
        self.chunker = MarkdownChunker(ChunkConfig(max_tokens=self.config.chunk_size))

    def _get_enhancement_prompt(self, is_first_chunk: bool = True) -> str:
        """Get the enhancement prompt from config files.

        Args:
            is_first_chunk: If True, returns the full prompt with frontmatter instructions.
                           If False, returns continuation prompt without frontmatter.

        Returns:
            Enhancement prompt string with {content} placeholder

        Raises:
            RuntimeError: If prompt file cannot be loaded
        """
        prompt_type = "enhancement" if is_first_chunk else "enhancement_continuation"
        prompt = self.prompt_config.get_prompt(prompt_type)
        if not prompt:
            raise RuntimeError(
                f"Failed to load {prompt_type} prompt. "
                f"Check that prompt files exist in markit.config.prompts package."
            )
        return prompt

    def _get_summary_prompt(self) -> str:
        """Get the summary prompt from config files.

        Returns:
            Summary prompt string with {content} placeholder

        Raises:
            RuntimeError: If prompt file cannot be loaded
        """
        prompt = self.prompt_config.get_prompt("summary")
        if not prompt:
            raise RuntimeError(
                "Failed to load summary prompt. "
                "Check that prompt files exist in markit.config.prompts package."
            )
        return prompt

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
        # Set file context for all LLM-related logs
        set_request_context(file_path=source_file.name)

        # Chunk if needed
        chunks = self.chunker.chunk(markdown)
        log.debug("Document split into chunks", count=len(chunks), file=source_file.name)

        # Track statistics
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        models_used: set[str] = set()

        # Process chunks concurrently (with optional rate limiting)
        # First chunk uses full prompt (with frontmatter), subsequent chunks use continuation prompt
        async def process_with_limit(chunk: str, chunk_index: int) -> tuple[str, dict]:
            is_first = chunk_index == 0
            if semaphore:
                async with semaphore:
                    return await self._process_chunk_with_stats(chunk, is_first_chunk=is_first)
            return await self._process_chunk_with_stats(chunk, is_first_chunk=is_first)

        chunk_results = await asyncio.gather(
            *[process_with_limit(chunk, i) for i, chunk in enumerate(chunks)]
        )

        # Collect results and stats, extract partial metadata from continuation chunks
        enhanced_chunks = []
        all_partial_entities: list[str] = []
        all_partial_topics: list[str] = []

        for i, (content, stats) in enumerate(chunk_results):
            # For non-first chunks, extract and remove PARTIAL_METADATA from content
            if i > 0:
                content, partial_meta = self._extract_partial_metadata(content)
                if partial_meta:
                    all_partial_entities.extend(partial_meta.get("entities", []))
                    all_partial_topics.extend(partial_meta.get("topics", []))

            enhanced_chunks.append(content)
            total_prompt_tokens += stats.get("prompt_tokens", 0)
            total_completion_tokens += stats.get("completion_tokens", 0)
            total_cost += stats.get("estimated_cost", 0.0)
            if stats.get("model"):
                models_used.add(stats["model"])

        # Merge chunks
        enhanced_markdown = self.chunker.merge(enhanced_chunks)

        # Merge partial metadata from all continuation chunks (deduplicated)
        merged_partial_metadata = {
            "entities": list(dict.fromkeys(all_partial_entities)),  # preserve order, dedupe
            "topics": list(dict.fromkeys(all_partial_topics)),
        }

        # Extract summary from LLM-generated frontmatter (description field)
        # No separate LLM call needed - summary is generated as part of enhancement
        summary = self._extract_summary_from_frontmatter(enhanced_markdown)

        # Inject frontmatter (with merged partial metadata from all chunks)
        if self.config.add_frontmatter:
            enhanced_markdown = self._inject_frontmatter(
                enhanced_markdown, source_file, summary, merged_partial_metadata
            )

        # provider and model are auto-injected from context set by manager.py
        log.info("Markdown enhancement complete", file=source_file.name)

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

    async def _process_chunk_with_stats(
        self, chunk: str, is_first_chunk: bool = True
    ) -> tuple[str, dict]:
        """Process a single chunk with LLM and return stats.

        Args:
            chunk: Markdown chunk to process
            is_first_chunk: If True, use full prompt with frontmatter instructions.
                           If False, use continuation prompt.

        Returns:
            Tuple of (enhanced_chunk, stats_dict)
        """
        prompt = self._get_enhancement_prompt(is_first_chunk).format(content=chunk)
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

    def _extract_partial_metadata(self, content: str) -> tuple[str, dict | None]:
        """Extract PARTIAL_METADATA from continuation chunk content.

        The continuation prompt instructs LLM to output metadata at the end as:
        <!-- PARTIAL_METADATA: {"entities": [...], "topics": [...]} -->

        Args:
            content: Enhanced chunk content

        Returns:
            Tuple of (content_without_metadata, metadata_dict or None)
        """
        import json
        import re

        # Pattern to match PARTIAL_METADATA HTML comment at the end
        pattern = re.compile(
            r"\s*<!--\s*PARTIAL_METADATA:\s*(\{.*?\})\s*-->\s*$",
            re.DOTALL,
        )

        match = pattern.search(content)
        if not match:
            return content, None

        try:
            metadata = json.loads(match.group(1))
            # Remove the metadata comment from content
            clean_content = content[: match.start()].rstrip()
            return clean_content, metadata
        except json.JSONDecodeError as e:
            log.warning("Failed to parse PARTIAL_METADATA JSON", error=str(e))
            return content, None

    def _extract_summary_from_frontmatter(self, markdown: str) -> str:
        """Extract description from LLM-generated frontmatter as summary.

        The LLM is instructed to generate a description field in the frontmatter,
        which serves as the document summary. This eliminates the need for a
        separate LLM call to generate the summary.

        Args:
            markdown: Enhanced Markdown content with LLM-generated frontmatter

        Returns:
            Summary string extracted from description field, or empty string
        """
        try:
            handler = FrontmatterHandler()
            frontmatter, _ = handler.parse(markdown)
            if frontmatter and frontmatter.description:
                # Limit to 100 characters
                return frontmatter.description[:100]
        except Exception as e:
            log.warning("Failed to extract summary from frontmatter", error=str(e))
        return ""

    async def _generate_summary_with_stats(self, markdown: str) -> tuple[str, dict]:
        """Generate a one-sentence summary with stats.

        Args:
            markdown: Full Markdown content

        Returns:
            Tuple of (summary_string, stats_dict)
        """
        # Use first 2000 characters for summary
        preview = markdown[:2000]
        prompt = self._get_summary_prompt().format(content=preview)
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
        partial_metadata: dict | None = None,
    ) -> str:
        """Inject YAML frontmatter into the document.

        Args:
            markdown: Enhanced Markdown content
            source_file: Original source file
            summary: Generated summary
            partial_metadata: Optional dict with entities/topics from continuation chunks

        Returns:
            Markdown with frontmatter
        """
        import re

        # 1. Strip outer code blocks if present (e.g., ```markdown ... ```)
        # Some LLMs wrap the entire response in code blocks despite instructions
        # Use regex that allows leading whitespace/newlines
        code_block_pattern = re.compile(r"^\s*```(?:markdown)?\s*\n(.*?)\n```\s*$", re.DOTALL)
        match = code_block_pattern.match(markdown)
        if match:
            markdown = match.group(1)

        # 2. Parse any existing frontmatter (generated by LLM)
        handler = FrontmatterHandler()
        llm_frontmatter, content = handler.parse(markdown)

        # 2.5 Fallback: If parse failed, check if frontmatter is wrapped in ```yaml
        if not llm_frontmatter:
            # Pattern to find frontmatter wrapped in code blocks at the start
            # Matches: ```yaml\n---\n...---\n``` followed by content
            # Group 1: frontmatter block (with trailing newline for FRONTMATTER_PATTERN)
            # Group 2: remaining content
            wrapped_fm_pattern = re.compile(
                r"^\s*```(?:yaml)?\s*\n(---\n[\s\S]*?\n---\n)```\s*\n?([\s\S]*)$"
            )
            match_wrapped = wrapped_fm_pattern.match(markdown)
            if match_wrapped:
                fm_str = match_wrapped.group(1)
                content_rest = match_wrapped.group(2).strip()
                # Try parsing the extracted frontmatter string
                llm_frontmatter, _ = handler.parse(fm_str)
                if llm_frontmatter:
                    # Successfully extracted frontmatter from wrapped block
                    content = content_rest
                else:
                    # Parsing failed even after extraction - log warning
                    log.warning(
                        "Failed to parse extracted frontmatter from code block",
                        fm_preview=fm_str[:100] if fm_str else None,
                    )

        # 3. Create system frontmatter
        system_frontmatter = create_frontmatter(source_file, summary)

        # 4. Merge fields if LLM frontmatter exists
        if llm_frontmatter:
            # Update system frontmatter with LLM extracted data
            # We want to keep system fields (title, processed, etc.) but add LLM fields (entities, etc.)
            if llm_frontmatter.entities:
                system_frontmatter.entities = llm_frontmatter.entities
            if llm_frontmatter.topics:
                system_frontmatter.topics = llm_frontmatter.topics
            if llm_frontmatter.domain:
                system_frontmatter.domain = llm_frontmatter.domain

            # Merge extra fields that aren't system fields
            for k, v in llm_frontmatter.extra.items():
                if k not in ["title", "processed", "description", "source"]:
                    system_frontmatter.extra[k] = v

        # 4.5 Merge partial metadata from continuation chunks
        if partial_metadata:
            # Merge entities (deduplicate while preserving order)
            if partial_metadata.get("entities"):
                existing_entities = system_frontmatter.entities or []
                all_entities = existing_entities + partial_metadata["entities"]
                # Deduplicate while preserving order
                system_frontmatter.entities = list(dict.fromkeys(all_entities))

            # Merge topics (deduplicate while preserving order)
            if partial_metadata.get("topics"):
                existing_topics = system_frontmatter.topics or []
                all_topics = existing_topics + partial_metadata["topics"]
                # Deduplicate while preserving order
                system_frontmatter.topics = list(dict.fromkeys(all_topics))

        # 5. Recombine
        return handler.add(content, system_frontmatter)


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
