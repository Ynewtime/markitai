"""Tests for LLM-powered Markdown enhancement."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from markit.llm.base import LLMResponse, TokenUsage
from markit.llm.enhancer import (
    ENHANCEMENT_PROMPT_EN,
    ENHANCEMENT_PROMPT_ZH,
    SUMMARY_PROMPT_EN,
    SUMMARY_PROMPT_ZH,
    EnhancedMarkdown,
    EnhancementConfig,
    MarkdownEnhancer,
    SimpleMarkdownCleaner,
    get_enhancement_prompt,
    get_summary_prompt,
)


class TestEnhancementConfig:
    """Tests for EnhancementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EnhancementConfig()

        assert config.remove_headers_footers is True
        assert config.fix_heading_levels is True
        assert config.normalize_blank_lines is True
        assert config.follow_gfm is True
        assert config.add_frontmatter is True
        assert config.generate_summary is True
        assert config.chunk_size == 32000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EnhancementConfig(
            remove_headers_footers=False,
            chunk_size=16000,
        )

        assert config.remove_headers_footers is False
        assert config.chunk_size == 16000


class TestEnhancedMarkdown:
    """Tests for EnhancedMarkdown dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = EnhancedMarkdown(content="# Test", summary="A test document")

        assert result.content == "# Test"
        assert result.summary == "A test document"
        assert result.total_prompt_tokens == 0
        assert result.total_completion_tokens == 0
        assert result.total_estimated_cost == 0.0
        assert result.models_used is None

    def test_with_stats(self):
        """Test with statistics."""
        result = EnhancedMarkdown(
            content="# Test",
            summary="Summary",
            total_prompt_tokens=100,
            total_completion_tokens=50,
            total_estimated_cost=0.001,
            models_used=["gpt-4o", "claude-3-sonnet"],
        )

        assert result.total_prompt_tokens == 100
        assert result.total_completion_tokens == 50
        assert result.total_estimated_cost == 0.001
        assert result.models_used == ["gpt-4o", "claude-3-sonnet"]


class TestPromptFunctions:
    """Tests for prompt helper functions."""

    def test_get_enhancement_prompt_zh(self):
        """Test getting Chinese enhancement prompt."""
        prompt = get_enhancement_prompt("zh")
        assert prompt == ENHANCEMENT_PROMPT_ZH
        assert "请按照以下规则优化" in prompt

    def test_get_enhancement_prompt_en(self):
        """Test getting English enhancement prompt."""
        prompt = get_enhancement_prompt("en")
        assert prompt == ENHANCEMENT_PROMPT_EN
        assert "Please optimize" in prompt

    def test_get_enhancement_prompt_default(self):
        """Test default is Chinese."""
        prompt = get_enhancement_prompt()
        assert prompt == ENHANCEMENT_PROMPT_ZH

    def test_get_summary_prompt_zh(self):
        """Test getting Chinese summary prompt."""
        prompt = get_summary_prompt("zh")
        assert prompt == SUMMARY_PROMPT_ZH
        assert "请用一句话概括" in prompt

    def test_get_summary_prompt_en(self):
        """Test getting English summary prompt."""
        prompt = get_summary_prompt("en")
        assert prompt == SUMMARY_PROMPT_EN
        assert "Summarize" in prompt


class TestMarkdownEnhancerInit:
    """Tests for MarkdownEnhancer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        mock_manager = MagicMock()
        enhancer = MarkdownEnhancer(mock_manager)

        assert enhancer.provider_manager is mock_manager
        assert isinstance(enhancer.config, EnhancementConfig)
        assert enhancer.use_concurrent_fallback is False
        assert enhancer.prompt_config is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        mock_manager = MagicMock()
        config = EnhancementConfig(chunk_size=16000)
        enhancer = MarkdownEnhancer(mock_manager, config=config)

        assert enhancer.config.chunk_size == 16000

    def test_init_with_concurrent_fallback(self):
        """Test initialization with concurrent fallback enabled."""
        mock_manager = MagicMock()
        enhancer = MarkdownEnhancer(mock_manager, use_concurrent_fallback=True)

        assert enhancer.use_concurrent_fallback is True


class TestMarkdownEnhancerPrompts:
    """Tests for MarkdownEnhancer prompt methods."""

    def test_get_enhancement_prompt_default(self):
        """Test getting default enhancement prompt."""
        mock_manager = MagicMock()
        enhancer = MarkdownEnhancer(mock_manager)

        prompt = enhancer._get_enhancement_prompt()
        assert "{content}" in prompt

    def test_get_enhancement_prompt_custom(self):
        """Test getting custom enhancement prompt from config."""
        mock_manager = MagicMock()
        mock_prompt_config = MagicMock()
        mock_prompt_config.get_prompt.return_value = "Custom: {content}"
        enhancer = MarkdownEnhancer(mock_manager, prompt_config=mock_prompt_config)

        prompt = enhancer._get_enhancement_prompt()
        assert prompt == "Custom: {content}"
        mock_prompt_config.get_prompt.assert_called_with("enhancement")

    def test_get_summary_prompt_default(self):
        """Test getting default summary prompt."""
        mock_manager = MagicMock()
        enhancer = MarkdownEnhancer(mock_manager)

        prompt = enhancer._get_summary_prompt()
        assert "{content}" in prompt

    def test_get_summary_prompt_custom(self):
        """Test getting custom summary prompt from config."""
        mock_manager = MagicMock()
        mock_prompt_config = MagicMock()
        mock_prompt_config.get_prompt.return_value = "Summarize: {content}"
        enhancer = MarkdownEnhancer(mock_manager, prompt_config=mock_prompt_config)

        prompt = enhancer._get_summary_prompt()
        assert prompt == "Summarize: {content}"


class TestMarkdownEnhancerEnhance:
    """Tests for MarkdownEnhancer.enhance method."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock provider manager."""
        manager = MagicMock()
        manager.complete_with_fallback = AsyncMock(
            return_value=LLMResponse(
                content="---\ndescription: Test summary\n---\n\n# Enhanced",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
                model="gpt-4o",
                finish_reason="stop",
                estimated_cost=0.001,
            )
        )
        return manager

    @pytest.mark.asyncio
    async def test_enhance_basic(self, mock_manager):
        """Test basic enhancement."""
        enhancer = MarkdownEnhancer(mock_manager)

        result = await enhancer.enhance(
            markdown="# Test\n\nContent",
            source_file=Path("test.md"),
        )

        assert isinstance(result, EnhancedMarkdown)
        assert result.total_prompt_tokens == 100
        assert result.total_completion_tokens == 50
        mock_manager.complete_with_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_with_return_stats(self, mock_manager):
        """Test enhancement with return_stats=True."""
        enhancer = MarkdownEnhancer(mock_manager)

        result = await enhancer.enhance(
            markdown="# Test",
            source_file=Path("test.md"),
            return_stats=True,
        )

        from markit.llm.base import LLMTaskResultWithStats

        assert isinstance(result, LLMTaskResultWithStats)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50

    @pytest.mark.asyncio
    async def test_enhance_with_concurrent_fallback(self):
        """Test enhancement with concurrent fallback enabled."""
        mock_manager = MagicMock()
        mock_manager.complete_with_concurrent_fallback = AsyncMock(
            return_value=LLMResponse(
                content="# Enhanced",
                usage=TokenUsage(prompt_tokens=50, completion_tokens=25),
                model="gpt-4o",
                finish_reason="stop",
            )
        )
        enhancer = MarkdownEnhancer(mock_manager, use_concurrent_fallback=True)

        await enhancer.enhance(markdown="# Test", source_file=Path("test.md"))

        mock_manager.complete_with_concurrent_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_with_semaphore(self, mock_manager):
        """Test enhancement with rate limiting semaphore."""
        import asyncio

        semaphore = asyncio.Semaphore(1)
        enhancer = MarkdownEnhancer(mock_manager)

        await enhancer.enhance(
            markdown="# Test",
            source_file=Path("test.md"),
            semaphore=semaphore,
        )

        mock_manager.complete_with_fallback.assert_called()


class TestMarkdownEnhancerProcessChunk:
    """Tests for MarkdownEnhancer chunk processing."""

    @pytest.fixture
    def enhancer(self):
        """Create an enhancer with mocked manager."""
        mock_manager = MagicMock()
        mock_manager.complete_with_fallback = AsyncMock(
            return_value=LLMResponse(
                content="Enhanced chunk",
                usage=TokenUsage(prompt_tokens=50, completion_tokens=25),
                model="gpt-4o",
                finish_reason="stop",
                estimated_cost=0.0005,
            )
        )
        return MarkdownEnhancer(mock_manager)

    @pytest.mark.asyncio
    async def test_process_chunk_with_stats(self, enhancer):
        """Test processing chunk returns stats."""
        content, stats = await enhancer._process_chunk_with_stats("Test chunk")

        assert content == "Enhanced chunk"
        assert stats["model"] == "gpt-4o"
        assert stats["prompt_tokens"] == 50
        assert stats["completion_tokens"] == 25
        assert stats["estimated_cost"] == 0.0005

    @pytest.mark.asyncio
    async def test_process_chunk_legacy(self, enhancer):
        """Test legacy process_chunk method."""
        content = await enhancer._process_chunk("Test chunk")

        assert content == "Enhanced chunk"

    @pytest.mark.asyncio
    async def test_process_chunk_error_returns_original(self):
        """Test that errors return original chunk."""
        mock_manager = MagicMock()
        mock_manager.complete_with_fallback = AsyncMock(side_effect=Exception("LLM Error"))
        enhancer = MarkdownEnhancer(mock_manager)

        content, stats = await enhancer._process_chunk_with_stats("Original chunk")

        assert content == "Original chunk"
        assert stats == {}


class TestMarkdownEnhancerFrontmatter:
    """Tests for MarkdownEnhancer frontmatter handling."""

    @pytest.fixture
    def enhancer(self):
        """Create an enhancer."""
        mock_manager = MagicMock()
        return MarkdownEnhancer(mock_manager)

    def test_extract_summary_from_frontmatter(self, enhancer):
        """Test extracting summary from frontmatter description."""
        markdown = """---
description: "This is the document summary"
entities:
  - Entity1
---

# Content"""

        summary = enhancer._extract_summary_from_frontmatter(markdown)
        assert summary == "This is the document summary"

    def test_extract_summary_truncates_long(self, enhancer):
        """Test that summary is truncated to 100 characters."""
        long_desc = "x" * 200
        markdown = f"""---
description: "{long_desc}"
---

# Content"""

        summary = enhancer._extract_summary_from_frontmatter(markdown)
        assert len(summary) == 100

    def test_extract_summary_invalid_frontmatter(self, enhancer):
        """Test extraction with invalid frontmatter returns empty."""
        markdown = "# No frontmatter"
        summary = enhancer._extract_summary_from_frontmatter(markdown)
        assert summary == ""

    def test_inject_frontmatter_basic(self, enhancer):
        """Test basic frontmatter injection."""
        markdown = "# Content\n\nParagraph"
        result = enhancer._inject_frontmatter(
            markdown,
            source_file=Path("test.md"),
            summary="Test summary",
        )

        assert "---" in result
        assert "title:" in result
        assert "source:" in result

    def test_inject_frontmatter_strips_code_block(self, enhancer):
        """Test that code block wrapper is stripped."""
        markdown = "```markdown\n# Content\n```"
        result = enhancer._inject_frontmatter(
            markdown,
            source_file=Path("test.md"),
            summary="",
        )

        assert "```markdown" not in result

    def test_inject_frontmatter_merges_llm_fields(self, enhancer):
        """Test that LLM-generated frontmatter fields are merged."""
        markdown = """---
description: "LLM summary"
entities:
  - Entity1
topics:
  - Topic1
domain: Technology
---

# Content"""

        result = enhancer._inject_frontmatter(
            markdown,
            source_file=Path("test.md"),
            summary="",
        )

        # Should contain both system and LLM fields
        assert "entities:" in result
        assert "topics:" in result
        assert "domain:" in result


class TestMarkdownEnhancerGenerateSummary:
    """Tests for MarkdownEnhancer summary generation."""

    @pytest.fixture
    def enhancer(self):
        """Create an enhancer with mocked manager."""
        mock_manager = MagicMock()
        mock_manager.complete_with_fallback = AsyncMock(
            return_value=LLMResponse(
                content="Generated summary text",
                usage=TokenUsage(prompt_tokens=30, completion_tokens=10),
                model="gpt-4o",
                finish_reason="stop",
                estimated_cost=0.0001,
            )
        )
        return MarkdownEnhancer(mock_manager)

    @pytest.mark.asyncio
    async def test_generate_summary_with_stats(self, enhancer):
        """Test summary generation returns stats."""
        summary, stats = await enhancer._generate_summary_with_stats("Long markdown content")

        assert summary == "Generated summary text"
        assert stats["model"] == "gpt-4o"
        assert stats["prompt_tokens"] == 30

    @pytest.mark.asyncio
    async def test_generate_summary_truncates_to_100(self):
        """Test that summary is truncated to 100 characters."""
        mock_manager = MagicMock()
        mock_manager.complete_with_fallback = AsyncMock(
            return_value=LLMResponse(
                content="x" * 200,
                usage=None,
                model="gpt-4o",
                finish_reason="stop",
            )
        )
        enhancer = MarkdownEnhancer(mock_manager)

        summary, _ = await enhancer._generate_summary_with_stats("Content")
        assert len(summary) == 100

    @pytest.mark.asyncio
    async def test_generate_summary_legacy(self, enhancer):
        """Test legacy generate_summary method."""
        summary = await enhancer._generate_summary("Content")
        assert summary == "Generated summary text"

    @pytest.mark.asyncio
    async def test_generate_summary_error_returns_empty(self):
        """Test that errors return empty summary."""
        mock_manager = MagicMock()
        mock_manager.complete_with_fallback = AsyncMock(side_effect=Exception("LLM Error"))
        enhancer = MarkdownEnhancer(mock_manager)

        summary, stats = await enhancer._generate_summary_with_stats("Content")
        assert summary == ""


class TestSimpleMarkdownCleaner:
    """Tests for SimpleMarkdownCleaner."""

    @pytest.fixture
    def cleaner(self):
        """Create a cleaner instance."""
        return SimpleMarkdownCleaner()

    def test_normalize_line_endings(self, cleaner):
        """Test line ending normalization."""
        result = cleaner.clean("Line1\r\nLine2\rLine3")
        assert "\r" not in result
        assert "Line1\nLine2\nLine3" in result

    def test_remove_excessive_blank_lines(self, cleaner):
        """Test removal of excessive blank lines."""
        result = cleaner.clean("Line1\n\n\n\n\nLine2")
        assert result == "Line1\n\nLine2\n"

    def test_blank_line_before_heading(self, cleaner):
        """Test blank line is added before headings."""
        result = cleaner.clean("Paragraph\n# Heading")
        assert "Paragraph\n\n# Heading" in result

    def test_blank_line_after_heading(self, cleaner):
        """Test blank line is added after headings."""
        result = cleaner.clean("# Heading\nContent")
        assert "# Heading\n\nContent" in result

    def test_standardize_list_markers(self, cleaner):
        """Test list markers are standardized to -."""
        result = cleaner.clean("* Item1\n+ Item2\n- Item3")
        assert "- Item1" in result
        assert "- Item2" in result
        assert "- Item3" in result

    def test_remove_trailing_whitespace(self, cleaner):
        """Test trailing whitespace is removed."""
        result = cleaner.clean("Line with trailing   \nAnother line  ")
        lines = result.strip().split("\n")
        for line in lines:
            assert line == line.rstrip()

    def test_ends_with_single_newline(self, cleaner):
        """Test file ends with single newline."""
        result = cleaner.clean("Content\n\n\n")
        assert result.endswith("\n")
        assert not result.endswith("\n\n")

    def test_complex_document(self, cleaner):
        """Test cleaning a complex document."""
        input_md = """* List item with asterisk
+ Another item with plus


# Heading with no blank before
Content right after heading
More content   \t

## Another heading


"""
        result = cleaner.clean(input_md)

        # List markers standardized
        assert "- List item" in result
        assert "- Another item" in result
        # No excessive blank lines
        assert "\n\n\n" not in result
        # Ends with single newline
        assert result.endswith("\n")
        assert not result.endswith("\n\n")
