"""LLM-powered Markdown enhancement."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from markit.llm.base import LLMMessage, LLMTaskResultWithStats
from markit.llm.manager import ProviderManager
from markit.markdown.chunker import ChunkConfig, MarkdownChunker
from markit.markdown.frontmatter import FrontmatterHandler, create_frontmatter
from markit.utils.logging import get_logger, set_request_context

if TYPE_CHECKING:
    from markit.config.settings import PromptConfig

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


# Chinese enhancement prompt
ENHANCEMENT_PROMPT_ZH = """请按照以下规则优化 Markdown 文档的格式。

## 核心原则：尊重原文

**严禁修改原文档中的任何实际内容**，包括但不限于：
- **表格列头/表头**: 必须保持原文，禁止翻译或改写（如 "First Name" 不能改为 "名字"）
- **正文文本**: 所有段落、句子必须保持原样
- **数据内容**: 表格数据、列表内容必须原样保留
- **专有名词**: 人名、地名、产品名等保持原文

**仅以下内容可以使用中文**：
- YAML Frontmatter 中的 `description`、`topics`、`domain` 等元数据值
- 图片的 alt 描述文本

## 清理规则

### 1. 垃圾内容清理
- **页眉页脚**: 移除文档顶部/底部重复出现的公司名称、日期、机密标识等
- **水印干扰**: 移除扫描件中的斜体水印文字（如"CONFIDENTIAL"、"内部资料"、"草稿"）
- **图表残留清理**: 移除从图表/图片中错误提取的坐标轴数据：
  - 连续的独立数字行（如 "12"、"10"、"8"...通常是Y轴刻度）
  - 图表轴标签（如 "Row 1"、"Column 1"、"Category A"）
  - 图例文字（如 "Series 1"、"数据系列1"）
  - 这些内容通常出现在图片标记 `![...]()` 的前后
- **页码清理**: 移除独立成行的页码（如 "1"、"Page 2"、"第X页"）
- **PPT特有**: 移除每页重复的公司logo说明、版权声明、日期时间戳
- **乱码字符**: 移除OCR识别产生的乱码（如"■■■"、"口口口"、连续特殊符号）

### 2. PPT/演示文稿特殊处理
对于 PPT/PPTX 转换的内容，必须保留页码结构：
- 使用 `## 第 N 页` 或 `## Slide N` 作为每页的标题
- 每页内容放在对应的页码标题下
- 保持幻灯片的原有顺序

### 3. 标题层级规范
- 文档内容从 `##` (h2) 开始，避免使用多个 `#` (h1)
- 保持标题层级的逻辑递进关系

### 4. 空行规范化
- 标题上方保留一个空行
- 标题下方内容前保留一个空行
- 段落之间保留一个空行
- 移除连续超过两个的空行

### 5. GFM 规范
- 无序列表使用 `-` 作为标记符
- 代码块使用 ``` 并标注语言标识符
- **代码块内容必须完整保留，不得删除或修改**
- 表格对齐规范化

### 6. 复杂表格处理
- 合并单元格转换为多行表示
- 嵌套表格展开为独立表格
- 超宽表格考虑转换为列表形式
- **表头必须保持原文，禁止翻译**

### 7. 知识图谱元数据抽取
从文档内容中抽取以下元数据，添加到 YAML Frontmatter：
- **description**: 一句话概括文档核心内容（最多100个汉字），突出主题和关键信息，避免使用"本文档"等冗余词汇
- **entities**: 文档中的关键实体（人名、组织、产品、技术术语、地名等）- 保持原文
- **topics**: 文档涉及的主题标签（3-5个最相关的主题）- 可使用中文
- **domain**: 文档所属领域（如：技术、商业、学术、医疗、法律、金融等）- 可使用中文

**Frontmatter 格式要求**：
- 位于文档最开头，使用 `---` 包裹
- **不要**使用 Markdown 代码块（如 ```yaml）包裹 Frontmatter
- 包含特殊字符的值使用双引号包裹（如冒号、引号、换行符）
- 列表项每行一个，使用 `- ` 前缀
- 确保 YAML 语法正确，可被标准解析器解析

**示例输出格式**：
```yaml
---
description: "Kubernetes 容器编排技术指南，涵盖部署、扩展和监控最佳实践。"
entities:
  - "Kubernetes"
  - "Docker"
  - "AWS"
topics:
  - "容器化"
  - "云原生"
  - "DevOps"
domain: "技术"
---
```

---

原始 Markdown:
```markdown
{content}
```

请输出优化后的 Markdown（包含 YAML Frontmatter，不要包含 ```markdown 标记）:"""

# English enhancement prompt
ENHANCEMENT_PROMPT_EN = """Please optimize the following Markdown document's format according to these rules:

## Core Principle: Respect Original Content

**DO NOT modify any actual content from the original document**, including:
- **Table headers/column names**: Must keep original text, do NOT translate (e.g., keep "First Name" as is)
- **Body text**: All paragraphs and sentences must remain unchanged
- **Data content**: Table data, list items must be preserved exactly
- **Proper nouns**: Names, locations, product names must stay in original language

**Only the following may be translated/localized**:
- YAML Frontmatter values like `description`, `topics`, `domain`
- Image alt text descriptions

## Cleanup Rules

1. **Clean up junk content**:
   - Remove headers, footers, watermarks, and meaningless repeated characters
   - For PowerPoint slides: remove repetitive footer text (company names, dates, slide numbers, copyright notices) that appear on every slide
   - **Chart artifact cleanup**: Remove incorrectly extracted chart axis data:
     - Consecutive standalone number lines (e.g., "12", "10", "8"... usually Y-axis ticks)
     - Chart axis labels (e.g., "Row 1", "Column 1", "Category A")
     - Legend text (e.g., "Series 1", "Data Series 1")
     - These typically appear before/after image markers `![...]()`
   - Remove standalone page/slide numbers (e.g., "1", "Page 2", "Slide 3")
   - Remove OCR artifacts (garbled characters like "■■■", consecutive special symbols)

2. **PowerPoint/Presentation special handling**:
   For PPT/PPTX converted content, preserve slide structure:
   - Use `## Slide N` or `## Page N` as heading for each slide
   - Place each slide's content under its corresponding slide heading
   - Maintain original slide order

3. **Heading levels**: Ensure headings start from ## (h2), avoid multiple # (h1)

4. **Blank line normalization**:
   - One blank line above headings
   - One blank line below headings before content
   - One blank line between paragraphs
   - Remove more than 2 consecutive blank lines

5. **Follow GFM specification**:
   - Use `-` for unordered list markers
   - Use ``` with language identifier for code blocks
   - **Code block content must be preserved completely**
   - Properly align tables

6. **Complex table handling**:
   - Convert merged cells to multi-row representation
   - Expand nested tables to independent tables
   - Consider converting very wide tables to list format
   - **Table headers must remain in original language, do NOT translate**

7. **Knowledge Graph Metadata Extraction**:
   Extract the following metadata from the document and add to YAML Frontmatter:
   - **description**: One-sentence summary of the document's core content (max 100 characters), highlighting the theme and key information, avoiding redundant phrases like "this document"
   - **entities**: Key entities in the document (person names, organizations, products, technical terms, locations) - keep original
   - **topics**: Topic tags the document covers (3-5 most relevant topics)
   - **domain**: The domain the document belongs to (e.g., technology, business, academic, medical, legal, finance)

   **Frontmatter format requirements**:
   - Place at the very beginning of the document, wrapped with `---`
   - Do **NOT** wrap Frontmatter in Markdown code blocks (e.g., ```yaml)
   - Use double quotes for values containing special characters (colons, quotes, newlines)
   - List items on separate lines with `- ` prefix
   - Ensure valid YAML syntax parseable by standard parsers

   **Example output format**:
   ```yaml
   ---
   description: "Kubernetes container orchestration guide covering deployment, scaling, and monitoring best practices."
   entities:
     - "Kubernetes"
     - "Docker"
     - "AWS"
   topics:
     - "Containerization"
     - "Cloud Native"
     - "DevOps"
   domain: "Technology"
   ---
   ```

Original Markdown:
```markdown
{content}
```

Output the optimized Markdown (including YAML Frontmatter, without ```markdown markers):"""

# Default prompt (uses Chinese)
ENHANCEMENT_PROMPT = ENHANCEMENT_PROMPT_ZH

# Chinese summary prompt
SUMMARY_PROMPT_ZH = """请用一句话概括以下文档的核心内容（最多100个汉字）。

要求：
- 使用中文输出
- 突出文档的主题和关键信息
- 避免使用"本文档"、"该文件"等冗余词汇
- 格式：直接陈述核心内容

文档内容:
{content}

摘要:"""

# English summary prompt
SUMMARY_PROMPT_EN = """Summarize the following document in one sentence (maximum 100 characters):

{content}

Summary:"""

# Default prompt (uses Chinese)
SUMMARY_PROMPT = SUMMARY_PROMPT_ZH


def get_enhancement_prompt(language: str = "zh") -> str:
    """Get the enhancement prompt for the specified language.

    Args:
        language: Language code ("zh" for Chinese, "en" for English)

    Returns:
        Enhancement prompt string
    """
    return ENHANCEMENT_PROMPT_ZH if language == "zh" else ENHANCEMENT_PROMPT_EN


def get_summary_prompt(language: str = "zh") -> str:
    """Get the summary prompt for the specified language.

    Args:
        language: Language code ("zh" for Chinese, "en" for English)

    Returns:
        Summary prompt string
    """
    return SUMMARY_PROMPT_ZH if language == "zh" else SUMMARY_PROMPT_EN


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
        self.provider_manager = provider_manager
        self.config = config or EnhancementConfig()
        self.use_concurrent_fallback = use_concurrent_fallback
        self.prompt_config = prompt_config
        self.chunker = MarkdownChunker(ChunkConfig(max_tokens=self.config.chunk_size))

    def _get_enhancement_prompt(self) -> str:
        """Get the enhancement prompt from config or builtin default.

        Returns:
            Enhancement prompt string with {content} placeholder
        """
        if self.prompt_config:
            custom_prompt = self.prompt_config.get_prompt("enhancement")
            if custom_prompt:
                return custom_prompt
        return ENHANCEMENT_PROMPT

    def _get_summary_prompt(self) -> str:
        """Get the summary prompt from config or builtin default.

        Returns:
            Summary prompt string with {content} placeholder
        """
        if self.prompt_config:
            custom_prompt = self.prompt_config.get_prompt("summary")
            if custom_prompt:
                return custom_prompt
        return SUMMARY_PROMPT

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

        # Extract summary from LLM-generated frontmatter (description field)
        # No separate LLM call needed - summary is generated as part of enhancement
        summary = self._extract_summary_from_frontmatter(enhanced_markdown)

        # Inject frontmatter
        if self.config.add_frontmatter:
            enhanced_markdown = self._inject_frontmatter(enhanced_markdown, source_file, summary)

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

    async def _process_chunk_with_stats(self, chunk: str) -> tuple[str, dict]:
        """Process a single chunk with LLM and return stats.

        Args:
            chunk: Markdown chunk to process

        Returns:
            Tuple of (enhanced_chunk, stats_dict)
        """
        prompt = self._get_enhancement_prompt().format(content=chunk)
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
    ) -> str:
        """Inject YAML frontmatter into the document.

        Args:
            markdown: Enhanced Markdown content
            source_file: Original source file
            summary: Generated summary

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
