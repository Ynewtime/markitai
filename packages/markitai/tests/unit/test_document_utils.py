"""Unit tests for document processing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from markitai.llm.document import (
    DocumentMixin,
)
from markitai.llm.models import get_response_cost
from markitai.llm.types import (
    DocumentProcessResult,
    EnhancedDocumentResult,
    Frontmatter,
)

if TYPE_CHECKING:
    from markitai.config import LLMConfig, PromptsConfig


class TestProtectImagePositions:
    """Tests for DocumentMixin._protect_image_positions static method."""

    def test_single_image(self) -> None:
        """Test protecting a single image reference."""
        text = "Before ![alt text](image.jpg) after"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert "![alt text](image.jpg)" not in protected
        assert "__MARKITAI_IMG_0__" in protected
        assert len(mapping) == 1
        assert mapping["__MARKITAI_IMG_0__"] == "![alt text](image.jpg)"

    def test_multiple_images(self) -> None:
        """Test protecting multiple image references."""
        text = "![img1](a.jpg) text ![img2](b.png) more ![img3](c.gif)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert len(mapping) == 3
        assert "__MARKITAI_IMG_0__" in protected
        assert "__MARKITAI_IMG_1__" in protected
        assert "__MARKITAI_IMG_2__" in protected

    def test_no_images(self) -> None:
        """Test with no images in text."""
        text = "Just plain text without images"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert protected == text
        assert len(mapping) == 0

    def test_screenshots_excluded(self) -> None:
        """Test that screenshots/ paths are excluded from protection."""
        text = "![screenshot](.markitai/screenshots/page.jpg) and ![regular](image.png)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        # Screenshot should remain, regular image should be protected
        assert "![screenshot](.markitai/screenshots/page.jpg)" in protected
        assert "![regular](image.png)" not in protected
        assert len(mapping) == 1

    def test_external_url_images(self) -> None:
        """Test protection of external URL images."""
        text = "![logo](https://example.com/logo.png)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert len(mapping) == 1
        assert "https://example.com/logo.png" in mapping["__MARKITAI_IMG_0__"]

    def test_empty_alt_text(self) -> None:
        """Test image with empty alt text."""
        text = "![](image.jpg)"
        protected, mapping = DocumentMixin._protect_image_positions(text)

        assert len(mapping) == 1
        assert mapping["__MARKITAI_IMG_0__"] == "![](image.jpg)"


class TestRestoreImagePositions:
    """Tests for DocumentMixin._restore_image_positions static method."""

    def test_restore_single_image(self) -> None:
        """Test restoring a single image."""
        mapping = {"__MARKITAI_IMG_0__": "![alt](image.jpg)"}
        text = "Before __MARKITAI_IMG_0__ after"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert restored == "Before ![alt](image.jpg) after"

    def test_restore_multiple_images(self) -> None:
        """Test restoring multiple images."""
        mapping = {
            "__MARKITAI_IMG_0__": "![img1](a.jpg)",
            "__MARKITAI_IMG_1__": "![img2](b.png)",
        }
        text = "Start __MARKITAI_IMG_0__ middle __MARKITAI_IMG_1__ end"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert "![img1](a.jpg)" in restored
        assert "![img2](b.png)" in restored
        assert "__MARKITAI_IMG_" not in restored

    def test_restore_empty_mapping(self) -> None:
        """Test restoration with empty mapping."""
        text = "No images here"
        restored = DocumentMixin._restore_image_positions(text, {})

        assert restored == text

    def test_roundtrip(self) -> None:
        """Test protect -> restore roundtrip."""
        original = "![first](1.jpg) text ![second](2.png) end"
        protected, mapping = DocumentMixin._protect_image_positions(original)
        restored = DocumentMixin._restore_image_positions(protected, mapping)

        assert restored == original


class TestGetResponseCost:
    """Tests for get_response_cost function."""

    def test_successful_cost_extraction(self) -> None:
        """Test successful cost extraction from valid response."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4"

        with patch("markitai.llm.models.completion_cost", return_value=0.05):
            cost = get_response_cost(mock_response)
            assert cost == 0.05

    def test_cost_returns_none(self) -> None:
        """Test when completion_cost returns None."""
        mock_response = MagicMock()

        with patch("markitai.llm.models.completion_cost", return_value=None):
            cost = get_response_cost(mock_response)
            assert cost == 0.0

    def test_cost_extraction_exception(self) -> None:
        """Test that exceptions return 0.0."""
        mock_response = MagicMock()

        with patch(
            "litellm.completion_cost",
            side_effect=Exception("Cost calculation failed"),
        ):
            cost = get_response_cost(mock_response)
            assert cost == 0.0

    def test_invalid_response_object(self) -> None:
        """Test with invalid response object."""
        with patch(
            "litellm.completion_cost",
            side_effect=AttributeError("No model attribute"),
        ):
            cost = get_response_cost(None)
            assert cost == 0.0


class TestSplitIntoBatches:
    """Tests for DocumentMixin._split_into_batches static method."""

    def test_single_batch(self) -> None:
        """Test when all images fit in one batch."""
        images = [Path(f"page{i}.jpg") for i in range(5)]
        batches = DocumentMixin._split_into_batches(images, batch_size=10)

        assert len(batches) == 1
        assert batches[0] == images

    def test_exact_batches(self) -> None:
        """Test when images divide evenly into batches."""
        images = [Path(f"page{i}.jpg") for i in range(10)]
        batches = DocumentMixin._split_into_batches(images, batch_size=5)

        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_uneven_batches(self) -> None:
        """Test when last batch has fewer images."""
        images = [Path(f"page{i}.jpg") for i in range(7)]
        batches = DocumentMixin._split_into_batches(images, batch_size=3)

        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_empty_list(self) -> None:
        """Test with empty image list."""
        batches = DocumentMixin._split_into_batches([], batch_size=5)

        assert len(batches) == 0

    def test_batch_size_one(self) -> None:
        """Test with batch size of 1."""
        images = [Path(f"page{i}.jpg") for i in range(3)]
        batches = DocumentMixin._split_into_batches(images, batch_size=1)

        assert len(batches) == 3
        assert all(len(batch) == 1 for batch in batches)

    def test_batch_larger_than_list(self) -> None:
        """Test when batch size exceeds total images."""
        images = [Path(f"page{i}.jpg") for i in range(3)]
        batches = DocumentMixin._split_into_batches(images, batch_size=100)

        assert len(batches) == 1
        assert batches[0] == images

    def test_preserves_path_objects(self) -> None:
        """Test that Path objects are preserved."""
        images = [Path("/tmp/page1.jpg"), Path("/home/user/page2.png")]
        batches = DocumentMixin._split_into_batches(images, batch_size=1)

        assert all(isinstance(p, Path) for batch in batches for p in batch)
        assert batches[0][0] == Path("/tmp/page1.jpg")


class TestBuildFallbackFrontmatter:
    """Tests for DocumentMixin._build_fallback_frontmatter method."""

    def test_basic_frontmatter(self) -> None:
        """Test basic frontmatter generation."""
        mixin = DocumentMixin()
        content = "# Test Document\n\nSome content here."
        result = mixin._build_fallback_frontmatter("test.pdf", content)

        assert "source: test.pdf" in result
        assert "title:" in result
        assert "description:" in result

    def test_title_extraction_from_content(self) -> None:
        """Test that title is extracted from content when not provided."""
        mixin = DocumentMixin()
        content = "# My Document Title\n\nContent here."
        result = mixin._build_fallback_frontmatter("doc.pdf", content)

        assert "title:" in result

    def test_provided_title_preserved(self) -> None:
        """Test that provided title is preserved."""
        mixin = DocumentMixin()
        content = "# Different Title\n\nContent."
        result = mixin._build_fallback_frontmatter(
            "doc.pdf", content, title="Custom Title"
        )

        assert "Custom Title" in result

    def test_empty_content(self) -> None:
        """Test with empty content."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter("empty.pdf", "")

        assert "source: empty.pdf" in result
        # Should still have valid YAML structure
        assert "title:" in result

    def test_source_preserved(self) -> None:
        """Test that source is preserved in frontmatter."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter(
            "path/to/document.docx", "Content", title="Title"
        )

        assert "source: path/to/document.docx" in result

    def test_returns_yaml_without_markers(self) -> None:
        """Test that result doesn't have YAML document markers."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter("test.pdf", "Content")

        # Should not start with --- (markers are added by format_llm_output)
        assert not result.startswith("---")

    def test_uses_extra_meta_description(self) -> None:
        """Fallback should use description from extra_meta when available."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter(
            "test.pdf",
            "Content",
            extra_meta={"description": "A great article about testing"},
        )
        assert "A great article about testing" in result

    def test_uses_extra_meta_tags(self) -> None:
        """Fallback should use tags from extra_meta when available."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter(
            "test.pdf",
            "Content",
            extra_meta={"tags": ["python", "testing"]},
        )
        assert "python" in result
        assert "testing" in result

    def test_ignores_empty_extra_meta_description(self) -> None:
        """Empty description in extra_meta should not be used."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter(
            "test.pdf",
            "Content",
            extra_meta={"description": "  "},
        )
        assert "description: ''" in result

    def test_ignores_non_string_extra_meta_description(self) -> None:
        """Non-string description in extra_meta should not be used."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter(
            "test.pdf",
            "Content",
            extra_meta={"description": 42},
        )
        assert "description: ''" in result

    def test_no_extra_meta_still_works(self) -> None:
        """No extra_meta should produce empty description as before."""
        mixin = DocumentMixin()
        result = mixin._build_fallback_frontmatter("test.pdf", "Content")
        assert "description:" in result


class TestStabilizePagedMarkdown:
    """Tests for DocumentMixin._stabilize_paged_markdown."""

    def test_restores_slide_section_when_body_text_is_dropped(self) -> None:
        """Slide documents should keep short body text that the LLM removes."""
        mixin = DocumentMixin()
        original = """<!-- Slide number: 1 -->

# FREE TEST DATA

PPT FILE

<!-- Slide number: 2 -->

# FREE TEST DATA

Lorem ipsum
"""
        cleaned = """<!-- Slide number: 1 -->

# FREE TEST DATA

<!-- Slide number: 2 -->

# FREE TEST DATA

Lorem ipsum
"""

        stabilized = mixin._stabilize_paged_markdown(original, cleaned, "sample.pptx")

        assert "PPT FILE" in stabilized
        assert stabilized == original.strip()

    def test_restores_image_only_slide_when_llm_injects_text(self) -> None:
        """Image-only slides should not gain extra OCR prose in the final output."""
        mixin = DocumentMixin()
        original = """<!-- Slide number: 8 -->

![Logo Description automatically generated](.markitai/assets/sample.pptx.0001.jpg)
"""
        cleaned = """<!-- Slide number: 8 -->

![Logo with text FreeTestData and slogan Your Slogan Here](.markitai/assets/sample.pptx.0001.jpg)

FTD
FREE TEST DATA
8
"""

        stabilized = mixin._stabilize_paged_markdown(original, cleaned, "sample.pptx")

        assert "FTD" not in stabilized
        assert stabilized == original.strip()


class TestValidateNoPromptLeakage:
    """Tests for DocumentMixin._validate_no_prompt_leakage method."""

    def test_clean_content_unchanged(self) -> None:
        """Test that clean content is returned unchanged."""
        mixin = DocumentMixin()
        content = "# Normal Document\n\nJust regular markdown content."
        result = mixin._validate_no_prompt_leakage(content, "test.md")

        assert result == content

    def test_detects_task_1_english(self) -> None:
        """Test detection of '## Task 1:' marker."""
        mixin = DocumentMixin()
        content = "## Task 1: Clean the markdown\n\nSome leaked prompt"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_task_2_english(self) -> None:
        """Test detection of '## Task 2:' marker."""
        mixin = DocumentMixin()
        content = "Some content\n## Task 2: Generate metadata"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_task_1_chinese(self) -> None:
        """Test detection of '## 任务 1:' marker."""
        mixin = DocumentMixin()
        content = "## 任务 1: 清理文档\n\n泄漏的提示词"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_task_2_chinese(self) -> None:
        """Test detection of '## 任务 2:' marker."""
        mixin = DocumentMixin()
        content = "内容\n## 任务 2: 生成元数据"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_core_principles_marker(self) -> None:
        """Test detection of '【核心原则】' marker."""
        mixin = DocumentMixin()
        content = "【核心原则】保持原始内容不变"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_cleaning_spec_marker(self) -> None:
        """Test detection of '【清理规范】' marker."""
        mixin = DocumentMixin()
        content = "【清理规范】删除多余空白"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_please_process_marker(self) -> None:
        """Test detection of '请处理以下' marker."""
        mixin = DocumentMixin()
        content = "请处理以下 markdown 内容"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_detects_you_are_professional_marker(self) -> None:
        """Test detection of '你是一个专业的' marker."""
        mixin = DocumentMixin()
        content = "你是一个专业的 markdown 清理工具"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")

    def test_recovery_with_frontmatter(self) -> None:
        """Test recovery when content has frontmatter structure."""
        mixin = DocumentMixin()
        # Simulates LLM outputting prompt then actual content in frontmatter-like structure
        content = "## Task 1:\n---\nprompt stuff\n---\nActual clean content"
        result = mixin._validate_no_prompt_leakage(content, "test.md")

        assert result == "Actual clean content"

    def test_recovery_extracts_after_second_separator(self) -> None:
        """Test that recovery extracts content after second --- separator."""
        mixin = DocumentMixin()
        content = "## Task 1:\n---\nfrontmatter\n---\n\n# Real Document\n\nContent"
        result = mixin._validate_no_prompt_leakage(content, "test.md")

        assert "# Real Document" in result
        assert "Content" in result

    def test_markers_in_code_block_still_detected(self) -> None:
        """Test that markers are detected even in unusual contexts.

        Note: The current implementation does simple string matching,
        so markers will be detected regardless of context.
        """
        mixin = DocumentMixin()
        # This would be detected as leakage
        content = "```\n## Task 1: example\n```"

        with pytest.raises(ValueError, match="LLM returned prompt text"):
            mixin._validate_no_prompt_leakage(content, "test.md")


class TestRestoreImagePositionsEdgeCases:
    """Additional edge case tests for _restore_image_positions."""

    def test_marker_appears_multiple_times(self) -> None:
        """Test when same marker appears multiple times (shouldn't happen normally)."""
        mapping = {"__MARKITAI_IMG_0__": "![img](a.jpg)"}
        text = "__MARKITAI_IMG_0__ and __MARKITAI_IMG_0__"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        # Both should be replaced
        assert restored == "![img](a.jpg) and ![img](a.jpg)"

    def test_partial_marker_not_replaced(self) -> None:
        """Test that partial markers are not replaced."""
        mapping = {"__MARKITAI_IMG_0__": "![img](a.jpg)"}
        text = "__MARKITAI_IMG_ and __MARKITAI_IMG_0__"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert "__MARKITAI_IMG_" in restored
        assert "![img](a.jpg)" in restored

    def test_missing_marker_in_text(self) -> None:
        """Test when mapping has marker not present in text."""
        mapping = {
            "__MARKITAI_IMG_0__": "![img](a.jpg)",
            "__MARKITAI_IMG_1__": "![img2](b.jpg)",
        }
        text = "Only __MARKITAI_IMG_0__ here"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert "![img](a.jpg)" in restored
        assert "![img2](b.jpg)" not in restored

    def test_special_characters_in_image_ref(self) -> None:
        """Test image refs with special characters are restored correctly."""
        mapping = {
            "__MARKITAI_IMG_0__": "![alt & text](image%20file.jpg?v=1&size=large)"
        }
        text = "Before __MARKITAI_IMG_0__ after"
        restored = DocumentMixin._restore_image_positions(text, mapping)

        assert "![alt & text](image%20file.jpg?v=1&size=large)" in restored


# =============================================================================
# Async Tests for Document Processing Methods
# =============================================================================


@pytest.fixture
def mock_processor(llm_config: LLMConfig, prompts_config: PromptsConfig) -> MagicMock:
    """Create a mock processor with all necessary attributes for DocumentMixin."""
    from markitai.llm import LLMProcessor

    processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
    return processor


@pytest.fixture
def mock_llm_response_factory():
    """Factory to create mock LLM responses."""

    def _create(
        content: str = "Cleaned content",
        model: str = "test-model",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        cost: float = 0.001,
    ) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=content))]
        response.model = model
        response.usage = MagicMock(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return response

    return _create


@pytest.fixture
def mock_instructor_result_factory():
    """Factory to create mock Instructor results."""

    def _create(
        cleaned_markdown: str = "# Cleaned Document\n\nCleaned content.",
        description: str = "A test document",
        tags: list[str] | None = None,
    ) -> tuple[Any, MagicMock]:
        if tags is None:
            tags = ["test", "document"]

        result = DocumentProcessResult(
            cleaned_markdown=cleaned_markdown,
            frontmatter=Frontmatter(description=description, tags=tags),
        )
        raw_response = MagicMock()
        raw_response.model = "test-model"
        raw_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        raw_response.choices = [MagicMock(finish_reason="stop")]
        return result, raw_response

    return _create


@pytest.fixture
def mock_enhanced_result_factory():
    """Factory to create mock EnhancedDocumentResult."""

    def _create(
        cleaned_markdown: str = "# Enhanced Document\n\nEnhanced content.",
        description: str = "An enhanced document",
        tags: list[str] | None = None,
    ) -> tuple[Any, MagicMock]:
        if tags is None:
            tags = ["enhanced", "vision"]

        result = EnhancedDocumentResult(
            cleaned_markdown=cleaned_markdown,
            frontmatter=Frontmatter(description=description, tags=tags),
        )
        raw_response = MagicMock()
        raw_response.model = "vision-model"
        raw_response.usage = MagicMock(prompt_tokens=200, completion_tokens=100)
        raw_response.choices = [MagicMock(finish_reason="stop")]
        return result, raw_response

    return _create


@pytest.fixture
def sample_test_image(tmp_path: Path) -> Path:
    """Create a sample test image file."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_page.jpg"
    img.save(img_path, "JPEG")
    return img_path


class TestCleanMarkdownAsync:
    """Async tests for clean_markdown method."""

    @pytest.mark.asyncio
    async def test_clean_markdown_cache_hit_memory(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ) -> None:
        """Test clean_markdown returns cached result from memory cache."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Pre-populate the in-memory cache
        content = "# Test Content\n\nSome text."
        cached_result = "# Cleaned Content\n\nCleaned text."
        processor._cache.set("cleaner", content, cached_result)

        result = await processor.clean_markdown(content, "test.md")

        assert result == cached_result
        assert processor._cache_hits == 1

    @pytest.mark.asyncio
    async def test_clean_markdown_cache_hit_persistent(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ) -> None:
        """Test clean_markdown returns cached result from persistent cache."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Test Content\n\nSome text."
        cached_result = "# Cleaned from persistent\n\nCleaned."

        # Mock the persistent cache to return a hit
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        result = await processor.clean_markdown(content, "test.md")

        assert result == cached_result
        assert processor._cache_hits == 1

    @pytest.mark.asyncio
    async def test_clean_markdown_cache_miss_calls_llm(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_llm_response_factory,
    ) -> None:
        """Test clean_markdown calls LLM on cache miss."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Raw Content\n\nUncleaned text."
        cleaned_content = "# Clean Content\n\nCleaned text."

        # Setup mock router
        mock_response = mock_llm_response_factory(content=cleaned_content)
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        processor._router = mock_router

        result = await processor.clean_markdown(content, "test.md")

        assert result == cleaned_content
        assert processor._cache_misses == 1
        mock_router.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_clean_markdown_preserves_protected_content(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_llm_response_factory,
    ) -> None:
        """Test that clean_markdown preserves page number markers."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "<!-- Page number: 1 -->\n# Title\n\nContent"
        # LLM returns content with placeholder
        llm_output = "__MARKITAI_PAGENUM_0__\n# Cleaned Title\n\nCleaned Content"

        mock_response = mock_llm_response_factory(content=llm_output)
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        processor._router = mock_router

        result = await processor.clean_markdown(content, "test.md")

        # Page marker should be restored
        assert "<!-- Page number: 1 -->" in result


class TestProcessDocumentAsync:
    """Async tests for process_document method."""

    @pytest.mark.asyncio
    async def test_process_document_combined_success(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Test process_document with successful combined Instructor call."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# Processed Document\n\nProcessed content.",
            description="A processed test document",
            tags=["processed", "test"],
        )

        # Mock instructor
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                "# Raw Document\n\nRaw content.", "test.md"
            )

        assert "Processed Document" in cleaned
        assert "description:" in frontmatter
        assert "source: test.md" in frontmatter

    @pytest.mark.asyncio
    async def test_process_document_fallback_on_instructor_failure(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_llm_response_factory,
    ) -> None:
        """Test process_document falls back when Instructor fails."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        clean_response = mock_llm_response_factory(
            content="# Fallback Cleaned\n\nFallback content."
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=clean_response)
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("Instructor failed")
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                "# Raw\n\nContent", "test.md"
            )

        assert "Fallback Cleaned" in cleaned
        # Frontmatter is generated programmatically
        assert "source: test.md" in frontmatter

    @pytest.mark.asyncio
    async def test_process_document_preserves_original_when_page_markers_are_lost(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Paged documents should fall back when the cleaned output loses markers."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        original = """<!-- Page number: 1 -->

# Page 1

Original first page.

<!-- Page number: 2 -->

# Page 2

Original second page.
"""
        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# Page 1\n\nOriginal first page.\n\nOriginal second page.",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.process_document(original, "test.pdf")

        assert cleaned.strip() == original.strip()

    @pytest.mark.asyncio
    async def test_process_document_logs_warning_on_instructor_failure(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_llm_response_factory,
    ) -> None:
        """process_document should log warning when structured processing fails."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        clean_response = mock_llm_response_factory(content="# Cleaned\n\nContent.")
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=clean_response)
        processor._router = mock_router

        with (
            patch("markitai.llm.document.instructor.from_litellm") as mock_instr,
            patch("markitai.llm.document.logger") as mock_logger,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=ValueError("Instructor parse failed")
            )
            mock_instr.return_value = mock_client

            await processor.process_document("# Raw", "test.md")

        # Verify warning was logged (not silently swallowed)
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "Structured document processing failed" in str(call)
        ]
        assert len(warning_calls) >= 1

    @pytest.mark.asyncio
    async def test_process_document_preserves_original_title(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Test that original title from frontmatter is preserved."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# New Title\n\nContent",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            # Input has frontmatter with title
            input_content = "---\ntitle: Original Title\n---\n\n# Content"
            cleaned, frontmatter = await processor.process_document(
                input_content, "test.md"
            )

        # Original title should be preserved in output frontmatter
        assert "Original Title" in frontmatter

    @pytest.mark.asyncio
    async def test_process_document_uses_original_content_when_page_markers_missing(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Paginated content should fall back to the original when markers are lost."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor.clean_markdown = AsyncMock(
            side_effect=AssertionError("cleaner should not run")
        )

        original_content = """<!-- Page number: 1 -->

# Page 1

Alpha

<!-- Page number: 2 -->

# Page 2

Beta

<!-- Page number: 3 -->

# Page 3

Gamma
"""
        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# Summary\n\nAlpha\n\nBeta",
            description="Preserved pagination",
            tags=["pdf", "pages"],
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                original_content, "test.pdf"
            )

        assert cleaned.rstrip() == original_content.rstrip()
        assert "# Summary" not in cleaned
        assert "<!-- Page number: 3 -->" in cleaned
        assert "description: Preserved pagination" in frontmatter
        assert "source: test.pdf" in frontmatter

    def test_stabilize_paged_markdown_restores_slide_text_when_cleaned_drops_body(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Slide-marked documents should restore slide body text if it disappears."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        original = """<!-- Slide number: 1 -->

# FREE TEST DATA

PPT FILE

<!-- Slide number: 2 -->

# FREE TEST DATA

Slide body
"""
        cleaned = """<!-- Slide number: 1 -->

# FREE TEST DATA

<!-- Slide number: 2 -->

# FREE TEST DATA

Slide body
"""

        stabilized = processor._stabilize_paged_markdown(
            original,
            cleaned,
            "sample.pptx",
        )

        assert "PPT FILE" in stabilized
        assert stabilized.rstrip() == original.rstrip()

    @pytest.mark.asyncio
    async def test_process_document_restores_screenshot_reference_comments(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """URL screenshot reference comments should survive document cleanup."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# Cleaned\n\nContent",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.process_document(
                "# Source\n\nContent\n\n<!-- Screenshot for reference -->\n"
                "<!-- ![Screenshot](.markitai/screenshots/example.full.jpg) -->",
                "https://example.com/article",
            )

        assert "<!-- Screenshot for reference -->" in cleaned
        assert (
            "<!-- ![Screenshot](.markitai/screenshots/example.full.jpg) -->" in cleaned
        )

    @pytest.mark.asyncio
    async def test_process_document_uses_stable_title_for_structured_files(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Structured files should not drift title based on cleaned markdown."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# Reminder\n\nStructured content.",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            _, frontmatter = await processor.process_document(
                "heading: Reminder\nbody: Structured content.", "sample.xml"
            )

        assert "title: sample.xml" in frontmatter

    @pytest.mark.asyncio
    async def test_process_document_uses_input_title_when_llm_heading_drifts(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Document titles should come from the input when no explicit title exists."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_instructor_result_factory(
            cleaned_markdown="# Drifted LLM Title\n\nContent",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            _, frontmatter = await processor.process_document(
                "# Canonical Input Title\n\nContent",
                "sample.pdf",
            )

        assert "title: Canonical Input Title" in frontmatter

    @pytest.mark.asyncio
    async def test_process_document_cache_hit(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test process_document returns cached result."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Test\n\nContent"
        cached_value = {
            "cleaned_markdown": "# Cached Cleaned",
            "description": "Cached description",
            "tags": ["cached"],
        }

        # Pre-populate in-memory cache
        cache_key = "document_process:test.md"
        processor._cache.set(cache_key, content, cached_value)

        # Also need to mock the instructor to fail so it uses cache
        mock_router = MagicMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            # Cache should be checked before instructor is called
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("Should not be called")
            )
            mock_instructor.return_value = mock_client

            # The combined call goes through _process_document_combined which checks cache
            result = await processor._process_document_combined(content, "test.md")

        assert result.cleaned_markdown == "# Cached Cleaned"
        assert processor._cache_hits == 1

    @pytest.mark.asyncio
    async def test_process_document_preserves_image_position_when_llm_moves_image(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Image at start must stay at start even if LLM moves it to end."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        original = (
            "![cover](.markitai/assets/article.0001.jpeg)\n\n第一段。\n\n第二段。"
        )
        # LLM moved the image to the end
        result, raw = mock_instructor_result_factory(
            cleaned_markdown="第一段。\n\n第二段。\n\n![cover](.markitai/assets/article.0001.jpeg)"
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.process_document(original, "article.md")

        # Image must remain at the beginning, not drift to the end
        assert cleaned.splitlines()[0].startswith("![cover]")

    @pytest.mark.asyncio
    async def test_process_document_does_not_append_missing_image_to_end(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """When LLM drops an image, it must be restored at its original position, not appended."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        original = (
            "![cover](.markitai/assets/article.0001.jpeg)\n\n第一段。\n\n第二段。"
        )
        # LLM completely dropped the image
        result, raw = mock_instructor_result_factory(
            cleaned_markdown="第一段。\n\n第二段。"
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.process_document(original, "article.md")

        # Image must be at start, not end
        assert cleaned.splitlines()[0].startswith("![cover]")
        assert cleaned.rstrip().endswith("第二段。")


class TestEnhanceDocumentWithVisionAsync:
    """Async tests for enhance_document_with_vision method."""

    @pytest.mark.asyncio
    async def test_enhance_document_with_vision_no_images(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test enhance_document_with_vision returns original text when no images."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Document\n\nNo images here."
        result = await processor.enhance_document_with_vision(content, [], "test.md")

        assert result == content

    @pytest.mark.asyncio
    async def test_enhance_document_with_vision_cache_hit(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        """Test enhance_document_with_vision returns cached result."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Document\n\nWith image."
        cached_result = "# Cached Enhanced\n\nCached content."

        # Mock persistent cache to return cached result
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        result = await processor.enhance_document_with_vision(
            content, [sample_test_image], "test.md"
        )

        assert result == cached_result

    @pytest.mark.asyncio
    async def test_enhance_document_with_vision_calls_llm(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_llm_response_factory,
    ) -> None:
        """Test enhance_document_with_vision calls LLM with vision."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "<!-- Slide number: 1 -->\n# Title\n\nContent"
        enhanced_content = "__MARKITAI_SLIDENUM_0__\n# Enhanced Title\n\nEnhanced"

        mock_response = mock_llm_response_factory(content=enhanced_content)
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        processor._router = mock_router
        processor._vision_router = mock_router

        result = await processor.enhance_document_with_vision(
            content, [sample_test_image], "test.md"
        )

        # Slide marker should be restored
        assert "<!-- Slide number: 1 -->" in result
        mock_router.acompletion.assert_called_once()


class TestEnhanceDocumentCompleteAsync:
    """Async tests for enhance_document_complete method."""

    @pytest.mark.asyncio
    async def test_enhance_document_complete_no_images_fallback(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Test enhance_document_complete falls back to process_document when no images."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_instructor_result_factory()

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.enhance_document_complete(
                "# Test\n\nContent", [], "test.md"
            )

        assert "Cleaned Document" in cleaned
        assert "source: test.md" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_document_complete_single_batch(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Test enhance_document_complete with single batch uses combined call."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Vision Enhanced\n\nEnhanced content."
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.enhance_document_complete(
                "# Test\n\nContent",
                [sample_test_image],
                "test.md",
                max_pages_per_batch=10,
            )

        assert "Vision Enhanced" in cleaned
        assert "source: test.md" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_document_complete_reverts_suspicious_page_expansion(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Vision output should fall back to original page chunks when later pages bloat."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        original = """<!-- Page number: 1 -->

# Page 1

Alpha.

<!-- Page number: 2 -->

# Page 2

Beta.

<!-- Page number: 3 -->

# Page 3

Borrowed paragraph from page 3.

<!-- Page number: 4 -->

![](.markitai/assets/page4.jpg)

# Sparse page

<!-- Page number: 5 -->

Tail.
"""
        bloated = """__MARKITAI_PAGENUM_0__

# Page 1

Alpha.

__MARKITAI_PAGENUM_1__

# Page 2

Beta.

__MARKITAI_PAGENUM_2__

# Page 3

Borrowed paragraph from page 3.

__MARKITAI_PAGENUM_3__

![Alt text](.markitai/assets/page4.jpg)

# Sparse page

Borrowed paragraph from page 3.

Borrowed paragraph from page 3.

Borrowed paragraph from page 3.

Borrowed paragraph from page 3.

Borrowed paragraph from page 3.

__MARKITAI_PAGENUM_4__

Borrowed paragraph from page 3.

Borrowed paragraph from page 3.

Borrowed paragraph from page 3.

Tail.
"""
        result, raw = mock_enhanced_result_factory(cleaned_markdown=bloated)

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.enhance_document_complete(
                original,
                [sample_test_image] * 5,
                "test.pdf",
                max_pages_per_batch=10,
            )

        assert cleaned.strip() == original.strip()

    @pytest.mark.asyncio
    async def test_enhance_document_complete_single_batch_preserves_original_title(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Single-batch vision enhancement should preserve explicit title."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Chapter 1: Test Content\n\nEnhanced content."
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            _, frontmatter = await processor.enhance_document_complete(
                "# Chapter 1: Test Content",
                [sample_test_image],
                "sample.epub",
                original_title="Canonical Document Title",
                max_pages_per_batch=10,
            )

        assert "Canonical Document Title" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_document_complete_cache_hit_preserves_original_title(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        """Cached vision enhancement should still preserve explicit title."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)
        processor._persistent_cache.get = MagicMock(
            return_value={
                "cleaned_markdown": "# Chapter 1: Test Content",
                "description": "Cached description",
                "tags": ["cached"],
            }
        )

        _, frontmatter = await processor.enhance_document_complete(
            "# Chapter 1: Test Content",
            [sample_test_image],
            "sample.epub",
            original_title="Canonical Document Title",
        )

        assert "Canonical Document Title" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_document_complete_fallback_on_combined_failure(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_llm_response_factory,
    ) -> None:
        """Test enhance_document_complete falls back when combined call fails."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Vision enhance fallback response
        vision_response = mock_llm_response_factory(
            content="# Fallback Vision\n\nFallback enhanced."
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=vision_response)
        processor._router = mock_router
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("Combined call failed")
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.enhance_document_complete(
                "# Test\n\nContent",
                [sample_test_image],
                "test.md",
            )

        assert "Fallback Vision" in cleaned
        assert "source: test.md" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_document_complete_multi_batch(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        tmp_path: Path,
        mock_enhanced_result_factory,
        mock_llm_response_factory,
    ) -> None:
        """Test enhance_document_complete with multiple batches."""
        from PIL import Image

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Create 5 test images (will split into 2 batches with max_pages=3)
        page_images = []
        for i in range(5):
            img = Image.new("RGB", (100, 100), color="blue")
            img_path = tmp_path / f"page_{i}.jpg"
            img.save(img_path, "JPEG")
            page_images.append(img_path)

        # Content with page markers
        content = "\n".join(
            [f"<!-- Page number: {i + 1} -->\n# Page {i + 1}" for i in range(5)]
        )

        # First batch uses instructor (combined), remaining use vision only
        first_result, first_raw = mock_enhanced_result_factory(
            cleaned_markdown="# Batch 1 Enhanced"
        )
        remaining_response = mock_llm_response_factory(content="# Batch 2 Enhanced")

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=remaining_response)
        processor._router = mock_router
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(first_result, first_raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.enhance_document_complete(
                content,
                page_images,
                "test.pdf",
                max_pages_per_batch=3,
            )

        # Should have content from both batches
        assert "source: test.pdf" in frontmatter


class TestEnhanceUrlWithVisionAsync:
    """Async tests for enhance_url_with_vision method."""

    @pytest.mark.asyncio
    async def test_enhance_url_with_vision_cache_hit(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        """Test enhance_url_with_vision returns cached result."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        cached_result = {
            "cleaned_markdown": "# Cached URL Content",
            "description": "A cached URL",
            "tags": ["web"],
        }
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        cleaned, frontmatter = await processor.enhance_url_with_vision(
            "# Original Content",
            sample_test_image,
            "https://example.com",
        )

        assert cleaned == "# Cached URL Content"
        assert "Cached URL Content" in frontmatter
        assert "markitai_processed" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_url_with_vision_cache_hit_preserves_source_metadata(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        """Cache-hit frontmatter should include threaded fetch metadata."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        cached_result = {
            "cleaned_markdown": "# Cached URL Content",
            "description": "A cached URL",
            "tags": ["web"],
        }
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        _, frontmatter = await processor.enhance_url_with_vision(
            "# Original Content",
            sample_test_image,
            "https://example.com",
            fetch_strategy="defuddle",
            extra_meta={"author": "Jane", "published": "2024-01-15"},
        )

        parsed = yaml.safe_load(frontmatter)
        assert parsed["fetch_strategy"] == "defuddle"
        assert parsed["author"] == "Jane"
        assert parsed["published"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_enhance_url_with_vision_calls_vision_llm(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Test enhance_url_with_vision calls vision LLM."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Enhanced URL\n\nContent from vision.",
            description="URL content description",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.enhance_url_with_vision(
                "# Original\n\nContent",
                sample_test_image,
                "https://example.com/page",
            )

        assert "Enhanced URL" in cleaned
        assert "description:" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_url_with_vision_includes_source_metadata(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Fresh vision frontmatter should include fetch metadata."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Enhanced URL\n\nContent from vision.",
            description="URL content description",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            _, frontmatter = await processor.enhance_url_with_vision(
                "# Original\n\nContent",
                sample_test_image,
                "https://example.com/page",
                fetch_strategy="static",
                extra_meta={"author": "Jane", "published": "2024-01-15"},
            )

        parsed = yaml.safe_load(frontmatter)
        assert parsed["fetch_strategy"] == "static"
        assert parsed["author"] == "Jane"
        assert parsed["published"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_enhance_url_with_vision_removes_hallucinated_markers(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Test that hallucinated slide/page markers are removed from URL content."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # LLM hallucinates slide markers which shouldn't be in URL content
        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Title\n<!-- Slide number: 1 -->\nContent",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor.enhance_url_with_vision(
                "# Original",
                sample_test_image,
                "https://example.com",
            )

        # Hallucinated slide marker should be removed
        assert "<!-- Slide number:" not in cleaned

    @pytest.mark.asyncio
    async def test_enhance_url_with_vision_preserves_original_title(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Test that provided original_title is preserved in frontmatter."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory()

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            _, frontmatter = await processor.enhance_url_with_vision(
                "# Content",
                sample_test_image,
                "https://example.com",
                original_title="My Custom Page Title",
            )

        assert "My Custom Page Title" in frontmatter


class TestExtractFromScreenshotAsync:
    """Async tests for extract_from_screenshot method."""

    @pytest.mark.asyncio
    async def test_extract_from_screenshot_cache_hit(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        """Test extract_from_screenshot returns cached result."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        cached_result = {
            "cleaned_markdown": "# Extracted Content",
            "description": "Screenshot extraction",
            "tags": ["screenshot"],
        }
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        cleaned, frontmatter = await processor.extract_from_screenshot(
            sample_test_image,
            "https://example.com",
        )

        assert cleaned == "# Extracted Content"
        assert "Extracted Content" in frontmatter
        assert "markitai_processed" in frontmatter

    @pytest.mark.asyncio
    async def test_extract_from_screenshot_calls_vision_llm(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Test extract_from_screenshot calls vision LLM."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Extracted from Screenshot\n\nContent.",
            description="Screenshot content",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.extract_from_screenshot(
                sample_test_image,
                "https://example.com/page",
            )

        assert "Extracted from Screenshot" in cleaned
        assert "description:" in frontmatter


class TestProcessDocumentCombinedAsync:
    """Async tests for _process_document_combined method."""

    @pytest.mark.asyncio
    async def test_process_document_combined_truncation_warning(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_instructor_result_factory,
    ) -> None:
        """Test that long content is truncated with warning."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Create very long content
        long_content = "# Title\n\n" + "x" * 200000

        result, raw = mock_instructor_result_factory()

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            # Should not raise, just truncate
            doc_result = await processor._process_document_combined(
                long_content, "test.md"
            )

        assert doc_result.cleaned_markdown is not None

    @pytest.mark.asyncio
    async def test_process_document_combined_validates_prompt_leakage(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test that prompt leakage is detected and handled."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # LLM returns content with prompt leakage that can be recovered
        leaked_result = DocumentProcessResult(
            cleaned_markdown="## Task 1:\n---\nprompt\n---\n# Actual Content",
            frontmatter=Frontmatter(description="Test", tags=["test"]),
        )
        raw = MagicMock()
        raw.model = "test-model"
        raw.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        raw.choices = [MagicMock(finish_reason="stop")]

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(leaked_result, raw)
            )
            mock_instructor.return_value = mock_client

            doc_result = await processor._process_document_combined("# Test", "test.md")

        # Should have recovered content after frontmatter
        assert "# Actual Content" in doc_result.cleaned_markdown

    @pytest.mark.asyncio
    async def test_process_document_combined_raises_on_truncation(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test that output truncation raises ValueError."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result = DocumentProcessResult(
            cleaned_markdown="# Content",
            frontmatter=Frontmatter(description="Test", tags=["test"]),
        )
        raw = MagicMock()
        raw.model = "test-model"
        raw.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        # Simulate truncation
        raw.choices = [MagicMock(finish_reason="length")]

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            with pytest.raises(ValueError, match="truncated"):
                await processor._process_document_combined("# Test", "test.md")


class TestEnhanceWithFrontmatterAsync:
    """Async tests for _enhance_with_frontmatter method."""

    @pytest.mark.asyncio
    async def test_enhance_with_frontmatter_cache_hit(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
    ) -> None:
        """Test _enhance_with_frontmatter returns cached result."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        cached_result = {
            "cleaned_markdown": "# Cached Enhanced",
            "description": "A cached doc",
            "tags": ["test"],
        }
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        cleaned, frontmatter = await processor._enhance_with_frontmatter(
            "# Original",
            [sample_test_image],
            "test.pdf",
        )

        assert cleaned == "# Cached Enhanced"
        assert "markitai_processed" in frontmatter

    @pytest.mark.asyncio
    async def test_enhance_with_frontmatter_restores_protected_content(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Test that protected content is restored after enhancement."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Result with placeholder
        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="__MARKITAI_SLIDENUM_0__\n# Enhanced Title"
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            cleaned, _ = await processor._enhance_with_frontmatter(
                "<!-- Slide number: 1 -->\n# Title",
                [sample_test_image],
                "test.pptx",
            )

        # Slide marker should be restored
        assert "<!-- Slide number: 1 -->" in cleaned

    @pytest.mark.asyncio
    async def test_enhance_with_frontmatter_uses_input_title_when_missing_explicit_title(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        sample_test_image: Path,
        mock_enhanced_result_factory,
    ) -> None:
        """Vision enhancement should derive title from input text before LLM drift."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        result, raw = mock_enhanced_result_factory(
            cleaned_markdown="# Drifted LLM Title\n\nEnhanced content.",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(result, raw)
            )
            mock_instructor.return_value = mock_client

            _, frontmatter = await processor._enhance_with_frontmatter(
                "# Canonical Input Title\n\nOriginal content.",
                [sample_test_image],
                "sample.pdf",
            )

        assert "title: Canonical Input Title" in frontmatter


class TestCacheInteractionPatterns:
    """Tests for cache interaction patterns."""

    @pytest.mark.asyncio
    async def test_cache_miss_populates_both_caches(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        mock_llm_response_factory,
    ) -> None:
        """Test that cache miss populates both memory and persistent caches."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Test\n\nContent"
        cleaned = "# Cleaned\n\nContent"

        mock_response = mock_llm_response_factory(content=cleaned)
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        processor._router = mock_router

        # Track cache.set calls
        memory_cache_set = MagicMock()
        persistent_cache_set = MagicMock()
        processor._cache.set = memory_cache_set
        processor._persistent_cache.set = persistent_cache_set

        await processor.clean_markdown(content, "test.md")

        # Both caches should have been populated
        memory_cache_set.assert_called_once()
        persistent_cache_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_persistent_cache_hit_populates_memory_cache(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test that persistent cache hit populates memory cache."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        content = "# Test\n\nContent"
        cached_result = "# Cached\n\nContent"

        # Memory cache miss, persistent cache hit
        processor._cache.get = MagicMock(return_value=None)
        processor._persistent_cache.get = MagicMock(return_value=cached_result)

        memory_cache_set = MagicMock()
        processor._cache.set = memory_cache_set

        await processor.clean_markdown(content, "test.md")

        # Memory cache should be populated from persistent cache
        memory_cache_set.assert_called_once()


class TestErrorHandlingAsync:
    """Async tests for error handling in document processing."""

    @pytest.mark.asyncio
    async def test_clean_markdown_fallback_on_error(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test that clean_markdown handles LLM errors gracefully."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(side_effect=Exception("LLM Error"))
        processor._router = mock_router

        # Should raise the exception (clean_markdown doesn't have fallback)
        with pytest.raises(Exception, match="LLM Error"):
            await processor.clean_markdown("# Test", "test.md")

    @pytest.mark.asyncio
    async def test_process_document_fallback_when_clean_fails(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Test process_document returns original when all calls fail."""
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        original_content = "# Original\n\nContent"

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(side_effect=Exception("All calls fail"))
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("Instructor failed")
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                original_content, "test.md"
            )

        # Should return original content when everything fails
        assert cleaned == original_content
        # Frontmatter should still be generated programmatically
        assert "source: test.md" in frontmatter

    @pytest.mark.asyncio
    async def test_process_document_skips_cleaner_after_non_retryable_provider_error(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """A fatal provider error should skip the cleaner fallback."""
        from markitai.llm import LLMProcessor
        from markitai.providers.errors import ProviderError

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        original_content = "# Original\n\nContent"
        fatal_error = ProviderError(
            "Session error: Execution failed: Error: Failed to list models",
            provider="copilot",
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=AssertionError("cleaner should not run")
        )
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=fatal_error
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                original_content, "test.md"
            )

        assert cleaned == original_content
        assert "source: test.md" in frontmatter
        mock_router.acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_document_skips_cleaner_after_wrapped_provider_error(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ) -> None:
        """Wrapped instructor retry failures should also skip the cleaner fallback."""
        from instructor.core.exceptions import FailedAttempt, InstructorRetryException

        from markitai.llm import LLMProcessor
        from markitai.providers.errors import ProviderError

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        original_content = "# Original\n\nContent"
        fatal_error = ProviderError(
            "Session error: Execution failed: Error: Failed to list models",
            provider="copilot",
        )
        wrapped_error = InstructorRetryException(
            "Session error: Execution failed: Error: Failed to list models",
            n_attempts=1,
            total_usage=0,
            failed_attempts=[FailedAttempt(1, fatal_error)],
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=AssertionError("cleaner should not run")
        )
        processor._router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=wrapped_error
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                original_content, "test.md"
            )

        assert cleaned == original_content
        assert "source: test.md" in frontmatter
        mock_router.acompletion.assert_not_called()

    @pytest.mark.asyncio
    async def test_enhance_document_complete_handles_batch_failure(
        self,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
        tmp_path: Path,
    ) -> None:
        """Test enhance_document_complete handles individual batch failures."""
        from PIL import Image

        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Create 5 test images
        page_images = []
        for i in range(5):
            img = Image.new("RGB", (100, 100), color="green")
            img_path = tmp_path / f"page_{i}.jpg"
            img.save(img_path, "JPEG")
            page_images.append(img_path)

        content = "\n".join(
            [f"<!-- Page number: {i + 1} -->\nContent {i}" for i in range(5)]
        )

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(side_effect=Exception("Vision failed"))
        processor._router = mock_router
        processor._vision_router = mock_router

        with patch("markitai.llm.document.instructor.from_litellm") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("All batches fail")
            )
            mock_instructor.return_value = mock_client

            cleaned, frontmatter = await processor.enhance_document_complete(
                content,
                page_images,
                "test.pdf",
                max_pages_per_batch=3,
            )

        # Should have fallback frontmatter
        assert "source: test.pdf" in frontmatter


class TestRemoveUncommentedScreenshots:
    """Tests for _remove_uncommented_screenshots static method."""

    def test_removes_page_screenshot_references(self) -> None:
        """Test that page screenshot references are removed."""
        content = (
            "# Title\n\n![Page 1](.markitai/screenshots/doc.page0001.jpg)\n\nContent"
        )
        result = DocumentMixin._remove_uncommented_screenshots(content)

        assert "![Page 1](.markitai/screenshots/doc.page0001.jpg)" not in result
        assert "# Title" in result
        assert "Content" in result

    def test_preserves_regular_screenshots(self) -> None:
        """Test that regular screenshots are preserved."""
        content = (
            "# Title\n\n![Screenshot](.markitai/screenshots/regular.jpg)\n\nContent"
        )
        result = DocumentMixin._remove_uncommented_screenshots(content)

        # Regular screenshots without .pageNNNN pattern should be preserved
        assert "![Screenshot](.markitai/screenshots/regular.jpg)" in result

    def test_removes_markitai_page_labels(self) -> None:
        """Test that MARKITAI page labels are removed."""
        content = "# Title\n__MARKITAI_PAGE_LABEL_1__\nContent"
        result = DocumentMixin._remove_uncommented_screenshots(content)

        assert "__MARKITAI_PAGE_LABEL_1__" not in result

    def test_fixes_uncommented_in_page_images_section(self) -> None:
        """Test that uncommented images in page images section are commented."""
        content = """# Title

Content

<!-- Page images for reference -->
![Page 1](.markitai/screenshots/doc.page0001.jpg)"""

        result = DocumentMixin._remove_uncommented_screenshots(content)

        # Should be converted to comment
        assert "<!-- ![Page 1](.markitai/screenshots/doc.page0001.jpg) -->" in result

    def test_preserves_already_commented_images(self) -> None:
        """Test that already commented images remain unchanged."""
        content = """# Title

<!-- Page images for reference -->
<!-- ![Page 1](.markitai/screenshots/doc.page0001.jpg) -->"""

        result = DocumentMixin._remove_uncommented_screenshots(content)

        assert "<!-- ![Page 1](.markitai/screenshots/doc.page0001.jpg) -->" in result


class TestSplitTextIntoBatches:
    """Tests for _split_text_into_batches method."""

    def test_splits_by_page_markers(self) -> None:
        """Test splitting text by page markers."""
        mixin = DocumentMixin()
        # Mock the split_text_by_pages method
        mixin.split_text_by_pages = MagicMock(
            return_value=["Page 1", "Page 2", "Page 3", "Page 4", "Page 5"]
        )

        images = [Path(f"page{i}.jpg") for i in range(5)]
        batches = mixin._split_text_into_batches("full text", images, batch_size=2)

        assert len(batches) == 3
        assert "Page 1" in batches[0]
        assert "Page 3" in batches[1]
        assert "Page 5" in batches[2]

    def test_handles_single_batch(self) -> None:
        """Test with content that fits in single batch."""
        mixin = DocumentMixin()
        mixin.split_text_by_pages = MagicMock(return_value=["Page 1", "Page 2"])

        images = [Path("page1.jpg"), Path("page2.jpg")]
        batches = mixin._split_text_into_batches("full text", images, batch_size=10)

        assert len(batches) == 1
        assert "Page 1" in batches[0]
        assert "Page 2" in batches[0]


class TestFallbackFrontmatterTitle:
    """Tests for fallback frontmatter using cleaned title after cleaner succeeds."""

    @pytest.mark.asyncio
    async def test_fallback_uses_cleaned_title_not_original(self) -> None:
        """When structured processing fails but cleaning changes the title,
        fallback frontmatter should reflect the cleaned title."""
        mixin = DocumentMixin()
        mixin._config = MagicMock()
        mixin._config.prompts = MagicMock()
        mixin._prompts_config = MagicMock()

        original_markdown = "# Old Tittle With Typo\n\nSome content here."
        cleaned_markdown = "# Corrected Title\n\nSome content here."
        original_title = "Old Tittle With Typo"

        # Mock _process_document_combined to fail (trigger fallback)
        mixin._process_document_combined = AsyncMock(
            side_effect=Exception("structured processing failed")
        )

        # Mock clean_markdown to return content with corrected title
        mixin.clean_markdown = AsyncMock(return_value=cleaned_markdown)

        # Mock content protection methods (no-op)
        mixin.extract_protected_content = MagicMock(return_value={})
        mixin.protect_content = MagicMock(return_value=(original_markdown, {}))

        cleaned, frontmatter = await mixin.process_document(
            original_markdown,
            source="test.pdf",
            title=original_title,
        )

        assert cleaned == cleaned_markdown
        # Frontmatter should contain the CLEANED title, not the original
        assert "Corrected Title" in frontmatter, (
            f"Fallback frontmatter should use cleaned title. Got: {frontmatter}"
        )
        assert "Old Tittle With Typo" not in frontmatter
