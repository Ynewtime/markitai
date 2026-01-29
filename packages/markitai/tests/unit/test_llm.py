"""Tests for LLM processor module."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import LiteLLMParams, LLMConfig, ModelConfig, PromptsConfig
from markitai.llm import (
    ContentCache,
    DocumentProcessResult,
    Frontmatter,
    LLMProcessor,
)

# Note: Uses llm_config and prompts_config from conftest.py
# llm_config provides a basic config; tests needing vision model create their own


class TestLLMProcessor:
    """Tests for LLMProcessor class."""

    def test_init(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test processor initialization."""
        processor = LLMProcessor(llm_config, prompts_config)
        assert processor.config == llm_config
        assert processor._router is None  # Lazy initialization

    def test_router_lazy_init(self, prompts_config: PromptsConfig):
        """Test that router is lazily initialized with valid model."""
        # Use a valid LiteLLM model format
        config = LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",  # Valid format
                        api_key="test-key",
                    ),
                ),
            ],
            concurrency=2,
        )
        processor = LLMProcessor(config, prompts_config)
        assert processor._router is None
        router = processor.router
        assert router is not None
        assert processor._router is router

    def test_no_models_error(self, prompts_config: PromptsConfig):
        """Test error when no models configured."""
        config = LLMConfig(enabled=True, model_list=[])
        processor = LLMProcessor(config, prompts_config)
        with pytest.raises(ValueError, match="No models configured"):
            _ = processor.router

    def test_usage_tracking(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test usage tracking."""
        processor = LLMProcessor(llm_config, prompts_config)

        # Simulate tracking
        processor._track_usage("test-model", 100, 50, 0.001)
        processor._track_usage("test-model", 200, 100, 0.002)
        processor._track_usage("other-model", 50, 25, 0.0005)

        usage = processor.get_usage()
        assert "test-model" in usage
        assert usage["test-model"]["requests"] == 2
        assert usage["test-model"]["input_tokens"] == 300
        assert usage["test-model"]["output_tokens"] == 150
        assert usage["test-model"]["cost_usd"] == pytest.approx(0.003)

        assert "other-model" in usage
        assert usage["other-model"]["requests"] == 1

        total_cost = processor.get_total_cost()
        assert total_cost == pytest.approx(0.0035)

    async def test_analyze_with_instructor_tracks_usage(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Ensure instructor path tracks usage when available."""
        processor = LLMProcessor(llm_config, prompts_config)

        mock_usage = MagicMock(prompt_tokens=123, completion_tokens=45)
        # Model response (Pydantic model)
        mock_model_response = MagicMock(
            caption="Alt", description="Desc", extracted_text=None
        )
        # Raw API response (has usage info and model name)
        mock_raw_response = MagicMock(usage=mock_usage, model="test-model")

        # Mock the router to avoid LiteLLM validation
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch("markitai.llm.vision.instructor.from_litellm") as mock_from_litellm:
            mock_client = MagicMock()
            # create_with_completion returns (model, raw_response)
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(mock_model_response, mock_raw_response)
            )
            mock_from_litellm.return_value = mock_client

            result = await processor._analyze_with_instructor(
                [{"role": "user", "content": "test"}], "vision"
            )

        assert result.caption == "Alt"
        # Verify llm_usage is populated on the result
        assert result.llm_usage is not None
        assert "test-model" in result.llm_usage
        assert result.llm_usage["test-model"]["requests"] == 1
        assert result.llm_usage["test-model"]["input_tokens"] == 123
        assert result.llm_usage["test-model"]["output_tokens"] == 45
        # Also verify processor's aggregate usage
        usage = processor.get_usage()
        assert "test-model" in usage
        assert usage["test-model"]["input_tokens"] == 123
        assert usage["test-model"]["output_tokens"] == 45


class TestLLMProcessorAsync:
    """Async tests for LLMProcessor."""

    async def test_clean_markdown(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test markdown cleaning."""
        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Mock the router's acompletion
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Cleaned markdown"))
        ]
        mock_response.model = "test-model"
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        with patch.object(processor, "_router") as mock_router:
            mock_router.acompletion = AsyncMock(return_value=mock_response)
            processor._router = mock_router

            result = await processor.clean_markdown("# Test\n\nSome content")

            assert result == "Cleaned markdown"
            mock_router.acompletion.assert_called_once()

    async def test_process_document(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test full document processing with fallback to parallel mode.

        When combined Instructor call fails, fallback uses:
        - clean_markdown() for cleaning (one LLM call)
        - _build_fallback_frontmatter() for programmatic frontmatter (no LLM call)
        """
        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Set response for clean_markdown only (frontmatter is generated programmatically)
        clean_response = MagicMock()
        clean_response.choices = [
            MagicMock(message=MagicMock(content="# Test Heading\n\nCleaned content"))
        ]
        clean_response.model = "test-model"
        clean_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        # Mock the router
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=clean_response)
        processor._router = mock_router

        # Mock instructor to fail, forcing fallback to parallel mode
        with patch(
            "markitai.llm.document.instructor.from_litellm"
        ) as mock_from_litellm:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("Instructor not available")
            )
            mock_from_litellm.return_value = mock_client

            cleaned, frontmatter = await processor.process_document(
                "# Test Heading\n\nSome content", "test.md"
            )

            # Cleaned content should be from LLM
            assert "Cleaned content" in cleaned
            # Frontmatter is now generated programmatically (title extracted from content)
            assert "title: Test Heading" in frontmatter
            assert "source: test.md" in frontmatter
            assert "markitai_processed:" in frontmatter

    async def test_process_document_combined_success(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test combined document processing with Instructor."""
        processor = LLMProcessor(llm_config, prompts_config)

        # Mock result from instructor
        mock_result = DocumentProcessResult(
            cleaned_markdown="# Cleaned Title\n\nClean content.",
            frontmatter=Frontmatter(
                description="A test document for testing",
                tags=["test", "document", "sample"],
            ),
        )

        # Create mock raw response with usage info and model name
        mock_raw_response = MagicMock()
        mock_raw_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        mock_raw_response.model = "openai/gpt-4o-mini"

        # Mock the router to avoid LiteLLM validation
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        processor._router = mock_router

        with patch(
            "markitai.llm.document.instructor.from_litellm"
        ) as mock_from_litellm:
            mock_client = MagicMock()
            # create_with_completion returns (model, raw_response) tuple
            mock_client.chat.completions.create_with_completion = AsyncMock(
                return_value=(mock_result, mock_raw_response)
            )
            mock_from_litellm.return_value = mock_client

            result = await processor._process_document_combined("# Raw Test", "test.md")

            assert result.cleaned_markdown == "# Cleaned Title\n\nClean content."
            assert result.frontmatter.description == "A test document for testing"
            assert "test" in result.frontmatter.tags

    async def test_process_document_with_combined_fallback(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that process_document falls back to parallel when combined fails."""
        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # Mock instructor to fail
        with patch(
            "markitai.llm.document.instructor.from_litellm"
        ) as mock_from_litellm:
            mock_client = MagicMock()
            mock_client.chat.completions.create_with_completion = AsyncMock(
                side_effect=Exception("Instructor failed")
            )
            mock_from_litellm.return_value = mock_client

            # Mock fallback responses
            clean_response = MagicMock()
            clean_response.choices = [
                MagicMock(message=MagicMock(content="Fallback cleaned"))
            ]
            clean_response.model = "test-model"
            clean_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

            frontmatter_response = MagicMock()
            frontmatter_response.choices = [
                MagicMock(message=MagicMock(content="title: Fallback\nsource: test.md"))
            ]
            frontmatter_response.model = "test-model"
            frontmatter_response.usage = MagicMock(
                prompt_tokens=100, completion_tokens=50
            )

            with patch.object(processor, "_router") as mock_router:
                mock_router.acompletion = AsyncMock(
                    side_effect=[clean_response, frontmatter_response]
                )
                processor._router = mock_router

                cleaned, frontmatter = await processor.process_document(
                    "# Test", "test.md"
                )

                # Should use fallback values
                assert cleaned == "Fallback cleaned"
                assert "title: Fallback" in frontmatter


class TestFormatLLMOutput:
    """Tests for format_llm_output method."""

    def test_basic_format(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test basic output formatting."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output(
            "# Content", "title: Test\nsource: test.md"
        )

        assert result.startswith("---\n")
        assert "title: Test" in result
        assert "---\n\n# Content" in result

    def test_strip_existing_markers(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test stripping existing --- markers."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output("# Content", "---\ntitle: Test\n---")

        # Should not have double markers
        assert result.count("---") == 2  # Start and end only
        assert result.startswith("---\n")

    def test_strip_code_block(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test stripping yaml code block markers."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output(
            "# Content", "```yaml\ntitle: Test\nsource: test.md\n```"
        )

        # Should not contain code block markers
        assert "```" not in result
        assert "title: Test" in result
        assert result.startswith("---\n")

    def test_strip_code_block_with_markers(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test stripping both code block and --- markers."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.format_llm_output(
            "# Content", "```yaml\n---\ntitle: Test\n---\n```"
        )

        assert "```" not in result
        assert result.count("---") == 2
        assert "title: Test" in result


class TestCleanFrontmatter:
    """Tests for _clean_frontmatter method."""

    def test_clean_plain_yaml(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test cleaning plain YAML."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.clean_frontmatter("title: Test\nsource: test.md")
        assert result == "title: Test\nsource: test.md"

    def test_clean_yaml_code_block(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test cleaning YAML in code block."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.clean_frontmatter("```yaml\ntitle: Test\n```")
        assert result == "title: Test"

    def test_clean_yml_code_block(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test cleaning YAML with yml code block."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.clean_frontmatter("```yml\ntitle: Test\n```")
        assert result == "title: Test"

    def test_clean_with_dashes(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test cleaning with --- markers."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.clean_frontmatter("---\ntitle: Test\n---")
        assert result == "title: Test"

    def test_clean_combined(self, llm_config: LLMConfig, prompts_config: PromptsConfig):
        """Test cleaning with both code block and dashes."""
        processor = LLMProcessor(llm_config, prompts_config)

        result = processor.clean_frontmatter("```yaml\n---\ntitle: Test\n---\n```")
        assert result == "title: Test"


class TestProtectedContent:
    """Tests for content protection methods."""

    def test_extract_images(self):
        """Test extracting image links."""
        content = "Text\n![Alt](path/to/image.jpg)\nMore text"
        protected = LLMProcessor.extract_protected_content(content)
        assert len(protected["images"]) == 1
        assert protected["images"][0] == "![Alt](path/to/image.jpg)"

    def test_extract_slides(self):
        """Test extracting slide comments."""
        content = "<!-- Slide 1 -->\nContent\n<!-- Slide 2 -->"
        protected = LLMProcessor.extract_protected_content(content)
        assert len(protected["slides"]) == 2

    def test_extract_page_comments(self):
        """Test extracting page image comments."""
        content = "<!-- Page images for reference -->\n<!-- ![Page 1](path) -->"
        protected = LLMProcessor.extract_protected_content(content)
        assert len(protected["page_comments"]) == 2

    def test_extract_page_numbers(self):
        """Test extracting page number comments."""
        content = "<!-- Page number: 1 -->\nContent\n<!-- Page number: 2 -->"
        protected = LLMProcessor.extract_protected_content(content)
        assert len(protected["page_numbers"]) == 2
        assert "<!-- Page number: 1 -->" in protected["page_numbers"]
        assert "<!-- Page number: 2 -->" in protected["page_numbers"]

    def test_restore_missing_image(self):
        """Test restoring missing images."""
        protected = {
            "images": ["![Test](img.jpg)"],
            "slides": [],
            "page_numbers": [],
            "page_comments": [],
        }
        result = LLMProcessor.restore_protected_content("Content only", protected)
        assert "![Test](img.jpg)" in result

    def test_preserve_existing_image(self):
        """Test that existing images are not duplicated."""
        protected = {
            "images": ["![Test](img.jpg)"],
            "slides": [],
            "page_numbers": [],
            "page_comments": [],
        }
        result = LLMProcessor.restore_protected_content(
            "Content ![Test](img.jpg) more", protected
        )
        assert result.count("![Test](img.jpg)") == 1


class TestPlaceholderProtection:
    """Tests for placeholder-based content protection."""

    def test_protect_images(self):
        """Test that images are NOT protected (they pass through as-is).

        Images are no longer replaced with placeholders because LLM would often
        delete placeholders, causing images to be incorrectly appended to the end.
        Instead, we let LLM process images directly, relying on prompts to preserve
        their positions.
        """
        content = "Text\n![Alt](path/to/image.jpg)\nMore text"
        protected, mapping = LLMProcessor.protect_content(content)
        # Images are NOT protected - they remain as is
        assert "![Alt](path/to/image.jpg)" in protected
        assert "__MARKITAI_IMG_" not in protected
        # No image mappings (only page comments are protected)
        assert not any("IMG" in k for k in mapping)

    def test_protect_slide_numbers(self):
        """Test that slide number comments are protected with placeholders."""
        content = "<!-- Slide number: 1 -->\nContent\n<!-- Slide number: 2 -->"
        protected, mapping = LLMProcessor.protect_content(content)
        # Slide numbers are protected with placeholders
        assert "__MARKITAI_SLIDENUM_" in protected
        assert "<!-- Slide number: 1 -->" not in protected
        assert "<!-- Slide number: 2 -->" not in protected
        assert len([k for k in mapping if "SLIDENUM" in k]) == 2

    def test_protect_page_numbers(self):
        """Test that page number comments are protected with placeholders."""
        content = "<!-- Page number: 1 -->\nContent\n<!-- Page number: 2 -->"
        protected, mapping = LLMProcessor.protect_content(content)
        # Page numbers are protected with placeholders
        assert "__MARKITAI_PAGENUM_" in protected
        assert "<!-- Page number: 1 -->" not in protected
        assert "<!-- Page number: 2 -->" not in protected
        assert len([k for k in mapping if "PAGENUM" in k]) == 2

    def test_protect_page_comments(self):
        """Test protecting page image comments with placeholders."""
        content = "<!-- Page images for reference -->\n<!-- ![Page 1](path) -->"
        protected, mapping = LLMProcessor.protect_content(content)
        assert "__MARKITAI_PAGE_" in protected
        assert len(mapping) == 2

    def test_unprotect_restores_content(self):
        """Test that unprotect restores page comments from placeholders."""
        content = (
            "Start\n<!-- Page images for reference -->\n<!-- ![Page 1](path) -->\nEnd"
        )
        protected, mapping = LLMProcessor.protect_content(content)
        restored = LLMProcessor.unprotect_content(protected, mapping)
        assert "<!-- Page images for reference -->" in restored
        assert "<!-- ![Page 1](path) -->" in restored

    def test_unprotect_preserves_position(self):
        """Test that restored content preserves placeholder position."""
        content = "# Title\n\n<!-- Page images for reference -->\n\nText after"
        protected, mapping = LLMProcessor.protect_content(content)
        # Simulate LLM modifying only the text
        modified = protected.replace("# Title", "# New Title")
        restored = LLMProcessor.unprotect_content(modified, mapping)
        # Page comment should still be between title and text
        assert "<!-- Page images for reference -->" in restored
        assert restored.index("# New Title") < restored.index("<!-- Page images")
        assert restored.index("<!-- Page images") < restored.index("Text after")

    def test_unprotect_fallback_for_removed_placeholder(self):
        """Test fallback when LLM removes a page comment placeholder."""
        content = "Text\n<!-- Page images for reference -->\nEnd"
        protected_data = LLMProcessor.extract_protected_content(content)
        _, mapping = LLMProcessor.protect_content(content)
        # Simulate LLM removing the placeholder
        modified = "Text\nEnd"
        restored = LLMProcessor.unprotect_content(modified, mapping, protected_data)
        # Page comment should be restored at end (fallback)
        assert "<!-- Page images for reference -->" in restored

    def test_unprotect_removes_hallucinated_slide_comments(self):
        """Test that hallucinated slide comments are removed.

        When LLM sees page images, it sometimes hallucinates slide comments
        like <!-- Slide number: 0 --> which are not from the original content.
        These should be removed before restoring the actual placeholders.
        """
        content = (
            "<!-- Slide number: 1 -->\n# Title\n<!-- Slide number: 2 -->\n# Title 2"
        )
        protected, mapping = LLMProcessor.protect_content(content)

        # Simulate LLM adding hallucinated slide comments
        hallucinated = (
            "__MARKITAI_SLIDENUM_0__\n# Title\n"
            "<!-- Slide number: 0 -->\n"  # Hallucinated!
            "__MARKITAI_SLIDENUM_1__\n# Title 2\n"
            "<!-- Slide number: 1 -->\n"  # Hallucinated!
        )

        restored = LLMProcessor.unprotect_content(hallucinated, mapping)

        # Should have exactly 2 slide comments (the original ones)
        import re

        slide_matches = re.findall(r"<!-- Slide number: \d+ -->", restored)
        assert len(slide_matches) == 2
        assert "<!-- Slide number: 1 -->" in restored
        assert "<!-- Slide number: 2 -->" in restored
        # Hallucinated slide 0 should be removed
        assert "<!-- Slide number: 0 -->" not in restored

    def test_unprotect_removes_hallucinated_page_comments(self):
        """Test that hallucinated page comments are removed."""
        content = "<!-- Page number: 1 -->\nContent\n<!-- Page number: 2 -->\nMore"
        protected, mapping = LLMProcessor.protect_content(content)

        # Simulate LLM adding hallucinated page comments
        hallucinated = (
            "__MARKITAI_PAGENUM_0__\nContent\n"
            "<!-- Page number: 99 -->\n"  # Hallucinated!
            "__MARKITAI_PAGENUM_1__\nMore"
        )

        restored = LLMProcessor.unprotect_content(hallucinated, mapping)

        # Should have exactly 2 page comments (the original ones)
        import re

        page_matches = re.findall(r"<!-- Page number: \d+ -->", restored)
        assert len(page_matches) == 2
        assert "<!-- Page number: 1 -->" in restored
        assert "<!-- Page number: 2 -->" in restored
        # Hallucinated page 99 should be removed
        assert "<!-- Page number: 99 -->" not in restored


class TestNormalizeWhitespace:
    """Tests for normalize_markdown_whitespace function."""

    def test_merge_multiple_blank_lines(self):
        """Test merging 3+ blank lines into 2."""
        from markitai.utils.text import normalize_markdown_whitespace

        content = "Line 1\n\n\n\nLine 2"
        result = normalize_markdown_whitespace(content)
        assert result == "Line 1\n\nLine 2\n"

    def test_preserve_double_blank_lines(self):
        """Test that exactly 2 blank lines are preserved."""
        from markitai.utils.text import normalize_markdown_whitespace

        content = "Line 1\n\nLine 2"
        result = normalize_markdown_whitespace(content)
        assert result == "Line 1\n\nLine 2\n"

    def test_strip_trailing_whitespace(self):
        """Test stripping trailing whitespace from lines."""
        from markitai.utils.text import normalize_markdown_whitespace

        content = "Line 1   \nLine 2  \n"
        result = normalize_markdown_whitespace(content)
        assert result == "Line 1\nLine 2\n"

    def test_ensure_single_newline_at_end(self):
        """Test that content ends with exactly one newline."""
        from markitai.utils.text import normalize_markdown_whitespace

        content = "Content\n\n\n"
        result = normalize_markdown_whitespace(content)
        assert result == "Content\n"

    def test_empty_content(self):
        """Test handling of empty content."""
        from markitai.utils.text import normalize_markdown_whitespace

        result = normalize_markdown_whitespace("")
        assert result == "\n"


class TestFixMalformedImageRefs:
    """Tests for _fix_malformed_image_refs static method."""

    def test_fix_extra_single_paren(self):
        """Test fixing single extra closing parenthesis."""
        content = "![alt](path.jpg))"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![alt](path.jpg)"

    def test_fix_extra_multiple_parens(self):
        """Test fixing multiple extra closing parentheses."""
        content = "![alt](path.jpg)))"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![alt](path.jpg)"

    def test_fix_real_world_case(self):
        """Test fixing a real-world malformed image reference."""
        content = "![Grouped bar chart](assets/file.pdf-0-0.jpg))"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![Grouped bar chart](assets/file.pdf-0-0.jpg)"

    def test_preserve_normal_image(self):
        """Test that normal images are not modified."""
        content = "![alt](path.jpg)"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![alt](path.jpg)"

    def test_preserve_clickable_image(self):
        """Test that clickable images are not modified."""
        content = "[![alt](img.jpg)](link)"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "[![alt](img.jpg)](link)"

    def test_preserve_image_in_parens(self):
        """Test that images inside parentheses are preserved."""
        content = "(see ![alt](img.jpg))"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "(see ![alt](img.jpg))"

    def test_preserve_image_with_text_in_parens(self):
        """Test that images with text inside parentheses are preserved."""
        content = "(text: ![alt](img.jpg))"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "(text: ![alt](img.jpg))"

    def test_preserve_image_followed_by_caption(self):
        """Test that images followed by caption in parens are preserved."""
        content = "![alt](path.jpg) (caption)"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![alt](path.jpg) (caption)"

    def test_preserve_image_with_parens_in_path(self):
        """Test that images with parens in path are preserved."""
        content = "![alt](path(1).jpg)"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![alt](path(1).jpg)"

    def test_fix_middle_of_text(self):
        """Test fixing extra paren in the middle of text."""
        content = "text ![alt](path.jpg)) more"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "text ![alt](path.jpg) more"

    def test_multiple_images_mixed(self):
        """Test handling multiple images with mixed issues."""
        content = "![good](a.jpg) and ![bad](b.jpg)) and [![link](c.jpg)](url)"
        result = LLMProcessor.fix_malformed_image_refs(content)
        assert result == "![good](a.jpg) and ![bad](b.jpg) and [![link](c.jpg)](url)"


class TestSplitTextByPages:
    """Tests for _split_text_by_pages static method."""

    def test_split_by_slide_markers(self):
        """Test splitting by slide markers."""
        content = """<!-- Slide number: 1 -->
# Title
<!-- Slide number: 2 -->
Content
<!-- Slide number: 3 -->
End"""
        chunks = LLMProcessor.split_text_by_pages(content, 3)
        assert len(chunks) == 3
        assert "<!-- Slide number: 1 -->" in chunks[0]
        assert "<!-- Slide number: 2 -->" in chunks[1]
        assert "<!-- Slide number: 3 -->" in chunks[2]

    def test_removes_page_reference_section(self):
        """Test that page image reference section is removed before splitting."""
        content = """<!-- Slide number: 1 -->
Content
<!-- Page images for reference -->
<!-- ![Page 1](screenshots/img.jpg) -->"""
        chunks = LLMProcessor.split_text_by_pages(content, 1)
        assert "<!-- Page images for reference -->" not in chunks[0]

    def test_fallback_to_paragraph_split(self):
        """Test fallback to paragraph splitting when no markers."""
        content = "Para 1\n\nPara 2\n\nPara 3\n\nPara 4"
        chunks = LLMProcessor.split_text_by_pages(content, 2)
        assert len(chunks) == 2
        assert "Para 1" in chunks[0]
        assert "Para 3" in chunks[1] or "Para 4" in chunks[1]

    def test_split_by_page_markers(self):
        """Test splitting by page markers (PDF)."""
        content = """<!-- Page number: 1 -->
# Title
<!-- Page number: 2 -->
Content
<!-- Page number: 3 -->
End"""
        chunks = LLMProcessor.split_text_by_pages(content, 3)
        assert len(chunks) == 3
        assert "<!-- Page number: 1 -->" in chunks[0]
        assert "<!-- Page number: 2 -->" in chunks[1]
        assert "<!-- Page number: 3 -->" in chunks[2]

    def test_slide_markers_priority_over_page_markers(self):
        """Test that slide markers take priority over page markers."""
        content = """<!-- Slide number: 1 -->
Slide content
<!-- Page number: 1 -->
Page content"""
        chunks = LLMProcessor.split_text_by_pages(content, 1)
        assert len(chunks) == 1
        # Should use slide marker, so chunk includes everything
        assert "<!-- Slide number: 1 -->" in chunks[0]
        assert "<!-- Page number: 1 -->" in chunks[0]


class TestContentCache:
    """Tests for ContentCache class."""

    def test_init_defaults(self):
        """Test cache initialization with defaults."""
        cache = ContentCache()
        assert cache._maxsize == 100
        assert cache._ttl == 300
        assert cache.size == 0

    def test_init_custom(self):
        """Test cache initialization with custom values."""
        cache = ContentCache(maxsize=50, ttl_seconds=60)
        assert cache._maxsize == 50
        assert cache._ttl == 60

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = ContentCache()

        cache.set("prompt1", "content1", "result1")
        assert cache.size == 1
        assert cache.get("prompt1", "content1") == "result1"

    def test_get_miss(self):
        """Test cache miss returns None."""
        cache = ContentCache()

        assert cache.get("nonexistent", "content") is None

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = ContentCache(ttl_seconds=1)

        cache.set("prompt", "content", "result")
        assert cache.get("prompt", "content") == "result"

        # Wait for TTL to expire
        time.sleep(1.1)
        assert cache.get("prompt", "content") is None
        assert cache.size == 0

    def test_maxsize_eviction(self):
        """Test that oldest entries are evicted when full."""
        cache = ContentCache(maxsize=3)

        cache.set("p1", "c1", "r1")
        time.sleep(0.01)  # Ensure different timestamps
        cache.set("p2", "c2", "r2")
        time.sleep(0.01)
        cache.set("p3", "c3", "r3")

        assert cache.size == 3

        # Adding fourth entry should evict oldest (p1, c1)
        time.sleep(0.01)
        cache.set("p4", "c4", "r4")

        assert cache.size == 3
        assert cache.get("p1", "c1") is None  # Evicted
        assert cache.get("p2", "c2") == "r2"
        assert cache.get("p3", "c3") == "r3"
        assert cache.get("p4", "c4") == "r4"

    def test_clear(self):
        """Test clearing the cache."""
        cache = ContentCache()

        cache.set("p1", "c1", "r1")
        cache.set("p2", "c2", "r2")
        assert cache.size == 2

        cache.clear()
        assert cache.size == 0
        assert cache.get("p1", "c1") is None

    def test_hash_consistency(self):
        """Test that same inputs produce same hash."""
        cache = ContentCache()

        hash1 = cache._compute_hash("prompt", "content")
        hash2 = cache._compute_hash("prompt", "content")
        assert hash1 == hash2

    def test_hash_uniqueness(self):
        """Test that different inputs produce different hashes."""
        cache = ContentCache()

        hash1 = cache._compute_hash("prompt1", "content")
        hash2 = cache._compute_hash("prompt2", "content")
        hash3 = cache._compute_hash("prompt1", "content2")
        assert hash1 != hash2
        assert hash1 != hash3

    def test_complex_values(self):
        """Test caching complex values like dicts."""
        cache = ContentCache()

        value = {"key": "value", "nested": {"a": 1}}
        cache.set("prompt", "content", value)

        result = cache.get("prompt", "content")
        assert result == value
        assert result is value  # Same object reference


class TestParallelImageBatchAnalysis:
    """Tests for parallel batch image analysis."""

    async def test_analyze_images_batch_empty(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test analyzing empty image list."""
        processor = LLMProcessor(llm_config, prompts_config)

        results = await processor.analyze_images_batch([], "en", 10, "test")
        assert results == []

    async def test_analyze_images_batch_single(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path
    ):
        """Test analyzing single image."""
        from PIL import Image

        from markitai.llm import ImageAnalysis

        processor = LLMProcessor(llm_config, prompts_config)

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.jpg"
        img.save(img_path, "JPEG")

        # Mock analyze_batch
        with patch.object(
            processor,
            "analyze_batch",
            new=AsyncMock(
                return_value=[ImageAnalysis(caption="Test", description="Test image")]
            ),
        ):
            results = await processor.analyze_images_batch([img_path], "en", 10, "test")

        assert len(results) == 1
        assert results[0].caption == "Test"

    async def test_analyze_images_batch_multiple_batches(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path
    ):
        """Test analyzing multiple batches in parallel."""
        from PIL import Image

        from markitai.llm import ImageAnalysis

        processor = LLMProcessor(llm_config, prompts_config)

        # Create 25 test images (will be split into 3 batches of 10, 10, 5)
        image_paths = []
        for i in range(25):
            img = Image.new("RGB", (100, 100), color="red")
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path, "JPEG")
            image_paths.append(img_path)

        # Track batch calls
        batch_calls = []

        async def mock_analyze_batch(paths, lang, ctx):
            batch_calls.append(len(paths))
            return [
                ImageAnalysis(caption=f"Image {i}", description=f"Description {i}")
                for i in range(len(paths))
            ]

        with patch.object(
            processor, "analyze_batch", new=AsyncMock(side_effect=mock_analyze_batch)
        ):
            results = await processor.analyze_images_batch(
                image_paths, "en", max_images_per_batch=10, context="test"
            )

        # Should have 3 batches
        assert len(batch_calls) == 3
        assert sorted(batch_calls) == [5, 10, 10]

        # Should have 25 results in correct order
        assert len(results) == 25

    async def test_analyze_images_batch_handles_failures(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig, tmp_path
    ):
        """Test that batch failures are handled gracefully."""
        from PIL import Image

        processor = LLMProcessor(llm_config, prompts_config)

        # Create 5 test images
        image_paths = []
        for i in range(5):
            img = Image.new("RGB", (100, 100), color="blue")
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path, "JPEG")
            image_paths.append(img_path)

        # Mock analyze_batch to fail
        with patch.object(
            processor,
            "analyze_batch",
            new=AsyncMock(side_effect=Exception("API Error")),
        ):
            results = await processor.analyze_images_batch(
                image_paths, "en", max_images_per_batch=10, context="test"
            )

        # Should return placeholder results
        assert len(results) == 5
        for result in results:
            assert result.description == "Analysis failed"


class TestImageCacheSize:
    """Tests for image cache configuration."""

    def test_image_cache_max_size_increased(
        self, llm_config: LLMConfig, prompts_config: PromptsConfig
    ):
        """Test that image cache max size is set to 200."""
        processor = LLMProcessor(llm_config, prompts_config)
        assert processor._image_cache_max_size == 200


class TestLLMSupportedImageFormat:
    """Tests for is_llm_supported_image function."""

    def test_supported_formats(self):
        """Test that common supported formats are recognized."""
        from markitai.utils.mime import is_llm_supported_image

        # Supported formats
        assert is_llm_supported_image(".jpg") is True
        assert is_llm_supported_image(".jpeg") is True
        assert is_llm_supported_image(".png") is True
        assert is_llm_supported_image(".gif") is True
        assert is_llm_supported_image(".webp") is True
        # Case insensitive
        assert is_llm_supported_image(".JPG") is True
        assert is_llm_supported_image(".PNG") is True

    def test_unsupported_formats(self):
        """Test that unsupported formats are rejected."""
        from markitai.utils.mime import is_llm_supported_image

        # Unsupported formats
        assert is_llm_supported_image(".svg") is False
        assert is_llm_supported_image(".bmp") is False
        assert is_llm_supported_image(".ico") is False
        assert is_llm_supported_image(".tiff") is False
        assert is_llm_supported_image(".unknown") is False
