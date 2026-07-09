"""Unit tests for LLM content protection and formatting utilities."""

from __future__ import annotations

from markitai.llm.content import (
    extract_protected_content,
    protect_content,
    smart_truncate,
    strip_prompt_echo,
    unprotect_content,
)


class TestSmartTruncate:
    """Tests for smart_truncate function."""

    def test_no_truncation_needed(self) -> None:
        """Test that short text is not truncated."""
        text = "This is a short text."
        result = smart_truncate(text, max_chars=100)
        assert result == text

    def test_truncate_at_sentence_boundary(self) -> None:
        """Test truncation at sentence boundary."""
        # Use a longer text to ensure truncation actually occurs
        # The function searches in [max_chars - 500 : max_chars + 200] for boundaries
        # so text must be longer than max_chars + 200 to force truncation
        text = "First sentence. " + "A" * 300 + ". Second sentence. Third sentence."
        result = smart_truncate(text, max_chars=30)
        # Should truncate at first sentence boundary (the period after "First sentence")
        assert result.endswith(".")
        assert len(result) <= 230  # Allow for boundary search within first 200+ chars

    def test_truncate_at_paragraph_boundary(self) -> None:
        """Test truncation at paragraph boundary."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = smart_truncate(text, max_chars=25)
        # Should prefer paragraph boundary
        assert "Third" not in result

    def test_truncate_chinese_text(self) -> None:
        """Test truncation with Chinese punctuation."""
        text = "第一句话。第二句话。第三句话。"
        result = smart_truncate(text, max_chars=15)
        assert result.endswith("。") or len(result) <= 20

    def test_truncate_preserve_end(self) -> None:
        """Test truncation preserving the end."""
        text = "First part. Middle part. Final part."
        result = smart_truncate(text, max_chars=20, preserve_end=True)
        assert "Final" in result or "part" in result

    def test_truncate_no_good_boundary(self) -> None:
        """Test truncation when no good boundary exists."""
        text = "a" * 100  # No natural boundaries
        result = smart_truncate(text, max_chars=50)
        assert len(result) == 50


class TestExtractProtectedContent:
    """Tests for extract_protected_content function."""

    def test_extract_images(self) -> None:
        """Test extracting image links."""
        content = "Text ![alt](image.jpg) more ![](other.png) end"
        protected = extract_protected_content(content)

        assert len(protected["images"]) == 2
        assert "![alt](image.jpg)" in protected["images"]
        assert "![](other.png)" in protected["images"]

    def test_extract_slide_comments(self) -> None:
        """Test extracting slide comments."""
        content = """
<!-- Slide 1 -->
Content 1

<!-- Slide number: 2 -->
Content 2
"""
        protected = extract_protected_content(content)

        assert len(protected["slides"]) == 2
        assert "<!-- Slide 1 -->" in protected["slides"]
        assert "<!-- Slide number: 2 -->" in protected["slides"]

    def test_extract_page_numbers(self) -> None:
        """Test extracting page number comments."""
        content = """
<!-- Page number: 1 -->
Page 1 content

<!-- Page number: 2 -->
Page 2 content
"""
        protected = extract_protected_content(content)

        assert len(protected["page_numbers"]) == 2
        assert "<!-- Page number: 1 -->" in protected["page_numbers"]

    def test_extract_page_image_comments(self) -> None:
        """Test extracting page image comments."""
        content = """
<!-- Page images for reference -->
<!-- ![Page 1](.markitai/screenshots/page1.png) -->
Content here
"""
        protected = extract_protected_content(content)

        assert len(protected["page_comments"]) == 2

    def test_extract_screenshot_reference_comments(self) -> None:
        """Test extracting screenshot reference comments used by URL outputs."""
        content = """
<!-- Screenshot for reference -->
<!-- ![Screenshot](.markitai/screenshots/page.full.jpg) -->
Content here
"""
        protected = extract_protected_content(content)

        assert len(protected["page_comments"]) == 2
        assert "<!-- Screenshot for reference -->" in protected["page_comments"]
        assert (
            "<!-- ![Screenshot](.markitai/screenshots/page.full.jpg) -->"
            in protected["page_comments"]
        )

    def test_extract_empty_content(self) -> None:
        """Test extracting from empty content."""
        protected = extract_protected_content("")

        assert protected["images"] == []
        assert protected["slides"] == []
        assert protected["page_numbers"] == []
        assert protected["page_comments"] == []


class TestProtectContent:
    """Tests for protect_content function."""

    def test_protect_page_numbers(self) -> None:
        """Test protecting page number comments."""
        content = "<!-- Page number: 1 -->\nContent here"
        protected_content, mapping = protect_content(content)

        # Should have placeholder
        assert "__MARKITAI_PAGENUM_0__" in protected_content
        # Original should be in mapping
        assert "<!-- Page number: 1 -->" in mapping.values()

    def test_protect_slide_numbers(self) -> None:
        """Test protecting slide number comments."""
        content = "<!-- Slide number: 1 -->\nSlide content"
        protected_content, mapping = protect_content(content)

        assert "__MARKITAI_SLIDENUM_0__" in protected_content
        assert "<!-- Slide number: 1 -->" in mapping.values()

    def test_protect_multiple_markers(self) -> None:
        """Test protecting multiple markers."""
        content = """
<!-- Page number: 1 -->
Page 1

<!-- Page number: 2 -->
Page 2

<!-- Slide number: 1 -->
Slide 1
"""
        protected_content, mapping = protect_content(content)

        assert len(mapping) == 3
        assert "__MARKITAI_PAGENUM_0__" in protected_content
        assert "__MARKITAI_PAGENUM_1__" in protected_content
        assert "__MARKITAI_SLIDENUM_0__" in protected_content

    def test_protect_page_image_comments(self) -> None:
        """Test protecting page image comments."""
        content = """<!-- Page images for reference -->
<!-- ![Page 1](.markitai/screenshots/page1.png) -->
Content"""
        protected_content, mapping = protect_content(content)

        assert "__MARKITAI_PAGE_" in protected_content
        assert len(mapping) == 2

    def test_protect_screenshot_reference_comments(self) -> None:
        """Test protecting screenshot reference comments."""
        content = """<!-- Screenshot for reference -->
<!-- ![Screenshot](.markitai/screenshots/page.full.jpg) -->
Content"""
        protected_content, mapping = protect_content(content)

        assert "__MARKITAI_PAGE_" in protected_content
        assert len(mapping) == 2

    def test_protect_no_markers(self) -> None:
        """Test content with no markers to protect."""
        content = "Just plain text without any markers."
        protected_content, mapping = protect_content(content)

        assert protected_content == content
        assert len(mapping) == 0


class TestStripPromptEcho:
    """Tests for strip_prompt_echo function."""

    # Verbatim tail of the pre-tag-delimiter document_vision_user.md as echoed
    # by the LLM (still present in old cached results)
    ECHOED_REMINDER = (
        "REMINDER: All `__MARKITAI_*__` placeholders must appear in your "
        "output exactly as in the input. Do not remove, modify, or merge "
        "any placeholder. Do not wrap output in a code block."
    )

    def test_strips_echoed_reminder_and_delimiter(self) -> None:
        """Regression: gpt-5.4-mini echoed the prompt tail into sample.pdf output."""
        text = "# Doc\n\nBody text.\n\n---\n\n" + self.ECHOED_REMINDER
        result = strip_prompt_echo(text)
        assert "REMINDER" not in result
        assert result == "# Doc\n\nBody text."

    def test_strips_incode_reminder_wording(self) -> None:
        """The in-code tail reminder (document.py) can be echoed too."""
        text = (
            "Content.\n\nREMINDER: Preserve ALL __MARKITAI_*__ placeholders "
            "exactly as-is. Do not remove or modify any placeholder."
        )
        result = strip_prompt_echo(text)
        assert "REMINDER" not in result
        assert "Content." in result

    def test_strips_reminder_in_middle(self) -> None:
        """Echo between batches must not swallow surrounding content."""
        text = "Page 1 text.\n\n---\n\n" + self.ECHOED_REMINDER + "\n\nPage 2 text."
        result = strip_prompt_echo(text)
        assert "REMINDER" not in result
        assert "Page 1 text." in result
        assert "Page 2 text." in result

    def test_keeps_content_without_reminder(self) -> None:
        text = "# Doc\n\nBody.\n\n---\n\nMore body."
        assert strip_prompt_echo(text) == text

    def test_keeps_regular_reminder_line(self) -> None:
        """A REMINDER line not mentioning placeholders is real content."""
        text = "# Notes\n\nREMINDER: Buy milk tomorrow."
        assert strip_prompt_echo(text) == text

    def test_keeps_trailing_rule_without_reminder(self) -> None:
        text = "Body.\n\n---"
        assert strip_prompt_echo(text) == text

    def test_unwraps_echoed_document_tags(self) -> None:
        """Models may mimic the prompt's <document> wrapper in their output."""
        text = "<document>\n# Doc\n\nBody.\n</document>"
        result = strip_prompt_echo(text)
        assert result == "# Doc\n\nBody."

    def test_strips_trailing_closing_tag_only(self) -> None:
        """A stray echoed </document> at the tail must be removed."""
        text = "# Doc\n\nBody.\n\n</document>"
        result = strip_prompt_echo(text)
        assert result == "# Doc\n\nBody."

    def test_keeps_document_tags_inside_content(self) -> None:
        """Tag lines mid-content (e.g. inside code blocks) are real content."""
        text = "# Doc\n\n```xml\n<document>\nx\n</document>\n```\n\nEnd."
        assert strip_prompt_echo(text) == text


class TestUnprotectContent:
    """Tests for unprotect_content function."""

    def test_unprotect_basic(self) -> None:
        """Test basic unprotection."""
        content = "__MARKITAI_PAGENUM_0__\nContent"
        mapping = {"__MARKITAI_PAGENUM_0__": "<!-- Page number: 1 -->"}

        result = unprotect_content(content, mapping)

        assert "<!-- Page number: 1 -->" in result
        assert "__MARKITAI_" not in result

    def test_unprotect_multiple_markers(self) -> None:
        """Test unprotecting multiple markers."""
        content = """
__MARKITAI_PAGENUM_0__
Page 1

__MARKITAI_SLIDENUM_0__
Slide 1
"""
        mapping = {
            "__MARKITAI_PAGENUM_0__": "<!-- Page number: 1 -->",
            "__MARKITAI_SLIDENUM_0__": "<!-- Slide number: 1 -->",
        }

        result = unprotect_content(content, mapping)

        assert "<!-- Page number: 1 -->" in result
        assert "<!-- Slide number: 1 -->" in result
        assert "__MARKITAI_" not in result

    def test_unprotect_removes_hallucinated_markers(self) -> None:
        """Test that hallucinated markers are removed."""
        content = """__MARKITAI_PAGENUM_0__
Real content
<!-- Page number: 99 -->
More content"""
        mapping = {"__MARKITAI_PAGENUM_0__": "<!-- Page number: 1 -->"}

        result = unprotect_content(content, mapping)

        # Real marker should be restored
        assert "<!-- Page number: 1 -->" in result
        # Hallucinated marker should be removed
        assert "<!-- Page number: 99 -->" not in result

    def test_unprotect_with_protected_fallback(self) -> None:
        """Test fallback restoration of missing images."""
        content = "Content without images"
        mapping = {}
        protected = {
            "images": ["![missing](image.jpg)"],
            "slides": [],
            "page_numbers": [],
            "page_comments": [],
        }

        result = unprotect_content(content, mapping, protected)

        # Missing image should be appended
        assert "![missing](image.jpg)" in result

    def test_unprotect_empty_mapping(self) -> None:
        """Test with empty mapping."""
        content = "Plain content"
        result = unprotect_content(content, {})

        assert result == content

    def test_residual_cleanup_preserves_img_placeholders(self) -> None:
        """Residual placeholder cleanup must not strip __MARKITAI_IMG_*__ placeholders.

        Image position placeholders are managed by _restore_images_or_fallback(),
        which runs AFTER unprotect_content(). If the residual cleanup removes them
        first, the fallback logic incorrectly concludes the LLM dropped them and
        discards the entire LLM output.
        """
        content = (
            "Paragraph one.\n\n"
            "__MARKITAI_IMG_0__\n\n"
            "Paragraph two.\n\n"
            "__MARKITAI_CODEBLOCK_99__\n"  # residual — should be cleaned
        )
        mapping: dict[str, str] = {}

        result = unprotect_content(content, mapping)

        # IMG placeholders must survive for downstream restoration
        assert "__MARKITAI_IMG_0__" in result
        # Other residual placeholders should be cleaned
        assert "__MARKITAI_CODEBLOCK_99__" not in result

    def test_roundtrip(self) -> None:
        """Test protect -> unprotect roundtrip."""
        original = """<!-- Page number: 1 -->
First page content

<!-- Page number: 2 -->
Second page content
"""
        protected_content, mapping = protect_content(original)
        restored = unprotect_content(protected_content, mapping)

        # Content should be equivalent (may have whitespace differences)
        assert "<!-- Page number: 1 -->" in restored
        assert "<!-- Page number: 2 -->" in restored
        assert "First page content" in restored
        assert "Second page content" in restored
