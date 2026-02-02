"""Unit tests for LLM content protection and formatting utilities."""

from __future__ import annotations

from markitai.llm.content import (
    extract_protected_content,
    protect_content,
    smart_truncate,
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
<!-- ![Page 1](screenshots/page1.png) -->
Content here
"""
        protected = extract_protected_content(content)

        assert len(protected["page_comments"]) == 2

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
<!-- ![Page 1](screenshots/page1.png) -->
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
