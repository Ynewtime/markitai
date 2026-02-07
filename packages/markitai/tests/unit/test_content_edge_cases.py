"""Tests for content protection and smart_truncate edge cases."""

from markitai.llm.content import (
    protect_content,
    smart_truncate,
    unprotect_content,
)


class TestSmartTruncate:
    """Test smart_truncate with edge cases."""

    def test_truncate_empty_string(self):
        """Empty string should return empty string."""
        assert smart_truncate("", 100) == ""

    def test_truncate_short_string(self):
        """String shorter than limit should return unchanged."""
        text = "Hello world"
        assert smart_truncate(text, 100) == text

    def test_truncate_exact_limit(self):
        """String exactly at limit should return unchanged."""
        text = "A" * 50
        assert smart_truncate(text, 50) == text

    def test_truncate_at_sentence_boundary(self):
        """Should truncate at sentence boundary when possible."""
        text = "First sentence. Second sentence. Third sentence is much longer and goes on."
        result = smart_truncate(text, 35)
        # Should end at a sentence boundary (period)
        assert result.endswith(".")

    def test_truncate_at_cjk_boundary(self):
        """Should handle CJK text boundaries correctly."""
        text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。"
        result = smart_truncate(text, 20)
        # Should try to end at a sentence boundary (。)
        assert result.endswith("。") or len(result) <= 21  # Allow slight overflow

    def test_truncate_preserves_end(self):
        """preserve_end=True should keep the end of the text."""
        text = "A" * 200 + "\n\nimportant ending paragraph."
        result = smart_truncate(text, 80, preserve_end=True)
        assert "important ending" in result

    def test_truncate_no_good_boundary(self):
        """Without sentence boundaries, falls back to simple truncation."""
        text = "a" * 200  # No sentence boundaries
        result = smart_truncate(text, 100)
        assert len(result) == 100

    def test_truncate_paragraph_boundary(self):
        """Should prefer paragraph boundaries over sentence boundaries."""
        text = "Paragraph one content.\n\nParagraph two has more." + "x" * 500
        result = smart_truncate(text, 50)
        # Should truncate at a natural boundary
        assert len(result) <= 60  # Allow some overflow for better breaks


class TestProtectContent:
    """Test content protection with edge cases."""

    def test_protect_no_markers(self):
        """Content with no markers should return unchanged."""
        text = "Just plain text without any markers."
        protected, mapping = protect_content(text)
        assert protected == text
        assert len(mapping) == 0

    def test_protect_page_number_markers(self):
        """Page number markers should be protected."""
        text = "Content before.\n\n<!-- Page number: 1 -->\n\nContent after."
        protected, mapping = protect_content(text)
        assert "<!-- Page number: 1 -->" not in protected
        assert "__MARKITAI_PAGENUM_" in protected
        assert len(mapping) == 1

    def test_protect_slide_number_markers(self):
        """Slide number markers should be protected."""
        text = "Content.\n\n<!-- Slide number: 3 -->\n\nSlide content."
        protected, mapping = protect_content(text)
        assert "<!-- Slide number: 3 -->" not in protected
        assert "__MARKITAI_SLIDENUM_" in protected
        assert len(mapping) == 1

    def test_protect_multiple_markers(self):
        """Multiple markers should all be protected."""
        text = (
            "<!-- Page number: 1 -->\n"
            "Page 1 content.\n"
            "<!-- Page number: 2 -->\n"
            "Page 2 content."
        )
        protected, mapping = protect_content(text)
        assert len(mapping) == 2

    def test_protect_and_unprotect_roundtrip(self):
        """Protect then unprotect should restore original markers."""
        original = (
            "Text before.\n\n"
            "<!-- Page number: 1 -->\n\n"
            "Middle text.\n\n"
            "<!-- Slide number: 2 -->\n\n"
            "Text after."
        )
        protected, mapping = protect_content(original)
        # Verify markers are replaced
        assert "<!-- Page number: 1 -->" not in protected
        assert "<!-- Slide number: 2 -->" not in protected
        # Restore
        restored = unprotect_content(protected, mapping)
        # Both markers should be restored
        assert "<!-- Page number: 1 -->" in restored
        assert "<!-- Slide number: 2 -->" in restored


class TestProtectContentImagesNotProtected:
    """Verify that images are NOT protected by protect_content (by design)."""

    def test_images_pass_through(self):
        """Images should not be replaced with placeholders."""
        text = "Text.\n\n![Image](assets/doc.0001.jpg)\n\nMore text."
        protected, mapping = protect_content(text)
        # Images should remain as-is (not protected)
        assert "![Image](assets/doc.0001.jpg)" in protected
        # No image-related mappings
        assert not any("IMG" in k for k in mapping)
