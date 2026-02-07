"""Tests verifying content.py is the single source of truth for image protection."""


class TestPatternConsolidation:
    """Verify document.py uses patterns from content.py (no duplication)."""

    def test_content_has_protect_image_positions(self):
        """content.py should export protect_image_positions."""
        from markitai.llm import content

        assert hasattr(content, "protect_image_positions")
        assert callable(content.protect_image_positions)

    def test_content_has_restore_image_positions(self):
        """content.py should export restore_image_positions."""
        from markitai.llm import content

        assert hasattr(content, "restore_image_positions")
        assert callable(content.restore_image_positions)

    def test_protect_image_positions_excludes_screenshots(self):
        """protect_image_positions should support exclude_screenshots parameter."""
        from markitai.llm.content import protect_image_positions

        md = "![alt](assets/doc.0001.jpg)\n![Page 1](screenshots/page1.jpg)"
        protected, mapping = protect_image_positions(md, exclude_screenshots=True)
        # Only the assets image should be protected, not the screenshot
        assert "screenshots/page1.jpg" in protected
        assert "assets/doc.0001.jpg" not in protected

    def test_protect_image_positions_includes_all_by_default(self):
        """Without exclude_screenshots, all images should be protected."""
        from markitai.llm.content import protect_image_positions

        md = "![alt](assets/doc.0001.jpg)\n![Page 1](screenshots/page1.jpg)"
        protected, mapping = protect_image_positions(md, exclude_screenshots=False)
        # Both images should be protected
        assert "assets/doc.0001.jpg" not in protected
        assert "screenshots/page1.jpg" not in protected
        assert len(mapping) == 2

    def test_protect_and_restore_roundtrip(self):
        """Protect then restore should return original content."""
        from markitai.llm.content import (
            protect_image_positions,
            restore_image_positions,
        )

        original = "Text\n![img](assets/test.jpg)\nMore text\n![img2](assets/test2.png)"
        protected, mapping = protect_image_positions(original)
        restored = restore_image_positions(protected, mapping)
        assert restored == original

    def test_document_mixin_delegates_to_content(self):
        """DocumentMixin._protect_image_positions should delegate to content module."""
        from markitai.llm.document import DocumentMixin

        md = "![alt](assets/doc.jpg)\n![Page 1](screenshots/page1.jpg)"
        protected, mapping = DocumentMixin._protect_image_positions(md)
        # Should exclude screenshots (like the old behavior)
        assert "screenshots/page1.jpg" in protected
        assert "assets/doc.jpg" not in protected
