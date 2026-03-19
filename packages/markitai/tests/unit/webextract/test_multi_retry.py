"""Tests for multi-level adaptive retry (Phase 2)."""

from __future__ import annotations

from markitai.webextract.pipeline import extract_web_content


class TestMultiLevelRetry:
    def test_extracts_content_from_minimal_page(self):
        """A page with very little content should still produce output."""
        html = """
        <html><body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <div style="display:none">hidden tracking</div>
            <article><p>Short content.</p></article>
            <footer>Copyright</footer>
        </body></html>
        """
        result = extract_web_content(html, "https://example.com/page")
        assert result.word_count >= 2
        assert "Short content" in result.markdown

    def test_handles_page_with_all_noise(self):
        """A page where everything looks like noise should fallback gracefully."""
        html = """
        <html><body>
            <nav><a href="/">Home</a></nav>
            <div class="sidebar"><a href="/1">Link</a></div>
            <footer>Footer text</footer>
        </body></html>
        """
        result = extract_web_content(html, "https://example.com/page")
        # Should produce something (even if minimal) via fallback
        assert isinstance(result.markdown, str)
