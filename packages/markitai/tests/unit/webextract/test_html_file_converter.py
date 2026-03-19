"""Tests for HTML file conversion going through webextract pipeline."""

from __future__ import annotations

from pathlib import Path

from markitai.converter.markitdown_ext import HtmlConverter


class TestHtmlFileWebextract:
    def test_html_file_uses_webextract_pipeline(self, tmp_path: Path):
        """HTML files should go through webextract noise removal."""
        html_file = tmp_path / "test.html"
        html_file.write_text(
            """<html><body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <article>
                <h1>Test Article</h1>
                <p>This is the main article content with enough substance
                to be meaningful and pass quality thresholds.</p>
                <p>A second paragraph with additional detail and context.</p>
            </article>
            <div class="sidebar">
                <h3>Popular Posts</h3>
                <ul><li><a href="/1">Post 1</a></li></ul>
            </div>
            <footer>Copyright 2026</footer>
            </body></html>""",
            encoding="utf-8",
        )

        converter = HtmlConverter()
        result = converter.convert(html_file)

        assert "Test Article" in result.markdown
        assert "main article content" in result.markdown
        # Noise should be cleaned by webextract
        assert "Popular Posts" not in result.markdown
        assert result.metadata.get("converter") == "webextract"

    def test_html_file_with_task_list(self, tmp_path: Path):
        html_file = tmp_path / "tasks.html"
        html_file.write_text(
            """<html><body><article>
            <ul>
            <li class="task-list-item"><input type="checkbox" checked> Done</li>
            <li class="task-list-item"><input type="checkbox"> Pending</li>
            </ul>
            <p>Content with enough words to pass threshold check.</p>
            </article></body></html>""",
            encoding="utf-8",
        )

        converter = HtmlConverter()
        result = converter.convert(html_file)

        assert "[x]" in result.markdown
        assert "[ ]" in result.markdown

    def test_html_file_fallback_to_markitdown(self, tmp_path: Path):
        """If webextract produces too little content, fall back to markitdown."""
        html_file = tmp_path / "minimal.html"
        html_file.write_text("<html><body><p>Hi</p></body></html>", encoding="utf-8")

        converter = HtmlConverter()
        result = converter.convert(html_file)

        # Should still produce something (via markitdown fallback)
        assert result.markdown is not None
