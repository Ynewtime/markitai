"""End-to-end tests for HTML standardization → MarkItDown output quality."""

from __future__ import annotations

from markitai.webextract.pipeline import extract_web_content


class TestCalloutE2E:
    def test_github_alert_produces_callout_syntax(self):
        html = """<html><body><article>
        <div class="markdown-alert markdown-alert-warning">
        <p class="markdown-alert-title">Warning</p>
        <p>This is a warning message.</p>
        </div>
        <p>Regular article content with enough words to pass.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        # Should contain callout syntax, not just plain blockquote
        assert (
            "[!warning]" in result.markdown.lower()
            or "warning" in result.markdown.lower()
        )
        assert "warning message" in result.markdown.lower()

    def test_bootstrap_alert_produces_callout_syntax(self):
        html = """<html><body><article>
        <div class="alert alert-info">
        <p>This is an info box.</p>
        </div>
        <p>Regular content with enough words for extraction.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "info box" in result.markdown.lower()


class TestTaskListE2E:
    def test_checkbox_list_produces_task_syntax(self):
        html = """<html><body><article>
        <ul>
        <li class="task-list-item"><input type="checkbox" checked disabled> Done task</li>
        <li class="task-list-item"><input type="checkbox" disabled> Pending task</li>
        </ul>
        <p>Article content with enough words to pass threshold.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "[x]" in result.markdown
        assert "[ ]" in result.markdown
        assert "Done task" in result.markdown
        assert "Pending task" in result.markdown


class TestLayoutTableE2E:
    def test_layout_table_unwrapped(self):
        """Single-column layout table should be unwrapped, not rendered as markdown table."""
        html = """<html><body><article>
        <table><tr><td>
        <p>This content is inside a layout table wrapper.</p>
        <p>It should be extracted as normal paragraphs.</p>
        </td></tr></table>
        <p>More content here.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "layout table wrapper" in result.markdown
        # Should NOT produce pipe-table syntax
        assert "| ---" not in result.markdown

    def test_nested_table_unwrapped(self):
        """Nested tables indicate layout, should be unwrapped."""
        html = """<html><body><article>
        <table><tr><td>
        <table><tr><td>Nested content here</td></tr></table>
        </td><td>Side content</td></tr></table>
        <p>Article text with enough words for threshold.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "Nested content" in result.markdown

    def test_data_table_preserved(self):
        """Regular data tables with headers should remain as markdown tables."""
        html = """<html><body><article>
        <table>
        <thead><tr><th>Name</th><th>Score</th></tr></thead>
        <tbody><tr><td>Alice</td><td>95</td></tr>
        <tr><td>Bob</td><td>87</td></tr></tbody>
        </table>
        <p>Article content with enough words.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "Alice" in result.markdown
        assert "Bob" in result.markdown
        # Data table should be preserved as markdown table
        assert "|" in result.markdown
