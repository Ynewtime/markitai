"""Tests for content pattern removal (Phase 3)."""

from __future__ import annotations

from markitai.webextract.dom import parse_html
from markitai.webextract.removals.content_patterns import remove_content_patterns


class TestReadTimeRemoval:
    def test_removes_read_time_metadata(self):
        soup = parse_html("<div><p>3 min read</p><p>Article content here.</p></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "min read" not in root.get_text()

    def test_removes_minutes_read_variant(self):
        soup = parse_html("<div><span>5 minutes read</span><p>Content.</p></div>")
        root = soup.find("div")
        assert root is not None
        remove_content_patterns(root)
        assert "minutes read" not in root.get_text()

    def test_keeps_non_matching_text(self):
        soup = parse_html("<div><p>Read this article about Python.</p></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed == 0
        assert "Read this article" in root.get_text()


class TestBoilerplateRemoval:
    def test_removes_originally_published(self):
        soup = parse_html(
            "<div><p>Content.</p><p>Originally published in The Atlantic.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "Originally published" not in root.get_text()

    def test_removes_article_appeared(self):
        soup = parse_html(
            "<div><p>Content.</p><p>This article appeared in Nature.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1

    def test_keeps_normal_sentences(self):
        soup = parse_html("<div><p>This article discusses important topics.</p></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed == 0


class TestAuthorBylineRemoval:
    def test_removes_byline_with_date(self):
        soup = parse_html(
            '<div><p class="byline">By John Smith · March 15, 2026</p>'
            "<p>Article content with enough words.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1

    def test_keeps_longer_paragraphs(self):
        """Long paragraphs that happen to contain 'by' should not be removed."""
        soup = parse_html(
            "<div><p>By implementing this approach, we achieved significant "
            "improvements across multiple metrics and validated our hypothesis "
            "through extensive experimentation.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed == 0
