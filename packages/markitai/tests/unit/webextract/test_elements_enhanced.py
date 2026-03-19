"""Tests for enhanced element handlers (Phase 3)."""

from __future__ import annotations

from markitai.webextract.dom import parse_html


class TestHeadingAnchorRemoval:
    def test_removes_permalink_hash(self):
        from markitai.webextract.elements.headings import normalize_headings

        soup = parse_html('<div><h2>Title <a href="#title">#</a></h2></div>')
        root = soup.find("div")
        normalize_headings(root)
        h2 = root.find("h2")
        assert h2 is not None
        assert "#" not in h2.get_text()
        assert "Title" in h2.get_text()

    def test_removes_paragraph_mark(self):
        from markitai.webextract.elements.headings import normalize_headings

        soup = parse_html('<div><h3>Section <a href="#section">¶</a></h3></div>')
        root = soup.find("div")
        normalize_headings(root)
        assert "¶" not in root.get_text()
        assert "Section" in root.get_text()

    def test_keeps_meaningful_links_in_headings(self):
        from markitai.webextract.elements.headings import normalize_headings

        soup = parse_html(
            '<div><h2><a href="https://example.com">External Link Title</a></h2></div>'
        )
        root = soup.find("div")
        normalize_headings(root)
        assert root.find("a") is not None


class TestCalloutStandardization:
    def test_github_alert(self):
        from markitai.webextract.elements.callouts import normalize_callouts

        soup = parse_html(
            '<div><div class="markdown-alert markdown-alert-note">'
            '<p class="markdown-alert-title">Note</p>'
            "<p>Important info here.</p></div></div>"
        )
        root = soup.find("div")
        normalize_callouts(root)
        bq = root.find("blockquote")
        assert bq is not None
        assert bq.get("data-callout") == "note"

    def test_bootstrap_alert(self):
        from markitai.webextract.elements.callouts import normalize_callouts

        soup = parse_html(
            '<div><div class="alert alert-warning"><p>Warning message.</p></div></div>'
        )
        root = soup.find("div")
        normalize_callouts(root)
        bq = root.find("blockquote")
        assert bq is not None
        assert bq.get("data-callout") == "warning"

    def test_keeps_non_callout_divs(self):
        from markitai.webextract.elements.callouts import normalize_callouts

        soup = parse_html(
            '<div><div class="regular-div"><p>Normal content.</p></div></div>'
        )
        root = soup.find("div")
        normalize_callouts(root)
        assert root.find("blockquote") is None
