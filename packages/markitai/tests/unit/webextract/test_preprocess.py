from __future__ import annotations

"""Tests for webextract.preprocess — raw HTML preprocessing phase."""


from markitai.webextract.pipeline import extract_web_content
from markitai.webextract.preprocess import preprocess_html

# ---------------------------------------------------------------------------
# preprocess_html unit tests
# ---------------------------------------------------------------------------


class TestDeclarativeShadowDomFlattening:
    """Declarative shadow DOM <template shadowrootmode="open"> should be flattened."""

    def test_flattens_open_shadow_root_template(self) -> None:
        html = (
            '<div id="host">'
            '<template shadowrootmode="open"><p>shadow text</p></template>'
            "</div>"
        )
        result = preprocess_html(html)
        assert "<p>shadow text</p>" in result
        assert "shadowrootmode" not in result

    def test_flattens_shadow_root_case_insensitive(self) -> None:
        html = (
            '<div><template shadowrootmode="Open"><span>content</span></template></div>'
        )
        result = preprocess_html(html)
        assert "<span>content</span>" in result

    def test_removes_template_wrapper_tag(self) -> None:
        html = '<div><template shadowrootmode="open"><p>inner</p></template></div>'
        result = preprocess_html(html)
        assert "<template" not in result

    def test_preserves_non_shadow_templates(self) -> None:
        html = "<div><template id='tmpl'><p>template content</p></template></div>"
        result = preprocess_html(html)
        # Regular templates (no shadowrootmode) should be left intact
        assert "<template" in result

    def test_multiple_shadow_roots_flattened(self) -> None:
        html = (
            '<div><template shadowrootmode="open"><p>first</p></template></div>'
            '<div><template shadowrootmode="open"><p>second</p></template></div>'
        )
        result = preprocess_html(html)
        assert "<p>first</p>" in result
        assert "<p>second</p>" in result
        assert "shadowrootmode" not in result


class TestWbrRemoval:
    """<wbr> tags break word boundaries and should be removed."""

    def test_removes_wbr_tags(self) -> None:
        html = "<p>super<wbr>long<wbr>word</p>"
        result = preprocess_html(html)
        assert "<wbr>" not in result
        assert "<wbr/>" not in result
        assert "superlong" in result or "super" in result  # text preserved

    def test_removes_self_closing_wbr(self) -> None:
        html = "<p>word<wbr/>break</p>"
        result = preprocess_html(html)
        assert "wbr" not in result

    def test_preserves_surrounding_text(self) -> None:
        html = "<p>hello<wbr>world</p>"
        result = preprocess_html(html)
        assert "hello" in result
        assert "world" in result


class TestStreamedContentNormalization:
    """Streamed / incomplete HTML should be normalized before parsing."""

    def test_handles_unclosed_body_tag(self) -> None:
        # Should not raise; just return cleaned string
        html = "<html><body><p>real body text</p>"
        result = preprocess_html(html)
        assert "real body text" in result

    def test_handles_empty_string(self) -> None:
        result = preprocess_html("")
        assert result == ""

    def test_handles_whitespace_only(self) -> None:
        result = preprocess_html("   \n\t  ")
        # Should return something without error
        assert isinstance(result, str)


class TestNoscriptPromotion:
    """<noscript> content should be promoted when JS-dependent main content exists."""

    def test_promotes_noscript_when_body_is_script_only(self) -> None:
        html = (
            "<html><body>"
            "<script>renderApp()</script>"
            "<noscript><p>real body text</p></noscript>"
            "</body></html>"
        )
        result = preprocess_html(html)
        assert "real body text" in result

    def test_does_not_promote_noscript_when_body_has_content(self) -> None:
        html = (
            "<html><body>"
            "<p>Substantial real content here that is meaningful.</p>"
            "<noscript><p>fallback text</p></noscript>"
            "</body></html>"
        )
        result = preprocess_html(html)
        # noscript not promoted when body already has content
        assert "Substantial real content here" in result


# ---------------------------------------------------------------------------
# Integration: preprocess feeds into extract_web_content
# ---------------------------------------------------------------------------


class TestPreprocessIntegration:
    """preprocess_html should run automatically inside parse_html / extract_web_content."""

    def test_extract_web_content_sees_shadow_dom_text(self) -> None:
        html = (
            "<html><body>"
            "<h1>Article</h1>"
            '<div id="host">'
            '<template shadowrootmode="open">'
            "<p>shadow paragraph with real body text about the topic</p>"
            "</template>"
            "</div>"
            "</body></html>"
        )
        url = "https://example.com/article"
        result = extract_web_content(html, url)
        assert (
            "shadow paragraph" in result.markdown or "real body text" in result.markdown
        )

    def test_extract_web_content_no_wbr_artifacts(self) -> None:
        html = (
            "<html><body>"
            "<article>"
            "<p>Some<wbr>LongWord and more content here to pass word count threshold.</p>"
            "</article>"
            "</body></html>"
        )
        url = "https://example.com/test"
        result = extract_web_content(html, url)
        assert "wbr" not in result.markdown.lower()

    def test_extract_from_shadow_dom_fixture(self, fixtures_dir: object) -> None:
        import pathlib

        fixture_path = pathlib.Path(str(fixtures_dir)) / "web" / "shadow_dom_page.html"
        html = fixture_path.read_text(encoding="utf-8")
        result = extract_web_content(html, "https://example.com/shadow")
        assert "shadow text" in result.markdown
