"""Tests for remaining defuddle gaps: complex tables, srcset, code language."""

from __future__ import annotations

from markitai.webextract.dom import parse_html
from markitai.webextract.pipeline import extract_web_content


class TestComplexTable:
    def test_colspan_table_preserved_as_html(self):
        """Tables with colspan should not be converted to broken markdown tables."""
        html = """<html><body><article>
        <table>
        <tr><th colspan="2">Full Width Header</th></tr>
        <tr><td>Cell A</td><td>Cell B</td></tr>
        </table>
        <p>Article content with enough words to pass threshold.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "Full Width Header" in result.markdown
        assert "Cell A" in result.markdown

    def test_rowspan_table_preserved_as_html(self):
        html = """<html><body><article>
        <table>
        <tr><td rowspan="2">Merged</td><td>A</td></tr>
        <tr><td>B</td></tr>
        </table>
        <p>Article content with enough words to pass threshold.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "Merged" in result.markdown
        assert "A" in result.markdown

    def test_simple_table_still_markdown(self):
        """Simple tables without colspan/rowspan should still be markdown."""
        html = """<html><body><article>
        <table>
        <thead><tr><th>X</th><th>Y</th></tr></thead>
        <tbody><tr><td>1</td><td>2</td></tr></tbody>
        </table>
        <p>Article content with enough words.</p>
        </article></body></html>"""
        result = extract_web_content(html, "https://example.com")
        assert "|" in result.markdown


class TestImageSrcset:
    def test_srcset_picks_best_resolution(self):
        from markitai.webextract.elements.images import normalize_images

        soup = parse_html(
            '<div><img src="small.jpg" srcset="medium.jpg 800w, large.jpg 1200w, small.jpg 400w"></div>'
        )
        root = soup.find("div")
        normalize_images(root, "https://example.com")
        img = root.find("img")
        assert img is not None
        # Should pick the largest width (1200w)
        assert "large.jpg" in str(img.get("src", ""))

    def test_srcset_with_density_descriptors(self):
        from markitai.webextract.elements.images import normalize_images

        soup = parse_html('<div><img src="1x.jpg" srcset="2x.jpg 2x, 3x.jpg 3x"></div>')
        root = soup.find("div")
        normalize_images(root, "https://example.com")
        img = root.find("img")
        assert img is not None
        # Should pick highest density (3x)
        assert "3x.jpg" in str(img.get("src", ""))

    def test_no_srcset_keeps_src(self):
        from markitai.webextract.elements.images import normalize_images

        soup = parse_html('<div><img src="photo.jpg" alt="photo"></div>')
        root = soup.find("div")
        normalize_images(root, "https://example.com")
        img = root.find("img")
        assert "photo.jpg" in str(img.get("src", ""))


class TestCodeLanguageDetection:
    def test_language_class_on_code(self):
        from markitai.webextract.elements.code import normalize_code_blocks

        soup = parse_html(
            '<div><pre><code class="language-python">print("hello")</code></pre></div>'
        )
        root = soup.find("div")
        normalize_code_blocks(root)
        code = root.find("code")
        assert code is not None
        classes = code.get("class", [])
        # language-python class should be preserved
        assert any("python" in c for c in classes)

    def test_lang_class_variant(self):
        from markitai.webextract.elements.code import normalize_code_blocks

        soup = parse_html(
            '<div><pre><code class="lang-javascript">var x = 1;</code></pre></div>'
        )
        root = soup.find("div")
        normalize_code_blocks(root)
        code = root.find("code")
        assert code is not None
        classes = code.get("class", [])
        # Should normalize to language-javascript
        assert any("javascript" in c for c in classes)

    def test_highlight_class_variant(self):
        from markitai.webextract.elements.code import normalize_code_blocks

        soup = parse_html(
            '<div><pre class="highlight-ruby"><code>puts "hello"</code></pre></div>'
        )
        root = soup.find("div")
        normalize_code_blocks(root)
        code = root.find("code")
        assert code is not None
        classes = code.get("class", [])
        assert any("ruby" in c for c in classes)

    def test_data_lang_attribute(self):
        from markitai.webextract.elements.code import normalize_code_blocks

        soup = parse_html(
            '<div><pre data-lang="rust"><code>fn main() {}</code></pre></div>'
        )
        root = soup.find("div")
        normalize_code_blocks(root)
        code = root.find("code")
        assert code is not None
        classes = code.get("class", [])
        assert any("rust" in c for c in classes)
