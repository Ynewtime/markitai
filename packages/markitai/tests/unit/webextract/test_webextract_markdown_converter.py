"""Tests for WebExtractMarkdownConverter code-block language detection."""

from __future__ import annotations

from markitai.converter.webextract_html_converter import WebExtractMarkdownConverter


def _convert(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    converter = WebExtractMarkdownConverter()
    return converter.convert_soup(soup).strip()


class TestCodeBlockLanguageDetection:
    def test_language_class_prefix(self) -> None:
        html = '<pre><code class="language-python">print("hello")</code></pre>'
        md = _convert(html)
        assert "```python" in md
        assert 'print("hello")' in md

    def test_lang_class_prefix(self) -> None:
        html = '<pre><code class="lang-javascript">const x = 1;</code></pre>'
        md = _convert(html)
        assert "```javascript" in md

    def test_highlight_class_prefix(self) -> None:
        html = '<pre><code class="highlight-ruby">puts "hi"</code></pre>'
        md = _convert(html)
        assert "```ruby" in md

    def test_data_lang_attribute(self) -> None:
        html = '<pre><code data-lang="rust">fn main() {}</code></pre>'
        md = _convert(html)
        assert "```rust" in md

    def test_prism_class_on_pre(self) -> None:
        html = '<pre class="language-typescript"><code>let x: number = 1;</code></pre>'
        md = _convert(html)
        assert "```typescript" in md

    def test_no_language_produces_plain_code_block(self) -> None:
        html = "<pre><code>plain code</code></pre>"
        md = _convert(html)
        assert "```" in md
        assert "plain code" in md

    def test_syntax_highlighter_brush(self) -> None:
        html = '<pre class="brush: java">public class Foo {}</pre>'
        md = _convert(html)
        assert "```java" in md

    def test_multiple_classes_picks_language(self) -> None:
        html = '<pre><code class="hljs language-go">func main() {}</code></pre>'
        md = _convert(html)
        assert "```go" in md

    def test_unknown_language_still_detected(self) -> None:
        html = (
            '<pre><code class="language-solidity">pragma solidity ^0.8.0;</code></pre>'
        )
        md = _convert(html)
        assert "```solidity" in md


class TestHighlightAndStrikethrough:
    def test_mark_to_highlight(self) -> None:
        html = "<p>This is <mark>highlighted</mark> text.</p>"
        md = _convert(html)
        assert "==highlighted==" in md

    def test_del_to_strikethrough(self) -> None:
        html = "<p>This is <del>deleted</del> text.</p>"
        md = _convert(html)
        assert "~~deleted~~" in md

    def test_s_to_strikethrough(self) -> None:
        html = "<p>This is <s>struck</s> text.</p>"
        md = _convert(html)
        assert "~~struck~~" in md


class TestFootnoteConversion:
    def test_sup_footnote_ref(self) -> None:
        html = '<p>Some text<sup><a href="#fn1" id="fnref1">1</a></sup> continues.</p>'
        md = _convert(html)
        assert "[^1]" in md

    def test_sup_non_footnote_preserved(self) -> None:
        html = "<p>x<sup>2</sup> + y<sup>2</sup></p>"
        md = _convert(html)
        assert "[^" not in md


class TestMathConversion:
    def test_katex_annotation(self) -> None:
        html = '<span class="katex"><span class="katex-mathml"><math><semantics><annotation encoding="application/x-tex">E = mc^2</annotation></semantics></math></span></span>'
        md = _convert(html)
        assert "$E = mc^2$" in md

    def test_mathjax_tex_script(self) -> None:
        html = '<script type="math/tex">\\alpha + \\beta</script>'
        md = _convert(html)
        assert "$\\alpha + \\beta$" in md

    def test_mathjax_display_script(self) -> None:
        html = '<script type="math/tex; mode=display">\\int_0^1 f(x) dx</script>'
        md = _convert(html)
        assert "$$\\int_0^1 f(x) dx$$" in md

    def test_mathml_with_alttext(self) -> None:
        html = '<math alttext="x^2 + y^2 = z^2"><mi>x</mi></math>'
        md = _convert(html)
        assert "$x^2 + y^2 = z^2$" in md

    def test_math_display_block(self) -> None:
        html = '<math display="block" alttext="\\sum_{i=1}^n i"><mi>sum</mi></math>'
        md = _convert(html)
        assert "$$\\sum_{i=1}^n i$$" in md
