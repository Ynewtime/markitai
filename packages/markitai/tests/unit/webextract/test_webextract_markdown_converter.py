"""Tests for WebExtractMarkdownConverter code-block language detection."""

from __future__ import annotations

from markitai.webextract.html_to_markdown import (
    WebExtractMarkdownConverter,
    _mathml_to_latex,
)


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

    def test_mathml_structural_fallback(self) -> None:
        """MathML without annotation/alttext uses structural conversion."""
        html = "<math><mi>a</mi><msup><mi>x</mi><mn>2</mn></msup><mo>+</mo><mi>b</mi></math>"
        md = _convert(html)
        assert "x^{2}" in md
        assert "+" in md

    def test_katex_display(self) -> None:
        html = (
            '<span class="katex katex-display">'
            '<span class="katex-mathml"><math><semantics>'
            '<annotation encoding="application/x-tex">E = mc^2</annotation>'
            "</semantics></math></span></span>"
        )
        md = _convert(html)
        assert "$$E = mc^2$$" in md


class TestMathMLToLatex:
    """Unit tests for the MathML → LaTeX structural converter."""

    def _latex(self, mathml_inner: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(f"<math>{mathml_inner}</math>", "html.parser")
        return _mathml_to_latex(soup.find("math"))

    # --- Leaf elements ---

    def test_mi_single_char(self) -> None:
        assert self._latex("<mi>x</mi>") == "x"

    def test_mi_multichar_becomes_mathrm(self) -> None:
        assert self._latex("<mi>sin</mi>") == r"\mathrm{sin}"

    def test_mn(self) -> None:
        assert self._latex("<mn>42</mn>") == "42"

    def test_mo_plain(self) -> None:
        assert self._latex("<mo>+</mo>") == "+"

    def test_mo_unicode_replacement(self) -> None:
        assert self._latex("<mo>≠</mo>") == r"\neq"
        assert self._latex("<mo>≤</mo>") == r"\leq"
        assert self._latex("<mo>∞</mo>") == r"\infty"

    def test_mo_greek_letter(self) -> None:
        assert self._latex("<mi>α</mi>") == r"\alpha"
        assert self._latex("<mi>Ω</mi>") == r"\Omega"

    def test_mtext(self) -> None:
        assert self._latex("<mtext>hello</mtext>") == r"\text{hello}"

    def test_mspace(self) -> None:
        assert self._latex("<mspace/>") == r"\;"

    # --- Structural elements ---

    def test_msup(self) -> None:
        assert self._latex("<msup><mi>x</mi><mn>2</mn></msup>") == "x^{2}"

    def test_msub(self) -> None:
        assert self._latex("<msub><mi>a</mi><mi>i</mi></msub>") == "a_{i}"

    def test_msubsup(self) -> None:
        result = self._latex("<msubsup><mi>x</mi><mn>0</mn><mn>1</mn></msubsup>")
        assert result == "x_{0}^{1}"

    def test_mfrac(self) -> None:
        result = self._latex(
            "<mfrac><mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow><mi>c</mi></mfrac>"
        )
        assert result == r"\frac{a + b}{c}"

    def test_msqrt(self) -> None:
        assert self._latex("<msqrt><mi>x</mi></msqrt>") == r"\sqrt{x}"

    def test_mroot(self) -> None:
        result = self._latex("<mroot><mi>x</mi><mn>3</mn></mroot>")
        assert result == r"\sqrt[3]{x}"

    # --- Overscripts ---

    def test_mover_dot(self) -> None:
        assert self._latex("<mover><mi>x</mi><mo>˙</mo></mover>") == r"\dot{x}"

    def test_mover_overline(self) -> None:
        assert self._latex("<mover><mi>x</mi><mo>¯</mo></mover>") == r"\overline{x}"

    def test_mover_hat(self) -> None:
        assert self._latex("<mover><mi>x</mi><mo>^</mo></mover>") == r"\hat{x}"

    def test_mover_vec_arrow(self) -> None:
        """Arrow in <mo> must produce \\vec, not \\overset{\\rightarrow}."""
        assert self._latex("<mover><mi>B</mi><mo>→</mo></mover>") == r"\vec{B}"

    def test_mover_generic(self) -> None:
        result = self._latex("<mover><mi>x</mi><mo>*</mo></mover>")
        assert result == r"\overset{*}{x}"

    # --- Underscript / underover ---

    def test_munder(self) -> None:
        result = self._latex("<munder><mo>∑</mo><mi>i</mi></munder>")
        assert result == r"\underset{i}{\sum}"

    def test_munderover(self) -> None:
        result = self._latex("<munderover><mo>∑</mo><mn>0</mn><mi>n</mi></munderover>")
        assert result == r"\sum_{0}^{n}"

    # --- Table (aligned) ---

    def test_mtable(self) -> None:
        result = self._latex(
            "<mtable>"
            "<mtr><mtd><mi>a</mi></mtd><mtd><mi>b</mi></mtd></mtr>"
            "<mtr><mtd><mi>c</mi></mtd><mtd><mi>d</mi></mtd></mtr>"
            "</mtable>"
        )
        assert r"\begin{aligned}" in result
        assert "a & b" in result
        assert r"\\" in result
        assert "c & d" in result

    # --- Semantics wrapper ---

    def test_semantics_uses_first_child(self) -> None:
        result = self._latex(
            "<semantics><mrow><mi>x</mi></mrow>"
            '<annotation encoding="application/x-tex">x</annotation>'
            "</semantics>"
        )
        assert result == "x"

    # --- mrow passthrough ---

    def test_mrow_concatenates(self) -> None:
        result = self._latex("<mrow><mi>a</mi><mo>+</mo><mi>b</mi></mrow>")
        assert result == "a + b"

    # --- Nested expression ---

    def test_quadratic_formula(self) -> None:
        mathml = (
            "<mfrac>"
            "<mrow><mo>-</mo><mi>b</mi><mo>±</mo>"
            "<msqrt><mrow><msup><mi>b</mi><mn>2</mn></msup>"
            "<mo>-</mo><mn>4</mn><mi>a</mi><mi>c</mi></mrow></msqrt>"
            "</mrow>"
            "<mrow><mn>2</mn><mi>a</mi></mrow>"
            "</mfrac>"
        )
        result = self._latex(mathml)
        assert r"\frac" in result
        assert r"\sqrt" in result
        assert "b^{2}" in result
        assert r"\pm" in result

    # --- mfenced ---

    def test_mfenced_default(self) -> None:
        result = self._latex("<mfenced><mi>a</mi><mi>b</mi></mfenced>")
        assert result == r"\left(a , b\right)"

    def test_mfenced_custom_delimiters(self) -> None:
        result = self._latex('<mfenced open="[" close="]"><mi>x</mi></mfenced>')
        assert result == r"\left[x\right]"


class TestDataLatexAttribute:
    def test_data_latex_preferred(self) -> None:
        html = '<math data-latex="a+b" alttext="ignored">a+b</math>'
        md = _convert(html)
        assert "$a+b$" in md

    def test_data_latex_display_block(self) -> None:
        html = '<math display="block" data-latex="\\sum_i x_i">\\sum_i x_i</math>'
        md = _convert(html)
        assert "$$\\sum_i x_i$$" in md

    def test_annotation_display_block(self) -> None:
        html = (
            '<math display="block"><semantics>'
            '<annotation encoding="application/x-tex">E = mc^2</annotation>'
            "</semantics></math>"
        )
        md = _convert(html)
        assert "$$E = mc^2$$" in md


class TestLanguageAllowlist:
    def test_bogus_language_token_rejected(self) -> None:
        # "CodeBlock-code" matches the (\w+)-code pattern but "codeblock"
        # is not a real language
        html = '<pre><code class="CodeBlock-code">const x = 1;</code></pre>'
        md = _convert(html)
        assert "```codeblock" not in md
        assert "const x = 1;" in md

    def test_real_language_token_accepted(self) -> None:
        html = '<pre><code class="python-code">print(1)</code></pre>'
        md = _convert(html)
        assert "```python" in md
