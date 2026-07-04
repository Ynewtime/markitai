"""Tests for math standardization (elements/math.py)."""

from __future__ import annotations

from bs4 import Tag

from markitai.webextract.dom import parse_html
from markitai.webextract.elements.math import normalize_math


def _root(html: str) -> Tag:
    soup = parse_html(f"<div id='root'>{html}</div>")
    root = soup.find("div", id="root")
    assert isinstance(root, Tag)
    return root


class TestTexScripts:
    def test_inline_tex_script_becomes_math(self):
        root = _root('<p>where <script type="math/tex">E = mc^2</script>.</p>')
        normalize_math(root)
        math = root.find("math")
        assert isinstance(math, Tag)
        assert math.get("data-latex") == "E = mc^2"
        assert math.get("display") == "inline"
        assert root.find("script") is None

    def test_display_tex_script_becomes_block_math(self):
        root = _root('<script type="math/tex; mode=display">\\int_0^1 f(x) dx</script>')
        normalize_math(root)
        math = root.find("math")
        assert isinstance(math, Tag)
        assert math.get("display") == "block"
        assert math.get("data-latex") == "\\int_0^1 f(x) dx"

    def test_empty_tex_script_removed(self):
        root = _root('<p><script type="math/tex"> </script>done</p>')
        normalize_math(root)
        assert root.find("script") is None
        assert root.find("math") is None

    def test_non_math_script_untouched(self):
        root = _root('<script type="application/json">{"a": 1}</script>')
        normalize_math(root)
        assert root.find("script") is not None

    def test_mathjax_v2_render_siblings_removed(self):
        root = _root(
            '<p><span class="MathJax_Preview">[math]</span>'
            '<span class="MathJax">rendered junk</span>'
            '<script type="math/tex">a+b</script></p>'
        )
        normalize_math(root)
        assert root.find(class_="MathJax_Preview") is None
        assert root.find(class_="MathJax") is None
        math = root.find("math")
        assert isinstance(math, Tag)
        assert math.get("data-latex") == "a+b"


class TestMediaWikiMath:
    def test_collapses_wrapper_to_math(self):
        root = _root(
            '<span class="mwe-math-element">'
            '<span class="mwe-math-mathml-inline" style="display: none;">'
            '<math alttext="x^2"><mi>x</mi></math></span>'
            '<img class="mwe-math-fallback-image-inline" src="x.svg" alt="x^2">'
            "</span>"
        )
        normalize_math(root)
        math = root.find("math")
        assert isinstance(math, Tag)
        assert math.get("data-latex") == "x^2"
        assert root.find("img") is None
        assert root.find(class_="mwe-math-element") is None

    def test_image_only_fallback_uses_alt(self):
        root = _root(
            '<span class="mwe-math-element">'
            '<img class="mwe-math-fallback-image-display" src="x.svg" alt="a+b">'
            "</span>"
        )
        normalize_math(root)
        math = root.find("math")
        assert isinstance(math, Tag)
        assert math.get("data-latex") == "a+b"
        assert math.get("display") == "block"


class TestMjxContainer:
    def test_hoists_assistive_mathml(self):
        root = _root(
            '<mjx-container display="true">'
            "<mjx-math><mjx-mi></mjx-mi></mjx-math>"
            '<mjx-assistive-mml><math alttext="y=x">'
            "<mi>y</mi><mo>=</mo><mi>x</mi></math></mjx-assistive-mml>"
            "</mjx-container>"
        )
        normalize_math(root)
        assert root.find("mjx-container") is None
        math = root.find("math")
        assert isinstance(math, Tag)
        assert math.get("display") == "block"
        assert math.get("data-latex") == "y=x"

    def test_container_without_math_untouched(self):
        root = _root("<mjx-container><mjx-math></mjx-math></mjx-container>")
        normalize_math(root)
        assert root.find("mjx-container") is not None


class TestPipelineIntegration:
    def test_tex_scripts_survive_extraction_to_markdown(self):
        from markitai.webextract import extract_web_content

        body = (
            "<p>Filler text to satisfy word thresholds. " + "word " * 60 + "</p>"
            '<p>The identity <script type="math/tex">e^{i\\pi} + 1 = 0</script> '
            "is famous.</p>"
            '<script type="math/tex; mode=display">B = \\gamma f</script>'
        )
        html = f"<html><head><title>T</title></head><body><article>{body}</article></body></html>"
        result = extract_web_content(html, "https://example.com/math")
        assert "$e^{i\\pi} + 1 = 0$" in result.markdown
        assert "$$B = \\gamma f$$" in result.markdown
