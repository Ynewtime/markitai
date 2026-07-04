"""End-to-end tests for defuddle parity gap fixes.

Each test locks in a user-visible extraction improvement verified against
the defuddle fixture corpus: math survival, code-block preservation, and
heading-anchor cleanup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from markitai.webextract import extract_web_content

_HTML_DIR = Path(__file__).parents[1] / "defuddle_fixtures" / "fixtures"


def _extract(stem: str, url: str) -> str:
    path = _HTML_DIR / f"{stem}.html"
    if not path.is_file():
        pytest.skip(f"Fixture {stem} not available")
    html = path.read_text(encoding="utf-8")
    return extract_web_content(html, url).markdown


class TestMathGapFixes:
    def test_mathjax_tex_scripts_preserved(self) -> None:
        md = _extract(
            "math--mathjax-tex-scripts", "https://example.com/math-tex-scripts"
        )
        # Inline math from <script type="math/tex">
        assert "$E_u$" in md
        assert "$\\theta = 0$" in md
        # Display math from mode=display scripts
        assert "$$" in md
        assert "\\sqrt{3}\\gamma" in md

    def test_wikipedia_mathml_preserved(self) -> None:
        md = _extract("math--wikipedia-mathml", "https://example.com/wiki-math")
        # Hidden MathML spans must survive and render as LaTeX, not images
        assert "$ax^{2}+bx+c=0$" in md
        assert "![" not in md  # no math fallback images


class TestCodeBlockGapFixes:
    def test_stripe_code_blocks_preserved(self) -> None:
        md = _extract("codeblocks--stripe", "https://docs.stripe.com/x402")
        # Code inside a "dropdown"-classed tab group must survive
        assert "paymentMiddleware" in md
        assert "curl http://localhost:3000/paid" in md
        # Whitespace-only token spans inside <pre> must be preserved
        assert 'import { paymentMiddleware } from "@x402/hono";' in md
        # No bogus language from the "CodeBlock-code" class
        assert "```codeblock" not in md

    def test_stripe_heading_anchors_unwrapped(self) -> None:
        md = _extract("codeblocks--stripe", "https://docs.stripe.com/x402")
        assert "## Create your endpoint" in md
        assert "[## Create your endpoint]" not in md
