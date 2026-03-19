"""Tests for review-discovered bugs (P3-D review fixes)."""

from __future__ import annotations

from markitai.webextract.dom import parse_html
from markitai.webextract.removals import apply_removals


class TestAlertCalloutConflict:
    """ISSUE-5/6: .alert and aside in EXACT_SELECTORS kills callouts."""

    def test_bootstrap_alert_survives_removals_and_becomes_callout(self):
        """Bootstrap .alert should be converted to blockquote, not removed."""
        soup = parse_html(
            '<div><div class="alert alert-warning">'
            "<p>Warning: this is important.</p></div>"
            "<p>Main content here.</p></div>"
        )
        root = soup.find("div")
        apply_removals(root)
        # After removals, the alert content should still exist
        assert "important" in root.get_text()

    def test_aside_callout_survives_removals(self):
        """aside with callout class should not be removed by selectors."""
        soup = parse_html(
            '<div><aside class="callout callout-info">'
            "<p>Helpful note here.</p></aside>"
            "<p>Main content.</p></div>"
        )
        root = soup.find("div")
        apply_removals(root)
        assert "Helpful note" in root.get_text()


class TestMathProtection:
    """ISSUE-8: [aria-hidden] selector bypasses math protection."""

    def test_math_katex_survives_full_removal_pipeline(self):
        """KaTeX elements with aria-hidden should survive all removals."""
        soup = parse_html(
            "<div><p>The formula is "
            '<span class="katex" aria-hidden="true">x^2 + y^2</span></p></div>'
        )
        root = soup.find("div")
        apply_removals(root)
        assert "x^2" in root.get_text()

    def test_mathjax_survives_full_removal_pipeline(self):
        soup = parse_html(
            "<div><p>Equation: "
            '<span class="MathJax" aria-hidden="true">E=mc^2</span></p></div>'
        )
        root = soup.find("div")
        apply_removals(root)
        assert "E=mc" in root.get_text()

    def test_math_element_survives(self):
        soup = parse_html(
            "<div><p>See <math aria-hidden='true'><mi>x</mi></math></p></div>"
        )
        root = soup.find("div")
        apply_removals(root)
        assert root.find("math") is not None


class TestBodyFallbackFreshParse:
    """BUG-4: body fallback should use fresh HTML, not mutated soup."""

    def test_body_fallback_preserves_all_content(self):
        from markitai.webextract.pipeline import extract_web_content

        # Page where initial root is very short, triggering retry + body fallback.
        # The body has content that should NOT be stripped by Level 1 mutations.
        html = """
        <html><body>
            <div id="wrapper">
                <p>Short.</p>
            </div>
            <section>
                <p>This is extra content that lives outside the main wrapper
                and should be captured by the body fallback path with enough
                words to pass the sparse threshold.</p>
            </section>
        </body></html>
        """
        result = extract_web_content(html, "https://example.com/page")
        assert "extra content" in result.markdown


class TestSelectorIdentity:
    """ISSUE-13: el in to_remove should use identity, not equality."""

    def test_identical_elements_both_removed(self):
        """Two different elements with same content should both be removed."""
        soup = parse_html("<div><nav>links</nav><p>content</p><nav>links</nav></div>")
        root = soup.find("div")
        apply_removals(root)
        # Both nav elements should be removed
        assert root.find("nav") is None


class TestBylineRegex:
    """BUG-3: IGNORECASE makes [A-Z] meaningless in byline regex."""

    def test_byline_requires_capitalized_name(self):
        from markitai.webextract.removals.content_patterns import (
            remove_content_patterns,
        )

        # Lowercase "by someone" with date should NOT match
        soup = parse_html("<div><p>by someone on Jan 15</p><p>Real content.</p></div>")
        root = soup.find("div")
        remove_content_patterns(root)
        # Should not be removed (lowercase name)
        assert "by someone" in root.get_text()
