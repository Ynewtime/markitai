"""Tests for enhanced scoring (Phase 2)."""

from __future__ import annotations

from markitai.webextract.dom import parse_html
from markitai.webextract.scoring import score_candidate


class TestEnhancedScoring:
    def test_link_density_multiplicative(self):
        """Link density should scale score multiplicatively, not linearly."""
        # High link density page (50% links)
        html = '<div><a href="/1">link text</a> normal text</div>'
        soup = parse_html(html)
        el = soup.find("div")
        score = score_candidate(el)
        # Score should be reduced by link density factor
        assert score < 4  # 4 words without penalty would be 4.0

    def test_comma_count_bonus(self):
        """Commas should boost score (prose indicator)."""
        html = "<div><p>First, second, third, fourth, fifth item here.</p></div>"
        soup = parse_html(html)
        el = soup.find("div")
        score = score_candidate(el)
        # 7 words + 4 commas + 1 paragraph bonus
        assert score > 7  # more than just word count

    def test_navigation_heading_penalty(self):
        """Navigation headings should be penalized."""
        html = """<div>
            <h3>Related Articles</h3>
            <a href="/1">Link 1</a>
            <a href="/2">Link 2</a>
        </div>"""
        soup = parse_html(html)
        el = soup.find("div")
        score = score_candidate(el)
        # Navigation heading + high link density should produce low score
        assert score < 10
