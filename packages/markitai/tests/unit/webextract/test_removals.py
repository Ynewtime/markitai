"""Tests for webextract removals modules (Phase 1)."""

from __future__ import annotations

from bs4 import BeautifulSoup

from markitai.webextract.dom import parse_html
from markitai.webextract.removals import apply_removals
from markitai.webextract.removals.hidden import remove_hidden_elements
from markitai.webextract.removals.scoring import score_and_remove
from markitai.webextract.removals.selectors import remove_by_selectors
from markitai.webextract.removals.small_images import remove_small_images


def _make_soup(html: str) -> BeautifulSoup:
    return parse_html(html)


# ─── small_images.py ────────────────────────────────────────────────


class TestRemoveSmallImages:
    def test_removes_tracking_pixel(self):
        soup = _make_soup('<div><img src="pixel.gif" width="1" height="1"></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_small_images(root)
        assert removed == 1
        assert root.find("img") is None

    def test_keeps_normal_image(self):
        soup = _make_soup('<div><img src="photo.jpg" width="800" height="600"></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_small_images(root)
        assert removed == 0
        assert root.find("img") is not None

    def test_removes_small_icon_by_inline_style(self):
        soup = _make_soup(
            '<div><img src="icon.png" style="width: 16px; height: 16px"></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_small_images(root)
        assert removed == 1

    def test_keeps_image_without_dimensions(self):
        """Images without known dimensions should be kept (conservative)."""
        soup = _make_soup('<div><img src="unknown.jpg"></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_small_images(root)
        assert removed == 0
        assert root.find("img") is not None

    def test_removes_small_svg(self):
        soup = _make_soup(
            '<div><svg width="10" height="10"><circle r="5"/></svg></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_small_images(root)
        assert removed == 1

    def test_keeps_large_svg(self):
        soup = _make_soup(
            '<div><svg width="200" height="200"><circle r="100"/></svg></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_small_images(root)
        assert removed == 0


# ─── hidden.py ───────────────────────────────────────────────────────


class TestRemoveHiddenElements:
    def test_removes_display_none(self):
        soup = _make_soup(
            '<div><p>visible</p><p style="display: none">hidden</p></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_hidden_elements(root)
        assert removed == 1
        assert "hidden" not in root.get_text()

    def test_removes_visibility_hidden(self):
        soup = _make_soup(
            '<div><p>visible</p><p style="visibility: hidden">hidden</p></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_hidden_elements(root)
        assert removed == 1

    def test_removes_opacity_zero(self):
        soup = _make_soup(
            '<div><p>visible</p><span style="opacity: 0">hidden</span></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_hidden_elements(root)
        assert removed == 1

    def test_removes_hidden_class(self):
        soup = _make_soup('<div><p>visible</p><div class="hidden">hidden</div></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_hidden_elements(root)
        assert removed == 1

    def test_preserves_math_elements(self):
        """Math elements should never be removed even if aria-hidden."""
        soup = _make_soup(
            '<div><span class="katex" aria-hidden="true">math</span></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_hidden_elements(root)
        assert removed == 0
        assert "math" in root.get_text()

    def test_preserves_visible_elements(self):
        soup = _make_soup('<div><p style="display: block">visible</p></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_hidden_elements(root)
        assert removed == 0


# ─── selectors.py ─────────────────────────────────────────────────────


class TestRemoveBySelectors:
    def test_removes_nav_element(self):
        soup = _make_soup("<div><nav>menu</nav><article>content</article></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_by_selectors(root, None)
        assert removed >= 1
        assert root.find("nav") is None
        assert root.find("article") is not None

    def test_removes_footer(self):
        soup = _make_soup("<div><p>content</p><footer>copyright 2026</footer></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_by_selectors(root, None)
        assert removed >= 1
        assert root.find("footer") is None

    def test_removes_sidebar(self):
        soup = _make_soup('<div><p>content</p><div class="sidebar">links</div></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_by_selectors(root, None)
        assert removed >= 1
        assert root.find("div", class_="sidebar") is None

    def test_removes_ad_class(self):
        soup = _make_soup('<div><p>content</p><div class="ad">sponsor</div></div>')
        root = soup.find("div")
        assert root is not None
        removed = remove_by_selectors(root, None)
        assert removed >= 1

    def test_partial_selector_removes_newsletter_signup(self):
        soup = _make_soup(
            '<div><p>content</p><div class="newsletter-signup">Subscribe!</div></div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_by_selectors(root, None, use_partial=True)
        assert removed >= 1
        assert root.find("div", class_="newsletter-signup") is None

    def test_partial_selectors_disabled(self):
        """When partial selectors disabled, only exact selectors apply."""
        soup = _make_soup(
            '<div><p>content</p><div class="newsletter-signup">Subscribe!</div></div>'
        )
        root = soup.find("div")
        assert root is not None
        remove_by_selectors(root, None, use_partial=False)
        # newsletter-signup is only a partial match, should not be removed
        assert root.find("div", class_="newsletter-signup") is not None

    def test_protects_main_content(self):
        """Elements containing main_content should not be removed."""
        soup = _make_soup('<div><nav><article id="main">content</article></nav></div>')
        root = soup.find("div")
        assert root is not None
        main = soup.find("article")
        remove_by_selectors(root, main)
        # nav should NOT be removed because it contains main content
        assert root.find("article") is not None

    def test_protects_code_blocks(self):
        """Elements inside <pre>/<code> should not be removed."""
        soup = _make_soup(
            '<div><pre><code><span class="comment">// test</span></code></pre></div>'
        )
        root = soup.find("div")
        assert root is not None
        remove_by_selectors(root, None)
        assert root.find("span", class_="comment") is not None

    def test_removes_form_elements(self):
        soup = _make_soup(
            "<div><p>content</p><form><input><button>go</button></form></div>"
        )
        root = soup.find("div")
        assert root is not None
        remove_by_selectors(root, None)
        assert root.find("form") is None

    def test_removes_hidden_attribute(self):
        soup = _make_soup("<div><p>content</p><div hidden>hidden</div></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_by_selectors(root, None)
        assert removed >= 1


# ─── scoring.py ──────────────────────────────────────────────────────


class TestScoreAndRemove:
    def test_removes_navigation_block(self):
        """Block with navigation indicators and high link density."""
        soup = _make_soup(
            """<div>
            <p>This is the real article content with enough words to be meaningful.
            It has multiple sentences and proper prose structure.</p>
            <div class="related-posts">
                <h3>Related Articles</h3>
                <ul>
                    <li><a href="/1">Link one</a></li>
                    <li><a href="/2">Link two</a></li>
                    <li><a href="/3">Link three</a></li>
                    <li><a href="/4">Link four</a></li>
                </ul>
            </div>
            </div>"""
        )
        root = soup.find("div")
        assert root is not None
        removed = score_and_remove(root)
        assert removed >= 1

    def test_keeps_prose_content(self):
        """Block with substantial prose should be kept."""
        soup = _make_soup(
            """<div>
            <div class="section">
                <p>This is a substantial paragraph with many words. It contains
                commas, periods, and other punctuation marks that indicate real
                prose content rather than navigation or metadata.</p>
                <p>Another paragraph continues the article with additional detail,
                explanation, and context for the reader.</p>
            </div>
            </div>"""
        )
        root = soup.find("div")
        assert root is not None
        removed = score_and_remove(root)
        assert removed == 0

    def test_removes_high_link_density_block(self):
        soup = _make_soup(
            """<div>
            <p>Main content here with real words.</p>
            <div class="links">
                <a href="/a">link</a> <a href="/b">link</a>
                <a href="/c">link</a> <a href="/d">link</a>
                <a href="/e">link</a> <a href="/f">link</a>
            </div>
            </div>"""
        )
        root = soup.find("div")
        assert root is not None
        removed = score_and_remove(root)
        assert removed >= 1

    def test_keeps_code_blocks(self):
        """Blocks containing <pre> should be protected."""
        soup = _make_soup(
            """<div>
            <div class="code-example">
                <pre><code>function hello() { return "world"; }</code></pre>
            </div>
            </div>"""
        )
        root = soup.find("div")
        assert root is not None
        removed = score_and_remove(root)
        assert removed == 0
        assert root.find("pre") is not None


# ─── apply_removals (integration) ────────────────────────────────────


class TestApplyRemovals:
    def test_full_pipeline_removes_noise(self):
        """Integration test: multiple noise types in one document."""
        html = """
        <div>
            <nav><a href="/">Home</a> <a href="/about">About</a></nav>
            <article>
                <p>This is the main article content with enough substance.</p>
                <p>A second paragraph with more detail and explanation.</p>
            </article>
            <div class="sidebar">
                <h3>Popular Posts</h3>
                <ul>
                    <li><a href="/1">Post 1</a></li>
                    <li><a href="/2">Post 2</a></li>
                </ul>
            </div>
            <footer>Copyright 2026</footer>
            <img src="pixel.gif" width="1" height="1">
            <div style="display: none">tracking code</div>
        </div>
        """
        soup = _make_soup(html)
        root = soup.find("div")
        assert root is not None
        stats = apply_removals(root)
        assert stats["small_images"] >= 1
        assert stats["hidden"] >= 1
        assert stats["selectors"] >= 1
        # Article content should survive
        assert root.find("article") is not None
        assert "main article content" in root.get_text()

    def test_disable_partial_selectors(self):
        html = '<div><p>content</p><div class="newsletter-signup">Sub</div></div>'
        soup = _make_soup(html)
        root = soup.find("div")
        assert root is not None
        apply_removals(root, use_partial_selectors=False, use_scoring=False)
        # Partial selector not applied and scoring disabled, so newsletter-signup stays
        assert root.find("div", class_="newsletter-signup") is not None

    def test_disable_hidden_removal(self):
        html = '<div><p>visible</p><p style="display:none">hidden</p></div>'
        soup = _make_soup(html)
        root = soup.find("div")
        assert root is not None
        stats = apply_removals(root, use_hidden_removal=False)
        assert "hidden" not in stats  # hidden stage not run
        assert "hidden" in root.get_text()


def test_joined_selector_removes_same_elements_as_individual() -> None:
    """Joining selectors into one query must produce identical results."""
    html = """
    <div>
        <nav class="navigation">nav</nav>
        <div class="ad">ad</div>
        <footer>footer</footer>
        <article>
            <p>Main content here with enough words to be meaningful.</p>
        </article>
        <aside class="sidebar">sidebar</aside>
    </div>
    """
    from bs4 import BeautifulSoup

    from markitai.webextract.removals.selectors import remove_by_selectors

    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div")
    assert root is not None

    remove_by_selectors(root, main_content=None, use_partial=False)
    remaining_text = root.get_text(strip=True)
    assert "Main content" in remaining_text
    assert "nav" not in remaining_text.lower().split()
    assert "sidebar" not in remaining_text
