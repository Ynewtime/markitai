"""Tests for content pattern removal (defuddle content-patterns port)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.dom import parse_html
from markitai.webextract.removals.content_patterns import (
    remove_content_patterns,
    remove_eyebrow_label,
)

# A prose paragraph long enough to anchor the content boundary (≥ 7 words,
# sentence punctuation) — pre-content heuristics need this to activate.
_PROSE = (
    "<p>This is the actual article content, with plenty of words to "
    "qualify as prose for boundary detection.</p>"
)

# ~340 words of prose: enough for the "substantial content" guards
# (boilerplate truncation needs 200+ chars; trailing-section removal
# needs 300+ words; related-section removal needs 500+ chars).
_LONG_PROSE = "".join(
    f"<p>Paragraph {i} carries substantial article prose so heuristics that "
    "require real content before them can activate during testing. It keeps "
    "going with more words to make the document realistically long.</p>"
    for i in range(12)
)


def _make_root(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div")
    assert root is not None
    return root


class TestReadTimeRemoval:
    def test_removes_read_time_metadata(self):
        soup = parse_html(f"<div><p>3 min read</p>{_PROSE}</div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "min read" not in root.get_text()

    def test_removes_minutes_read_variant(self):
        soup = parse_html(f"<div><span>5 minutes read</span>{_PROSE}</div>")
        root = soup.find("div")
        assert root is not None
        remove_content_patterns(root)
        assert "minutes read" not in root.get_text()

    def test_removes_read_time_with_date_anywhere(self):
        soup = parse_html(f"<div>{_LONG_PROSE}<p>Mar 4th 2026 | 3 min read</p></div>")
        root = soup.find("div")
        assert root is not None
        remove_content_patterns(root)
        assert "min read" not in root.get_text()

    def test_keeps_non_matching_text(self):
        soup = parse_html("<div><p>Read this article about Python.</p></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed == 0
        assert "Read this article" in root.get_text()


class TestBoilerplateRemoval:
    def test_removes_originally_published(self):
        soup = parse_html(
            f"<div>{_LONG_PROSE}<p>Originally published in The Atlantic.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "Originally published" not in root.get_text()

    def test_removes_article_appeared(self):
        soup = parse_html(
            f"<div>{_LONG_PROSE}<p>This article appeared in Nature.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1

    def test_truncates_trailing_siblings_after_boilerplate(self):
        soup = parse_html(
            f"<div>{_LONG_PROSE}"
            "<p>Originally published in The Atlantic.</p>"
            "<p>Share on social media</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        remove_content_patterns(root)
        assert "Share on social" not in root.get_text()

    def test_keeps_normal_sentences(self):
        soup = parse_html("<div><p>This article discusses important topics.</p></div>")
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed == 0

    def test_no_truncation_without_substantial_preceding_content(self):
        """Boilerplate needs 200+ chars of preceding content to truncate."""
        soup = parse_html(
            "<div><p>Short intro.</p>"
            "<p>Originally published in The Atlantic.</p>"
            "<p>Real closing content.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        remove_content_patterns(root)
        assert "Real closing content" in root.get_text()


class TestAuthorBylineRemoval:
    def test_removes_byline_with_date(self):
        soup = parse_html(
            f'<div><p class="byline">By John Smith · March 15, 2026</p>{_PROSE}</div>'
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "John Smith" not in root.get_text()

    def test_removes_standalone_date(self):
        soup = parse_html(f"<div><p>March 15, 2026</p>{_PROSE}</div>")
        root = soup.find("div")
        assert root is not None
        remove_content_patterns(root)
        assert "March 15" not in root.get_text()

    def test_keeps_longer_paragraphs(self):
        """Long paragraphs that happen to contain 'by' should not be removed."""
        soup = parse_html(
            "<div><p>By implementing this approach, we achieved significant "
            "improvements across multiple metrics and validated our hypothesis "
            "through extensive experimentation.</p></div>"
        )
        root = soup.find("div")
        assert root is not None
        removed = remove_content_patterns(root)
        assert removed == 0


class TestHeroHeaderRemoval:
    def test_removes_hero_header_with_time(self) -> None:
        html = f"""<div>
        <header><h1>Article Title</h1><time>March 23, 2026</time></header>
        {_PROSE}
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "actual article content" in text
        assert removed > 0

    def test_preserves_substantial_header(self) -> None:
        words = " ".join(f"word{i}" for i in range(35))
        html = f"""<div>
        <header><h1>Title</h1><p>{words}.</p></header>
        {_PROSE}
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "word0" in text


class TestTrailingThinSectionRemoval:
    def test_removes_trailing_cta(self) -> None:
        html = f"""<div>
        {_LONG_PROSE}
        <div><h3>Subscribe</h3><p>Get updates</p></div>
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "Paragraph 3" in text
        assert "Get updates" not in text
        assert removed > 0

    def test_preserves_substantial_trailing_section(self) -> None:
        words = " ".join(f"word{i}" for i in range(30))
        html = f"""<div>
        {_LONG_PROSE}
        <div><h3>Conclusion</h3><p>{words}</p></div>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        text = root.get_text(strip=True)
        assert "word0" in text

    def test_skips_small_documents(self) -> None:
        """The 300-word guard keeps small docs intact."""
        html = """<div>
        <p>Main article content with enough words to be meaningful here.</p>
        <div><h3>Subscribe</h3><p>Get updates</p></div>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        assert "Get updates" in root.get_text()


class TestBreadcrumbRemoval:
    def test_removes_breadcrumb_list(self) -> None:
        html = f"""<div>
        <ul><li><a href="/">Home</a></li><li><a href="/blog">Blog</a></li>
        <li>Post Title</li></ul>
        <h1>Post Title</h1>
        {_PROSE}
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "Home" not in root.get_text()

    def test_keeps_content_lists(self) -> None:
        html = f"""<div>
        <h1>Post Title</h1>
        {_PROSE}
        <ul><li>First point with a longer prose explanation inside it</li>
        <li>Second point with more prose words</li></ul>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        assert "First point" in root.get_text()

    def test_removes_section_back_link(self) -> None:
        html = f"""<div>
        <div><a href="/blog/">← Back to blog</a></div>
        <h1>Title</h1>
        {_PROSE}
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root, url="https://example.com/blog/2026/my-post")
        assert "Back to blog" not in root.get_text()


class TestNewsletterRemoval:
    def test_removes_newsletter_section(self) -> None:
        html = f"""<div>
        {_LONG_PROSE}
        <div><h3>Never miss the latest updates</h3>
        <p>Sign up for our newsletter and get stories weekly</p></div>
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "newsletter" not in root.get_text().lower()

    def test_removes_newsletter_list(self) -> None:
        html = f"""<div>
        {_PROSE}
        <ul><li><a href="/signup">Sign up for our daily newsletter</a></li></ul>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        assert "newsletter" not in root.get_text().lower()


class TestSocialCounterRemoval:
    def test_removes_trailing_like_counter(self) -> None:
        html = f"<div>{_PROSE}<p>9 Likes</p><p>3 Comments</p></div>"
        root = _make_root(html)
        removed = remove_content_patterns(root)
        assert removed >= 2
        assert "Likes" not in root.get_text()

    def test_keeps_counter_like_prose_mid_content(self) -> None:
        html = f"<div><p>42 Likes</p>{_LONG_PROSE}</div>"
        root = _make_root(html)
        remove_content_patterns(root)
        # Far from the end of content — preserved
        assert "42 Likes" in root.get_text()


class TestRelatedSectionRemoval:
    def test_removes_wrapped_related_section(self) -> None:
        html = f"""<div>
        {_LONG_PROSE}
        <section><h2>Related Posts</h2>
        <p><a href="/a">Another article title</a></p>
        <p><a href="/b">Yet another article</a></p></section>
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "Related Posts" not in root.get_text()

    def test_removes_cta_heading_and_trailing(self) -> None:
        html = f"""<div>
        {_LONG_PROSE}
        <h2>Subscribe</h2>
        <p>Join our community for more</p>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        assert "Join our community" not in root.get_text()

    def test_keeps_early_related_mention(self) -> None:
        html = f"""<div>
        <h2>Read More</h2>
        {_PROSE}
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        # Within the first 500 chars — preserved
        assert "Read More" in root.get_text()

    def test_removes_related_intro_paragraph(self) -> None:
        html = f"""<div>
        {_LONG_PROSE}
        <p>For more on this topic, check out our other coverage</p>
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        assert "For more on" not in root.get_text()


class TestTrailingTagBlockRemoval:
    def test_removes_tag_link_block(self) -> None:
        html = f"""<div>
        {_PROSE}
        <div><a href="/tag/features">Features</a> <a href="/tag/amazon">Amazon</a></div>
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert "Features" not in root.get_text()


class TestTocRemoval:
    def test_removes_anchor_link_toc(self) -> None:
        html = f"""<div>
        <h1>Guide</h1>
        <ul>
        <li><a href="#intro">Introduction</a></li>
        <li><a href="#setup">Setup</a></li>
        <li><a href="#usage">Usage</a></li>
        </ul>
        {_LONG_PROSE}
        </div>"""
        root = _make_root(html)
        removed = remove_content_patterns(root)
        assert removed >= 1
        assert root.select_one('a[href="#intro"]') is None

    def test_keeps_external_link_lists(self) -> None:
        html = f"""<div>
        <h1>Guide</h1>
        <ul>
        <li><a href="/a">First resource</a></li>
        <li><a href="/b">Second resource</a></li>
        <li><a href="/c">Third resource</a></li>
        </ul>
        {_LONG_PROSE}
        </div>"""
        root = _make_root(html)
        remove_content_patterns(root)
        assert "First resource" in root.get_text()


class TestEyebrowRemoval:
    def test_removes_category_label_before_h1(self) -> None:
        html = f"<div><p>Announcements</p><h1>Big Title</h1>{_PROSE}</div>"
        root = _make_root(html)
        removed = remove_eyebrow_label(root)
        assert removed == 1
        assert "Announcements" not in root.get_text()

    def test_keeps_prose_before_h1(self) -> None:
        html = f"<div><p>A real sentence appears here first.</p><h1>Title</h1>{_PROSE}</div>"
        root = _make_root(html)
        removed = remove_eyebrow_label(root)
        assert removed == 0
