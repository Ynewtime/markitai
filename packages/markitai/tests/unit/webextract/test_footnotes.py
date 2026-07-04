"""Unit tests for footnote standardization (elements/footnotes.py).

Covers the canonical DOM rewrite (``standardize_footnotes``) and the
Markdown emission (``[^N]`` references and ``[^N]: ...`` definitions).
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.elements.footnotes import standardize_footnotes
from markitai.webextract.pipeline import _html_fragment_to_markdown


def _standardize(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div", id="content")
    assert isinstance(root, Tag)
    standardize_footnotes(root)
    return root


def _to_markdown(root: Tag) -> str:
    return _html_fragment_to_markdown(str(root))


WIKIPEDIA_HTML = """
<div id="content">
<p>A well-known fact.<sup class="reference" id="cite_ref-alpha_1-0">
<a href="#cite_note-alpha-1">[1]</a></sup>
Another fact.<sup class="reference" id="cite_ref-beta_2-0">
<a href="#cite_note-beta-2">[2]</a></sup></p>
<ol class="references">
<li id="cite_note-alpha-1"><span class="mw-cite-backlink">
<a href="#cite_ref-alpha_1-0">^</a></span>
<span class="reference-text">First reference text.</span></li>
<li id="cite_note-beta-2"><span class="mw-cite-backlink">
<a href="#cite_ref-beta_2-0">^</a></span>
<span class="reference-text">Second reference text.</span></li>
</ol>
</div>
"""


class TestWikipediaStyle:
    def test_references_are_standardized(self) -> None:
        root = _standardize(WIKIPEDIA_HTML)
        refs = root.select('sup[id^="fnref:"] > a[href^="#fn:"]')
        assert [a.get_text() for a in refs] == ["1", "2"]
        items = root.select("#footnotes ol > li")
        assert [li.get("id") for li in items] == ["fn:1", "fn:2"]
        assert "First reference text." in items[0].get_text()

    def test_markdown_emission(self) -> None:
        md = _to_markdown(_standardize(WIKIPEDIA_HTML))
        assert "fact.[^1]" in md
        assert "fact.[^2]" in md
        assert "[^1]: First reference text." in md
        assert "[^2]: Second reference text." in md


SECTION_FOOTNOTES_HTML = """
<div id="content">
<p>Body text.<sup id="fnref:5"><a href="#fn:5">5</a></sup>
More body.<sup id="fnref:9"><a href="#fn:9">9</a></sup></p>
<section class="footnotes"><ol>
<li id="fn:5"><p>Fifth note. <a href="#fnref:5" class="footnote-backref">↩</a></p></li>
<li id="fn:9"><p>Ninth note. <a href="#fnref:9" class="footnote-backref">↩</a></p></li>
</ol></section>
</div>
"""


class TestSectionFootnotesStyle:
    def test_notes_are_renumbered_sequentially(self) -> None:
        root = _standardize(SECTION_FOOTNOTES_HTML)
        items = root.select("#footnotes ol > li")
        # Original numbers 5 and 9 are renumbered from 1 in collection order
        assert [li.get("id") for li in items] == ["fn:1", "fn:2"]
        refs = root.select('sup[id^="fnref:"] > a')
        assert [a.get_text() for a in refs] == ["1", "2"]

    def test_markdown_renumbered(self) -> None:
        md = _to_markdown(_standardize(SECTION_FOOTNOTES_HTML))
        assert "[^1]" in md and "[^2]" in md
        assert "[^5]" not in md and "[^9]" not in md
        assert "[^1]: Fifth note." in md
        assert "[^2]: Ninth note." in md


MULTI_REF_HTML = """
<div id="content">
<p>First mention <a href="#note1">[1]</a> and later a second mention
<a href="#note1">[1]</a> of the same note. Another <a href="#note2">[2]</a>.</p>
<h4>Notes</h4>
<ol>
<li><a id="note1"></a>Shared note body.</li>
<li><a id="note2"></a>Second note body.</li>
</ol>
</div>
"""


class TestMultipleRefsToOneNote:
    def test_duplicate_refs_get_distinct_ids(self) -> None:
        root = _standardize(MULTI_REF_HTML)
        sups = root.select('sup[id^="fnref:1"]')
        assert [s.get("id") for s in sups] == ["fnref:1", "fnref:1-2"]
        # Definition carries one backlink per reference
        item = root.select_one("#footnotes li[id='fn:1']")
        assert item is not None
        assert len(item.select("a.footnote-backref")) == 2

    def test_markdown_repeats_reference(self) -> None:
        md = _to_markdown(_standardize(MULTI_REF_HTML))
        assert md.count("[^1]") == 3  # two refs + one definition
        assert md.count("[^1]:") == 1
        assert "[^2]: Second note body." in md


class TestBacklinkStripping:
    def test_arrows_and_backrefs_are_stripped_from_definitions(self) -> None:
        html = """
        <div id="content">
        <p>Text.<sup id="fnref:1"><a href="#fn:1">1</a></sup></p>
        <section class="footnotes"><ol>
        <li id="fn:1"><p>Note body. <a href="#fnref:1">↩︎</a></p></li>
        </ol></section>
        </div>
        """
        root = _standardize(html)
        item = root.select_one("#footnotes li")
        assert item is not None
        # The original arrow backlink is stripped; only the generated
        # backref (with class footnote-backref) remains.
        arrows = [
            a
            for a in item.select("a")
            if "footnote-backref" not in (a.get("class") or [])
        ]
        assert arrows == []

    def test_markdown_has_no_arrows(self) -> None:
        html = """
        <div id="content">
        <p>Text.<sup id="fnref:1"><a href="#fn:1">1</a></sup></p>
        <section class="footnotes"><ol>
        <li id="fn:1"><p>Note body. <a href="#fnref:1">↩</a></p></li>
        </ol></section>
        </div>
        """
        md = _to_markdown(_standardize(html))
        assert "↩" not in md
        assert "#fnref" not in md
        assert "[^1]: Note body." in md


class TestMultiParagraphDefinition:
    def test_paragraphs_survive_as_blocks(self) -> None:
        html = """
        <div id="content">
        <p>Claim.<sup id="fnref:1"><a href="#fn:1">1</a></sup></p>
        <section class="footnotes"><ol>
        <li id="fn:1"><p>First paragraph.</p><p>Second paragraph.</p></li>
        </ol></section>
        </div>
        """
        root = _standardize(html)
        item = root.select_one("#footnotes li")
        assert item is not None
        assert [p.get_text().strip().rstrip("↩").strip() for p in item.select("p")][
            :2
        ] == ["First paragraph.", "Second paragraph."]

        md = _to_markdown(root)
        assert "[^1]: First paragraph.\n\nSecond paragraph." in md


class TestNoFootnotesPassThrough:
    def test_document_is_untouched(self) -> None:
        html = """
        <div id="content">
        <h2>Title</h2>
        <p>Paragraph with a <a href="https://example.com">link</a>.</p>
        <ul><li>item one</li><li>item two</li></ul>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        root = soup.find("div", id="content")
        assert isinstance(root, Tag)
        before = str(root)
        standardize_footnotes(root)
        assert str(root) == before

    def test_plain_ordered_list_is_not_a_footnote_list(self) -> None:
        html = """
        <div id="content">
        <p>Intro.</p>
        <ol><li>step one</li><li>step two</li></ol>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        root = soup.find("div", id="content")
        assert isinstance(root, Tag)
        standardize_footnotes(root)
        md = _html_fragment_to_markdown(str(root))
        assert "[^" not in md
        assert "1. step one" in md
