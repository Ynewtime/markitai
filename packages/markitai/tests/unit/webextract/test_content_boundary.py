"""Tests for content boundary detection (defuddle content-boundary port)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.content_boundary import (
    element_precedes,
    find_content_start,
    is_above_content_start,
)


def _root(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div")
    assert root is not None
    return root


class TestFindContentStart:
    def test_finds_first_prose_paragraph(self) -> None:
        root = _root(
            "<div><span>3 min read</span>"
            "<p>This opening paragraph has enough words to count as prose.</p></div>"
        )
        start = find_content_start(root, "")
        assert start is not None
        assert start.name == "p"
        assert "opening paragraph" in start.get_text()

    def test_anchors_on_title_heading(self) -> None:
        root = _root(
            "<div><p>A promo sentence with plenty of words appears up here.</p>"
            "<h1>The Actual Title</h1>"
            "<p>The body text starts after the title with enough words.</p></div>"
        )
        start = find_content_start(root, "The Actual Title")
        assert start is not None
        assert "body text starts" in start.get_text()

    def test_rejects_bylines_and_dates(self) -> None:
        root = _root(
            "<div><p>By Jane Smith, senior technology reporter here</p>"
            "<p>Published March 15, 2026 in the morning edition today</p>"
            "<p>Real prose finally begins in this paragraph with many words.</p></div>"
        )
        start = find_content_start(root, "")
        assert start is not None
        assert "Real prose" in start.get_text()

    def test_returns_none_without_prose(self) -> None:
        root = _root("<div><span>Nav</span><span>Menu</span></div>")
        assert find_content_start(root, "") is None

    def test_skips_nav_and_header_ancestors(self) -> None:
        root = _root(
            "<div><header><p>A header sentence with plenty of words inside it.</p>"
            "</header>"
            "<p>The real body paragraph carries the actual article prose.</p></div>"
        )
        start = find_content_start(root, "")
        assert start is not None
        assert "real body" in start.get_text()


class TestDocumentOrder:
    def test_sibling_order(self) -> None:
        root = _root(
            "<div><p id='a'>First one here.</p><p id='b'>Second one.</p></div>"
        )
        a = root.find(id="a")
        b = root.find(id="b")
        assert isinstance(a, Tag) and isinstance(b, Tag)
        assert element_precedes(a, b)
        assert not element_precedes(b, a)

    def test_nested_order(self) -> None:
        root = _root(
            "<div><section id='s'><p id='a'>Text.</p></section><p id='b'>After.</p></div>"
        )
        s = root.find(id="s")
        a = root.find(id="a")
        b = root.find(id="b")
        assert isinstance(s, Tag) and isinstance(a, Tag) and isinstance(b, Tag)
        assert element_precedes(a, b)
        # Container precedes its own descendant (FOLLOWING semantics)
        assert element_precedes(s, a)
        assert not element_precedes(a, s)

    def test_self_is_not_above(self) -> None:
        root = _root("<div><p id='a'>Text.</p></div>")
        a = root.find(id="a")
        assert isinstance(a, Tag)
        assert not element_precedes(a, a)
        assert not is_above_content_start(a, a)

    def test_none_boundary_is_false(self) -> None:
        root = _root("<div><p id='a'>Text.</p></div>")
        a = root.find(id="a")
        assert isinstance(a, Tag)
        assert not is_above_content_start(a, None)
