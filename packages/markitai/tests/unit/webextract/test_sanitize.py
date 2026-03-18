from __future__ import annotations


def test_sanitize_tag_tree_handles_none_attrs() -> None:
    """Tag with attrs=None should not crash sanitize_tag_tree."""
    from bs4 import BeautifulSoup, Tag

    from markitai.webextract.sanitize import sanitize_tag_tree

    soup = BeautifulSoup("<div><p>hello</p></div>", "html.parser")
    div = soup.find("div")
    assert isinstance(div, Tag)

    # Simulate the edge case: BeautifulSoup can produce tags with attrs=None
    # in certain parsing contexts (e.g., complex X/Twitter HTML)
    p = div.find("p")
    assert isinstance(p, Tag)
    p.attrs = None  # type: ignore[assignment]

    # Should not raise TypeError: 'NoneType' object is not iterable
    sanitize_tag_tree(div)
    assert "hello" in div.get_text()


def test_sanitize_html_removes_event_handlers_and_javascript_links() -> None:
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment(
        '<div onclick="evil()"><a href="javascript:evil()">x</a><p>safe</p></div>'
    )

    assert "onclick" not in sanitized
    assert "javascript:" not in sanitized
    assert "safe" in sanitized
