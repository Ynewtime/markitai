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


def test_sanitize_removes_url_encoded_javascript() -> None:
    """URL-encoded javascript: should be caught after decoding."""
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment('<a href="javascript%3Aalert(1)">click</a>')
    assert "javascript" not in sanitized.lower() or "href" not in sanitized


def test_sanitize_removes_vbscript_links() -> None:
    """vbscript: scheme should be removed."""
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment('<a href="vbscript:evil()">click</a>')
    assert "vbscript" not in sanitized.lower() or "href" not in sanitized


def test_sanitize_checks_formaction_attribute() -> None:
    """formaction with javascript: should be removed."""
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment(
        '<div formaction="javascript:evil()"><p>safe</p></div>'
    )
    assert "formaction" not in sanitized
