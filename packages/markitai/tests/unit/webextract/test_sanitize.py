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


def _sanitize_fragment(html: str) -> str:
    """Parse a fragment, sanitize the tree in place, and serialize it back."""
    from bs4 import BeautifulSoup

    from markitai.webextract.sanitize import sanitize_tag_tree

    soup = BeautifulSoup(html, "html.parser")
    sanitize_tag_tree(soup)
    return str(soup)


def test_sanitize_removes_event_handlers_and_javascript_links() -> None:
    sanitized = _sanitize_fragment(
        '<div onclick="evil()"><a href="javascript:evil()">x</a><p>safe</p></div>'
    )

    assert "onclick" not in sanitized
    assert "javascript:" not in sanitized
    assert "safe" in sanitized


def test_sanitize_removes_url_encoded_javascript() -> None:
    """URL-encoded javascript: should be caught after decoding."""
    sanitized = _sanitize_fragment('<a href="javascript%3Aalert(1)">click</a>')
    assert "javascript" not in sanitized.lower() or "href" not in sanitized


def test_sanitize_removes_vbscript_links() -> None:
    """vbscript: scheme should be removed."""
    sanitized = _sanitize_fragment('<a href="vbscript:evil()">click</a>')
    assert "vbscript" not in sanitized.lower() or "href" not in sanitized


def test_sanitize_checks_formaction_attribute() -> None:
    """formaction with javascript: should be removed."""
    sanitized = _sanitize_fragment(
        '<div formaction="javascript:evil()"><p>safe</p></div>'
    )
    assert "formaction" not in sanitized


def test_sanitize_removes_template_and_svg_smil_elements() -> None:
    """Inert template fragments and SMIL animations can mutate the output."""
    html = (
        "<div><p>a</p><template><img src=x onerror=alert(1)></template>"
        '<svg><a href="#ok"><animate attributeName="href" to="javascript:alert(1)"/>'
        '<set attributeName="href" to="javascript:alert(1)"/></a></svg></div>'
    )
    out = _sanitize_fragment(html)
    assert "<template" not in out
    assert "onerror" not in out
    assert "<animate" not in out
    assert "<set" not in out
    assert "<p>a</p>" in out


def test_sanitize_empties_an_unsafe_root() -> None:
    """find_all omits the root, so an unsafe root is neutralized in place."""
    from bs4 import BeautifulSoup, Tag

    from markitai.webextract.sanitize import sanitize_tag_tree

    soup = BeautifulSoup('<template id="t"><p>leak</p></template>', "html.parser")
    root = soup.find("template")
    assert isinstance(root, Tag)
    sanitize_tag_tree(root)
    assert str(root) == "<template></template>"


def test_sanitize_checks_svg_xlink_href() -> None:
    """Inline SVG is emitted raw into the Markdown, so its links must be safe."""
    out = _sanitize_fragment(
        '<svg><a xlink:href="javascript:alert(1)"><text>x</text></a>'
        '<a xlink:href="#ok"><text>y</text></a></svg>'
    )
    assert "javascript" not in out
    assert 'xlink:href="#ok"' in out


def test_sanitize_ignores_whitespace_and_controls_inside_scheme() -> None:
    """Browsers drop tabs/newlines anywhere and C0 controls at the ends."""
    for href in (
        "java&#9;script:alert(1)",
        "java&#10;script:alert(1)",
        "&#1;javascript:alert(1)",
        "\x00 javascript:alert(1)",
        "jav%09ascript:alert(1)",
    ):
        out = _sanitize_fragment(f'<svg><a href="{href}"><text>x</text></a></svg>')
        assert "script" not in out, href


def test_sanitize_checks_srcset_poster_and_data() -> None:
    out = _sanitize_fragment(
        '<video poster="javascript:alert(1)" src="a.mp4"></video>'
        '<img srcset="a.png 1x, javascript:alert(2) 2x" src="a.png">'
        '<img srcset="b.png 1x, b@2x.png 2x">'
        '<svg><image data="javascript:alert(3)"/></svg>'
    )
    assert "javascript" not in out
    assert 'srcset="b.png 1x, b@2x.png 2x"' in out
    assert 'src="a.mp4"' in out
