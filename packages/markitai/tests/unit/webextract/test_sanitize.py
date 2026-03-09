from __future__ import annotations


def test_sanitize_html_removes_event_handlers_and_javascript_links() -> None:
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment(
        '<div onclick="evil()"><a href="javascript:evil()">x</a><p>safe</p></div>'
    )

    assert "onclick" not in sanitized
    assert "javascript:" not in sanitized
    assert "safe" in sanitized
