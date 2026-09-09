from __future__ import annotations


def test_normalize_code_blocks_wraps_preformatted_code_and_keeps_language() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.code import normalize_code_blocks

    soup = parse_html(
        """
        <article>
          <code class="language-python" style="white-space: pre;">
          print("hello")
          </code>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_code_blocks(article)

    html = str(article)
    assert "<pre>" in html
    assert "language-python" in html
    assert 'print("hello")' in html


def test_numeric_gutters_go_but_numeric_literals_stay() -> None:
    """Only rows that announce themselves as line rows lose a leading number."""
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.code import _strip_inline_numeric_gutters

    soup = parse_html(
        "<pre><code>"
        '<span style="display:flex"><span>1</span><span>p = 61</span></span>\n'
        '<span><span class="mi">42</span><span> + x</span></span>'
        "</code></pre>"
    )
    pre = soup.pre
    assert pre is not None
    _strip_inline_numeric_gutters(pre)
    text = pre.get_text()
    assert "1p" not in text and "p = 61" in text
    assert "42 + x" in text
