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
