from __future__ import annotations


def test_normalize_footnotes_creates_canonical_ordered_list() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.elements.footnotes import normalize_footnotes

    soup = parse_html(
        """
        <article>
          <p>Text<a href="#fn1" id="fnref1">1</a></p>
          <section class="footnotes">
            <p id="fn1">1. Note body <a href="#fnref1">↩</a></p>
          </section>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    normalize_footnotes(article)

    html = str(article)
    assert "<ol" in html
    assert "Note body" in html
