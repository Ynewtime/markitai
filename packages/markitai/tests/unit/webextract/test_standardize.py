from __future__ import annotations


def test_standardize_content_dedupes_title_heading_and_removes_comments() -> None:
    from markitai.webextract.dom import parse_html
    from markitai.webextract.standardize import standardize_content

    soup = parse_html(
        """
        <article>
          <!-- remove me -->
          <h1>How Arrays Work</h1>
          <h2>How Arrays Work</h2>
          <div><div><p>Body text.</p></div></div>
        </article>
        """
    )

    article = soup.article
    assert article is not None
    standardize_content(
        article, title="How Arrays Work", base_url="https://example.com"
    )

    html = str(article)
    assert "<!--" not in html
    assert html.count("How Arrays Work") == 1
    assert "Body text." in html


def test_standardize_content_resolves_relative_urls_and_unwraps_javascript_links() -> (
    None
):
    from markitai.webextract.dom import parse_html
    from markitai.webextract.standardize import standardize_content

    soup = parse_html(
        '<article><a href="/docs">Docs</a><a href="javascript:void(0)">Click</a></article>'
    )

    article = soup.article
    assert article is not None
    standardize_content(article, title=None, base_url="https://example.com/base")

    html = str(article)
    assert 'href="https://example.com/docs"' in html
    assert "javascript:" not in html
