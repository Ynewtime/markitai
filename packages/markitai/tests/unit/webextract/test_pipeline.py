from __future__ import annotations


def test_extract_web_content_returns_article_markdown() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <head><title>Example Post</title></head>
      <body>
        <nav>menu</nav>
        <article><h1>Example Post</h1><p>Hello extraction.</p></article>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "Hello extraction." in result.markdown
    assert "<article" in result.clean_html
    assert result.metadata.title == "Example Post"
    assert result.word_count >= 2


def test_extract_web_content_preserves_jsonld_metadata_inside_article() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <body>
        <article>
          <script type="application/ld+json">
            {
              "@type": "Article",
              "headline": "Structured Post",
              "author": {"name": "Jane Doe"},
              "datePublished": "2026-03-01"
            }
          </script>
          <h1>Structured Post</h1>
          <p>Hello extraction.</p>
        </article>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert result.metadata.title == "Structured Post"
    assert result.metadata.author == "Jane Doe"
    assert result.metadata.published == "2026-03-01"
