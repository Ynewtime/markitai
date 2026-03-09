from __future__ import annotations


def test_extract_web_content_prefers_article_over_nav_sidebar_and_footer() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <body>
        <aside><p>signup signup signup</p></aside>
        <main>
          <article>
            <h1>Real Title</h1>
            <p>This is the real article body with enough text to win.</p>
            <p>It has paragraphs, low link density, and real content.</p>
          </article>
        </main>
        <footer><a href="/privacy">privacy</a></footer>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "Real Title" in result.markdown
    assert "signup signup signup" not in result.markdown


def test_extract_web_content_retries_without_partial_selectors_when_too_short() -> None:
    from markitai.webextract.pipeline import extract_web_content

    html = """
    <html>
      <body>
        <article class="post content story">
          <p>Short but valid statement that should survive fallback retry.</p>
        </article>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "should survive fallback retry" in result.markdown
    assert result.diagnostics["adaptive_retry_used"] is True
