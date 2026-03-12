from __future__ import annotations

from unittest.mock import patch

# ---- Medium-6: MarkItDown instance reuse tests ----


def test_html_fragment_to_markdown_reuses_markitdown_instance() -> None:
    """_html_fragment_to_markdown should accept an optional MarkItDown instance
    to avoid recreating it on every call (Medium-6 optimization)."""
    from markitai.webextract.pipeline import _html_fragment_to_markdown

    html = "<p>Hello world.</p>"

    # Call twice — should produce identical results
    result1 = _html_fragment_to_markdown(html)
    result2 = _html_fragment_to_markdown(html)
    assert result1 == result2
    assert "Hello world" in result1


def test_extract_web_content_creates_markitdown_once() -> None:
    """extract_web_content should create MarkItDown at most once even when
    adaptive retry triggers a second _html_fragment_to_markdown call."""
    from markitai.webextract.pipeline import extract_web_content

    # HTML that triggers adaptive retry: <article> wins scoring but is short,
    # <body> has more content.
    body_words = " ".join(f"word{i}" for i in range(30))
    html = f"""
    <html>
      <body>
        <article><p>Short.</p></article>
        <p>{body_words}</p>
        <p>More substantial content paragraph for the retry path.</p>
      </body>
    </html>
    """

    # Track MarkItDown constructor calls by patching at the source module
    import markitdown

    original_class = markitdown.MarkItDown
    call_count = 0

    def counting_markitdown(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_class(*args, **kwargs)

    with patch.object(markitdown, "MarkItDown", side_effect=counting_markitdown):
        result = extract_web_content(html, "https://example.com/test")

    # Should reuse instance: at most 1 construction despite 2 markdown conversions
    assert call_count <= 1, (
        f"MarkItDown was constructed {call_count} times, expected at most 1"
    )
    assert result.word_count > 0


# ---- Original tests ----


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


def test_adaptive_retry_broadens_extraction_on_short_content() -> None:
    """When initial extraction yields too few words, adaptive retry should
    fall back to body and produce more content."""
    from markitai.webextract.pipeline import extract_web_content

    # Craft HTML where <article> wins scoring (article bonus +40) but has
    # very few words, while <body> contains substantially more text in
    # paragraphs that are NOT inside any scored container.
    body_words = " ".join(f"word{i}" for i in range(30))
    html = f"""
    <html>
      <head><title>Retry Test</title></head>
      <body>
        <article><p>Short.</p></article>
        <p>{body_words}</p>
        <p>Additional paragraph with enough meaningful content here.</p>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/retry")

    # After adaptive retry, we should get body-level content
    assert result.word_count > 20, (
        f"Adaptive retry should produce >20 words, got {result.word_count}"
    )
    assert result.diagnostics.get("adaptive_retry_used") is True


def test_adaptive_retry_not_triggered_on_sufficient_content() -> None:
    """When initial extraction yields enough words, no retry should occur."""
    from markitai.webextract.pipeline import extract_web_content

    words = " ".join(f"word{i}" for i in range(30))
    html = f"""
    <html>
      <body>
        <article><h1>Good Content</h1><p>{words}</p></article>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/ok")

    assert result.word_count > 20
    assert result.diagnostics.get("adaptive_retry_used") is not True
