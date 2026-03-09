from __future__ import annotations


def test_schema_fallback_uses_smallest_matching_element_when_schema_text_is_longer() -> (
    None
):
    from markitai.webextract.pipeline import extract_web_content

    schema_text = (
        "This is the target post content with enough words to beat the short extracted "
        "result and trigger schema fallback."
    )

    html = f"""
    <html>
      <head>
        <script type="application/ld+json">
          {{"@type":"SocialMediaPosting","text":"{schema_text}"}}
        </script>
      </head>
      <body>
        <div id="feed">
          <div class="post"><p>Other post.</p></div>
          <div class="post" id="target"><p>{schema_text}</p></div>
        </div>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://x.com/post/1")

    assert "target post content" in result.markdown
    assert "Other post." not in result.markdown
    assert result.diagnostics["schema_fallback_used"] is True
