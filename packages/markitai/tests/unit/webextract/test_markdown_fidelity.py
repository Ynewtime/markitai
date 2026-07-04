from __future__ import annotations

"""Tests for the narrow Markdown fidelity layer (webextract/markdown.py)."""


from markitai.webextract.markdown import render_markdown

# ---------------------------------------------------------------------------
# Test fixtures: HTML inputs
# ---------------------------------------------------------------------------

_FIGURE_HTML = """
<article>
  <p>Some intro text.</p>
  <figure>
    <img src="https://example.com/image.jpg" alt="Alt">
    <figcaption>Figure caption</figcaption>
  </figure>
  <p>Trailing paragraph.</p>
</article>
"""

_FIGURE_SRCSET_HTML = """
<article>
  <figure>
    <img src="https://example.com/image-400.jpg"
         srcset="https://example.com/image-400.jpg 400w, https://example.com/image-1200.jpg 1200w, https://example.com/image-800.jpg 800w"
         alt="Responsive image">
    <figcaption>Responsive caption</figcaption>
  </figure>
</article>
"""

_EMBED_IFRAME_HTML = """
<article>
  <p>Check this tweet.</p>
  <iframe src="https://x.com/i/status/123" width="550" height="300"></iframe>
  <p>More text.</p>
</article>
"""

_YOUTUBE_IFRAME_HTML = """
<article>
  <p>Watch the video.</p>
  <iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ" width="640" height="480"></iframe>
</article>
"""

_VIMEO_IFRAME_HTML = """
<article>
  <p>Watch on Vimeo.</p>
  <iframe src="https://player.vimeo.com/video/76979871" width="640" height="360"></iframe>
</article>
"""

_BLANK_LINES_HTML = """
<article>
  <p>First paragraph.</p>
  <p>Second paragraph.</p>
  <p>Third paragraph.</p>
</article>
"""


# ---------------------------------------------------------------------------
# Figure / caption tests
# ---------------------------------------------------------------------------


def test_figure_caption_is_preserved_under_image() -> None:
    """Figcaption text must appear in output alongside the image."""
    markdown = render_markdown(_FIGURE_HTML)
    assert "![Alt](https://example.com/image.jpg)" in markdown
    assert "Figure caption" in markdown


def test_figure_caption_appears_near_image() -> None:
    """Caption should follow the image, not appear elsewhere."""
    markdown = render_markdown(_FIGURE_HTML)
    img_pos = markdown.index("![Alt](https://example.com/image.jpg)")
    caption_pos = markdown.index("Figure caption")
    # Caption must appear after the image
    assert caption_pos > img_pos


def test_figure_surrounding_text_preserved() -> None:
    """Non-figure content in the same article is not dropped."""
    markdown = render_markdown(_FIGURE_HTML)
    assert "Some intro text" in markdown
    assert "Trailing paragraph" in markdown


# ---------------------------------------------------------------------------
# srcset best-URL selection
# ---------------------------------------------------------------------------


def test_srcset_best_url_is_selected() -> None:
    """The highest-resolution URL from srcset is used as the image src."""
    markdown = render_markdown(_FIGURE_SRCSET_HTML)
    # 1200w is the highest resolution, so it should be selected
    assert "image-1200.jpg" in markdown


def test_srcset_caption_preserved() -> None:
    """Caption is still rendered even when srcset processing occurs."""
    markdown = render_markdown(_FIGURE_SRCSET_HTML)
    assert "Responsive caption" in markdown


# ---------------------------------------------------------------------------
# Embed canonicalization
# ---------------------------------------------------------------------------


def test_embed_iframe_is_reduced_to_canonical_link() -> None:
    """Social embed iframes must be reduced to a plain link."""
    markdown = render_markdown(_EMBED_IFRAME_HTML)
    assert "https://x.com/i/status/123" in markdown


def test_youtube_iframe_is_reduced_to_canonical_link() -> None:
    """YouTube embed iframes must become a canonical watch link."""
    markdown = render_markdown(_YOUTUBE_IFRAME_HTML)
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in markdown


def test_vimeo_iframe_is_reduced_to_canonical_link() -> None:
    """Vimeo embed iframes must become a canonical link."""
    markdown = render_markdown(_VIMEO_IFRAME_HTML)
    assert "https://vimeo.com/76979871" in markdown


def test_embed_surrounding_text_preserved() -> None:
    """Text around embed iframes is not dropped."""
    markdown = render_markdown(_EMBED_IFRAME_HTML)
    assert "Check this tweet" in markdown
    assert "More text" in markdown


def test_bilibili_iframe_is_reduced_to_canonical_link() -> None:
    """Bilibili player iframes (protocol-relative src) become video links."""
    html = """
    <article>
      <p>Watch on bilibili.</p>
      <iframe src="//player.bilibili.com/player.html?aid=22325156&amp;cid=36966181&amp;page=1"
              frameborder="no" allowfullscreen="true"></iframe>
    </article>
    """
    markdown = render_markdown(html)
    assert "https://www.bilibili.com/video/av22325156" in markdown


def test_bilibili_iframe_prefers_bvid_over_aid() -> None:
    """When a bvid parameter is present it wins over the numeric aid."""
    html = """
    <article>
      <iframe src="https://player.bilibili.com/player.html?bvid=BV1xx411c7mD&amp;aid=170001"></iframe>
    </article>
    """
    markdown = render_markdown(html)
    assert "https://www.bilibili.com/video/BV1xx411c7mD" in markdown


def test_twitter_widget_iframe_is_reduced_to_status_link() -> None:
    """platform.twitter.com widget iframes become x.com status links."""
    html = """
    <article>
      <iframe src="https://platform.twitter.com/embed/Tweet.html?id=1675626836821409792"
              width="550" height="300"></iframe>
    </article>
    """
    markdown = render_markdown(html)
    assert "https://x.com/i/status/1675626836821409792" in markdown


def test_video_embed_survives_full_extraction_pipeline() -> None:
    """extract_web_content must keep known video embeds as links.

    Regression guard: sanitize_tag_tree strips every <iframe>, so the
    pipeline must canonicalize embeds before sanitization.
    """
    from markitai.webextract import extract_web_content

    html = """
    <html><head><title>Post</title></head><body>
    <article>
      <h1>Post</h1>
      <p>Intro paragraph with enough words to look like real content for
      the extraction pipeline to keep this article as the root node.</p>
      <iframe src="//player.bilibili.com/player.html?aid=22325156&amp;cid=1&amp;page=1"></iframe>
      <p>Closing paragraph after the embedded video player element.</p>
    </article>
    </body></html>
    """
    result = extract_web_content(html, "https://example.com/post")
    assert "https://www.bilibili.com/video/av22325156" in result.markdown
    assert "<iframe" not in result.markdown


# ---------------------------------------------------------------------------
# Post-processing: blank line collapsing
# ---------------------------------------------------------------------------


def test_no_more_than_two_consecutive_blank_lines() -> None:
    """Output must never have more than 2 consecutive blank lines."""
    markdown = render_markdown(_BLANK_LINES_HTML)
    assert "\n\n\n\n" not in markdown


def test_trailing_whitespace_stripped() -> None:
    """No line in the output should have trailing whitespace."""
    markdown = render_markdown(_FIGURE_HTML)
    for line in markdown.splitlines():
        assert line == line.rstrip(), f"Trailing whitespace in line: {line!r}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_html_returns_empty_string() -> None:
    """Empty HTML should produce an empty or near-empty string."""
    markdown = render_markdown("")
    assert markdown.strip() == ""


def test_plain_text_html_passthrough() -> None:
    """Simple paragraph HTML works without errors."""
    markdown = render_markdown("<p>Hello world</p>")
    assert "Hello world" in markdown
