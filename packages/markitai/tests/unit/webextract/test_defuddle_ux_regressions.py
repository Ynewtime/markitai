"""User-visible gaps found by running the shared defuddle corpus locally."""

from pathlib import Path
from unittest.mock import patch

from markitai.webextract import extract_web_content

FIXTURES = Path(__file__).parents[2] / "defuddle_fixtures" / "fixtures"


def test_collapsed_callouts_keep_their_body_type_and_fold_state():
    result = extract_web_content(
        (FIXTURES / "callouts--obsidian-publish-callouts.html").read_text(),
        "https://example.com/callouts",
    )
    assert "> [!faq]- Is this foldable?" in result.markdown
    assert "> Yes, the content is hidden when collapsed." in result.markdown
    assert "> [!abstract]- Abstract" in result.markdown
    assert "> [!tip]- Tip" in result.markdown
    assert result.markdown.count("Lorem ipsum dolor sit amet") == 3


def test_hackernews_comment_permalink_keeps_div_commtext_body():
    result = extract_web_content(
        (FIXTURES / "general--news.ycombinator.com-item-id=12345678.html").read_text(),
        "https://news.ycombinator.com/item?id=12345678",
    )
    assert "This is the main comment text that should be extracted" in result.markdown
    assert (
        "It has multiple paragraphs to test proper content extraction."
        in result.markdown
    )
    assert "https://example.com" in result.markdown
    assert result.metadata.author == "testuser"
    assert result.metadata.published == "2025-06-15T12:00:00"
    assert "**testuser**" in result.markdown
    assert "2025-06-15" in result.markdown
    assert "# This is a comment" not in result.markdown


def test_native_html_does_not_initialize_general_file_detection():
    with patch(
        "markitdown.MarkItDown",
        side_effect=AssertionError("HTML needs no file detector"),
    ):
        result = extract_web_content(
            "<article><p>Known HTML can be converted directly without loading a file format detection model.</p></article>",
            "https://example.com/article",
        )
    assert "Known HTML" in result.markdown


def test_nested_collapsed_callouts_preserve_body_without_unhiding_other_elements():
    html = """<article><p>Callout examples with nested content and separate hidden UI.</p>
    <div class="callout is-collapsed" data-callout="note">
      <div class="callout-title"><div class="callout-title-inner">Outer</div></div>
      <div class="callout-content" style="display:none">
        <p>Outer body survives.</p>
        <span style="display:none">Hidden interface text</span>
        <div class="callout is-collapsed" data-callout="tip">
          <div class="callout-title"><div class="callout-title-inner">Inner</div></div>
          <div class="callout-content" style="display:none"><p>Inner body survives.</p></div>
        </div>
      </div>
    </div></article>"""
    result = extract_web_content(html, "https://example.com/nested-callouts")
    assert "[!note]- Outer" in result.markdown
    assert "[!tip]- Inner" in result.markdown
    assert result.markdown.count("Outer body survives.") == 1
    assert result.markdown.count("Inner body survives.") == 1
    assert "Hidden interface text" not in result.markdown
