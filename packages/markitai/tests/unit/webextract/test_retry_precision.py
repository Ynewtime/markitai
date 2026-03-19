from __future__ import annotations

from markitai.webextract.pipeline import extract_web_content


def test_short_article_does_not_pull_in_sidebar_noise() -> None:
    """Body fallback should not trade precision for word count on short valid articles."""
    html = """
    <html><body>
        <article>
            <h1>Note</h1>
            <p>Short valid post.</p>
        </article>
        <div class="sidebar">
            <p>
                This sidebar contains more words than the article and should never
                replace the main content, but it can dominate the retry path.
            </p>
        </div>
    </body></html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "Short valid post." in result.markdown
    assert "should never replace the main content" not in result.markdown


def test_short_github_issue_retry_keeps_extractor_content() -> None:
    """Retry should keep the GitHub issue extractor's root instead of switching to sidebar."""
    html = """
    <html><body>
        <div class="gh-header-show"><h1>Bug title</h1></div>
        <div class="comment-body"><p>Short bug.</p></div>
        <div class="sidebar">
            <p>
                This sidebar has many many many words and links and extra text to
                outweigh the short issue body and tempt the scorer into picking the
                wrong root during retry logic.
            </p>
        </div>
    </body></html>
    """

    result = extract_web_content(html, "https://github.com/org/repo/issues/1")

    assert "Bug title" in result.markdown
    assert "Short bug." in result.markdown
    assert "tempt the scorer" not in result.markdown
