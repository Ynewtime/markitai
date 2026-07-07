"""Tests for X tweet thread classification (defuddle-parity behaviour).

Author self-replies at the top of the reply timeline continue the post
(joined by horizontal rules); replies after the first third-party reply are
comments and follow the thread policy (third-party replies excluded by
default).
"""

from __future__ import annotations

from markitai.webextract.pipeline import extract_web_content

_URL = "https://x.com/dotey/status/2073286406558949828"

_THREAD_HTML = """
<html><body>
<div data-testid="primaryColumn">
  <article data-testid="tweet">
    <div data-testid="User-Name">
      <a href="/dotey"><span>Baoyu</span></a>
      <a href="/dotey"><span>@dotey</span></a>
    </div>
    <div data-testid="tweetText"><span>Main tweet part 1.</span></div>
    <time datetime="2025-12-24T10:00:00.000Z">10:00 AM</time>
  </article>
  <section aria-labelledby="list">
    <article data-testid="tweet">
      <div data-testid="User-Name">
        <a href="/dotey"><span>Baoyu</span></a>
        <a href="/dotey"><span>@dotey</span></a>
      </div>
      <div data-testid="tweetText"><span>Thread continuation part 2.</span></div>
    </article>
    <article data-testid="tweet">
      <div data-testid="User-Name">
        <a href="/someone"><span>Some One</span></a>
        <a href="/someone"><span>@someone</span></a>
      </div>
      <div data-testid="tweetText"><span>A third party reply.</span></div>
    </article>
    <article data-testid="tweet">
      <div data-testid="User-Name">
        <a href="/dotey"><span>Baoyu</span></a>
        <a href="/dotey"><span>@dotey</span></a>
      </div>
      <div data-testid="tweetText"><span>Author reply after thread ended.</span></div>
    </article>
  </section>
</div>
</body></html>
"""


def test_author_thread_rendered_as_post_continuation() -> None:
    result = extract_web_content(_THREAD_HTML, _URL)
    assert "Main tweet part 1." in result.markdown
    assert "Thread continuation part 2." in result.markdown
    # Continuation is part of the post body, not a Comments section
    assert "Comments" not in result.markdown
    # Separated by a horizontal rule
    assert "---" in result.markdown or "***" in result.markdown


def test_semantic_thread_has_continuation_items() -> None:
    result = extract_web_content(_THREAD_HTML, _URL)
    assert result.semantic is not None
    assert result.semantic.thread is not None
    thread = result.semantic.thread
    assert len(thread.continuation_items) == 1
    assert thread.continuation_items[0].text == "Thread continuation part 2."


def test_third_party_replies_excluded_by_default_policy() -> None:
    result = extract_web_content(_THREAD_HTML, _URL)
    assert "A third party reply." not in result.markdown


def test_author_reply_after_thread_end_is_not_continuation() -> None:
    """Mirrors defuddle: the first third-party reply ends the thread."""
    result = extract_web_content(_THREAD_HTML, _URL)
    assert "Author reply after thread ended." not in result.markdown


def test_main_author_meta_not_in_body_for_tweets() -> None:
    """Main item author meta is not rendered for tweets — in frontmatter only."""
    result = extract_web_content(_THREAD_HTML, _URL)
    assert "**Baoyu @dotey** · 2025-12-24" not in result.markdown
