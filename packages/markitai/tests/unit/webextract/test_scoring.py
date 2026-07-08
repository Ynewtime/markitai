from __future__ import annotations

from bs4 import Tag

from markitai.webextract.dom import parse_html
from markitai.webextract.scoring import score_candidate, select_best_candidate

# ---- Behavior-preserving scoring tests (Medium-9) ----


def test_score_candidate_deterministic_for_identical_input() -> None:
    """Scoring the same node twice must return the same value."""
    html = """
    <html><body>
      <article>
        <h1>Title</h1>
        <p>Paragraph one with some words.</p>
        <p>Paragraph two with different words.</p>
        <a href="/link1">link</a>
      </article>
    </body></html>
    """
    soup = parse_html(html)
    article = soup.find("article")
    assert isinstance(article, Tag)
    score1 = score_candidate(article)
    score2 = score_candidate(article)
    assert score1 == score2


def test_score_candidate_values_match_manual_calculation() -> None:
    """Verify that the scoring formula produces expected values so we can
    safely refactor internals without changing outcomes."""
    html = """
    <div class="content">
      <p>Alpha bravo charlie delta echo.</p>
      <p>Foxtrot golf hotel india juliet.</p>
      <a href="/x">link1</a>
      <a href="/y">link2</a>
    </div>
    """
    soup = parse_html(html)
    div = soup.find("div")
    assert isinstance(div, Tag)

    # Manual calculation (defuddle scoreElement):
    # 12 words => score starts at 12.0
    # 2 paragraphs => +20
    # 0 commas
    # class "content" matches => +15
    # base = 12 + 20 + 0 + 15 = 47
    # link_text = "link1" + "link2" = 10 chars
    # link_density = min(10 / len(text), 0.5) ≈ 0.12
    # score = 47 * (1 - 0.12) ≈ 41
    score = score_candidate(div)
    assert 35 < score < 47, f"Expected ~41, got {score}"


def test_select_best_candidate_prefers_entry_point_selectors() -> None:
    """An entry-point match (.post-content) must beat a generic wrapper."""
    html = """
    <html><body>
      <div class="sidebar"><p>Short sidebar.</p></div>
      <div class="post-content">
        <p>This is a long article body with many words to ensure high score.</p>
        <p>Another paragraph with substantial content for scoring.</p>
      </div>
      <div class="footer"><a href="/a">a</a><a href="/b">b</a></div>
    </body></html>
    """
    soup = parse_html(html)
    best = select_best_candidate(soup)
    assert best is not None
    assert "post-content" in best.get("class", [])  # type: ignore[reportArgumentType, reportOperatorIssue]


def test_select_best_candidate_falls_back_to_body() -> None:
    """With no entry-point match besides body, body is the candidate
    (clutter is handled later by removals, mirroring defuddle)."""
    html = """
    <html><body>
      <div class="wrapper">
        <p>Prose paragraph with enough words to score reasonably well here.</p>
      </div>
    </body></html>
    """
    soup = parse_html(html)
    best = select_best_candidate(soup)
    assert best is not None
    assert best.name == "body"


def test_select_best_candidate_prefers_deepest_child() -> None:
    """A contained <article> with >50 words beats its <main> wrapper."""
    words = " ".join(f"word{i}" for i in range(60))
    html = f"""
    <html><body>
      <main>
        <div class="promo"><p>Sidebar noise with several words here.</p></div>
        <article><p>{words}</p></article>
      </main>
    </body></html>
    """
    soup = parse_html(html)
    best = select_best_candidate(soup)
    assert best is not None
    assert best.name == "article"


def test_select_best_candidate_keeps_listing_parent() -> None:
    """Multiple sibling <article> cards mean a listing page — keep parent."""
    card = " ".join(f"word{i}" for i in range(60))
    html = f"""
    <html><body>
      <main>
        <article><p>{card}</p></article>
        <article><p>{card}</p></article>
        <article><p>{card}</p></article>
      </main>
    </body></html>
    """
    soup = parse_html(html)
    best = select_best_candidate(soup)
    assert best is not None
    assert best.name == "main"


def test_score_candidate_large_dom_many_candidates() -> None:
    """Scoring many candidates must produce the same winner regardless of
    internal caching/optimization strategy."""
    # Build a DOM with 50 div candidates + one clear winner (article)
    divs = "\n".join(
        f'<div id="d{i}"><p>Filler paragraph {i}.</p></div>' for i in range(50)
    )
    winner_words = " ".join(f"word{i}" for i in range(100))
    html = f"""
    <html><body>
      {divs}
      <article class="post">
        <p>{winner_words}</p>
        <p>Extra paragraph for bonus.</p>
      </article>
    </body></html>
    """
    soup = parse_html(html)
    best = select_best_candidate(soup)
    assert best is not None
    assert best.name == "article"


# ---- Original tests ----


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
          <p>Short.</p>
        </article>
        <section class="related">
          <p>This extra paragraph lives outside the article and provides
          enough words so that falling back to the body element yields
          a longer extraction than the initial short candidate.</p>
        </section>
      </body>
    </html>
    """

    result = extract_web_content(html, "https://example.com/post")

    assert "extra paragraph" in result.markdown
    assert result.diagnostics["adaptive_retry_used"] is True
