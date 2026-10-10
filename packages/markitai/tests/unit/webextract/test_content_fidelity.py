"""Content that extraction used to lose or corrupt (found against markitai-rust)."""

from __future__ import annotations

from pathlib import Path

from markitai.webextract.pipeline import extract_web_content

_PROSE = "The article explains the idea in ordinary sentences for the reader. " * 4
_SIDEBAR = (
    "A sidebar box of the article itself, written as prose: it develops a side "
    "argument in several sentences, with no list of links and no promotion. " * 2
)


def _markdown(body: str, url: str = "https://example.com/a") -> str:
    html = f"<html><head><title>T</title></head><body>{body}</body></html>"
    return extract_web_content(html, url).markdown


def test_an_aside_among_the_articles_paragraphs_is_kept() -> None:
    """martinfowler.com: "How big is a microservice?" and six more sidebars."""
    markdown = _markdown(
        f"<main><article><h1>T</h1><p>{_PROSE}</p>"
        f'<aside class="sidebar"><h3>How big?</h3><p>{_SIDEBAR}</p></aside>'
        f"<p>{_PROSE}</p><p>{_PROSE}</p></article></main>"
    )
    assert "How big?" in markdown
    assert "develops a side argument" in markdown


def test_a_page_sidebar_and_a_promo_aside_still_go() -> None:
    links = "".join(
        f'<li><a href="/post/{n}">Another post {n}</a></li>' for n in range(8)
    )
    markdown = _markdown(
        f"<main><article><h1>T</h1><p>{_PROSE}</p>"
        '<aside class="sidebar"><p>My <a href="/guide">Resource Guide</a> '
        "has links.</p></aside>"
        f"<p>{_PROSE}</p><p>{_PROSE}</p></article>"
        f'<aside class="sidebar"><h3>Recent posts</h3><ul>{links}</ul></aside></main>'
    )
    assert "Resource Guide" not in markdown
    assert "Recent posts" not in markdown and "Another post" not in markdown


def test_the_abstract_survives_its_front_matter_block() -> None:
    """martinfowler.com's section.frontMatter: abstract, date, author bios."""
    markdown = _markdown(
        "<main><h1>T</h1>"
        '<section class="frontMatter"><p class="abstract"><i>The term has sprung '
        "up over the last few years to describe a particular way of designing "
        "software applications as suites of services.</i></p>"
        '<div class="author"><p>The author is a consultant with many years of '
        "experience in integrating enterprise systems at scale.</p></div></section>"
        f"<p>{_PROSE}</p><p>{_PROSE}</p></main>"
    )
    assert "has sprung up over the last few years" in markdown
    assert "consultant" not in markdown


def test_a_caption_keeps_inline_code_and_links_with_their_spaces() -> None:
    """react.dev: "Initially,MyApp’scountstate is0and" before."""
    markdown = _markdown(
        f"<main><article><h1>T</h1><p>{_PROSE}</p>"
        '<figure><img src="/d.png" alt="diagram"><figcaption><p>Initially, '
        "<code>MyApp</code>’s <code>count</code> state is <code>0</code> and "
        '<a href="/next">passed</a> down</p></figcaption></figure></article></main>'
    )
    assert (
        "*Initially, `MyApp`’s `count` state is `0` and "
        "[passed](https://example.com/next) down*"
    ) in markdown


def test_a_space_written_as_its_own_element_stays_a_space() -> None:
    """obsidian.md's span.widont, MediaWiki's span[typeof=mw:Entity]."""
    markdown = _markdown(
        f"<main><article><h1>T</h1><p>{_PROSE}</p><p>mixing it with your password "
        'before<span class="widont">&nbsp;</span>hashing, in December'
        '<span typeof="mw:Entity">&nbsp;</span>1989.</p></article></main>'
    )
    assert "before\xa0hashing, in December\xa01989." in markdown


def test_line_breaks_in_a_code_block_stay_line_breaks() -> None:
    """Hexo and react.dev write code lines as <span>...</span><br>."""
    markdown = _markdown(
        f"<main><article><h1>T</h1><p>{_PROSE}</p>"
        '<pre><code><span class="line">foo bar</span><br>'
        '<span class="line">baz qux</span><br></code></pre></article></main>'
    )
    assert "```\nfoo bar\nbaz qux\n```" in markdown


_FIXTURES = Path(__file__).parents[2] / "defuddle_fixtures" / "fixtures"


def test_mediawiki_edit_links_and_print_footer_are_selector_removals() -> None:
    from bs4 import BeautifulSoup

    from markitai.webextract.removals.selectors import remove_by_selectors

    soup = BeautifulSoup(
        '<main><div class="mw-heading mw-heading2"><h2 id="History">History</h2>'
        '<span class="mw-editsection"><span>[</span><a href="/w/index.php?title=T'
        '&amp;action=edit&amp;section=1">edit</a><span>]</span></span></div>'
        f"<p>{_PROSE}</p>"
        '<div class="printfooter">Retrieved from "<a href="https://example.org/w/'
        'index.php?oldid=1">https://example.org/w/index.php?oldid=1</a>"</div></main>',
        "lxml",
    )
    assert soup.main is not None
    # The exact selectors alone: partial matching is fuzzy and did not catch
    # them on Wikipedia's own pages
    remove_by_selectors(soup.main, None, use_partial=False)
    text = soup.main.get_text(" ", strip=True)
    assert "History" in text
    assert "edit" not in text and "Retrieved from" not in text


def test_wikipedia_headings_come_without_edit_links() -> None:
    html = (_FIXTURES / "general--wikipedia.html").read_text(encoding="utf-8")
    markdown = extract_web_content(
        html, "https://en.wikipedia.org/wiki/Obsidian_(software)"
    ).markdown
    assert "## History\n" in markdown
    assert "action=edit" not in markdown
