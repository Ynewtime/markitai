"""Tests for X tweet internal cleanup."""

from bs4 import BeautifulSoup, Tag

from markitai.webextract.extractors.x_tweet import _clean_tweet_internals


def _make_tweet_with_username() -> Tag:
    html = """
    <article data-testid="tweet">
      <div data-testid="User-Name">
        <a href="/user"><span>Display Name</span></a>
        <a href="/user"><span>@handle</span></a>
      </div>
      <div data-testid="tweetText">
        <span>Tweet content here.</span>
      </div>
    </article>
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    assert isinstance(article, Tag)
    return article


def test_dedup_author_keeps_only_one_link() -> None:
    """After cleanup, User-Name should have exactly one link (display name only)."""
    tweet = _make_tweet_with_username()
    _clean_tweet_internals(tweet)
    user_name = tweet.find(attrs={"data-testid": "User-Name"})
    assert user_name is not None
    links = user_name.find_all("a")
    assert len(links) == 1, f"Expected 1 link, got {len(links)}"
    assert "Display Name" in links[0].get_text()
