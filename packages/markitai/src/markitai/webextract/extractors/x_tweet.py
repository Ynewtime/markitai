"""Extractor for X/Twitter tweet pages (rendered HTML from Playwright)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

# data-testid values for tweet-internal noise (action buttons, metadata)
_NOISE_TESTIDS = frozenset(
    {
        "caret",
        "reply",
        "retweet",
        "like",
        "bookmark",
        "app-text-transition-container",
        "logged_out_read_replies_pivot",
    }
)


class XTweetExtractor:
    """Extract tweet content from X.com rendered HTML.

    Targets the tweet timeline/conversation view. Finds ``<article>``
    elements with ``data-testid="tweet"`` and extracts the conversation
    thread, discarding navigation, trending, signup prompts, and
    tweet-internal noise (action buttons, analytics, timestamps).
    """

    name = "x_tweet"

    def matches_url(self, url: str) -> bool:
        """Match x.com and twitter.com status URLs."""
        return ("x.com/" in url or "twitter.com/" in url) and "/status/" in url

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Extract the conversation timeline or single tweet."""
        # Strategy 1: Find the conversation timeline
        timeline = soup.find(  # type: ignore[call-overload]
            attrs={
                "aria-label": lambda v: v
                and "Timeline" in str(v)
                and "Conversation" in str(v)
            }
        )
        if isinstance(timeline, Tag):
            return _extract_from_timeline(timeline, soup)

        # Strategy 2: Find individual tweet articles
        tweets = soup.find_all("article", attrs={"data-testid": "tweet"})
        if tweets:
            return _wrap_tweets(tweets, soup)

        # Strategy 3: Try role="main" as fallback
        main = soup.find(attrs={"role": "main"})
        if isinstance(main, Tag):
            return main

        return None


def _extract_from_timeline(timeline: Tag, soup: BeautifulSoup) -> Tag:
    """Extract tweets from a conversation timeline."""
    tweets = timeline.find_all("article", attrs={"data-testid": "tweet"})
    if not tweets:
        return timeline
    return _wrap_tweets(list(tweets), soup)


def _wrap_tweets(tweets: list[Tag], soup: BeautifulSoup) -> Tag:
    """Wrap tweet articles in a container, cleaning internal noise."""
    doc = BeautifulSoup("", "html.parser")
    container = doc.new_tag("div", attrs={"class": "tweet-thread"})

    for i, tweet in enumerate(tweets):
        cloned = tweet.extract()
        _clean_tweet_internals(cloned)
        container.append(cloned)

        if i < len(tweets) - 1:
            hr = doc.new_tag("hr")
            container.append(hr)

    return container


def _clean_tweet_internals(tweet: Tag) -> None:
    """Remove noise elements inside a tweet article.

    Strips action buttons, analytics, timestamps, and other
    non-content elements while preserving the tweet text, author,
    and media.
    """
    # Remove action buttons and metadata by data-testid
    for el in list(tweet.find_all(attrs={"data-testid": True})):
        if not el.attrs:
            continue
        testid = el.get("data-testid", "")
        if testid in _NOISE_TESTIDS or isinstance(testid, str) and "Avatar" in testid:
            el.decompose()

    # Remove action button groups (role="group")
    for el in list(tweet.find_all(attrs={"role": "group"})):
        el.decompose()

    # Remove timestamp and analytics links
    for link in list(tweet.find_all("a", href=True)):
        if not link.attrs:
            continue
        href = str(link.get("href", ""))
        text = link.get_text(strip=True)
        if "/analytics" in href:
            link.decompose()
            continue
        if "/status/" in href and _is_timestamp_text(text):
            link.decompose()
            continue

    # Remove "Translate post" text
    for el in list(
        tweet.find_all(string=lambda s: s and s.strip() == "Translate post")  # type: ignore[call-overload]
    ):
        if el.parent:
            el.parent.decompose()


def _is_timestamp_text(text: str) -> bool:
    """Check if text looks like a tweet timestamp."""
    if not text:
        return False
    # Patterns: "10:17 AM · Mar 7, 2026", "3:45 PM · Jan 15", "Mar 7"
    time_indicators = ("AM", "PM", "am", "pm", "·")
    return any(indicator in text for indicator in time_indicators) and len(text) < 50
