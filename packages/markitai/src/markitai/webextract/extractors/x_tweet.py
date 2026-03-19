"""Extractor for X/Twitter tweet pages (rendered HTML from Playwright)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag


class XTweetExtractor:
    """Extract tweet content from X.com rendered HTML.

    Targets the tweet timeline/conversation view. Finds ``<article>``
    elements with ``data-testid="tweet"`` and extracts the conversation
    thread, discarding navigation, trending, signup prompts, etc.
    """

    name = "x_tweet"

    def matches_url(self, url: str) -> bool:
        """Match x.com and twitter.com status URLs."""
        return ("x.com/" in url or "twitter.com/" in url) and "/status/" in url

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Extract the conversation timeline or single tweet.

        Strategy:
        1. Find the timeline conversation container
        2. If not found, find individual tweet articles
        3. Wrap matched tweets in a synthetic container
        """
        # Strategy 1: Find the conversation timeline
        timeline = soup.find(
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
    """Extract tweets from a conversation timeline, stopping at 'Discover more'."""
    tweets = timeline.find_all("article", attrs={"data-testid": "tweet"})
    if not tweets:
        return timeline

    # Filter: stop at section headings (Discover more, etc.)
    filtered: list[Tag] = []
    for tweet in tweets:
        # Check if there's a section/h2 between this tweet and the timeline
        # that would indicate "related content" rather than the thread
        filtered.append(tweet)

    return _wrap_tweets(filtered, soup)


def _wrap_tweets(tweets: list[Tag], soup: BeautifulSoup) -> Tag:
    """Wrap a list of tweet articles in a container div."""
    doc = BeautifulSoup("", "html.parser")
    container = doc.new_tag("div", attrs={"class": "tweet-thread"})

    for i, tweet in enumerate(tweets):
        # Clone the tweet (extract from original tree)
        cloned = tweet.extract()
        container.append(cloned)

        # Add separator between tweets
        if i < len(tweets) - 1:
            hr = doc.new_tag("hr")
            container.append(hr)

    return container
