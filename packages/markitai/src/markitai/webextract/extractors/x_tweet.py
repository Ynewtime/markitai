"""Extractor for X/Twitter tweet pages (rendered HTML from Playwright)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.extractors.x_common import (
    extract_tweet_id_from_url,
    find_primary_tweet,
    parse_tweet_article,
)
from markitai.webextract.render import render_semantic_content
from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.semantics import ConversationItem, ConversationThread
from markitai.webextract.thread_policy import get_thread_policy
from markitai.webextract.types import SemanticExtraction

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
                "aria-label": lambda v: (  # type: ignore[dict-item]
                    v and "Timeline" in str(v) and "Conversation" in str(v)
                )
            }
        )
        if isinstance(timeline, Tag):
            return _extract_from_timeline(timeline, soup)

        # Strategy 2: Find individual tweet articles
        tweets = soup.find_all("article", attrs={"data-testid": "tweet"})
        if tweets:
            return _wrap_tweets(tweets, soup)

        # Strategy 3: Try role="main" as fallback
        main = soup.find(True, attrs={"role": "main"})
        if isinstance(main, Tag):
            return main

        return None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve an X tweet page into a structured ResolvedPage.

        Finds the primary tweet, builds a ``ConversationThread``, renders
        it through the shared renderer, and returns metadata overrides.

        Args:
            soup: Parsed HTML of the page.
            url: Source URL.

        Returns:
            A ``ResolvedPage`` with semantic content and metadata overrides.
        """
        tweet_id = extract_tweet_id_from_url(url)

        # Find primaryColumn to scope our search
        primary_col = soup.find(True, attrs={"data-testid": "primaryColumn"})
        if not isinstance(primary_col, Tag):
            # Fallback: search the whole document
            primary_col = soup  # type: ignore[assignment]

        main_article = find_primary_tweet(primary_col)
        if main_article is None:
            # Fall back: can't find primary tweet, let generic pipeline handle
            return ResolvedPage(
                diagnostics={"x_resolve": "no_primary_tweet_found"},
            )

        main_item = parse_tweet_article(main_article, tweet_id=tweet_id)

        # Build title from handle or name
        handle = main_item.author_handle or ""
        display_name = main_item.author_name or ""
        title = f"Post by {handle}" if handle else f"Post by {display_name}"

        # Collect replies governed by thread policy. Mirrors defuddle's
        # classification: consecutive self-replies by the main author at the
        # top of the reply timeline continue the post (thread); everything
        # after the first third-party reply is a comment.
        policy = get_thread_policy(url)
        continuation_items: list[ConversationItem] = []
        reply_items: list[ConversationItem] = []
        if policy is not None:
            reply_section = primary_col.find("section")  # type: ignore[union-attr]
            if isinstance(reply_section, Tag):
                reply_articles = reply_section.find_all(
                    "article", attrs={"data-testid": "tweet"}
                )
                thread_ended = False
                for i, reply_art in enumerate(reply_articles):
                    if not isinstance(reply_art, Tag):
                        continue
                    item = parse_tweet_article(reply_art, tweet_id=f"reply-{i}")
                    item_is_author = bool(
                        (handle and item.author_handle == handle)
                        or (display_name and item.author_name == display_name)
                    )
                    if not item_is_author:
                        thread_ended = True
                    if item_is_author and not thread_ended:
                        if policy.include_author_thread:
                            continuation_items.append(item)
                    elif policy.include_third_party_replies:
                        reply_items.append(item)

        thread = ConversationThread(
            title=title,
            main_item=main_item,
            items=reply_items,
            continuation_items=continuation_items,
        )

        semantic = SemanticExtraction(thread=thread)
        content_html = render_semantic_content(semantic)

        # Metadata overrides
        description = main_item.text[:200] if main_item.text else ""
        metadata_overrides: dict[str, object] = {
            "title": title,
            "author": display_name or handle,
            "site": "X (Twitter)",
        }
        if description:
            metadata_overrides["description"] = description

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            semantic=semantic,
            diagnostics={"x_resolve": "success", "tweet_id": tweet_id},
        )


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
    for el in list(tweet.find_all(True, attrs={"data-testid": True})):
        if not el.attrs:
            continue
        testid = el.get("data-testid", "")
        if testid in _NOISE_TESTIDS or isinstance(testid, str) and "Avatar" in testid:
            el.decompose()

    # Remove action button groups (role="group")
    for el in list(tweet.find_all(True, attrs={"role": "group"})):
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

    # Deduplicate User-Name links: keep display name, remove @handle link
    for user_name_el in list(tweet.find_all(True, attrs={"data-testid": "User-Name"})):
        links = user_name_el.find_all("a")
        if len(links) >= 2:
            # Second link is typically the @handle duplicate
            for link in links[1:]:
                link.decompose()


def _is_timestamp_text(text: str) -> bool:
    """Check if text looks like a tweet timestamp."""
    if not text:
        return False
    # Patterns: "10:17 AM · Mar 7, 2026", "3:45 PM · Jan 15", "Mar 7"
    time_indicators = ("AM", "PM", "am", "pm", "\u00b7")
    return any(indicator in text for indicator in time_indicators) and len(text) < 50
