"""Shared X/Twitter HTML parsing utilities.

Reusable functions for extracting structured data from X.com rendered HTML.
These are consumed by ``XTweetExtractor.resolve()`` and potentially other
X-related extractors.
"""

from __future__ import annotations

import re

from bs4 import Tag

from markitai.webextract.semantics import (
    ConversationItem,
    EmbeddedQuote,
    MediaAttachment,
)

# Sections that indicate the end of the main conversation content.
_STOP_LABELS = frozenset({"Discover more", "Timeline: Trending now"})


def parse_tweet_article(article: Tag, *, tweet_id: str = "") -> ConversationItem:
    """Parse a single ``<article data-testid="tweet">`` into a ConversationItem.

    Args:
        article: The ``<article>`` tag to parse.
        tweet_id: Stable ID for the item; defaults to empty string.

    Returns:
        A populated ``ConversationItem``.
    """
    author_name, author_handle = _parse_author(article)
    text = _parse_text(article)
    timestamp = _parse_timestamp(article)
    media = _parse_media(article)
    quoted = _parse_quoted_item(article)

    return ConversationItem(
        id=tweet_id,
        author_name=author_name,
        author_handle=author_handle,
        text=text,
        timestamp=timestamp,
        quoted_item=quoted,
        media=media,
    )


def find_primary_tweet(primary_column: Tag) -> Tag | None:
    """Find the main tweet article, excluding quotes and reply sections.

    The primary tweet is the first ``<article data-testid="tweet">`` that is
    NOT inside a ``data-testid="card.wrapper"`` (quote) or a ``<section>``
    (replies).

    Args:
        primary_column: The ``data-testid="primaryColumn"`` container.

    Returns:
        The main tweet ``<article>`` tag, or ``None`` if not found.
    """
    for article in primary_column.find_all("article", attrs={"data-testid": "tweet"}):
        if _is_inside_quote_card(article):
            continue
        if _is_inside_reply_section(article):
            continue
        return article  # type: ignore[return-value]
    return None


def is_recommendation_section(tag: Tag) -> bool:
    """Check whether a tag is a recommendation / noise section.

    Args:
        tag: A BeautifulSoup Tag to check.

    Returns:
        True if the tag is a recommendation section that should be excluded.
    """
    aria_label = tag.get("aria-label", "")
    if isinstance(aria_label, str) and aria_label in _STOP_LABELS:
        return True
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_author(article: Tag) -> tuple[str | None, str | None]:
    """Extract display name and @handle from User-Name element.

    Args:
        article: Tweet article tag.

    Returns:
        Tuple of (display_name, handle). Either may be None.
    """
    user_name = article.find(attrs={"data-testid": "User-Name"})
    if not isinstance(user_name, Tag):
        return None, None

    display_name: str | None = None
    handle: str | None = None

    links = user_name.find_all("a", href=True)
    for link in links:
        spans = link.find_all("span")
        for span in spans:
            text = span.get_text(strip=True)
            if not text:
                continue
            if text.startswith("@"):
                handle = text
            elif display_name is None:
                display_name = text

    return display_name, handle


def _parse_text(article: Tag) -> str:
    """Extract tweet text content.

    Args:
        article: Tweet article tag.

    Returns:
        Plain text content of the tweet.
    """
    text_el = article.find(attrs={"data-testid": "tweetText"})
    if not isinstance(text_el, Tag):
        return ""
    return text_el.get_text(" ", strip=True)


def _parse_timestamp(article: Tag) -> str | None:
    """Extract ISO timestamp from ``<time>`` element.

    Args:
        article: Tweet article tag.

    Returns:
        ISO-8601 datetime string, or None if not found.
    """
    time_el = article.find("time", attrs={"datetime": True})
    if not isinstance(time_el, Tag):
        return None
    dt = time_el.get("datetime")
    return str(dt) if dt else None


def _parse_media(article: Tag) -> list[MediaAttachment]:
    """Extract media attachments from a tweet.

    Args:
        article: Tweet article tag.

    Returns:
        List of media attachments found.
    """
    media: list[MediaAttachment] = []
    for img in article.find_all("img", src=True):
        src = str(img.get("src", ""))
        alt = str(img.get("alt", ""))
        # Skip profile avatars and emoji images
        if "profile_images" in src or "emoji" in src:
            continue
        if src.startswith("http"):
            media.append(MediaAttachment(url=src, alt=alt))
    return media


def _parse_quoted_item(article: Tag) -> EmbeddedQuote | None:
    """Extract an embedded quote from inside a tweet article.

    Args:
        article: Tweet article tag.

    Returns:
        An EmbeddedQuote if found, otherwise None.
    """
    card = article.find(attrs={"data-testid": "card.wrapper"})
    if not isinstance(card, Tag):
        return None

    # Find the quoted tweet inside the card
    quoted_article = card.find("article", attrs={"data-testid": "tweet"})
    if not isinstance(quoted_article, Tag):
        return None

    q_name, q_handle = _parse_author(quoted_article)
    q_text = _parse_text(quoted_article)

    return EmbeddedQuote(
        author_name=q_name,
        author_handle=q_handle,
        text=q_text,
    )


def _is_inside_quote_card(article: Tag) -> bool:
    """Check if an article tag is nested inside a quote card wrapper."""
    parent = article.parent
    while parent is not None:
        if isinstance(parent, Tag):
            testid = parent.get("data-testid", "")
            if testid == "card.wrapper":
                return True
        parent = parent.parent
    return False


def _is_inside_reply_section(article: Tag) -> bool:
    """Check if an article tag is nested inside a reply section."""
    parent = article.parent
    while parent is not None:
        if isinstance(parent, Tag) and parent.name == "section":
            return True
        parent = parent.parent
    return False


def extract_tweet_id_from_url(url: str) -> str:
    """Extract the tweet/status ID from an X.com URL.

    Args:
        url: A URL like ``https://x.com/user/status/123456``.

    Returns:
        The status ID string, or an empty string if not found.
    """
    match = re.search(r"/status/(\d+)", url)
    return match.group(1) if match else ""
