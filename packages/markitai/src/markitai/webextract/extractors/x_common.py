"""Shared X/Twitter HTML parsing utilities.

Reusable functions for extracting structured data from X.com rendered HTML.
These are consumed by ``XTweetExtractor.resolve()`` and potentially other
X-related extractors.
"""

from __future__ import annotations

import copy
import re

from bs4 import Tag
from bs4.element import NavigableString

from markitai.webextract.semantics import (
    ConversationItem,
    EmbeddedQuote,
    MediaAttachment,
)

# Sections that indicate the end of the main conversation content.
_STOP_LABELS = frozenset({"Discover more", "Timeline: Trending now"})

_HANDLE_RE = re.compile(r"(@\w+)")

# Upgrade X image CDN size parameter, e.g. "...&name=small" -> "...&name=large"
_IMG_NAME_PARAM_RE = re.compile(r"([?&]name=)\w+")


def parse_tweet_article(article: Tag, *, tweet_id: str = "") -> ConversationItem:
    """Parse a single ``<article data-testid="tweet">`` into a ConversationItem.

    Args:
        article: The ``<article>`` tag to parse.
        tweet_id: Stable ID for the item; defaults to empty string.

    Returns:
        A populated ``ConversationItem``.
    """
    author_name, author_handle = _parse_author(article)
    quoted_el = _find_quoted_element(article)
    text = _parse_text(article, exclude=quoted_el)
    timestamp = _parse_timestamp(article, exclude=quoted_el)
    media = _parse_media(article, exclude=quoted_el)
    quoted = _parse_quoted_item(quoted_el) if quoted_el is not None else None
    card_url, card_title = _parse_card(article)

    return ConversationItem(
        id=tweet_id,
        author_name=author_name,
        author_handle=author_handle,
        text=text,
        timestamp=timestamp,
        quoted_item=quoted,
        media=media,
        card_url=card_url,
        card_title=card_title,
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
    """Extract display name and @handle from the User-Name element.

    Handles both the main-tweet structure (name and handle inside separate
    ``<a>`` links) and the quoted-tweet structure (plain text children like
    ``"Name"`` and ``"@handle·date"`` without links).

    Args:
        article: Tweet article tag.

    Returns:
        Tuple of (display_name, handle). Either may be None.
    """
    user_name = article.find(True, attrs={"data-testid": "User-Name"})
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

    # Fallback for quoted-tweet / linkless structure
    if display_name is None or handle is None:
        full_text = user_name.get_text(" ", strip=True)
        if handle is None:
            handle_match = _HANDLE_RE.search(full_text)
            if handle_match:
                handle = handle_match.group(1)
        if display_name is None:
            candidate = re.sub(r"[\s·…]+$", "", full_text.split("@", 1)[0])
            if candidate:
                display_name = candidate

    return display_name, handle


def _parse_text(article: Tag, *, exclude: Tag | None = None) -> str:
    """Extract tweet text content, preserving paragraph breaks.

    Inline emoji images are replaced by their alt text; links contribute
    their visible display text (X renders expanded/truncated URLs and
    @mentions as anchor text). Newlines in the tweet are preserved so the
    renderer can restore paragraph structure.

    Args:
        article: Tweet article tag.
        exclude: A descendant element (e.g. the quoted tweet container)
            whose text must not leak into the parent tweet's text.

    Returns:
        Plain text content of the tweet with ``\\n`` paragraph breaks.
    """
    text_el = article.find(True, attrs={"data-testid": "tweetText"})
    if not isinstance(text_el, Tag):
        return ""
    if exclude is not None and any(p is exclude for p in text_el.parents):
        # The only tweetText found belongs to the quoted tweet.
        return ""

    clone = copy.copy(text_el)
    for img in clone.find_all("img"):
        alt = img.get("alt", "")
        img.replace_with(str(alt) if alt else "")
    # Whitespace-only text nodes are HTML formatting (indentation between
    # tags), not tweet line breaks; collapse them to a single space. Must
    # run before <br> conversion so intentional breaks survive.
    for node in clone.find_all(string=True):
        if isinstance(node, NavigableString) and not node.strip() and "\n" in node:
            node.replace_with(" ")
    for br in clone.find_all("br"):
        br.replace_with("\n")

    text = clone.get_text("")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _parse_timestamp(article: Tag, *, exclude: Tag | None = None) -> str | None:
    """Extract ISO timestamp from the first own ``<time>`` element.

    Args:
        article: Tweet article tag.
        exclude: A descendant element (e.g. the quoted tweet container)
            whose ``<time>`` must not be attributed to this tweet.

    Returns:
        ISO-8601 datetime string, or None if not found.
    """
    for time_el in article.find_all("time", attrs={"datetime": True}):
        if not isinstance(time_el, Tag):
            continue
        if exclude is not None and any(p is exclude for p in time_el.parents):
            continue
        dt = time_el.get("datetime")
        if dt:
            return str(dt)
    return None


def _upgrade_image_url(src: str) -> str:
    """Request the large variant of an X image CDN URL."""
    return _IMG_NAME_PARAM_RE.sub(r"\g<1>large", src)


def _parse_media(article: Tag, *, exclude: Tag | None = None) -> list[MediaAttachment]:
    """Extract media attachments from a tweet.

    Collects tweet photos (upgraded to the ``name=large`` CDN variant) and
    videos (as poster + video URL). Profile avatars, emoji images, card
    thumbnails, and media belonging to an excluded element (e.g. a quoted
    tweet) are skipped.

    Args:
        article: Tweet article tag.
        exclude: A descendant element whose media must not be attributed
            to this tweet.

    Returns:
        List of media attachments found.
    """
    media: list[MediaAttachment] = []
    seen: set[str] = set()

    def _excluded(el: Tag) -> bool:
        if exclude is None:
            return False
        return el is exclude or any(p is exclude for p in el.parents)

    for img in article.find_all("img", src=True):
        if _excluded(img):
            continue
        src = str(img.get("src", ""))
        alt = re.sub(r"\s+", " ", str(img.get("alt", ""))).strip()
        if "profile_images" in src or "emoji" in src:
            continue
        if not src.startswith("http"):
            continue
        # Skip card thumbnails; the card is rendered as a link instead.
        if _is_inside_card(img):
            continue
        # Skip video poster frames; videos are collected separately below.
        if img.find_parent("video") is not None:
            continue
        src = _upgrade_image_url(src)
        if src in seen:
            continue
        seen.add(src)
        media.append(MediaAttachment(url=src, alt=alt))

    for video in article.find_all("video"):
        if not isinstance(video, Tag) or _excluded(video):
            continue
        poster = str(video.get("poster", "") or "")
        src = str(video.get("src", "") or "")
        if not src:
            source = video.find("source", src=True)
            if isinstance(source, Tag):
                src = str(source.get("src", ""))
        url = src or poster
        if not url or url in seen:
            continue
        seen.add(url)
        media.append(MediaAttachment(url=url, media_type="video", poster=poster))

    return media


def _find_quoted_element(article: Tag) -> Tag | None:
    """Locate the quoted-tweet container inside a tweet article.

    Quoted tweets on X are wrapped in an element whose ``aria-labelledby``
    references generated ``id__*`` ids and which contains its own
    ``User-Name`` block. Falls back to a nested ``<article>`` inside a
    ``card.wrapper`` for older markup.

    Args:
        article: Tweet article tag.

    Returns:
        The quoted-tweet container tag, or None.
    """
    main_text = article.find(True, attrs={"data-testid": "tweetText"})
    for el in article.find_all(True, attrs={"aria-labelledby": True}):
        if not isinstance(el, Tag) or el is article:
            continue
        labelledby = str(el.get("aria-labelledby", ""))
        if "id__" not in labelledby:
            continue
        if el.find(True, attrs={"data-testid": "User-Name"}) is None:
            continue
        # A wrapper around the whole tweet would contain the main tweet
        # text; the quote container never does.
        if isinstance(main_text, Tag) and any(p is el for p in main_text.parents):
            continue
        return el

    card = article.find(True, attrs={"data-testid": "card.wrapper"})
    if isinstance(card, Tag):
        quoted_article = card.find("article", attrs={"data-testid": "tweet"})
        if isinstance(quoted_article, Tag):
            return quoted_article
    return None


def _parse_quoted_item(quoted_el: Tag) -> EmbeddedQuote | None:
    """Parse a quoted-tweet container into an EmbeddedQuote.

    Args:
        quoted_el: The quoted-tweet container tag.

    Returns:
        An EmbeddedQuote, or None if the container has no usable content.
    """
    q_name, q_handle = _parse_author(quoted_el)
    q_text = _parse_text(quoted_el)
    q_timestamp = _parse_timestamp(quoted_el)
    q_media = _parse_media(quoted_el)

    if not (q_name or q_handle or q_text or q_media):
        return None

    return EmbeddedQuote(
        author_name=q_name,
        author_handle=q_handle,
        text=q_text,
        timestamp=q_timestamp,
        media=q_media,
    )


def _parse_card(article: Tag) -> tuple[str | None, str]:
    """Extract a link-preview card from a tweet.

    Args:
        article: Tweet article tag.

    Returns:
        Tuple of (card_url, card_title). ``card_url`` is None when the
        tweet has no card with a link.
    """
    card = article.find(True, attrs={"data-testid": "card.wrapper"})
    if not isinstance(card, Tag):
        return None, ""
    # A card wrapping a quoted tweet is not a link-preview card.
    if card.find("article", attrs={"data-testid": "tweet"}) is not None:
        return None, ""
    link = card.find("a", href=True)
    if not isinstance(link, Tag):
        return None, ""
    href = str(link.get("href", ""))
    if not href:
        return None, ""
    label = str(link.get("aria-label", "") or "")
    title = label.split("\n")[0].strip() if label else ""
    if not title:
        title = link.get_text(" ", strip=True)
    return href, title


def _is_inside_card(tag: Tag) -> bool:
    """Check if a tag is nested inside a card wrapper."""
    for parent in tag.parents:
        if isinstance(parent, Tag) and parent.get("data-testid") == "card.wrapper":
            return True
    return False


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
