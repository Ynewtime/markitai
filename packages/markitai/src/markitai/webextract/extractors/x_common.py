"""Shared X/Twitter HTML parsing utilities.

Reusable functions for extracting structured data from X.com rendered HTML.
These are consumed by ``XTweetExtractor.resolve()`` and potentially other
X-related extractors.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime

from bs4 import Tag
from bs4.element import NavigableString

from markitai.webextract.semantics import (
    ConversationItem,
    EmbeddedQuote,
    MediaAttachment,
)

_HANDLE_RE = re.compile(r"(@\w+)")

# Upgrade X image CDN size parameter, e.g. "...&name=small" -> "...&name=orig"
_IMG_NAME_PARAM_RE = re.compile(r"([?&]name=)\w+")

# 2026 X.com DOM: permalink text like "6:02 AM · Jul 4, 2026"
_NEW_FULL_TIME_FORMAT = "%I:%M %p · %b %d, %Y"
# 2026 X.com DOM: quoted-tweet date text like "Jun 19" or "Jun 19, 2025"
_NEW_SHORT_DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}(?:, \d{4})?$")


def parse_tweet_article(article: Tag, *, tweet_id: str = "") -> ConversationItem:
    """Parse a single ``<article data-testid="tweet">`` into a ConversationItem.

    Args:
        article: The ``<article>`` tag to parse.
        tweet_id: Stable ID for the item; defaults to empty string.

    Returns:
        A populated ``ConversationItem``.
    """
    if is_new_dom_article(article):
        return _parse_tweet_article_new(article, tweet_id=tweet_id)

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

    Supports both DOM generations:

    - Legacy DOM: ``<article data-testid="tweet">``
    - 2026 DOM: ``<article data-tweet-id="...">``

    The primary tweet is the first matching ``<article>`` that is NOT inside
    a ``data-testid="card.wrapper"`` (quote) or a ``<section>`` (replies).

    Args:
        primary_column: The tweet page content container.

    Returns:
        The main tweet ``<article>`` tag, or ``None`` if not found.
    """
    for article in primary_column.find_all("article"):
        if not isinstance(article, Tag):
            continue
        if article.get("data-testid") != "tweet" and not article.has_attr(
            "data-tweet-id"
        ):
            continue
        if _is_inside_quote_card(article):
            continue
        if _is_inside_reply_section(article):
            continue
        return article
    return None


def is_new_dom_article(article: Tag) -> bool:
    """Check whether a tweet article uses the 2026 X.com DOM shape.

    The 2026 redesign dropped all ``data-testid`` attributes; tweet articles
    carry a ``data-tweet-id`` attribute instead.

    Args:
        article: A tweet ``<article>`` tag.

    Returns:
        True for 2026-DOM articles.
    """
    return article.has_attr("data-tweet-id")


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
    """Request the original-quality variant of an X image CDN URL.

    Upgrades ``name=medium`` → ``name=orig``, strips ``format=webp``,
    and ensures a ``.jpg`` extension is present so the CDN returns a
    loadable Content-Type (matching defuddle's fxtwitter URL format).
    """
    # Upgrade name=small|medium|large → name=orig
    src = _IMG_NAME_PARAM_RE.sub(r"\g<1>orig", src)
    # Strip format=webp while preserving query-string structure
    src = re.sub(r"&format=webp", "", src)
    src = re.sub(r"\?format=webp&", "?", src)
    src = re.sub(r"\?format=webp$", "", src)
    # Ensure .jpg extension for correct CDN Content-Type
    if not re.search(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", src):
        src = re.sub(r"(\?(.+))", r".jpg\1", src) if "?" in src else src + ".jpg"
    return src


def _parse_media(article: Tag, *, exclude: Tag | None = None) -> list[MediaAttachment]:
    """Extract media attachments from a tweet.

    Collects tweet photos (upgraded to the ``name=orig`` CDN variant) and
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

    # Collect videos first so their poster URLs can be used to skip the
    # poster-overlay <img> duplicates the 2026 DOM renders next to <video>.
    videos: list[MediaAttachment] = []
    posters: set[str] = set()
    for video in article.find_all("video"):
        if not isinstance(video, Tag) or _excluded(video):
            continue
        poster = str(video.get("poster", "") or "")
        src = str(video.get("src", "") or "")
        if not src:
            source = video.find("source", src=True)
            if isinstance(source, Tag):
                src = str(source.get("src", ""))
        # blob: URLs are session-local MediaSource handles — useless outside
        # the browser. Keep the poster and drop the unusable video URL.
        if src.startswith("blob:"):
            src = ""
        if poster:
            posters.add(poster)
        key = src or poster
        if not key or key in seen:
            continue
        seen.add(key)
        videos.append(MediaAttachment(url=src, media_type="video", poster=poster))

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
        # Skip video poster frames; videos are collected separately.
        if img.find_parent("video") is not None:
            continue
        if src in posters:
            continue
        src = _upgrade_image_url(src)
        if src in seen:
            continue
        seen.add(src)
        media.append(MediaAttachment(url=src, alt=alt))

    media.extend(videos)
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


_QUOTE_TRUNCATION_MARKERS = {
    "...",
    "…",
    "show more",
    "显示更多",
    "顯示更多",
    "さらに表示",
    "더 보기",
    "voir plus",
    "mehr anzeigen",
    "mostrar más",
    "mostrar mais",
    "mostra altro",
    "показать ещё",
}


def _mark_truncated_quote(text: str, quoted_el: Tag) -> str:
    """Append an explicit marker when X exposes a clipped quoted post."""
    if not text or text.rstrip().endswith(("...", "…")):
        return text
    for element in quoted_el.find_all(["button", "span"]):
        marker = " ".join(element.get_text(" ", strip=True).casefold().split())
        if marker in _QUOTE_TRUNCATION_MARKERS:
            return f"{text.rstrip()}..."
    return text


def _parse_quoted_item(quoted_el: Tag) -> EmbeddedQuote | None:
    """Parse a quoted-tweet container into an EmbeddedQuote.

    Args:
        quoted_el: The quoted-tweet container tag.

    Returns:
        An EmbeddedQuote, or None if the container has no usable content.
    """
    q_name, q_handle = _parse_author(quoted_el)
    q_text = _mark_truncated_quote(_parse_text(quoted_el), quoted_el)
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


# ---------------------------------------------------------------------------
# 2026 X.com DOM (no data-testid attributes)
# ---------------------------------------------------------------------------


def _parse_tweet_article_new(article: Tag, *, tweet_id: str = "") -> ConversationItem:
    """Parse a 2026-DOM ``<article data-tweet-id>`` into a ConversationItem.

    Args:
        article: The ``<article>`` tag to parse.
        tweet_id: Stable ID for the item; defaults to the article's
            ``data-tweet-id`` when empty.

    Returns:
        A populated ``ConversationItem``.
    """
    if not tweet_id:
        tweet_id = str(article.get("data-tweet-id", "") or "")

    quoted_el = _find_quoted_element_new(article)
    author_name, author_handle = _parse_author_new(article, exclude=quoted_el)
    text = _parse_text_new(article, exclude=quoted_el)
    timestamp = _parse_timestamp_new(article, exclude=quoted_el)
    media = _parse_media(article, exclude=quoted_el)
    quoted = _parse_quoted_item_new(quoted_el) if quoted_el is not None else None

    return ConversationItem(
        id=tweet_id,
        author_name=author_name,
        author_handle=author_handle,
        text=text,
        timestamp=timestamp,
        quoted_item=quoted,
        media=media,
    )


def _new_excluded(el: Tag, exclude: Tag | None) -> bool:
    """Check whether ``el`` is (inside) the excluded subtree."""
    if exclude is None:
        return False
    return el is exclude or any(p is exclude for p in el.parents)


def _parse_author_new(
    scope: Tag, *, exclude: Tag | None = None
) -> tuple[str | None, str | None]:
    """Extract display name and @handle from a 2026-DOM author block.

    The 2026 DOM has two variants:

    1. Slot-based (older 2026): ``<div data-slot="hover-card-trigger">``
       containing ``<a>`` links for name and handle.

    2. Link-only (current): plain ``<a href="https://x.com/user">`` links
       scattered in the article — the display name (no @) and the @handle.
       Both point to the same user profile URL (no ``/status/`` in href).

    Strategy: first try slot-based, then fall back to link-based.

    Args:
        scope: Tweet article or quoted-tweet container.
        exclude: Subtree (e.g. the quoted tweet) to ignore.

    Returns:
        Tuple of (display_name, handle). Either may be None.
    """
    # Strategy 1: slot-based (data-slot="hover-card-trigger")
    for block in scope.find_all("div", attrs={"data-slot": "hover-card-trigger"}):
        if not isinstance(block, Tag) or _new_excluded(block, exclude):
            continue
        display_name: str | None = None
        handle: str | None = None
        for link in block.find_all("a"):
            text = link.get_text(strip=True)
            if not text:
                continue
            if text.startswith("@"):
                if handle is None:
                    handle = text
            elif display_name is None:
                display_name = text
        if display_name or handle:
            return display_name, handle

    # Strategy 2: link-based — find profile links (no /status/ in href)
    display_name: str | None = None
    handle: str | None = None
    for link in scope.find_all("a", href=True):
        if isinstance(link, Tag) and _new_excluded(link, exclude):
            continue
        href = str(link.get("href", ""))
        # Profile links point to user page, not status page
        if "/status/" in href:
            continue
        text = link.get_text(strip=True)
        if not text:
            continue
        if text.startswith("@"):
            if handle is None:
                handle = text
        else:
            if display_name is None:
                display_name = text
    return display_name, handle


def _parse_text_new(scope: Tag, *, exclude: Tag | None = None) -> str:
    """Extract tweet text from a 2026-DOM tweet.

    The tweet body is the first ``<div dir="auto">`` in the scope.  For
    X *Articles* (long-form posts), the first div is empty (the repurposed
    tweet-text slot) and the real article body lives in the *last*
    ``<div dir="auto">``.  We detect this case and fall back accordingly.

    Line breaks are literal ``\\n`` characters inside text nodes (the
    element is styled ``whitespace-pre-wrap``).  Truncated external links
    (anchor text ending in ``\u2026``) are expanded to their full ``href``;
    "Show more" affordances are dropped.

    Args:
        scope: Tweet article or quoted-tweet container.
        exclude: Subtree (e.g. the quoted tweet) to ignore.

    Returns:
        Plain text content with ``\\n`` paragraph breaks.
    """
    text_els: list[Tag] = []
    for div in scope.find_all("div", dir="auto"):
        if not isinstance(div, Tag) or _new_excluded(div, exclude):
            continue
        text_els.append(div)

    if not text_els:
        return ""

    # Article detection: first div empty AND 3+ divs → use the last one
    if len(text_els) >= 3 and not text_els[0].get_text(strip=True):
        text_el = text_els[-1]
    else:
        text_el = text_els[0]

    clone = copy.copy(text_el)
    for el in clone.find_all(["a", "button", "span"]):
        if el.get_text(strip=True) == "Show more":
            el.decompose()
    for img in clone.find_all("img"):
        alt = img.get("alt", "")
        img.replace_with(str(alt) if alt else "")
    # Expand truncated display URLs to the full target URL.
    for link in clone.find_all("a", href=True):
        display = link.get_text(strip=True)
        href = str(link.get("href", ""))
        if display.endswith("…") and href.startswith("http"):
            link.replace_with(href)
    # Whitespace-only text nodes are HTML formatting, not tweet line breaks.
    for node in clone.find_all(string=True):
        if isinstance(node, NavigableString) and not node.strip() and "\n" in node:
            node.replace_with(" ")

    text = clone.get_text("")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _parse_timestamp_new(scope: Tag, *, exclude: Tag | None = None) -> str | None:
    """Extract the timestamp from a 2026-DOM tweet's permalink text.

    The 2026 DOM has no ``<time>`` element; the timestamp is the visible
    text of the status permalink, e.g. ``"6:02 AM · Jul 4, 2026"`` on the
    main tweet or ``"Jun 19"`` on a quoted tweet. Parseable values are
    normalized to ISO-8601; short quote dates are returned verbatim.

    Args:
        scope: Tweet article or quoted-tweet container.
        exclude: Subtree (e.g. the quoted tweet) to ignore.

    Returns:
        Timestamp string, or None if not found.
    """
    for link in scope.find_all("a", href=True):
        if not isinstance(link, Tag) or _new_excluded(link, exclude):
            continue
        if "/status/" not in str(link.get("href", "")):
            continue
        text = link.get_text(" ", strip=True)
        if not text:
            continue
        try:
            return datetime.strptime(text, _NEW_FULL_TIME_FORMAT).isoformat()
        except ValueError:
            pass
        if _NEW_SHORT_DATE_RE.match(text):
            try:
                return datetime.strptime(text, "%b %d, %Y").date().isoformat()
            except ValueError:
                return text
    return None


def _find_quoted_element_new(article: Tag) -> Tag | None:
    """Locate the quoted-tweet container in a 2026-DOM tweet article.

    Quoted tweets render as ``<div role="link" data-href="/user/status/id">``
    embedded cards.

    Args:
        article: Tweet article tag.

    Returns:
        The quoted-tweet container tag, or None.
    """
    for el in article.find_all("div", attrs={"role": "link", "data-href": True}):
        if not isinstance(el, Tag):
            continue
        if "/status/" in str(el.get("data-href", "")):
            return el
    return None


def _parse_quoted_item_new(quoted_el: Tag) -> EmbeddedQuote | None:
    """Parse a 2026-DOM quoted-tweet container into an EmbeddedQuote.

    Args:
        quoted_el: The ``div[role="link"][data-href]`` container.

    Returns:
        An EmbeddedQuote, or None if the container has no usable content.
    """
    q_name, q_handle = _parse_author_new(quoted_el)
    q_text = _mark_truncated_quote(_parse_text_new(quoted_el), quoted_el)
    q_timestamp = _parse_timestamp_new(quoted_el)
    q_media = _parse_media(quoted_el)

    if not (q_name or q_handle or q_text or q_media):
        return None

    href = str(quoted_el.get("data-href", "") or "")
    url: str | None = None
    if href.startswith("http"):
        url = href
    elif href.startswith("/"):
        url = f"https://x.com{href}"

    return EmbeddedQuote(
        author_name=q_name,
        author_handle=q_handle,
        text=q_text,
        url=url,
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


def _extract_article_title(article: Tag) -> str | None:
    """Extract the article title from a 2026-DOM tweet article.

    For X Articles, the second ``<div dir="auto">`` holds the article title
    (the first is always empty — it's the repurposed tweet-text slot).
    Returns ``None`` when the article doesn't match this pattern.

    Args:
        article: The tweet article tag.

    Returns:
        Article title string, or ``None`` if not an article.
    """
    divs = [d for d in article.find_all("div", dir="auto") if isinstance(d, Tag)]
    # Article pattern: first div empty, second div has the title
    if len(divs) >= 3 and not divs[0].get_text(strip=True):
        title = divs[1].get_text(strip=True)
        if title:
            return title
    return None
