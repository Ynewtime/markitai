"""FxTwitter API client for fetching tweet data.

Fetches tweet content via the FxTwitter API and converts it to markitai's
semantic ConversationThread structure, then renders to markdown.
"""

from __future__ import annotations

import re

import httpx
from loguru import logger

from markitai.fetch_types import FetchResult
from markitai.webextract.markdown import html_to_markdown, postprocess_markdown
from markitai.webextract.render import render_semantic_content
from markitai.webextract.semantics import (
    ConversationItem,
    ConversationThread,
    EmbeddedQuote,
    MediaAttachment,
)
from markitai.webextract.types import SemanticExtraction

_TWITTER_STATUS_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/([^/]+)/status/(\d+)"
)

_FXTWITTER_API = "https://api.fxtwitter.com"
_TIMEOUT = 10
_USER_AGENT = "Mozilla/5.0 (compatible; MarkitAI/1.0)"


def _extract_twitter_url_parts(url: str) -> tuple[str, str] | None:
    """Extract (username, tweet_id) from a Twitter/X status URL.

    Returns None if the URL does not match the expected pattern.
    """
    m = _TWITTER_STATUS_RE.match(url)
    if not m:
        return None
    return m.group(1), m.group(2)


def _build_conversation_thread(
    tweet_data: dict,
    tweet_id: str,
    url: str = "",  # noqa: ARG001
) -> ConversationThread:
    """Convert FxTwitter API tweet JSON to a ConversationThread."""
    author = tweet_data.get("author", {})
    author_name = author.get("name")
    author_handle = f"@{author['screen_name']}" if author.get("screen_name") else None

    # Prefer raw_text.text over text
    raw_text = tweet_data.get("raw_text", {})
    text = raw_text.get("text") if isinstance(raw_text, dict) else None
    if not text:
        text = tweet_data.get("text", "")

    timestamp = tweet_data.get("created_at")

    # Media
    media_list: list[MediaAttachment] = []
    media_data = tweet_data.get("media", {})
    if isinstance(media_data, dict):
        for item in media_data.get("all", []):
            media_type = item.get("type", "image")
            # Normalize fxtwitter types to our types
            if media_type == "photo":
                media_type = "image"
            media_list.append(
                MediaAttachment(
                    url=item.get("url", ""),
                    alt=item.get("altText", ""),
                    media_type=media_type,
                )
            )

    # Quoted tweet
    quoted_item: EmbeddedQuote | None = None
    quote_data = tweet_data.get("quote")
    if isinstance(quote_data, dict):
        q_author = quote_data.get("author", {})
        quoted_item = EmbeddedQuote(
            author_name=q_author.get("name"),
            author_handle=f"@{q_author['screen_name']}"
            if q_author.get("screen_name")
            else None,
            text=quote_data.get("text", ""),
        )

    main_item = ConversationItem(
        id=tweet_id,
        author_name=author_name,
        author_handle=author_handle,
        text=text,
        timestamp=timestamp,
        quoted_item=quoted_item,
        media=media_list,
    )

    title = f"Post by {author_handle or author_name or 'Unknown'}"

    return ConversationThread(
        title=title,
        main_item=main_item,
    )


async def fetch_with_fxtwitter(url: str) -> FetchResult | None:
    """Fetch tweet content via the FxTwitter API.

    Returns a FetchResult with rendered markdown content, or None if the URL
    is not a Twitter/X status URL or if the API request fails.
    """
    parts = _extract_twitter_url_parts(url)
    if parts is None:
        return None

    username, tweet_id = parts
    api_url = f"{_FXTWITTER_API}/{username}/status/{tweet_id}"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(
                api_url,
                headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
    except Exception:
        logger.opt(exception=True).debug(f"FxTwitter API request failed for {url}")
        return None

    tweet_data = data.get("tweet")
    if not tweet_data:
        logger.debug(f"FxTwitter response missing tweet data for {url}")
        return None

    thread = _build_conversation_thread(tweet_data, tweet_id)
    extraction = SemanticExtraction(thread=thread)
    html = render_semantic_content(extraction)

    # Use the pipeline's MarkItDown instance (with custom converter)
    from markitai.webextract.pipeline import _create_markitdown

    md_instance = _create_markitdown()
    markdown = html_to_markdown(html, md_instance)
    markdown = postprocess_markdown(markdown)

    return FetchResult(
        content=markdown,
        strategy_used="fxtwitter",
        title=thread.title,
        url=url,
        final_url=url,
    )
