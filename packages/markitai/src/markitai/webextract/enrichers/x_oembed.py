"""X/Twitter oEmbed + FxTwitter enricher.

Reference: defuddle's XOembedExtractor — tries FxTwitter API first (full tweet
text with facets, media, quoted tweets, articles), falls back to X's public
oEmbed API (truncates long tweets but always available).

This enricher runs when the sync DOM-based resolver fails to produce
acceptable content for an X/Twitter status URL.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from markitai.webextract.enrichers.base import EnrichmentPolicy
from markitai.webextract.markdown import html_to_markdown, postprocess_markdown
from markitai.webextract.render import render_semantic_content
from markitai.webextract.semantics import (
    ConversationItem,
    ConversationThread,
    EmbeddedQuote,
    MediaAttachment,
)
from markitai.webextract.types import SemanticExtraction

if TYPE_CHECKING:
    from markitai.webextract.resolver import ResolvedPage

# X oEmbed endpoint (public, no auth required for public tweets)
_OEMBED_URL = "https://publish.twitter.com/oembed"
# FxTwitter API (third-party, no auth required)
_FXTWITTER_API = "https://api.fxtwitter.com"

# Matches x.com/twitter.com status and article URLs. Singular "article"
# only — confirmed against defuddle's reference XOembedExtractor/XArticleExtractor
# (github.com/kepano/defuddle), which uses the same `/(status|article)/` pattern
# with no plural form.
_STATUS_RE = re.compile(
    r"https?://(?:www\.)?(?:x\.com|twitter\.com)/([^/]+)/(status|article)/(\d+)"
)
_TIMEOUT = 10.0
_USER_AGENT = "Mozilla/5.0 (compatible; MarkitAI/1.0)"


class XOEmbedEnricher:
    """Enrich X/Twitter pages using FxTwitter API, falling back to oEmbed.

    Mirrors defuddle's async extraction: FxTwitter for full fidelity,
    oEmbed for broad availability.
    """

    name = "x_oembed"

    def should_run(self, url: str, policy: EnrichmentPolicy) -> bool:
        """Return True for X/Twitter status and article URLs when the policy permits it."""
        if not policy.allow_network or not policy.allow_async:
            return False
        is_x = "x.com/" in url or "twitter.com/" in url
        is_status_or_article = "/status/" in url or "/article/" in url
        return bool(is_x and is_status_or_article)

    async def enrich(
        self,
        url: str,
        sync_result: ResolvedPage | None,
    ) -> ResolvedPage | None:
        """Attempt FxTwitter then oEmbed, returning best result."""
        # Strategy 1: FxTwitter API (full fidelity)
        try:
            fx_result = await self._try_fxtwitter(url)
            if fx_result is not None:
                return fx_result
        except Exception as exc:
            logger.debug("[XOEmbedEnricher] FxTwitter failed: {}", exc)

        # Strategy 2: X oEmbed (broadly available, truncated)
        try:
            oembed_result = await self._try_oembed(url)
            if oembed_result is not None:
                return oembed_result
        except Exception as exc:
            logger.debug("[XOEmbedEnricher] oEmbed failed: {}", exc)

        return None

    async def _try_fxtwitter(self, url: str) -> ResolvedPage | None:
        """Fetch via FxTwitter API with retry on transient failures."""
        m = _STATUS_RE.match(url)
        if not m:
            return None

        username, _kind, tweet_id = m.group(1), m.group(2), m.group(3)
        api_url = f"{_FXTWITTER_API}/{username}/status/{tweet_id}"

        import asyncio as _asyncio

        import httpx

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(
                    timeout=_TIMEOUT, follow_redirects=True
                ) as client:
                    resp = await client.get(
                        api_url, headers={"User-Agent": _USER_AGENT}
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    wait_s = 2**attempt * 0.5
                    logger.debug(
                        "[XOEmbedEnricher] FxTwitter attempt {} failed, "
                        "retrying in {:.1f}s: {}",
                        attempt + 1,
                        wait_s,
                        exc,
                    )
                    await _asyncio.sleep(wait_s)
                    continue
                logger.warning(
                    "[XOEmbedEnricher] FxTwitter failed after 3 attempts: {}",
                    last_error,
                )
                return None
            else:
                break

        tweet_data = data.get("tweet")
        if not tweet_data:
            return None

        # Check if this is an article (long-form post)
        article_data = tweet_data.get("article")
        if article_data:
            return self._build_article_result(tweet_data, article_data)

        # Regular tweet
        thread = self._build_thread(tweet_data, tweet_id, url)
        extraction = SemanticExtraction(thread=thread)
        html = render_semantic_content(extraction)

        from markitai.webextract.pipeline import _create_markitdown

        md_instance = _create_markitdown()
        markdown = html_to_markdown(html, md_instance)
        markdown = postprocess_markdown(markdown)

        return self._build_resolved_from_html(html, thread.title, markdown, tweet_data)

    async def _try_oembed(self, url: str) -> ResolvedPage | None:
        """Fetch via X oEmbed API and build a ResolvedPage."""
        import httpx

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                _OEMBED_URL,
                params={"url": url, "omit_script": "true"},
            )
            resp.raise_for_status()
            data = resp.json()

        html_embed: str = data.get("html", "")
        if not html_embed:
            return None

        author_name: str = data.get("author_name", "")
        author_url: str = data.get("author_url", "")
        handle = (
            f"@{author_url.split('/')[-1]}" if author_url and "/" in author_url else ""
        )
        title = f"Post by {handle or author_name or 'Unknown'}"

        return self._build_resolved_from_html(
            html_embed, title, "", {}, author_name=author_name
        )

    def _build_thread(
        self, tweet_data: dict, tweet_id: str, url: str
    ) -> ConversationThread:
        """Convert FxTwitter API tweet JSON to a ConversationThread."""
        author = tweet_data.get("author", {})
        author_name = author.get("name")
        author_handle = (
            f"@{author['screen_name']}" if author.get("screen_name") else None
        )

        # Prefer text over raw_text.text: FxTwitter expands t.co links
        text = tweet_data.get("text", "")
        if not text:
            raw_text = tweet_data.get("raw_text", {})
            if isinstance(raw_text, dict):
                text = raw_text.get("text") or ""

        timestamp = tweet_data.get("created_at")
        media_list = self._build_media(tweet_data)

        # Quoted tweet
        quoted_item: EmbeddedQuote | None = None
        quote_data = tweet_data.get("quote")
        if isinstance(quote_data, dict):
            q_author = quote_data.get("author", {})
            quoted_item = EmbeddedQuote(
                author_name=q_author.get("name"),
                author_handle=(
                    f"@{q_author['screen_name']}"
                    if q_author.get("screen_name")
                    else None
                ),
                text=quote_data.get("text", ""),
                url=quote_data.get("url"),
                timestamp=quote_data.get("created_at"),
                media=self._build_media(quote_data),
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
            show_title_in_body=False,
            show_author_meta=False,
        )

    def _build_media(self, tweet_data: dict) -> list[MediaAttachment]:
        """Convert FxTwitter media JSON to MediaAttachment objects."""
        media_list: list[MediaAttachment] = []
        media_data = tweet_data.get("media", {})
        if not isinstance(media_data, dict):
            return media_list
        for item in media_data.get("all", []):
            media_type = item.get("type", "image")
            if media_type == "photo":
                media_type = "image"
            media_list.append(
                MediaAttachment(
                    url=item.get("url", ""),
                    alt=item.get("altText", ""),
                    media_type=media_type,
                    poster=item.get("thumbnail_url", "") or "",
                )
            )
        return media_list

    def _build_article_result(
        self, tweet_data: dict, article_data: dict
    ) -> ResolvedPage | None:
        """Build result for X article (long-form post) from FxTwitter data.

        Renders Draft.js content blocks into HTML, including:
        - Paragraphs, headers, unordered lists
        - Embedded media (images, videos) from atomic blocks
        - Code blocks from MARKDOWN entities
        - Cover image
        """
        author = tweet_data.get("author", {})
        handle = f"@{author['screen_name']}" if author.get("screen_name") else ""
        title = article_data.get("title", "")
        published = self._to_date_string(
            article_data.get("created_at") or tweet_data.get("created_at")
        )

        content = article_data.get("content", {})
        blocks: list[dict] = content.get("blocks", [])
        entity_map = self._normalize_entity_map(content.get("entityMap", []))
        media_entities: list[dict] = article_data.get("media_entities", [])
        cover_media: dict | None = article_data.get("cover_media")

        html_parts: list[str] = []
        html_parts.append('<article class="x-article">')

        # Cover image
        cover_url = self._get_cover_url(cover_media)
        if cover_url:
            html_parts.append(
                f'<img src="{self._escape(cover_url)}" alt="Cover image">'
            )

        # Render content blocks
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.get("type") == "unordered-list-item":
                items: list[str] = []
                while (
                    i < len(blocks) and blocks[i].get("type") == "unordered-list-item"
                ):
                    items.append(
                        f"<li>{self._render_inline(blocks[i], entity_map)}</li>"
                    )
                    i += 1
                html_parts.append(f"<ul>{''.join(items)}</ul>")
                continue

            rendered = self._render_block(block, entity_map, media_entities)
            if rendered:
                html_parts.append(rendered)
            i += 1

        html_parts.append("</article>")
        content_html = "\n".join(html_parts)

        from markitai.webextract.resolver import ResolvedPage

        metadata_overrides: dict[str, object] = {
            "title": title,
            "author": handle,
            "site": "X (Twitter)",
        }
        if published:
            metadata_overrides["published"] = published

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            diagnostics={"enricher_name": self.name, "source": "fxtwitter_article"},
        )

    # -- Article rendering helpers --

    @staticmethod
    def _escape(text: str) -> str:
        """Minimal HTML escape."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _to_date_string(date_str: str | None) -> str | None:
        """Convert ISO date to YYYY-MM-DD."""
        if not date_str:
            return None
        try:
            from datetime import datetime as dt

            return dt.fromisoformat(date_str.replace("Z", "+00:00")).date().isoformat()
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _get_cover_url(cover_media: dict | None) -> str | None:
        """Extract cover image URL from FxTwitter cover_media."""
        if not cover_media:
            return None
        return cover_media.get("media_info", {}).get("original_img_url")

    @staticmethod
    def _normalize_entity_map(raw: object) -> dict[str, dict]:
        """Convert FxTwitter's ``entityMap`` array to a ``{str(key): value}``
        dict for O(1) lookup instead of a linear scan per entity reference.

        FxTwitter serializes ``entityMap`` as a list of ``{"key", "value"}``
        pairs (confirmed against defuddle's reference ``XOembedExtractor``,
        github.com/kepano/defuddle) — not a plain Draft.js
        ``RawDraftContentState`` object keyed by id, despite that being the
        more commonly documented Draft.js shape.
        """
        if not isinstance(raw, list):
            return {}
        result: dict[str, dict] = {}
        for item in raw:
            if isinstance(item, dict) and "key" in item and "value" in item:
                result[str(item["key"])] = item["value"]
        return result

    def _render_block(
        self,
        block: dict,
        entity_map: dict[str, dict],
        media_entities: list[dict],
    ) -> str:
        """Render a single Draft.js block to HTML."""
        block_type = block.get("type", "")
        text = block.get("text", "")

        if block_type == "unstyled":
            if not text.strip():
                return ""
            return f"<p>{self._render_inline(block, entity_map)}</p>"
        elif block_type == "header-two":
            return f"<h2>{self._render_inline(block, entity_map)}</h2>"
        elif block_type == "header-three":
            return f"<h3>{self._render_inline(block, entity_map)}</h3>"
        elif block_type == "atomic":
            return self._render_atomic(block, entity_map, media_entities)
        else:
            if not text.strip():
                return ""
            return f"<p>{self._render_inline(block, entity_map)}</p>"

    def _render_inline(self, block: dict, entity_map: dict[str, dict]) -> str:
        """Render inline content with bold, links, and mentions."""
        text = block.get("text", "")
        if not text:
            return ""

        # Collect markers: (offset, type, tag) tuples
        markers: list[tuple[int, str, str]] = []

        # Bold ranges
        for rng in block.get("inlineStyleRanges", []):
            if rng.get("style") == "Bold":
                markers.append((rng["offset"], "open", "<strong>"))
                markers.append((rng["offset"] + rng["length"], "close", "</strong>"))

        # Entity ranges (links)
        for rng in block.get("entityRanges", []):
            entity = entity_map.get(str(rng.get("key")))
            if entity and entity.get("type") == "LINK":
                url = entity.get("data", {}).get("url", "")
                if url:
                    escaped_url = self._escape(url)
                    markers.append((rng["offset"], "open", f'<a href="{escaped_url}">'))
                    markers.append((rng["offset"] + rng["length"], "close", "</a>"))

        # Mentions in data
        for mention in block.get("data", {}).get("mentions", []):
            mention_url = f"https://x.com/{self._escape(mention.get('text', ''))}"
            markers.append((mention["fromIndex"], "open", f'<a href="{mention_url}">'))
            markers.append((mention["toIndex"], "close", "</a>"))

        if not markers:
            return self._escape(text)

        # Sort markers: by offset, close before open at same offset
        markers.sort(key=lambda m: (m[0], 0 if m[1] == "close" else 1))

        result: list[str] = []
        pos = 0
        for offset, _kind, tag in markers:
            if offset > pos:
                result.append(self._escape(text[pos:offset]))
            result.append(tag)
            pos = offset
        if pos < len(text):
            result.append(self._escape(text[pos:]))
        return "".join(result)

    def _render_atomic(
        self,
        block: dict,
        entity_map: dict[str, dict],
        media_entities: list[dict],
    ) -> str:
        """Render an atomic block (media, code, etc.)."""
        entity_ranges = block.get("entityRanges", [])
        if not entity_ranges:
            return ""

        entity = entity_map.get(str(entity_ranges[0].get("key")))
        if not entity:
            return ""

        etype = entity.get("type", "")
        data = entity.get("data", {})

        if etype == "MEDIA":
            media_items = data.get("mediaItems", [])
            caption = data.get("caption", "")
            images: list[str] = []

            for item in media_items:
                media_ent = next(
                    (
                        me
                        for me in media_entities
                        if str(me.get("media_id")) == str(item.get("mediaId"))
                    ),
                    None,
                )
                if not media_ent:
                    continue

                info = media_ent.get("media_info", {})
                typename = info.get("__typename", "")

                if typename == "ApiImage" and info.get("original_img_url"):
                    url = info["original_img_url"]
                    alt = self._escape(caption) if caption else ""
                    images.append(f'<img src="{self._escape(url)}" alt="{alt}">')
                elif typename == "ApiVideo":
                    variants = sorted(
                        [
                            v
                            for v in info.get("variants", [])
                            if v.get("content_type") == "video/mp4"
                            and v.get("bit_rate")
                        ],
                        key=lambda v: v.get("bit_rate", 0),
                        reverse=True,
                    )
                    poster = info.get("preview_image", {}).get("original_img_url", "")
                    video_url = variants[0]["url"] if variants else ""
                    if video_url:
                        images.append(
                            f'<video src="{self._escape(video_url)}" '
                            f'poster="{self._escape(poster)}" controls></video>'
                        )
                    elif poster:
                        images.append(
                            f'<img src="{self._escape(poster)}" '
                            f'alt="{self._escape(caption) if caption else ""}">'
                        )

            if images:
                if caption:
                    return f"<figure>{''.join(images)}<figcaption>{self._escape(caption)}</figcaption></figure>"
                return "\n".join(f"<figure>{img}</figure>" for img in images)
            elif caption:
                return (
                    f"<figure><figcaption>{self._escape(caption)}</figcaption></figure>"
                )
            return ""

        elif etype == "MARKDOWN":
            md = data.get("markdown", "")
            import re as _re

            code_match = _re.match(r"^```(\w*)\n([\s\S]*?)\n?```$", md)
            if code_match:
                lang = code_match.group(1)
                code = code_match.group(2)
                lang_attr = (
                    f' class="language-{self._escape(lang)}" data-lang="{self._escape(lang)}"'
                    if lang
                    else ""
                )
                return f"<pre><code{lang_attr}>{self._escape(code)}</code></pre>"
            return f"<pre><code>{self._escape(md)}</code></pre>"

        return ""

    def _build_resolved_from_html(
        self,
        html: str,
        title: str,
        markdown: str,
        tweet_data: dict | None,
        author_name: str = "",
    ) -> ResolvedPage | None:
        """Build a ResolvedPage from HTML content."""
        from markitai.webextract.resolver import ResolvedPage

        if not html:
            return None

        metadata_overrides: dict[str, object] = {"site": "X (Twitter)"}
        if title:
            metadata_overrides["title"] = title
        if author_name:
            metadata_overrides["author"] = author_name
        elif tweet_data and tweet_data.get("author", {}).get("screen_name"):
            metadata_overrides["author"] = f"@{tweet_data['author']['screen_name']}"

        return ResolvedPage(
            content_html=html,
            metadata_overrides=metadata_overrides,
            diagnostics={
                "enricher_name": self.name,
                "source": "fxtwitter" if tweet_data else "oembed",
            },
        )
