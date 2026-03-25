"""Extractor for Steam store news/announcement pages.

Steam embeds announcement content as JSON with BBCode markup inside
``data-partnereventstore`` attributes on ``#application_config``.
This extractor parses the JSON, converts BBCode to HTML, and extracts
metadata (title, author, published date).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from html import escape

from bs4 import BeautifulSoup, Tag

from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.types import ContentProfile


class SteamNewsExtractor:
    """Extract Steam store news/announcement pages.

    Targets store.steampowered.com/news/ URLs where the actual content
    lives inside a ``data-partnereventstore`` JSON attribute using BBCode.
    """

    name = "steam_news"

    def matches_url(self, url: str) -> bool:
        """Match Steam store news URLs."""
        return bool(
            re.search(r"store\.steampowered\.com/news/", url)
            or re.search(r"/news/app/\d+/view/\d+", url)
        )

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the application config div (legacy protocol compliance)."""
        el = soup.find(id="application_config")
        return el if isinstance(el, Tag) else None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a Steam news page into structured content.

        Extracts announcement data from JSON data attributes and converts
        BBCode body to HTML.
        """
        config_el = soup.find(id="application_config")

        announcement = None
        group_name = None

        if isinstance(config_el, Tag):
            announcement = _parse_event_store(config_el)
            group_name = _parse_group_name(config_el)

        if announcement is None:
            return ResolvedPage(
                content_html="",
                diagnostics={"steam_resolve": "no_announcement_data"},
            )

        body_bbcode = announcement.get("announcement_body", {}).get("body", "")
        headline = announcement.get("announcement_body", {}).get("headline", "")
        posttime = announcement.get("announcement_body", {}).get("posttime")

        content_html = _bbcode_to_html(body_bbcode)

        metadata_overrides: dict[str, object] = {}
        if headline:
            metadata_overrides["title"] = headline
        if group_name:
            metadata_overrides["author"] = group_name
        if posttime:
            metadata_overrides["published"] = datetime.fromtimestamp(
                posttime, tz=UTC
            ).isoformat()

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            diagnostics={
                "steam_resolve": "success",
                "content_profile": ContentProfile.GENERIC_ARTICLE.value,
                "extractor_name": self.name,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_event_store(el: Tag) -> dict | None:
    """Parse the first announcement from data-partnereventstore JSON."""
    raw = el.get("data-partnereventstore")
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            return data[0]
    except (json.JSONDecodeError, IndexError):
        pass
    return None


def _parse_group_name(el: Tag) -> str | None:
    """Extract group/app name from data-groupvanityinfo JSON."""
    raw = el.get("data-groupvanityinfo")
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            return data[0].get("group_name")
    except (json.JSONDecodeError, IndexError, AttributeError):
        pass
    return None


def _bbcode_to_html(bbcode: str) -> str:
    """Convert Steam BBCode markup to HTML.

    Handles the most common Steam BBCode tags:
    - [p]...[/p] -> <p>...</p>
    - [b]...[/b] -> <strong>...</strong>
    - [i]...[/i] -> <em>...</em>
    - [u]...[/u] -> <u>...</u>
    - [url=href]text[/url] -> <a href="href">text</a>
    - [url]href[/url] -> <a href="href">href</a>
    - [img]src[/img] -> <img src="src">
    - [h1]...[/h1] through [h3] -> <h1>...</h1>
    - [list]...[/list] -> <ul>...</ul>
    - [*]item -> <li>item</li>
    - [previewyoutube=id;opts][/previewyoutube] -> <img> with YouTube URL
    - [code]...[/code] -> <code>...</code>
    """
    text = bbcode

    # Unescape JSON-escaped sequences
    text = text.replace("\\/", "/")
    text = text.replace('\\"', '"')

    # [p]...[/p]
    text = re.sub(r"\[p\]", "<p>", text, flags=re.IGNORECASE)
    text = re.sub(r"\[/p\]", "</p>", text, flags=re.IGNORECASE)

    # Headings
    for level in range(1, 4):
        text = re.sub(
            rf"\[h{level}\](.*?)\[/h{level}\]",
            rf"<h{level}>\1</h{level}>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    # Bold, italic, underline, strikethrough
    text = re.sub(
        r"\[b\](.*?)\[/b\]", r"<strong>\1</strong>", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"\[i\](.*?)\[/i\]", r"<em>\1</em>", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"\[u\](.*?)\[/u\]", r"<u>\1</u>", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"\[s\](.*?)\[/s\]", r"<del>\1</del>", text, flags=re.IGNORECASE | re.DOTALL
    )

    # [url=href]text[/url]
    text = re.sub(
        r'\[url="?(.*?)"?\](.*?)\[/url\]',
        lambda m: f'<a href="{escape(m.group(1))}">{m.group(2)}</a>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # [url]href[/url]
    text = re.sub(
        r"\[url\](.*?)\[/url\]",
        lambda m: f'<a href="{escape(m.group(1))}">{escape(m.group(1))}</a>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # [img]src[/img]
    text = re.sub(
        r"\[img\](.*?)\[/img\]",
        lambda m: f'<img src="{escape(m.group(1))}">',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # [previewyoutube=id;opts][/previewyoutube]
    text = re.sub(
        r'\[previewyoutube="?([^;"\]]+)[^"]*"?\]\[/previewyoutube\]',
        lambda m: f'<img src="https://www.youtube.com/watch?v={escape(m.group(1))}">',
        text,
        flags=re.IGNORECASE,
    )

    # [list]...[/list]
    text = re.sub(r"\[list\]", "<ul>", text, flags=re.IGNORECASE)
    text = re.sub(r"\[/list\]", "</ul>", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\*\]", "<li>", text, flags=re.IGNORECASE)

    # [code]...[/code]
    text = re.sub(
        r"\[code\](.*?)\[/code\]", r"<code>\1</code>", text, flags=re.IGNORECASE | re.DOTALL
    )

    # Clean up remaining BBCode tags we don't handle
    text = re.sub(r"\[/?[a-zA-Z][^\]]*\]", "", text)

    # Convert newlines to <br> within paragraphs
    text = text.replace("\n\n", "</p><p>")
    text = text.replace("\n", "<br>")

    return text
