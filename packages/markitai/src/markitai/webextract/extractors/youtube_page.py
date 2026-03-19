"""Extractor for YouTube video watch pages (Playwright-rendered HTML).

YouTube video pages are rich media pages, NOT conversation threads.
This extractor sets content_profile=RICH_MEDIA_PAGE and semantic=None.
Comments are explicitly excluded from the thread model.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.types import ContentProfile


class YouTubePageExtractor:
    """Extract YouTube video page metadata and description.

    Targets youtube.com/watch and youtu.be URLs rendered by Playwright.
    Produces a simple HTML representation of the video title, channel,
    and description — without treating YouTube comments as a thread model.

    YouTube HTML structure targeted:
    - Title: ``<h1 class="ytd-watch-metadata">`` or ``<meta property="og:title">``
    - Channel: ``<yt-formatted-string id="channel-name">`` or ``<meta property="og:site_name">``
    - Description: ``<yt-attributed-string>`` in ``#description-inner``
      or ``<meta property="og:description">``
    """

    name = "youtube_page"

    def matches_url(self, url: str) -> bool:
        """Match YouTube video watch URLs.

        Args:
            url: Source URL to test.

        Returns:
            True when the URL is a YouTube watch or short URL.
        """
        return ("youtube.com/watch" in url and "v=" in url) or "youtu.be/" in url

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the primary content area (legacy protocol compliance path).

        The resolver path via ``resolve()`` is preferred and produces
        richer, more structured output.

        Args:
            soup: Parsed HTML of the page.

        Returns:
            The primary content div, or ``None``.
        """
        primary = soup.find(id="primary")
        return primary if isinstance(primary, Tag) else None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a YouTube watch page into a structured ResolvedPage.

        Extracts video title, channel name, and description, then builds a
        minimal HTML representation.  Sets ``semantic=None`` to indicate this
        is NOT a threaded conversation — comments are not modelled as a thread.

        Args:
            soup: Parsed HTML of the page.
            url: Source URL.

        Returns:
            A ``ResolvedPage`` with metadata overrides and no thread semantic.
        """
        title = _extract_title(soup)
        channel = _extract_channel(soup)
        description = _extract_description(soup)

        content_html = _build_content_html(title, channel, description, url)

        metadata_overrides: dict[str, object] = {
            "site": "YouTube",
        }
        if title:
            metadata_overrides["title"] = title
        if channel:
            metadata_overrides["author"] = channel
        if description:
            metadata_overrides["description"] = description[:500]

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            semantic=None,
            diagnostics={
                "youtube_resolve": "success",
                "content_profile": ContentProfile.RICH_MEDIA_PAGE.value,
                "extractor_name": self.name,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract the video title from the page.

    Tries multiple selectors in priority order.

    Args:
        soup: Parsed page HTML.

    Returns:
        The video title string, or ``None`` if not found.
    """
    # Strategy 1: ytd-watch-metadata h1
    h1 = soup.find("h1", class_="ytd-watch-metadata")
    if isinstance(h1, Tag):
        text = h1.get_text(strip=True)
        if text:
            return text

    # Strategy 2: any h1 inside ytd-watch-metadata
    watch_meta = soup.find("ytd-watch-metadata")
    if isinstance(watch_meta, Tag):
        h1_inner = watch_meta.find("h1")
        if isinstance(h1_inner, Tag):
            text = h1_inner.get_text(strip=True)
            if text:
                return text

    # Strategy 3: og:title meta tag
    og_title = soup.find("meta", property="og:title")
    if isinstance(og_title, Tag):
        content = og_title.get("content")
        if isinstance(content, str) and content:
            return content

    # Strategy 4: page <title> (strip " - YouTube" suffix)
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        text = title_tag.get_text(strip=True)
        if text:
            suffix = " - YouTube"
            if text.endswith(suffix):
                text = text[: -len(suffix)]
            return text

    return None


def _extract_channel(soup: BeautifulSoup) -> str | None:
    """Extract the channel name from the page.

    Args:
        soup: Parsed page HTML.

    Returns:
        The channel name string, or ``None`` if not found.
    """
    # Strategy 1: yt-formatted-string#channel-name
    channel_el = soup.find(id="channel-name")
    if isinstance(channel_el, Tag):
        # Prefer text of an <a> link inside it (channel name link)
        link = channel_el.find("a")
        if isinstance(link, Tag):
            text = link.get_text(strip=True)
            if text:
                return text
        text = channel_el.get_text(strip=True)
        if text:
            return text

    # Strategy 2: og:site_name is "YouTube" — not useful for channel name
    # Skip og:site_name since it returns "YouTube" not the channel

    return None


def _extract_description(soup: BeautifulSoup) -> str | None:
    """Extract the video description from the page.

    Args:
        soup: Parsed page HTML.

    Returns:
        The description string, or ``None`` if not found.
    """
    # Strategy 1: #description-inner yt-attributed-string
    desc_inner = soup.find(id="description-inner")
    if isinstance(desc_inner, Tag):
        yt_attr = desc_inner.find("yt-attributed-string")
        if isinstance(yt_attr, Tag):
            text = yt_attr.get_text(separator="\n", strip=True)
            if text:
                return text
        text = desc_inner.get_text(separator="\n", strip=True)
        if text:
            return text

    # Strategy 2: og:description meta tag
    og_desc = soup.find("meta", property="og:description")
    if isinstance(og_desc, Tag):
        content = og_desc.get("content")
        if isinstance(content, str) and content:
            return content

    return None


def _build_content_html(
    title: str | None,
    channel: str | None,
    description: str | None,
    url: str,
) -> str:
    """Build a minimal HTML representation of the video page.

    Args:
        title: Video title.
        channel: Channel name.
        description: Video description.
        url: Source URL.

    Returns:
        HTML string representing the key video content.
    """
    parts: list[str] = []

    if title:
        parts.append(f"<h1>{title}</h1>")

    if channel:
        parts.append(f"<p><strong>Channel:</strong> {channel}</p>")

    parts.append(f'<p><a href="{url}">Watch on YouTube</a></p>')

    if description:
        # Preserve line breaks in description as paragraphs
        paras = [p.strip() for p in description.split("\n\n") if p.strip()]
        if paras:
            parts.append("<div>")
            for para in paras:
                # Replace single newlines with <br>
                para_html = para.replace("\n", "<br>")
                parts.append(f"<p>{para_html}</p>")
            parts.append("</div>")

    return "\n".join(parts)
