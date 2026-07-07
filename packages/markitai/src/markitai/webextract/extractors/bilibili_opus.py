"""Extractor for Bilibili opus (专栏/动态) posts.

Bilibili renders opus posts as a card of typed modules
(``opus-module-title``, ``opus-module-author``, ``opus-module-content``,
``opus-module-copyright``, ``opus-module-bottom``) inside a
``.bili-opus-view`` container.  The container's *siblings* — a comments tab
pane, a right-hand stat/share sidebar, a back-to-top button, a coin-donation
popup — hold all the page chrome (login prompts, like/comment/coin counts,
share links) that generic extraction otherwise pulls into the output
verbatim, since markitai has no Bilibili-specific noise selectors.

This extractor scopes to ``.bili-opus-view`` and, within it, keeps only the
title/author/content modules — dropping ``opus-module-copyright`` (an
internal content-ID tag, e.g. ``cv41273131``) and ``opus-module-bottom``
(share widget + feedback link), which are real modules of the same card but
carry no reader-facing content.
"""

from __future__ import annotations

import re
from html import escape

from bs4 import BeautifulSoup, Tag

from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.types import ContentProfile

# Bilibili opus/dynamic post URLs: bilibili.com/opus/<id>
_OPUS_URL_RE = re.compile(r"bilibili\.com/opus/\d+")

# "编辑于 2026年06月10日 19:48" or "发布于 2026年06月10日 19:48" -> 2026-06-10
_PUB_TIME_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


class BilibiliOpusExtractor:
    """Extract Bilibili opus (专栏/动态) posts from the rendered DOM."""

    name = "bilibili_opus"

    def matches_url(self, url: str) -> bool:
        """Match Bilibili opus post URLs."""
        return bool(_OPUS_URL_RE.search(url))

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the content module (legacy protocol compliance path).

        The resolver path via ``resolve()`` is preferred and produces
        richer, more structured output.
        """
        content = soup.find(class_="opus-module-content")
        return content if isinstance(content, Tag) else None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a Bilibili opus page into a structured ResolvedPage."""
        view = soup.find(class_="bili-opus-view")
        content_module = view.find(class_="opus-module-content") if isinstance(view, Tag) else None

        if not isinstance(content_module, Tag):
            return ResolvedPage(
                content_html="",
                diagnostics={"bilibili_opus_resolve": "no_content_module"},
            )

        title = _extract_title(soup, view if isinstance(view, Tag) else soup)
        author = _extract_author(view) if isinstance(view, Tag) else None
        published = _extract_published(view) if isinstance(view, Tag) else None

        content_html = _build_content_html(title, author, published, content_module)

        metadata_overrides: dict[str, object] = {}
        if title:
            metadata_overrides["title"] = title
        if author:
            metadata_overrides["author"] = author
        if published:
            metadata_overrides["published"] = published

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            diagnostics={
                "bilibili_opus_resolve": "success",
                "content_profile": ContentProfile.SOCIAL_POST.value,
                "extractor_name": self.name,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_title(soup: BeautifulSoup, scope: Tag) -> str | None:
    """Extract the opus title.

    Not every opus post has a distinct headline (a quick text+image post
    may skip ``opus-module-title`` entirely) — falls back to the page
    ``<title>`` with the " - 哔哩哔哩" site suffix stripped.
    """
    title_el = scope.find(class_="opus-module-title__text")
    if isinstance(title_el, Tag):
        text = title_el.get_text(strip=True)
        if text:
            return text

    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        text = title_tag.get_text(strip=True)
        if text:
            suffix = " - 哔哩哔哩"
            if text.endswith(suffix):
                text = text[: -len(suffix)]
            return text

    return None


def _extract_author(view: Tag) -> str | None:
    """Extract the author display name."""
    name_el = view.find(class_="opus-module-author__name")
    if isinstance(name_el, Tag):
        text = name_el.get_text(strip=True)
        if text:
            return text
    return None


def _extract_published(view: Tag) -> str | None:
    """Extract the publish date as YYYY-MM-DD.

    The author module shows e.g. "编辑于 2026年06月10日 19:48" (edited) or
    "发布于 ..." (published) — the prefix varies, so match the date pattern
    directly rather than anchoring on either verb.
    """
    pub_el = view.find(class_="opus-module-author__pub__text")
    if not isinstance(pub_el, Tag):
        return None
    text = pub_el.get_text(strip=True)
    match = _PUB_TIME_RE.search(text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _build_content_html(
    title: str | None,
    author: str | None,
    published: str | None,
    content_module: Tag,
) -> str:
    """Build clean content HTML from the title/author/content modules.

    Deliberately excludes ``opus-module-copyright`` (internal content-ID
    tag) and ``opus-module-bottom`` (share widget + feedback link) — real
    modules of the same card, but not reader-facing content.
    """
    parts: list[str] = []

    if title:
        parts.append(f"<h1>{escape(title)}</h1>")
    if author:
        parts.append(f"<p>{escape(author)}</p>")
    if published:
        parts.append(f"<p>{escape(published)}</p>")

    parts.append(content_module.decode_contents())

    return "\n".join(parts)
