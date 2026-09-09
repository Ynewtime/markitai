"""Extract Substack articles and note permalinks.

Ported from defuddle ``extractors/substack.ts`` (commit pinned in
``webextract/PORT_MANIFEST.md``). Prefer rendered article bodies, then
``window._preloads`` JSON, with Notes and generic extraction as fallbacks.
The historical class/module name is retained for compatibility.
"""

from __future__ import annotations

import json
import re
from datetime import date
from html import escape
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.types import ContentProfile

_PRELOAD_RE = re.compile(r"\bwindow\._preloads\s*=\s*(?:JSON\.parse\(\s*)?")
_MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
_BYLINE_DATE_RE = re.compile(rf"\b({'|'.join(_MONTHS)})\s+(\d{{1,2}}),?\s+(\d{{4}})\b")
_HANDLE_SUFFIX_RE = re.compile(r"\s*\(@[^)]+\)\s*$")
_SRCSET_ENTRY_RE = re.compile(r"(\S+)\s+(\d+(?:\.\d+)?)w")


class SubstackNoteExtractor:
    """Extract Substack publications and note permalinks."""

    name = "substack_note"

    def matches_url(self, url: str) -> bool:
        """Match Substack hosts without accepting lookalike domains."""
        parts = urlsplit(url)
        host = parts.hostname or ""
        return parts.scheme in {"http", "https"} and (
            host == "substack.com" or host.endswith(".substack.com")
        )

    def matches_document(self, soup: BeautifulSoup) -> bool:
        """Recognize custom-domain articles, not arbitrary Substack links."""
        if _extract_preload_post(soup) is not None:
            return True
        if soup.select_one("div.body.markup") is None:
            return False
        for asset in soup.select("link[href], script[src]"):
            source = str(asset.get("href") or asset.get("src") or "")
            try:
                host = urlsplit(source).hostname or ""
            except ValueError:
                continue
            if host == "substackcdn.com" or host.endswith(".substackcdn.com"):
                return True
        return False

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return rendered content (legacy protocol compliance)."""
        return soup.select_one("div.body.markup") or _find_note_text(soup)

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve an article body or the main note text plus image.

        Args:
            soup: Parsed document.
            url: Page URL (unused; matching already happened).

        Returns:
            ResolvedPage with the note HTML, or empty content to fall back
            to the generic pipeline when no note element is present.
        """
        post = _extract_preload_post(soup)
        rendered = soup.select_one("div.body.markup")
        if rendered is not None or (post and _post_string(post, "body_html")):
            return ResolvedPage(
                content_root=rendered,
                content_html=None
                if rendered is not None
                else _post_string(post or {}, "body_html"),
                metadata_overrides=_post_metadata(soup, post or {}),
                diagnostics={
                    "substack_resolve": "post",
                    "content_profile": ContentProfile.GENERIC_ARTICLE.value,
                    "extractor_name": self.name,
                },
            )

        note = _find_note_text(soup)
        if note is None:
            return ResolvedPage(
                content_html="",
                diagnostics={"substack_resolve": "no_note_element"},
            )

        image_html = _build_image_html(soup, note)
        content_html = f"{note}\n{image_html}" if image_html else str(note)

        title = _meta_content(soup, "og:title")
        metadata_overrides: dict[str, object] = {"site": "Substack"}
        if title:
            metadata_overrides["title"] = title
            author = _HANDLE_SUFFIX_RE.sub("", title).strip()
            if author:
                metadata_overrides["author"] = author

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            diagnostics={
                "substack_resolve": "note",
                "content_profile": ContentProfile.SOCIAL_POST.value,
                "extractor_name": self.name,
            },
        )


def _extract_preload_post(soup: BeautifulSoup) -> dict[str, object] | None:
    """Decode preload JSON without evaluating JavaScript or unrelated payloads."""
    decoder = json.JSONDecoder()
    metadata_post: dict[str, object] | None = None
    for script in soup.find_all("script"):
        text = script.get_text()
        for match in _PRELOAD_RE.finditer(text):
            try:
                data, _ = decoder.raw_decode(text, match.end())
                if isinstance(data, str):
                    data = json.loads(data)
            except (ValueError, RecursionError):
                continue
            if not isinstance(data, dict):
                continue
            # The app feed and publication pages use different envelopes.
            feed = data.get("feedData")
            initial = feed.get("initialPost") if isinstance(feed, dict) else None
            candidates = (
                initial.get("post") if isinstance(initial, dict) else None,
                data.get("post"),
            )
            for post in candidates:
                if not isinstance(post, dict):
                    continue
                if _post_string(post, "body_html"):
                    return post
                # Keep metadata for rendered articles, but do not let an empty
                # or metadata-only preload hide a later usable article body.
                if post and metadata_post is None:
                    metadata_post = post
    return metadata_post


def _post_string(post: dict[str, object], key: str) -> str:
    value = post.get(key)
    return value.strip() if isinstance(value, str) else ""


def _post_metadata(soup: BeautifulSoup, post: dict[str, object]) -> dict[str, object]:
    """Only override metadata when the article supplies a nonempty value."""
    bylines = post.get("publishedBylines")
    first = bylines[0] if isinstance(bylines, list) and bylines else None
    author = _post_string(first, "name") if isinstance(first, dict) else ""
    if not author:
        link = soup.select_one('a[href*="substack.com/@"]')
        author = link.get_text(" ", strip=True) if link else ""
    values = {
        "site": "Substack",
        "title": _post_string(post, "title") or _meta_content(soup, "og:title"),
        "description": _post_string(post, "subtitle")
        or _meta_content(soup, "og:description"),
        "author": author,
        "published": _post_string(post, "post_date") or _parse_byline_date(soup),
    }
    return {key: value for key, value in values.items() if value}


def _parse_byline_date(soup: BeautifulSoup) -> str:
    """Parse Substack's abbreviated English date, including adjacent DOM text."""
    byline = soup.select_one('[class*="byline-wrapper"]')
    if byline is None:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", byline.get_text(" ", strip=True))
    match = _BYLINE_DATE_RE.search(text)
    if match is None:
        return ""
    month, day, year = match.groups()
    try:
        published = date(int(year), _MONTHS.index(month) + 1, int(day))
    except ValueError:
        return ""
    return f"{published.isoformat()}T00:00:00+00:00"


def _find_note_text(soup: BeautifulSoup) -> Tag | None:
    """Find the main note's ProseMirror element.

    On permalink pages the main note sits inside a ``feedPermalinkUnit``
    container; other ProseMirror elements are feed recommendations.
    """
    scope: Tag | BeautifulSoup = soup.select_one('[class*="feedPermalinkUnit"]') or soup
    return scope.select_one("div.ProseMirror.FeedProseMirror")


def _build_image_html(soup: BeautifulSoup, note: Tag) -> str:
    """Build an ``<img>`` for the note's attached image, if any.

    Prefers the ``og:image`` (full resolution); falls back to the largest
    srcset entry of the sibling ``imageGrid`` element.
    """
    og_image = _meta_content(soup, "og:image")
    if og_image:
        return f'<img src="{escape(og_image)}" alt=""/>'

    grid = _find_image_grid(note)
    img = grid.find("img") if grid is not None else None
    if not isinstance(img, Tag):
        return ""
    src = _largest_src(img)
    return f'<img src="{escape(src)}" alt=""/>' if src else ""


def _find_image_grid(note: Tag) -> Tag | None:
    """Find the imageGrid sibling of the note's feedCommentBody wrapper."""
    body: Tag | None = None
    for parent in note.parents:
        if not isinstance(parent, Tag):
            break
        classes = " ".join(parent.get("class") or [])
        if "feedCommentBody" in classes and "feedCommentBodyInner" not in classes:
            body = parent
            break
    if body is None:
        return None
    candidates = [body.find_next_sibling(True)]
    if isinstance(body.parent, Tag):
        candidates.append(body.parent.find_next_sibling(True))
    for el in candidates:
        if isinstance(el, Tag) and "imageGrid" in " ".join(el.get("class") or []):
            return el
    return None


def _largest_src(img: Tag) -> str:
    """Pick the largest-width srcset entry, falling back to ``src``."""
    srcset = str(img.get("srcset") or "")
    best_url, best_width = "", 0.0
    for match in _SRCSET_ENTRY_RE.finditer(srcset):
        width = float(match.group(2))
        if width > best_width:
            best_url, best_width = match.group(1), width
    return best_url or str(img.get("src") or "")


def _meta_content(soup: BeautifulSoup, prop: str) -> str:
    """Read a meta property content value."""
    el = soup.select_one(f'meta[property="{prop}"]')
    return str(el.get("content") or "") if isinstance(el, Tag) else ""
