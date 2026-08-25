"""Extractor for Substack note permalink pages.

Ported from the Notes branch of defuddle ``extractors/substack.ts``
(commit pinned in ``webextract/PORT_MANIFEST.md``): on a note permalink
the main note lives in a ``feedPermalinkUnit`` container while the other
ProseMirror editors on the page belong to feed recommendations. Focused
subset — the ``window._preloads`` post path and the byline-date parser
are not ported (post pages fall back to the generic pipeline).
"""

from __future__ import annotations

import re
from html import escape

from bs4 import BeautifulSoup, Tag

from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.types import ContentProfile

_NOTE_URL_RE = re.compile(r"https?://substack\.com/@[^/]+/note/")
_HANDLE_SUFFIX_RE = re.compile(r"\s*\(@[^)]+\)\s*$")
_SRCSET_ENTRY_RE = re.compile(r"(\S+)\s+(\d+(?:\.\d+)?)w")


class SubstackNoteExtractor:
    """Extract Substack note permalinks (substack.com/@user/note/...)."""

    name = "substack_note"

    def matches_url(self, url: str) -> bool:
        """Match Substack note permalink URLs."""
        return bool(_NOTE_URL_RE.search(url))

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the note text element (legacy protocol compliance)."""
        return _find_note_text(soup)

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a note permalink into the main note text plus image.

        Args:
            soup: Parsed document.
            url: Page URL (unused; matching already happened).

        Returns:
            ResolvedPage with the note HTML, or empty content to fall back
            to the generic pipeline when no note element is present.
        """
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
