from __future__ import annotations

from urllib.parse import urljoin

from bs4 import Comment, Tag

from markitai.webextract.elements.code import normalize_code_blocks
from markitai.webextract.elements.footnotes import normalize_footnotes
from markitai.webextract.elements.images import normalize_images


def standardize_content(root: Tag, title: str | None, base_url: str) -> None:
    """Normalize extracted content in place.

    Args:
        root: Content root.
        title: Extracted title.
        base_url: Base URL for relative link resolution.
    """

    _remove_comments(root)
    _dedupe_title_headings(root, title)
    _resolve_relative_urls(root, base_url)
    _remove_javascript_links(root)
    normalize_code_blocks(root)
    normalize_footnotes(root)
    normalize_images(root, base_url)


def _remove_comments(root: Tag) -> None:
    for comment in root.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _dedupe_title_headings(root: Tag, title: str | None) -> None:
    if not title:
        return

    seen = False
    normalized_title = " ".join(title.split())
    for heading in root.find_all(["h1", "h2", "h3"]):
        text = " ".join(heading.get_text(" ", strip=True).split())
        if text != normalized_title:
            continue
        if seen:
            heading.decompose()
        else:
            seen = True


def _resolve_relative_urls(root: Tag, base_url: str) -> None:
    for tag in root.find_all(["a", "img"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if (
            not isinstance(value, str)
            or not value
            or value.startswith(("#", "http", "data:"))
        ):
            continue
        tag[attr] = urljoin(base_url, value)


def _remove_javascript_links(root: Tag) -> None:
    for link in root.find_all("a", href=True):
        href = str(link["href"]).strip().lower()
        if href.startswith("javascript:"):
            del link["href"]
