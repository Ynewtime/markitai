from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

_SRCSET_WIDTH_RE = re.compile(r"(.+)\s+(\d+(?:\.\d+)?)w\s*$")
_SRCSET_DENSITY_RE = re.compile(r"(.+)\s+(\d+(?:\.\d+)?)x\s*$")


def normalize_images(root: Tag, base_url: str) -> None:
    """Normalize lazy-loaded images, srcset, and captions.

    Args:
        root: Content root.
        base_url: Base URL for relative asset resolution.
    """
    _remove_lightbox_duplicates(root)

    for img in list(root.find_all("img")):
        # Resolve lazy-loading attributes
        src = img.get("src") or img.get("data-src") or img.get("data-original")

        # Optimize srcset: pick best resolution image
        srcset = img.get("srcset")
        if srcset and isinstance(srcset, str):
            best = _pick_best_srcset(str(srcset))
            if best:
                src = best

        if src:
            img["src"] = urljoin(base_url, str(src))

        # Remove srcset after optimization (MarkItDown doesn't use it)
        if img.has_attr("srcset"):
            del img["srcset"]

        # Wrap with figure/figcaption if adjacent caption exists
        caption = img.find_next_sibling(class_="caption")
        if isinstance(caption, Tag):
            soup = BeautifulSoup("", "html.parser")
            figure = soup.new_tag("figure")
            caption_text = caption.get_text(" ", strip=True)
            img.replace_with(figure)
            caption.extract()
            figure.append(img)
            figcaption = soup.new_tag("figcaption")
            figcaption.string = caption_text
            figure.append(figcaption)


def _remove_lightbox_duplicates(root: Tag) -> None:
    """Remove standalone full-size duplicates of linked lightbox thumbnails.

    Pattern: ``<a href="full.jpg"><img src="thumb.jpg"></a><img
    src="full.jpg">`` — the bare image duplicates the link target and only
    exists for the lightbox overlay. Ported from defuddle ``defuddle.ts``
    ``_deduplicateImages`` (lightbox branch).
    """
    for img in list(root.find_all("img")):
        if img.parent is None or img.find_parent(("a", "figure", "noscript")):
            continue
        src = str(img.get("src") or "")
        if not src or src.startswith("data:"):
            continue
        parent = img.parent
        if not isinstance(parent, Tag):
            continue
        normalized = _normalize_src(src)
        for link in parent.find_all("a", href=True, recursive=False):
            if link.find("img") is None:
                continue
            if normalized == _normalize_src(str(link.get("href") or "")):
                img.decompose()
                break


def _normalize_src(url: str) -> str:
    """Strip protocol and query string for loose URL comparison."""
    return re.sub(r"^https?://", "", url).split("?")[0]


def _pick_best_srcset(srcset: str) -> str | None:
    """Pick the best image URL from a srcset attribute.

    Prefers highest width descriptor (e.g., 1200w), falls back to
    highest density descriptor (e.g., 3x).

    Args:
        srcset: srcset attribute value.

    Returns:
        Best URL or None if no valid entries found.
    """
    best_url: str | None = None
    best_value = 0.0
    best_type = ""  # "w" or "x"

    # Split by comma, parse each entry
    for entry in srcset.split(","):
        entry = entry.strip()
        if not entry:
            continue

        # Match "url descriptor" pattern
        match = _SRCSET_WIDTH_RE.fullmatch(entry)
        if match:
            url = match.group(1).strip()
            value = float(match.group(2))
            if best_type != "w" or value > best_value:
                best_url = url
                best_value = value
                best_type = "w"
            continue

        match = _SRCSET_DENSITY_RE.fullmatch(entry)
        if match:
            url = match.group(1).strip()
            value = float(match.group(2))
            if best_type == "w":
                continue  # width descriptors take priority
            if value > best_value:
                best_url = url
                best_value = value
                best_type = "x"

    return best_url
