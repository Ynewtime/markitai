"""Heading normalization: remove anchor/permalink links."""

from __future__ import annotations

from bs4 import Tag

_PERMALINK_SYMBOLS = frozenset({"#", "¶", "§", "\U0001f517"})  # 🔗


def normalize_headings(root: Tag) -> None:
    """Remove permalink/anchor links from headings.

    Detects and removes:
    - Links with href containing '#' and single-symbol text (#, ¶, §, 🔗)
    - Links with class/title containing 'permalink', 'anchor-link', 'heading-anchor'

    Args:
        root: Content root element.
    """
    for heading in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        for link in heading.find_all("a"):
            if _is_permalink(link):
                link.decompose()


def _is_permalink(link: Tag) -> bool:
    """Check if a link is a permalink/anchor."""
    href = str(link.get("href") or "")

    # Single-symbol text with anchor href
    text = link.get_text(strip=True)
    if text in _PERMALINK_SYMBOLS and "#" in href:
        return True

    # Class/title contains permalink indicators
    classes = " ".join(link.get("class", [])).lower() if link.get("class") else ""  # type: ignore[arg-type]
    title = str(link.get("title") or "").lower()
    combined = f"{classes} {title}"
    if any(
        kw in combined
        for kw in ("permalink", "anchor-link", "heading-anchor", "anchor_link")
    ):
        return True

    return False
