from __future__ import annotations

from typing import Protocol

from bs4 import BeautifulSoup, Tag


class BaseSiteExtractor(Protocol):
    """Protocol for site-specific extractors."""

    name: str

    def matches_url(self, url: str) -> bool:
        """Return whether this extractor handles the given URL."""

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the root content element for the document."""
