from __future__ import annotations

from bs4 import BeautifulSoup, Tag


class XArticleExtractor:
    """Extractor for X article pages."""

    name = "x_article"

    def matches_url(self, url: str) -> bool:
        return "x.com/i/articles/" in url

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        return soup.find("article") or soup.find(id="target")
