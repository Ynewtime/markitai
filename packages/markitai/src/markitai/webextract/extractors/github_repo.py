"""Extractor for GitHub repository root pages (rendered README).

A repository home page (``github.com/<owner>/<repo>``) wraps the README in
``<div id="readme"> … <article class="markdown-body">`` surrounded by heavy
chrome: the file-tree table, action menus, the About sidebar, star/fork
counts, and Releases/Languages sections. Generic scoring keeps the whole
main region (~950 words of chrome); defuddle extracts only the README
article. This extractor pins the content root to the README article via
``extract_root`` and falls back to the generic pipeline when the page has
no rendered README.

Implemented as a site-specific extractor (rather than a generic
"prefer markdown-body" scoring rule) so the 83-fixture defuddle corpus is
untouched: the ``markdown-body`` class is GitHub's README container, and a
registry extractor scopes the rule to GitHub repo URLs only.
"""

from __future__ import annotations

from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

# First path segments that are GitHub product pages, not repository owners.
_RESERVED_FIRST_SEGMENTS = frozenset(
    {
        "about",
        "apps",
        "collections",
        "contact",
        "customer-stories",
        "enterprise",
        "explore",
        "features",
        "join",
        "login",
        "marketplace",
        "notifications",
        "orgs",
        "pricing",
        "readme",
        "search",
        "settings",
        "site",
        "sponsors",
        "topics",
        "trending",
        "users",
    }
)


class GitHubRepoExtractor:
    """Select the rendered README article on GitHub repository home pages."""

    name = "github_repo"

    def matches_url(self, url: str) -> bool:
        """Match ``github.com/<owner>/<repo>`` repository root URLs.

        Args:
            url: Source URL to test.

        Returns:
            True for two-segment github.com paths whose first segment is not
            a reserved GitHub product page.
        """
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host not in {"github.com", "www.github.com"}:
            return False
        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) != 2:
            return False
        return segments[0].lower() not in _RESERVED_FIRST_SEGMENTS

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the README ``article.markdown-body`` as the content root.

        Args:
            soup: Parsed HTML of the repository page.

        Returns:
            The README article Tag, or ``None`` to fall back to the generic
            pipeline (e.g. repositories without a README).
        """
        readme = soup.find(id="readme")
        if isinstance(readme, Tag):
            article = readme.find("article")
            if isinstance(article, Tag):
                return article
        article = soup.find("article", class_="markdown-body")
        return article if isinstance(article, Tag) else None
