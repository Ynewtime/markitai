"""Extractor for X/Twitter article pages (long-form posts).

X.com serves article content at ``/article/`` URLs (e.g.
``https://x.com/user/article/123``).  Unlike regular ``/status/`` pages,
``/article/`` pages are gated behind a login wall for anonymous visitors,
so DOM-based extraction only yields the sign-up page.

The extractor detects article URLs and signals the pipeline that an
async enricher upgrade is needed.  The ``XOEmbedEnricher`` handles the
upgrade via the FxTwitter API (same approach as defuddle's
``XOembedExtractor``).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from markitai.webextract.extractors.base import site_url

# Match x.com article URLs: /user/article/ID (singular — confirmed against
# defuddle's reference XArticleExtractor, which uses the same
# `/(article|status)/` pattern with no plural form) or the older /i/articles/ID
# system-path form (plural; predates this extractor, left as originally found).
_ARTICLE_PATH_RE = re.compile(r"/[^/]+/article/\d+(?:/.*)?|/i/articles?/\d+(?:/.*)?")


class XArticleExtractor:
    """Extractor for X article pages (long-form posts)."""

    name = "x_article"

    def matches_url(self, url: str) -> bool:
        """Match ``/article/`` URLs on x.com (including ``/i/article/``)."""
        parts = site_url(url, "x.com")
        return parts is not None and bool(_ARTICLE_PATH_RE.fullmatch(parts.path))

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the ``<article>`` tag or the ``id="target"`` element.

        For article pages behind the login wall, no usable content is
        available — the resolver layer will fall back to the enricher.
        """
        article = soup.find("article")
        if isinstance(article, Tag):
            return article
        return soup.find(id="target")
