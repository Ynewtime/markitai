from __future__ import annotations

from markitai.webextract.extractors.base import BaseSiteExtractor
from markitai.webextract.extractors.github_issue import GitHubIssueExtractor
from markitai.webextract.extractors.x_article import XArticleExtractor

_EXTRACTORS: tuple[BaseSiteExtractor, ...] = (
    GitHubIssueExtractor(),
    XArticleExtractor(),
)


def find_extractor(url: str) -> BaseSiteExtractor | None:
    """Return a site-specific extractor for the given URL.

    Args:
        url: Source URL.

    Returns:
        Matching extractor if one exists, otherwise None.
    """

    for extractor in _EXTRACTORS:
        if extractor.matches_url(url):
            return extractor
    return None
