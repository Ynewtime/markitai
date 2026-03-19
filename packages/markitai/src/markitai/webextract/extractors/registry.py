from __future__ import annotations

from markitai.webextract.extractors.base import BaseSiteExtractor
from markitai.webextract.extractors.github_thread import GitHubThreadExtractor
from markitai.webextract.extractors.x_article import XArticleExtractor
from markitai.webextract.extractors.x_tweet import XTweetExtractor

_EXTRACTORS: tuple[BaseSiteExtractor, ...] = (
    GitHubThreadExtractor(),
    XArticleExtractor(),  # x.com/i/articles/ (long-form articles)
    XTweetExtractor(),  # x.com/user/status/ (regular tweets)
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
