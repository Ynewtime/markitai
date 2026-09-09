from __future__ import annotations

import re

from bs4 import BeautifulSoup

from markitai.webextract.extractors.base import BaseSiteExtractor
from markitai.webextract.extractors.bilibili_opus import BilibiliOpusExtractor
from markitai.webextract.extractors.github_repo import GitHubRepoExtractor
from markitai.webextract.extractors.github_thread import GitHubThreadExtractor
from markitai.webextract.extractors.hackernews_thread import HackerNewsThreadExtractor
from markitai.webextract.extractors.reddit_post import RedditPostExtractor
from markitai.webextract.extractors.steam_news import SteamNewsExtractor
from markitai.webextract.extractors.substack_note import SubstackNoteExtractor
from markitai.webextract.extractors.x_article import XArticleExtractor
from markitai.webextract.extractors.x_tweet import XTweetExtractor
from markitai.webextract.extractors.youtube_page import YouTubePageExtractor

_EXTRACTORS: tuple[BaseSiteExtractor, ...] = (
    GitHubThreadExtractor(),
    GitHubRepoExtractor(),  # github.com/<owner>/<repo> (rendered README)
    RedditPostExtractor(),
    HackerNewsThreadExtractor(),
    SteamNewsExtractor(),  # store.steampowered.com/news (BBCode announcements)
    SubstackNoteExtractor(),  # Substack articles and note permalinks
    XArticleExtractor(),  # x.com/i/articles/ (long-form articles)
    XTweetExtractor(),  # x.com/user/status/ (regular tweets)
    YouTubePageExtractor(),  # youtube.com/watch and youtu.be (video pages)
    BilibiliOpusExtractor(),  # bilibili.com/opus/<id> (专栏/动态 posts)
)


# Raw-HTML markers for the document sniff below. ``matches_document`` can only
# succeed when the page carries a Substack preload payload or a substackcdn
# asset, so a page without either marker never needs to be parsed just to be
# rejected. Deliberately a superset of the real check: it only decides whether
# parsing is worth it, never whether the extractor matches.
_DOCUMENT_SNIFF_MARKERS = re.compile(r"substackcdn|_preloads", re.IGNORECASE)

# Reused instance: ``matches_document`` is stateless, so constructing a fresh
# extractor per lookup only allocated garbage.
_SUBSTACK_EXTRACTOR = next(
    extractor
    for extractor in _EXTRACTORS
    if isinstance(extractor, SubstackNoteExtractor)
)


def document_sniff_may_apply(html: str) -> bool:
    """Return whether *html* can possibly match a document-sniffing extractor.

    Cheap raw-string pre-check that lets callers skip a full BeautifulSoup
    parse for the overwhelming majority of pages, which no extractor claims.

    Args:
        html: Raw HTML source.

    Returns:
        True when the document sniff is worth running.
    """
    return _DOCUMENT_SNIFF_MARKERS.search(html) is not None


def find_extractor_for_url(url: str) -> BaseSiteExtractor | None:
    """Return the extractor claiming *url*, without needing a parsed document.

    Args:
        url: Source URL.

    Returns:
        Matching extractor if one exists, otherwise None.
    """

    for extractor in _EXTRACTORS:
        if extractor.matches_url(url):
            return extractor
    return None


def find_extractor_for_document(soup: BeautifulSoup) -> BaseSiteExtractor | None:
    """Return an extractor recognizing *soup* by its content (custom domains).

    Args:
        soup: Parsed document.

    Returns:
        Matching extractor if one exists, otherwise None.
    """

    if _SUBSTACK_EXTRACTOR.matches_document(soup):
        return _SUBSTACK_EXTRACTOR
    return None


def find_extractor(
    url: str, soup: BeautifulSoup | None = None
) -> BaseSiteExtractor | None:
    """Return a site-specific extractor for the given URL.

    Args:
        url: Source URL.
        soup: Optional document for Substack custom-domain detection.

    Returns:
        Matching extractor if one exists, otherwise None.
    """

    extractor = find_extractor_for_url(url)
    if extractor is not None:
        return extractor
    if soup is not None:
        return find_extractor_for_document(soup)
    return None
