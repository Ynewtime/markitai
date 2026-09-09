from __future__ import annotations

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

    for extractor in _EXTRACTORS:
        if extractor.matches_url(url):
            return extractor
    if soup is not None:
        substack = SubstackNoteExtractor()
        if substack.matches_document(soup):
            return substack
    return None
