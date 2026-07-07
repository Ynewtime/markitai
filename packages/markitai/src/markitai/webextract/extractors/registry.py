from __future__ import annotations

from markitai.webextract.extractors.base import BaseSiteExtractor
from markitai.webextract.extractors.bilibili_opus import BilibiliOpusExtractor
from markitai.webextract.extractors.github_repo import GitHubRepoExtractor
from markitai.webextract.extractors.github_thread import GitHubThreadExtractor
from markitai.webextract.extractors.hackernews_thread import HackerNewsThreadExtractor
from markitai.webextract.extractors.reddit_post import RedditPostExtractor
from markitai.webextract.extractors.steam_news import SteamNewsExtractor
from markitai.webextract.extractors.x_article import XArticleExtractor
from markitai.webextract.extractors.x_tweet import XTweetExtractor
from markitai.webextract.extractors.youtube_page import YouTubePageExtractor

_EXTRACTORS: tuple[BaseSiteExtractor, ...] = (
    GitHubThreadExtractor(),
    GitHubRepoExtractor(),  # github.com/<owner>/<repo> (rendered README)
    RedditPostExtractor(),
    HackerNewsThreadExtractor(),
    SteamNewsExtractor(),  # store.steampowered.com/news (BBCode announcements)
    XArticleExtractor(),  # x.com/i/articles/ (long-form articles)
    XTweetExtractor(),  # x.com/user/status/ (regular tweets)
    YouTubePageExtractor(),  # youtube.com/watch and youtu.be (video pages)
    BilibiliOpusExtractor(),  # bilibili.com/opus/<id> (专栏/动态 posts)
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
