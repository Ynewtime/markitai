"""Thread inclusion policy for threaded page types.

Defines default rules for which items to include when extracting threaded
conversations (e.g. X/Twitter, GitHub, Reddit, Hacker News).  Generic
(non-threaded) URLs return ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ThreadPolicy:
    """Controls which items are included from a threaded conversation.

    Attributes:
        include_main_item: Whether to include the focal/root post.
        include_author_thread: Whether to include the original author's
            own follow-up replies.
        include_third_party_replies: Whether to include replies from
            other users.
    """

    include_main_item: bool = True
    include_author_thread: bool = True
    include_third_party_replies: bool = False


# Extractor names that are considered threaded sites.
_THREADED_EXTRACTORS = frozenset(
    {
        "x_tweet",
        "github_thread",
        "reddit_post",
        "hackernews_thread",
    }
)


def get_thread_policy(url: str) -> ThreadPolicy | None:
    """Return the default thread policy for a URL, or None for non-threaded sites.

    Args:
        url: Source URL to evaluate.

    Returns:
        A ``ThreadPolicy`` with sensible defaults for the matched site,
        or ``None`` if the URL does not belong to a threaded site.
    """
    # Lazy import to avoid circular dependency:
    # registry → x_tweet → thread_policy → registry
    from markitai.webextract.extractors.registry import find_extractor

    extractor = find_extractor(url)
    if extractor is None:
        return None

    if extractor.name in _THREADED_EXTRACTORS:
        return ThreadPolicy()

    return None
