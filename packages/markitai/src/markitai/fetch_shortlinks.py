"""X's ``t.co`` shortlinks, replaced by where they lead.

X wraps every link of a post in ``https://t.co/<code>``. The post data
(FxTwitter's facets) usually names the destination and the extractors use it;
a shortlink still left in the Markdown is resolved here with one ``HEAD``
request to ``t.co``, which answers with a redirect and no body. Bounded: at
most ``MAX_LINKS`` distinct links per page, ``CONCURRENCY`` at a time, each
with a ``TIMEOUT_S`` timeout and no redirect following, and the whole step
within ``BUDGET_S``. Answers (also failures) are kept for the process, so a
page converted again costs nothing.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from urllib.parse import urlsplit

from loguru import logger

MAX_LINKS = 20
CONCURRENCY = 4
TIMEOUT_S = 5.0
BUDGET_S = 10.0
CACHE_SIZE = 1024

_SHORTLINK = re.compile(r"https?://t\.co/[A-Za-z0-9]{4,32}\b")
_X_HOSTS = frozenset(
    {
        "x.com",
        "www.x.com",
        "mobile.x.com",
        "twitter.com",
        "www.twitter.com",
        "mobile.twitter.com",
    }
)
_USER_AGENT = (
    "Mozilla/5.0 (compatible; markitai; +https://github.com/Ynewtime/markitai)"
)

# shortlink -> destination ("" when it could not be resolved)
_resolved: OrderedDict[str, str] = OrderedDict()


def is_x_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in _X_HOSTS


def _remember(link: str, destination: str) -> None:
    _resolved[link] = destination
    _resolved.move_to_end(link)
    while len(_resolved) > CACHE_SIZE:
        _resolved.popitem(last=False)


def _usable(destination: str) -> bool:
    parts = urlsplit(destination)
    return (
        parts.scheme in {"http", "https"}
        and bool(parts.hostname)
        and (parts.hostname or "").lower() != "t.co"
    )


async def _resolve(client: object, link: str) -> str:
    try:
        response = await client.head(link)  # type: ignore[attr-defined]
    except Exception as e:
        logger.debug("[t.co] {} not resolved: {}", link, e)
        return ""
    location = response.headers.get("location", "")
    if response.is_redirect and _usable(location):
        return location
    return ""


async def expand_shortlinks(
    markdown: str, *, proxy: str | None = None, transport: object = None
) -> str:
    """Replace each resolvable ``t.co`` link in *markdown* by its destination.

    ``transport`` is an ``httpx`` transport for tests.
    """
    links = list(dict.fromkeys(_SHORTLINK.findall(markdown)))[:MAX_LINKS]
    if not links:
        return markdown
    pending = [link for link in links if link not in _resolved]
    if pending:
        import httpx

        gate = asyncio.Semaphore(CONCURRENCY)

        async with httpx.AsyncClient(
            timeout=TIMEOUT_S,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
            proxy=None if transport is not None else proxy or None,
            transport=transport,  # type: ignore[arg-type]
        ) as client:

            async def one(link: str) -> None:
                async with gate:
                    _remember(link, await _resolve(client, link))

            try:
                await asyncio.wait_for(
                    asyncio.gather(*(one(link) for link in pending)), BUDGET_S
                )
            except TimeoutError:
                logger.debug("[t.co] resolution stopped after {}s", BUDGET_S)
    for link in links:
        destination = _resolved.get(link, "")
        if destination:
            markdown = re.sub(
                re.escape(link) + r"\b", lambda _m, d=destination: d, markdown
            )
    return markdown
