from __future__ import annotations

from typing import Protocol
from urllib.parse import SplitResult, urlsplit

from bs4 import BeautifulSoup, Tag


def site_url(url: str, *domains: str) -> SplitResult | None:
    """Parse *url* when its host is one of *domains* or a subdomain of one.

    Site extractors match on the hostname, never on a substring of the
    whole URL: ``dropbox.com/status/1`` contains ``x.com/``, ``linux.com``
    contains ``x.com`` too, and any path can name another site.

    Returns:
        The split URL, or None for another host or a non-http(s) URL.
    """
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").rstrip(".")
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"}:
        return None
    if any(host == domain or host.endswith(f".{domain}") for domain in domains):
        return parts
    return None


class BaseSiteExtractor(Protocol):
    """Protocol for site-specific extractors."""

    name: str

    def matches_url(self, url: str) -> bool:
        """Return whether this extractor handles the given URL."""
        ...

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the root content element for the document."""
