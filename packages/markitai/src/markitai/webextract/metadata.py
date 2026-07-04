from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

from markitai.webextract.types import WebMetadata

# Generous cap: only guards against pathological titles. Real-world titles
# (e.g. GitHub repo descriptions ~170 chars) must survive intact for parity
# with defuddle output.
_MAX_TITLE_LENGTH = 300

_TITLE_SEPARATORS = (" | ", " - ", " -- ", " · ", " — ", " – ")

_ELLIPSIS_SUFFIXES = ("…", "...")

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}")


def clean_title(value: str | None, site: str | None = None) -> str | None:
    """Normalize a page title and strip repeated site affixes.

    Args:
        value: Raw title text.
        site: Optional site name for prefix/suffix removal.

    Returns:
        Cleaned title string, truncated to _MAX_TITLE_LENGTH if needed.
    """
    if not value:
        return None

    title = " ".join(value.split())
    if site:
        site = " ".join(site.split())
        # Strip repeatedly: pages like GitHub carry both a site prefix and a
        # site suffix ("GitHub - owner/repo: description · GitHub").
        stripped = True
        while stripped and title:
            stripped = False
            for sep in _TITLE_SEPARATORS:
                if title.endswith(f"{sep}{site}"):
                    title = title[: -len(f"{sep}{site}")]
                    stripped = True
                    break
                if title.startswith(f"{site}{sep}"):
                    title = title[len(f"{site}{sep}") :]
                    stripped = True
                    break

    if len(title) > _MAX_TITLE_LENGTH:
        cut = title[:_MAX_TITLE_LENGTH].rfind(" ")
        if cut > 0:
            title = title[:cut] + "…"
        else:
            title = title[:_MAX_TITLE_LENGTH] + "…"

    return title


def extract_metadata(soup: BeautifulSoup, url: str) -> WebMetadata:
    """Extract structured metadata from a parsed HTML document.

    Args:
        soup: Parsed HTML document.
        url: Source URL.

    Returns:
        Extracted metadata object.
    """

    site = _meta_content(soup, property_name="og:site_name") or _meta_content(
        soup, name="application-name"
    )
    jsonld_docs = _jsonld_documents(soup)
    title = clean_title(_jsonld_value(jsonld_docs, "headline"), site=site)
    if not title:
        title = clean_title(_jsonld_value(jsonld_docs, "name"), site=site)
    if not title:
        title = clean_title(_title_text(soup), site=site)
    if not title:
        title = clean_title(_meta_content(soup, property_name="og:title"), site=site)
    title = _prefer_untruncated_title(title, soup, site)
    title = _strip_suffix_by_headline(title, soup, jsonld_docs)

    return WebMetadata(
        title=title,
        author=_meta_content(soup, name="author") or _jsonld_author(jsonld_docs),
        site=site,
        published=_published(soup, jsonld_docs),
        description=_meta_content(soup, name="description")
        or _meta_content(soup, property_name="og:description"),
        canonical_url=_page_canonical_url(soup, url),
    )


def _published(soup: BeautifulSoup, jsonld_docs: list[dict[str, Any]]) -> str | None:
    """Extract the publication date from meta, JSON-LD, or a <time> element."""
    published = (
        _meta_content(soup, property_name="article:published_time")
        or _jsonld_value(jsonld_docs, "datePublished")
        or _time_datetime(soup)
    )
    return published


def _time_datetime(soup: BeautifulSoup) -> str | None:
    """Return the first ``<time datetime>`` value that looks like a date."""
    for time_tag in soup.find_all("time"):
        if not isinstance(time_tag, Tag):
            continue
        value = time_tag.get("datetime")
        if isinstance(value, str) and _ISO_DATE_RE.match(value.strip()):
            return value.strip()
    return None


def _prefer_untruncated_title(
    title: str | None, soup: BeautifulSoup, site: str | None
) -> str | None:
    """Swap a truncated title (trailing ellipsis) for a full-length source.

    Sites like GitHub truncate ``og:title``/JSON-LD values with a trailing
    ellipsis while the ``<title>`` tag carries the full text. When the
    selected title ends in an ellipsis and the document title is a longer
    string sharing the same prefix, prefer the document title.
    """
    if not title:
        return title
    if not title.endswith(_ELLIPSIS_SUFFIXES):
        return title

    prefix = title
    for suffix in _ELLIPSIS_SUFFIXES:
        prefix = prefix.removesuffix(suffix)
    prefix = prefix.rstrip()

    for candidate_raw in (
        _title_text(soup),
        _meta_content(soup, property_name="og:title"),
    ):
        candidate = clean_title(candidate_raw, site=site)
        if not candidate or candidate.endswith(_ELLIPSIS_SUFFIXES):
            continue
        if len(candidate) > len(title) and candidate.startswith(prefix):
            return candidate
    return title


def _strip_suffix_by_headline(
    title: str | None, soup: BeautifulSoup, jsonld_docs: list[dict[str, Any]]
) -> str | None:
    """Strip a trailing ``| SiteName`` when a cleaner headline matches.

    Handles pages that do not declare ``og:site_name`` (so the site-based
    strip in :func:`clean_title` cannot fire) but whose ``<h1>`` or JSON-LD
    headline equals the title minus a site suffix. Restricted to the
    ``" | "`` separator: looser separators like ``" - "`` also appear in
    legitimate subtitles (e.g. an ``<h1>`` of "Foo" with the title
    "Foo - a retrospective") and must not be stripped on h1 evidence alone.
    """
    if not title or " | " not in title:
        return title

    candidates: list[str] = []
    headline = _jsonld_value(jsonld_docs, "headline")
    if headline:
        candidates.append(" ".join(headline.split()))
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        h1_text = " ".join(h1.get_text(" ", strip=True).split())
        if h1_text:
            candidates.append(h1_text)

    for candidate in candidates:
        if not candidate or candidate == title:
            continue
        if title.startswith(f"{candidate} | "):
            return candidate
    return title


def _title_text(soup: BeautifulSoup) -> str | None:
    return soup.title.string if soup.title and soup.title.string else None


def _page_canonical_url(soup: BeautifulSoup, url: str) -> str | None:
    """Return the canonical URL only when it is page-specific.

    A ``rel=canonical`` pointing at the site root while the current URL is
    deeper is a template artifact (every page "canonicalizes" to the
    homepage) and would be misleading in frontmatter, so it is dropped.
    Absent or empty canonical links yield ``None`` — the source URL is NOT
    used as a fallback.
    """
    link = soup.find("link", rel=lambda value: value and "canonical" in value)  # type: ignore[arg-type]
    if not link or not link.get("href"):
        return None
    canonical = str(link["href"]).strip()
    if not canonical:
        return None

    canonical_path = urlparse(canonical).path.strip("/")
    current_path = urlparse(url).path.strip("/")
    if not canonical_path and current_path:
        return None
    return canonical


def _meta_content(
    soup: BeautifulSoup, *, name: str | None = None, property_name: str | None = None
) -> str | None:
    if name:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    if property_name:
        tag = soup.find("meta", attrs={"property": property_name})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
        # Common authoring mistake (e.g. Jekyll themes): OpenGraph values
        # emitted with name= instead of property=. Defuddle reads both.
        tag = soup.find("meta", attrs={"name": property_name})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    return None


def _jsonld_documents(soup: BeautifulSoup) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        _collect_jsonld_documents(parsed, docs)
    return docs


def _collect_jsonld_documents(
    payload: dict[str, Any] | list[Any], docs: list[dict[str, Any]]
) -> None:
    if isinstance(payload, dict):
        docs.append(payload)
        graph = payload.get("@graph")
        if isinstance(graph, dict):
            _collect_jsonld_documents(graph, docs)
        elif isinstance(graph, list):
            for item in graph:
                if isinstance(item, (dict, list)):
                    _collect_jsonld_documents(item, docs)
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                _collect_jsonld_documents(item, docs)


def _jsonld_value(docs: list[dict[str, Any]], key: str) -> str | None:
    for doc in docs:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _jsonld_author(docs: list[dict[str, Any]]) -> str | None:
    for doc in docs:
        author = doc.get("author")
        if isinstance(author, str) and author.strip():
            return author.strip()
        if isinstance(author, dict):
            name = author.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None
