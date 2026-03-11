from __future__ import annotations

import json
from typing import Any

from bs4 import BeautifulSoup

from markitai.webextract.types import WebMetadata


def clean_title(value: str | None, site: str | None = None) -> str | None:
    """Normalize a page title and strip repeated site affixes.

    Args:
        value: Raw title text.
        site: Optional site name for prefix/suffix removal.

    Returns:
        Cleaned title string.
    """

    if not value:
        return None

    title = " ".join(value.split())
    if site:
        site = " ".join(site.split())
        for sep in (" | ", " - ", " -- ", " · ", " — "):
            if title.endswith(f"{sep}{site}"):
                return title[: -len(f"{sep}{site}")]
            if title.startswith(f"{site}{sep}"):
                return title[len(f"{site}{sep}") :]
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

    return WebMetadata(
        title=title,
        author=_meta_content(soup, name="author") or _jsonld_author(jsonld_docs),
        site=site,
        published=_jsonld_value(jsonld_docs, "datePublished"),
        description=_meta_content(soup, name="description")
        or _meta_content(soup, property_name="og:description"),
        canonical_url=_canonical_url(soup) or url,
    )


def _title_text(soup: BeautifulSoup) -> str | None:
    return soup.title.string if soup.title and soup.title.string else None


def _canonical_url(soup: BeautifulSoup) -> str | None:
    link = soup.find("link", rel=lambda value: value and "canonical" in value)  # type: ignore[arg-type]
    if link and link.get("href"):
        return str(link["href"]).strip()
    return None


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
