from __future__ import annotations

import json

from bs4 import BeautifulSoup, Tag

from markitai.webextract.constants import SCHEMA_FALLBACK_MIN_GAIN


def extract_schema_text(soup: BeautifulSoup) -> str | None:
    """Extract text-like content from JSON-LD blocks.

    Args:
        soup: Parsed HTML document.

    Returns:
        Best schema text candidate, if any.
    """

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        docs = parsed if isinstance(parsed, list) else [parsed]
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            for key in ("articleBody", "text"):
                value = doc.get(key)
                if isinstance(value, str) and value.strip():
                    return " ".join(value.split())
    return None


def should_use_schema_fallback(schema_text: str | None, extracted_text: str) -> bool:
    """Return whether schema fallback should override extracted text."""

    if not schema_text:
        return False
    return (
        len(schema_text.split())
        >= len(extracted_text.split()) + SCHEMA_FALLBACK_MIN_GAIN
    )


def find_smallest_matching_element(soup: BeautifulSoup, schema_text: str) -> Tag | None:
    """Find the smallest DOM element containing the schema text.

    Args:
        soup: Parsed HTML document.
        schema_text: Text extracted from schema.

    Returns:
        Smallest matching tag, if any.
    """

    normalized = " ".join(schema_text.split())
    best: Tag | None = None
    best_length: int | None = None

    for node in soup.find_all(["article", "section", "main", "div", "p"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if normalized and normalized in text:
            length = len(text)
            if best is None or length < (best_length or length + 1):
                best = node
                best_length = length

    return best
