from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WebMetadata:
    """Structured metadata extracted from a web page."""

    title: str | None = None
    author: str | None = None
    site: str | None = None
    published: str | None = None
    description: str | None = None
    canonical_url: str | None = None


@dataclass(slots=True)
class ExtractedWebContent:
    """Native extraction result for web content."""

    clean_html: str
    markdown: str
    metadata: WebMetadata
    word_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
