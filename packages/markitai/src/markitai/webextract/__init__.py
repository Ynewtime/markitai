from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

from markitai.webextract.pipeline import extract_web_content
from markitai.webextract.types import ExtractedWebContent, WebMetadata

__all__ = [
    "ExtractedWebContent",
    "WebMetadata",
    "coerce_source_frontmatter",
    "extract_web_content",
    "is_native_markdown_acceptable",
]


def coerce_source_frontmatter(metadata: Any) -> dict[str, Any]:
    """Convert extracted metadata objects into a serializable frontmatter dict."""
    if metadata is None:
        return {}
    if is_dataclass(metadata) and not isinstance(metadata, type):
        data = asdict(metadata)
    elif isinstance(metadata, dict):
        data = metadata
    else:
        data = {
            key: getattr(metadata, key)
            for key in (
                "title",
                "author",
                "site",
                "published",
                "description",
                "canonical_url",
            )
            if getattr(metadata, key, None) is not None
        }
    return {key: value for key, value in data.items() if value is not None}


def is_native_markdown_acceptable(markdown: str) -> bool:
    """Check whether native extraction produced minimally usable markdown."""
    if not markdown or not markdown.strip():
        return False

    text_only = re.sub(r"[#*_\[\]()>`\-|!]", "", markdown)
    text_only = " ".join(text_only.split())
    return len(text_only) >= 10
