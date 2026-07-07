from __future__ import annotations

from typing import Any

from markitai.webextract.types import ExtractedWebContent


def build_source_frontmatter(result: ExtractedWebContent) -> dict[str, Any]:
    """Build a user-facing frontmatter dict from a typed extraction result.

    Exports page metadata (title, author, site, published, description,
    canonical_url) and extraction facts (word_count, content_profile) but
    deliberately excludes internal quality diagnostics (score, accepted,
    reasons).

    Args:
        result: A fully typed extraction result produced by the pipeline.

    Returns:
        A dict suitable for use as YAML frontmatter. None values and internal
        quality fields are omitted.
    """
    fm: dict[str, Any] = {}

    # Page-level metadata
    meta = result.metadata
    for key in ("title", "author", "site", "published", "description", "canonical_url", "domain"):
        value = getattr(meta, key, None)
        if value is not None:
            fm[key] = value

    # Extraction facts from info (when available)
    if result.info is not None:
        fm["word_count"] = result.info.word_count
        fm["content_profile"] = result.info.content_profile.value

    return fm
