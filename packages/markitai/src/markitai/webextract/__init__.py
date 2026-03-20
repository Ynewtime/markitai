from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from markitai.webextract.frontmatter import build_source_frontmatter
from markitai.webextract.pipeline import extract_web_content
from markitai.webextract.quality import assess_native_markdown
from markitai.webextract.types import ExtractedWebContent, WebMetadata

__all__ = [
    "ExtractedWebContent",
    "WebMetadata",
    "assess_native_markdown",
    "build_source_frontmatter",
    "coerce_source_frontmatter",
    "extract_web_content",
    "is_native_extraction_acceptable",
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
    """Check whether native extraction produced minimally usable markdown.

    Delegates to :func:`assess_native_markdown` with the ``generic_article``
    profile, preserving the original acceptance semantics.

    Args:
        markdown: Markdown text produced by native extraction.

    Returns:
        ``True`` if the markdown is considered acceptable.
    """
    return assess_native_markdown(markdown, profile="generic_article").accepted


def is_native_extraction_acceptable(extracted: Any) -> bool:
    """Check whether a typed native extraction result is acceptable.

    Prefers the profile-aware quality gate when the extraction result carries
    ``quality`` or ``info.content_profile``. Falls back to the historical
    generic markdown gate for legacy callers and mocks.

    Args:
        extracted: Native extraction result object.

    Returns:
        ``True`` if the extraction should be preferred over legacy fallback
        conversion.
    """
    markdown = getattr(extracted, "markdown", "")

    quality = getattr(extracted, "quality", None)
    accepted = getattr(quality, "accepted", None)
    if isinstance(accepted, bool):
        return accepted

    info = getattr(extracted, "info", None)
    content_profile = getattr(info, "content_profile", None)
    if content_profile is not None:
        profile = getattr(content_profile, "value", content_profile)
        return assess_native_markdown(markdown, profile=str(profile)).accepted

    return is_native_markdown_acceptable(markdown)
