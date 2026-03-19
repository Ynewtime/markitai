from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markitai.webextract.semantics import ConversationThread


class ContentProfile(Enum):
    """Semantic classification of a page's primary content type.

    Used to guide rendering and quality assessment decisions.
    """

    GENERIC_ARTICLE = "generic_article"
    SOCIAL_POST = "social_post"


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
class ExtractionInfo:
    """Facts about how the extraction was performed.

    These are diagnostic in nature and separate from page-level metadata.
    They inform quality assessment and downstream routing decisions.

    Attributes:
        content_profile: Semantic classification of the page content.
        extractor_name: Name of the site extractor used (e.g. "x_tweet", "generic").
        word_count: Number of words in the extracted markdown.
        enricher_name: Name of the async enricher used, if any.
        source_kind: Kind of source that produced the content (default "html").
    """

    content_profile: ContentProfile
    extractor_name: str
    word_count: int
    enricher_name: str | None = None
    source_kind: str = "html"


@dataclass(slots=True)
class QualityAssessment:
    """Outcome of the quality gate applied to extracted content.

    These fields are internal diagnostics and must NOT be included in
    user-facing frontmatter output.

    Attributes:
        accepted: Whether the extraction passed the quality gate.
        score: Numeric quality score in [0, 1].
        reasons: List of human-readable reasons describing why the extraction
            was accepted or rejected.
    """

    accepted: bool
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SemanticExtraction:
    """Optional semantic models derived from the page content.

    Attributes:
        thread: An optional conversation thread. Set by site-specific
            extractors that produce threaded conversation structures.
    """

    thread: ConversationThread | None = None


@dataclass(slots=True)
class ExtractedWebContent:
    """Native extraction result for web content.

    Combines cleaned HTML, markdown, page metadata, extraction facts, and
    optional quality/semantic information.

    Backward compatibility: The ``word_count`` field is retained as a
    top-level attribute so that existing callers (e.g. ``pipeline.py``) do
    not require changes. The ``info`` and ``quality`` fields are optional so
    that existing construction still works without modification.

    Attributes:
        clean_html: Sanitised HTML fragment used as the canonical representation.
        markdown: Markdown derived from ``clean_html``.
        metadata: Page-level metadata (title, author, site, etc.).
        word_count: Word count of the extracted markdown (legacy field).
        info: Extraction facts (content profile, extractor, word count).
            Optional for backward compatibility with existing construction.
        quality: Quality gate outcome. Optional for backward compatibility.
        semantic: Optional semantic models (e.g. conversation thread).
        diagnostics: Internal debug dict populated by the pipeline.
    """

    clean_html: str
    markdown: str
    metadata: WebMetadata
    word_count: int
    info: ExtractionInfo | None = None
    quality: QualityAssessment | None = None
    semantic: SemanticExtraction | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
