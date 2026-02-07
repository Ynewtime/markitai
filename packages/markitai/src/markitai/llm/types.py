"""Type definitions for LLM module.

This module contains all data classes and Pydantic models used by the LLM system.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from markitai.types import LLMUsageByModel

from markitai.constants import DEFAULT_IO_CONCURRENCY
from markitai.utils.text import clean_control_characters


@dataclass
class LLMRuntime:
    """Global LLM runtime with shared concurrency control.

    This allows multiple LLMProcessor instances to share semaphores
    for rate limiting across the entire application.

    Supports separate concurrency limits for:
    - LLM API calls (rate-limited by provider)
    - I/O operations (disk reads, can be higher)

    Usage:
        runtime = LLMRuntime(concurrency=10, io_concurrency=20)
        processor1 = LLMProcessor(config, runtime=runtime)
        processor2 = LLMProcessor(config, runtime=runtime)
        # Both processors share the same semaphores
    """

    concurrency: int
    io_concurrency: int = DEFAULT_IO_CONCURRENCY
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _io_semaphore: asyncio.Semaphore | None = field(
        default=None, init=False, repr=False
    )

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get or create the shared LLM concurrency semaphore."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
        return self._semaphore

    @property
    def io_semaphore(self) -> asyncio.Semaphore:
        """Get or create the shared I/O concurrency semaphore."""
        if self._io_semaphore is None:
            self._io_semaphore = asyncio.Semaphore(self.io_concurrency)
        return self._io_semaphore


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class ImageAnalysis:
    """Result of image analysis.

    Attributes:
        caption: Short alt text for accessibility
        description: Detailed markdown description
        extracted_text: Text extracted from image (OCR)
        llm_usage: LLM usage statistics in format:
            {"<model-name>": {"requests": N, "input_tokens": N,
             "output_tokens": N, "cost_usd": N}}
    """

    caption: str  # Short alt text
    description: str  # Detailed description
    extracted_text: str | None = None  # Text extracted from image
    llm_usage: LLMUsageByModel | None = None  # LLM usage stats


class _CleanedStringMixin(BaseModel):
    """Mixin that cleans control characters from string fields before validation.

    Subclasses should list their string fields in the ``clean_control_chars``
    validator decorator so that incoming values are sanitized automatically.
    """

    @field_validator("*", mode="before")
    @classmethod
    def clean_control_chars(cls, v: str | None) -> str | None:
        """Remove control characters that can cause JSON parsing errors."""
        if not isinstance(v, str):
            return v
        return clean_control_characters(v)


class ImageAnalysisResult(_CleanedStringMixin):
    """Pydantic model for structured image analysis output."""

    caption: str = Field(description="Short alt text for the image (10-30 characters)")
    description: str = Field(description="Detailed markdown description of the image")
    extracted_text: str | None = Field(
        default=None,
        description="Text extracted from the image, preserving original layout",
    )


class SingleImageResult(_CleanedStringMixin):
    """Result for a single image in batch analysis."""

    image_index: int = Field(description="Index of the image (1-based)")
    caption: str = Field(description="Short alt text for the image (10-30 characters)")
    description: str = Field(description="Detailed markdown description of the image")
    extracted_text: str | None = Field(
        default=None,
        description="Text extracted from the image, preserving original layout",
    )


class BatchImageAnalysisResult(BaseModel):
    """Result for batch image analysis."""

    images: list[SingleImageResult] = Field(
        description="Analysis results for each image"
    )


class Frontmatter(BaseModel):
    """Pydantic model for LLM-generated frontmatter fields.

    Note: title, source, and markitai_processed are added programmatically
    by build_frontmatter_dict(), not part of this LLM response model.
    Only description and tags are LLM-generated.
    """

    description: str = Field(
        description="Brief summary of the document (100 chars max)"
    )
    tags: list[str] = Field(description="Related tags (3-5 items)")

    @field_validator("description", mode="before")
    @classmethod
    def clean_control_chars(cls, v: str | None) -> str:
        """Remove control characters and validate non-empty description."""
        if v is None:
            raise ValueError("description is required and cannot be null")
        cleaned = clean_control_characters(v)
        if not cleaned or not cleaned.strip():
            raise ValueError(
                "description cannot be empty - provide a brief summary of the document"
            )
        return cleaned

    @field_validator("tags", mode="before")
    @classmethod
    def clean_tags_control_chars(cls, v: list[str] | None) -> list[str]:
        """Remove control characters and validate non-empty tags."""
        if v is None:
            raise ValueError("tags is required and cannot be null")
        if not v:
            raise ValueError(
                "tags cannot be empty - provide 3-5 related tags for the document"
            )
        cleaned = [clean_control_characters(tag) for tag in v if tag and tag.strip()]
        if not cleaned:
            raise ValueError(
                "tags cannot be empty after cleaning - provide 3-5 valid tags"
            )
        return cleaned


class DocumentProcessResult(_CleanedStringMixin):
    """LLM document processing result."""

    cleaned_markdown: str = Field(
        description=(
            "Format-optimized Markdown document content. "
            "Contains only actual document content, without any processing instructions or prompt text."
        )
    )
    frontmatter: Frontmatter = Field(
        description="Document metadata: title, summary, tags"
    )


# EnhancedDocumentResult is structurally identical to DocumentProcessResult
# (same fields: cleaned_markdown, frontmatter). Kept as an alias for backward
# compatibility with code that references the name explicitly.
EnhancedDocumentResult = DocumentProcessResult
