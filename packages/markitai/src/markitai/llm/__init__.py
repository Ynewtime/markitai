"""LLM integration package.

This package provides LLM integration functionality including:
- Type definitions for LLM responses and analysis results
- Caching system for LLM responses
- Model information and cost tracking
- Main LLM processor
- Content protection utilities

Usage:
    from markitai.llm import LLMProcessor, LLMResponse, ImageAnalysis
"""

from __future__ import annotations

# Re-export cache classes from refactored module
from markitai.llm.cache import (
    ContentCache,
    PersistentCache,
    SQLiteCache,
)

# Re-export content utilities from refactored module
from markitai.llm.content import (
    clean_frontmatter,
    extract_protected_content,
    fix_malformed_image_refs,
    protect_content,
    protect_image_positions,
    remove_uncommented_screenshots,
    restore_image_positions,
    smart_truncate,
    split_text_by_pages,
    unprotect_content,
)

# Re-export model utilities from refactored module
from markitai.llm.models import (
    MarkitaiLLMLogger,
    context_display_name,
    get_model_info_cached,
    get_model_max_output_tokens,
    get_response_cost,
)

# Re-export main processor from processor module
from markitai.llm.processor import LLMProcessor

# Re-export all public types from refactored module
from markitai.llm.types import (
    BatchImageAnalysisResult,
    DocumentProcessResult,
    EnhancedDocumentResult,
    Frontmatter,
    ImageAnalysis,
    ImageAnalysisResult,
    LLMResponse,
    LLMRuntime,
    SingleImageResult,
)

__all__ = [
    # Processor
    "LLMProcessor",
    # Types
    "LLMRuntime",
    "LLMResponse",
    "ImageAnalysis",
    "ImageAnalysisResult",
    "SingleImageResult",
    "BatchImageAnalysisResult",
    "Frontmatter",
    "DocumentProcessResult",
    "EnhancedDocumentResult",
    # Cache
    "SQLiteCache",
    "PersistentCache",
    "ContentCache",
    # Content utilities
    "extract_protected_content",
    "protect_content",
    "unprotect_content",
    "protect_image_positions",
    "restore_image_positions",
    "fix_malformed_image_refs",
    "remove_uncommented_screenshots",
    "clean_frontmatter",
    "smart_truncate",
    "split_text_by_pages",
    # Models
    "get_model_info_cached",
    "get_model_max_output_tokens",
    "get_response_cost",
    "context_display_name",
    "MarkitaiLLMLogger",
]
