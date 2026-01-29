"""CLI processors package.

This package contains processing functions for different input types:
- file: Single file processing
- url: URL and batch URL processing
- llm: LLM-based processing helpers
- validators: Validation helpers
- batch: Batch processing orchestration
"""

from __future__ import annotations

from markitai.cli.processors.batch import process_batch
from markitai.cli.processors.file import process_single_file
from markitai.cli.processors.llm import (
    analyze_images_with_llm,
    enhance_document_with_vision,
    format_standalone_image_markdown,
    process_with_llm,
)
from markitai.cli.processors.url import process_url, process_url_batch
from markitai.cli.processors.validators import (
    check_agent_browser_for_urls,
    check_vision_model_config,
    warn_case_sensitivity_mismatches,
)

__all__ = [
    # File processing
    "process_single_file",
    # URL processing
    "process_url",
    "process_url_batch",
    # LLM processing
    "process_with_llm",
    "analyze_images_with_llm",
    "enhance_document_with_vision",
    "format_standalone_image_markdown",
    # Validators
    "check_vision_model_config",
    "check_agent_browser_for_urls",
    "warn_case_sensitivity_mismatches",
    # Batch processing
    "process_batch",
]
