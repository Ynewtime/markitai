"""Workflow module for document processing pipelines."""

from __future__ import annotations

from markitai.workflow.core import (
    ConversionContext,
    ConversionStepResult,
    DocumentConversionError,
    FileSizeError,
    UnsupportedFormatError,
    convert_document_core,
)
from markitai.workflow.helpers import (
    add_basic_frontmatter,
    detect_language,
    merge_llm_usage,
    write_images_json,
)
from markitai.workflow.single import ImageAnalysisResult, SingleFileWorkflow

__all__ = [
    "ConversionContext",
    "ConversionStepResult",
    "DocumentConversionError",
    "FileSizeError",
    "ImageAnalysisResult",
    "SingleFileWorkflow",
    "UnsupportedFormatError",
    "add_basic_frontmatter",
    "convert_document_core",
    "detect_language",
    "merge_llm_usage",
    "write_images_json",
]

# Backward compatibility alias
write_assets_json = write_images_json
