"""Workflow module for document processing pipelines."""

from __future__ import annotations

from markit.workflow.helpers import (
    add_basic_frontmatter,
    detect_language,
    merge_llm_usage,
    write_assets_json,
)
from markit.workflow.single import ImageAnalysisResult, SingleFileWorkflow

__all__ = [
    "ImageAnalysisResult",
    "SingleFileWorkflow",
    "add_basic_frontmatter",
    "detect_language",
    "merge_llm_usage",
    "write_assets_json",
]
