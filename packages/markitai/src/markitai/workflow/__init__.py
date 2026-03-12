"""Workflow module for document processing pipelines."""

from __future__ import annotations

from typing import Any

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
    "merge_llm_usage",
    "write_assets_json",
    "write_images_json",
]


def __getattr__(name: str) -> Any:
    """Lazy import heavy submodules to avoid pulling in converters at CLI startup."""
    if name in {
        "ConversionContext",
        "ConversionStepResult",
        "DocumentConversionError",
        "FileSizeError",
        "UnsupportedFormatError",
        "convert_document_core",
    }:
        from markitai.workflow import core as _core

        return getattr(_core, name)
    if name in {"add_basic_frontmatter", "merge_llm_usage", "write_images_json"}:
        from markitai.workflow import helpers as _helpers

        return getattr(_helpers, name)
    if name == "write_assets_json":
        from markitai.workflow.helpers import write_images_json

        return write_images_json
    if name in {"ImageAnalysisResult", "SingleFileWorkflow"}:
        from markitai.workflow import single as _single

        return getattr(_single, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
