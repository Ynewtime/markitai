"""CLI package for Markitai.

This package provides the command-line interface for Markitai.

Usage:
    from markitai.cli import app
    from markitai.cli import ui
    from markitai.cli.i18n import t
"""

from __future__ import annotations

from typing import Any

# Re-export UI components and i18n (lightweight, always needed)
from markitai.cli import i18n, ui

# Re-export CLI app
from markitai.cli.main import app

# Re-export utilities from refactored modules (lightweight)
from markitai.utils.cli_helpers import (
    compute_task_hash,
    get_report_file_path,
    is_url,
    sanitize_filename,
    url_to_filename,
)
from markitai.utils.output import resolve_output_path
from markitai.utils.progress import ProgressReporter

# Backward compatibility alias (deprecated, use sanitize_filename instead)
_sanitize_filename = sanitize_filename

__all__ = [
    "ImageAnalysisResult",
    "ProgressReporter",
    "_sanitize_filename",  # Deprecated alias
    "_warn_case_sensitivity_mismatches",
    "app",
    "compute_task_hash",
    "get_report_file_path",
    "i18n",
    "is_url",
    "resolve_output_path",
    "sanitize_filename",
    "ui",
    "url_to_filename",
    "write_images_json",
]


def __getattr__(name: str) -> Any:
    """Lazy import heavy re-exports to avoid pulling in converters at CLI startup."""
    if name == "_warn_case_sensitivity_mismatches":
        from markitai.cli.processors.validators import (
            warn_case_sensitivity_mismatches,
        )

        return warn_case_sensitivity_mismatches
    if name == "write_images_json":
        from markitai.workflow.helpers import write_images_json

        return write_images_json
    if name == "ImageAnalysisResult":
        from markitai.workflow.single import ImageAnalysisResult

        return ImageAnalysisResult
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
