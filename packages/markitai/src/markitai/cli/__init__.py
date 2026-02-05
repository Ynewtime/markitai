"""CLI package for Markitai.

This package provides the command-line interface for Markitai.

Usage:
    from markitai.cli import app
    from markitai.cli import ui
    from markitai.cli.i18n import t
"""

from __future__ import annotations

# Re-export UI components and i18n
from markitai.cli import i18n, ui

# Re-export CLI app
from markitai.cli.main import app

# Re-export validators from processors
from markitai.cli.processors.validators import (
    warn_case_sensitivity_mismatches as _warn_case_sensitivity_mismatches,
)

# Re-export utilities from refactored modules
from markitai.utils.cli_helpers import (
    compute_task_hash,
    get_report_file_path,
    is_url,
    sanitize_filename,
    url_to_filename,
)
from markitai.utils.output import resolve_output_path
from markitai.utils.progress import ProgressReporter

# Re-export from workflow helpers
from markitai.workflow.helpers import write_images_json

# Re-export types from workflow for backward compatibility
from markitai.workflow.single import ImageAnalysisResult

# Backward compatibility alias (deprecated, use sanitize_filename instead)
_sanitize_filename = sanitize_filename

__all__ = [
    "app",
    "ui",
    "i18n",
    "ProgressReporter",
    "is_url",
    "url_to_filename",
    "sanitize_filename",
    "_sanitize_filename",  # Deprecated alias
    "_warn_case_sensitivity_mismatches",
    "compute_task_hash",
    "get_report_file_path",
    "resolve_output_path",
    "write_images_json",
    "ImageAnalysisResult",
]
