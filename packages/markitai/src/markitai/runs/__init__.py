"""Run orchestration layer.

First slice (Phase 4A): the report schema and exit-code matrix shared by
the single-file, single-URL, and URL-batch processing paths, single-sourced
so the CLI processors stop carrying near-identical inline copies.

Layering: ``markitai.runs`` sits below ``markitai.cli`` and must never
import it (enforced by the import-linter contracts in the root
pyproject.toml).
"""

from markitai.runs.report import (
    REPORT_VERSION,
    build_single_report,
    build_url_batch_report,
    resolve_exit_code,
)
from markitai.runs.types import Outcome

__all__ = [
    "REPORT_VERSION",
    "Outcome",
    "build_single_report",
    "build_url_batch_report",
    "resolve_exit_code",
]
