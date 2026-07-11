"""Run orchestration layer.

First slice (Phase 4A): the report schema and exit-code matrix shared by
the single-file, single-URL, and URL-batch processing paths, single-sourced
so the CLI processors stop carrying near-identical inline copies.

Second slice (Phase 4B): the shared output-target skeleton (``-o``
interpretation, stdout-mode temp dirs, explicit-file finalization) and the
stdout asset-reference helpers used by both single-input processors.

Layering: ``markitai.runs`` sits below ``markitai.cli`` and must never
import it (enforced by the import-linter contracts in the root
pyproject.toml).
"""

from markitai.runs.output import (
    ASSET_REF_PATTERN,
    OutputTarget,
    finalize_explicit_output,
    normalize_temp_asset_refs,
    prepare_output_target,
    resolve_asset_references,
    split_output_file_target,
    warn_ephemeral_links,
)
from markitai.runs.report import (
    REPORT_VERSION,
    build_single_report,
    build_url_batch_report,
    resolve_exit_code,
)
from markitai.runs.types import Outcome

__all__ = [
    "ASSET_REF_PATTERN",
    "REPORT_VERSION",
    "Outcome",
    "OutputTarget",
    "build_single_report",
    "build_url_batch_report",
    "finalize_explicit_output",
    "normalize_temp_asset_refs",
    "prepare_output_target",
    "resolve_asset_references",
    "resolve_exit_code",
    "split_output_file_target",
    "warn_ephemeral_links",
]
