"""Markitai utilities."""

from markitai.utils.cli_helpers import (
    compute_task_hash,
    get_report_file_path,
    is_url,
    sanitize_filename,
    url_to_filename,
)
from markitai.utils.executor import (
    get_converter_executor,
    run_in_converter_thread,
    shutdown_converter_executor,
)
from markitai.utils.frontmatter import (
    build_frontmatter_dict,
    extract_title_from_content,
    frontmatter_to_yaml,
)
from markitai.utils.mime import get_extension_from_mime, get_mime_type
from markitai.utils.office import find_libreoffice, has_ms_office
from markitai.utils.output import resolve_output_path
from markitai.utils.paths import (
    ensure_assets_dir,
    ensure_dir,
    ensure_screenshots_dir,
    ensure_subdir,
)
from markitai.utils.progress import ProgressReporter
from markitai.utils.text import (
    clean_control_characters,
    format_error_message,
    normalize_markdown_whitespace,
)

__all__ = [
    # CLI helpers
    "compute_task_hash",
    "get_report_file_path",
    "is_url",
    "ProgressReporter",
    "sanitize_filename",
    "url_to_filename",
    # Executor
    "get_converter_executor",
    "run_in_converter_thread",
    "shutdown_converter_executor",
    # MIME
    "get_extension_from_mime",
    "get_mime_type",
    # Office
    "find_libreoffice",
    "has_ms_office",
    # Output
    "resolve_output_path",
    # Paths
    "ensure_assets_dir",
    "ensure_dir",
    "ensure_screenshots_dir",
    "ensure_subdir",
    # Text
    "clean_control_characters",
    "format_error_message",
    "normalize_markdown_whitespace",
    # Frontmatter
    "build_frontmatter_dict",
    "extract_title_from_content",
    "frontmatter_to_yaml",
]
