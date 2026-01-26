"""Markitai utilities."""

from markitai.utils.executor import (
    get_converter_executor,
    run_in_converter_thread,
    shutdown_converter_executor,
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
from markitai.utils.text import normalize_markdown_whitespace

__all__ = [
    "ensure_assets_dir",
    "ensure_dir",
    "ensure_screenshots_dir",
    "ensure_subdir",
    "find_libreoffice",
    "get_converter_executor",
    "get_extension_from_mime",
    "get_mime_type",
    "has_ms_office",
    "normalize_markdown_whitespace",
    "resolve_output_path",
    "run_in_converter_thread",
    "shutdown_converter_executor",
]
