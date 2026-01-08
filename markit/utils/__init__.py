"""Utility module for MarkIt."""

from markit.utils.concurrency import BatchProcessor, ConcurrencyManager, TaskResult
from markit.utils.fs import (
    atomic_write,
    clean_empty_directories,
    compute_file_hash,
    compute_quick_hash,
    copy_file_safe,
    discover_files,
    ensure_directory,
    format_size,
    get_file_size_human,
    get_relative_path,
    get_unique_path,
    is_hidden,
    iter_files,
    move_file_safe,
    safe_filename,
    temporary_directory,
)

__all__ = [
    # Concurrency
    "ConcurrencyManager",
    "BatchProcessor",
    "TaskResult",
    # File system
    "ensure_directory",
    "safe_filename",
    "get_unique_path",
    "compute_file_hash",
    "compute_quick_hash",
    "discover_files",
    "iter_files",
    "get_relative_path",
    "copy_file_safe",
    "move_file_safe",
    "atomic_write",
    "temporary_directory",
    "get_file_size_human",
    "format_size",
    "is_hidden",
    "clean_empty_directories",
]
