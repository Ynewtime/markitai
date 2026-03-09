"""Path utilities for directory management.

This module provides helper functions for creating and managing
output directories used throughout markitai.
"""

from __future__ import annotations

from pathlib import Path

from markitai.constants import MARKITAI_META_DIR


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The same path (for chaining)

    Examples:
        >>> ensure_dir(Path("/tmp/output"))
        PosixPath('/tmp/output')
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_subdir(parent: Path, name: str) -> Path:
    """Ensure a subdirectory exists under the parent directory.

    Args:
        parent: Parent directory path
        name: Subdirectory name

    Returns:
        Path to the created subdirectory

    Examples:
        >>> ensure_subdir(Path("/tmp/output"), "assets")
        PosixPath('/tmp/output/assets')
    """
    subdir = parent / name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def ensure_assets_dir(output_dir: Path) -> Path:
    """Ensure the assets subdirectory exists under .markitai/.

    Args:
        output_dir: Output directory path

    Returns:
        Path to the assets directory

    Examples:
        >>> ensure_assets_dir(Path("/tmp/output"))
        PosixPath('/tmp/output/.markitai/assets')
    """
    return ensure_subdir(output_dir / MARKITAI_META_DIR, "assets")


def ensure_screenshots_dir(output_dir: Path) -> Path:
    """Ensure the screenshots subdirectory exists under .markitai/.

    Args:
        output_dir: Output directory path

    Returns:
        Path to the screenshots directory

    Examples:
        >>> ensure_screenshots_dir(Path("/tmp/output"))
        PosixPath('/tmp/output/.markitai/screenshots')
    """
    return ensure_subdir(output_dir / MARKITAI_META_DIR, "screenshots")


def ensure_reports_dir(output_dir: Path) -> Path:
    """Ensure the reports subdirectory exists under .markitai/.

    Args:
        output_dir: Output directory path

    Returns:
        Path to the reports directory

    Examples:
        >>> ensure_reports_dir(Path("/tmp/output"))
        PosixPath('/tmp/output/.markitai/reports')
    """
    return ensure_subdir(output_dir / MARKITAI_META_DIR, "reports")
