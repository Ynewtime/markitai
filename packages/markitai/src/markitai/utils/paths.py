"""Path utilities for directory management.

This module provides helper functions for creating and managing
output directories used throughout markitai.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from markitai.constants import MARKITAI_META_DIR

_tracked_temp_dirs: list[Path] = []


def _cleanup_tracked_temp_dirs() -> None:
    for temp_dir in _tracked_temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_tracked_temp_dir(prefix: str = "markitai-") -> Path:
    """Create a temp directory that is removed automatically at process exit.

    Used by converter paths that render intermediate images without an
    output directory: the files must stay readable for the rest of the
    conversion, so cleanup is deferred to interpreter shutdown.
    """
    if not _tracked_temp_dirs:
        atexit.register(_cleanup_tracked_temp_dirs)
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    _tracked_temp_dirs.append(temp_dir)
    return temp_dir


def derive_output_name(input_name: str) -> str:
    """Derive the output markdown filename for an input filename.

    Appends ``.md`` to the full input filename so the source format stays
    visible and distinct inputs can never map to the same output:
    ``sample.pdf`` -> ``sample.pdf.md``, ``archive.tar.gz`` ->
    ``archive.tar.gz.md``, ``README`` -> ``README.md``. LLM outputs swap
    the final ``.md`` for ``.llm.md`` (``sample.pdf.llm.md``).

    Args:
        input_name: Input filename (no directory components).

    Returns:
        Output markdown filename.

    Examples:
        >>> derive_output_name("sample.pdf")
        'sample.pdf.md'
        >>> derive_output_name("README")
        'README.md'
    """
    return f"{input_name}.md"


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
