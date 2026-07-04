"""Path utilities for directory management.

This module provides helper functions for creating and managing
output directories used throughout markitai.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from collections import Counter
from collections.abc import Collection, Sequence
from pathlib import Path

from loguru import logger

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


def derive_output_name(input_name: str, *, avoid: Collection[str] = ()) -> str:
    """Derive the output markdown filename for an input filename.

    Replaces the input extension with ``.md``:
    ``sample.pdf`` -> ``sample.md``, ``note.md`` -> ``note.md``,
    ``README`` -> ``README.md``.

    If the derived name appears in ``avoid`` (it collides with the source
    file itself or with another input mapping to the same name), falls back
    to the legacy append scheme: ``sample.pdf`` -> ``sample.pdf.md``.

    Args:
        input_name: Input filename (no directory components).
        avoid: Output names that must not be produced by extension
            replacement; triggers the legacy fallback.

    Returns:
        Output markdown filename.

    Examples:
        >>> derive_output_name("sample.pdf")
        'sample.md'
        >>> derive_output_name("note.md")
        'note.md'
        >>> derive_output_name("sample.pdf", avoid={"sample.md"})
        'sample.pdf.md'
    """
    stem = Path(input_name).stem
    candidate = f"{stem}.md" if stem else f"{input_name}.md"
    if candidate in avoid:
        return f"{input_name}.md"
    return candidate


def plan_output_names(inputs: Sequence[tuple[Path, Path]]) -> dict[Path, str]:
    """Plan output markdown filenames for one or more conversions.

    Uses extension replacement (see :func:`derive_output_name`) and detects
    collisions up front:

    - Two inputs mapping to the same output name in the same output
      directory (``a.pdf`` + ``a.docx`` -> ``a.md``).
    - The output name resolving to the source file itself (``note.md``
      converted into its own directory).

    Colliding inputs fall back to the legacy ``<name>.md`` append scheme;
    non-colliding inputs keep the replaced name.

    Args:
        inputs: Sequence of ``(input_path, output_dir)`` pairs.

    Returns:
        Mapping of input path to output markdown filename.
    """
    candidates = {
        input_path: derive_output_name(input_path.name)
        for input_path, _output_dir in inputs
    }
    counts = Counter(
        (str(output_dir.resolve()), candidates[input_path])
        for input_path, output_dir in inputs
    )

    planned: dict[Path, str] = {}
    for input_path, output_dir in inputs:
        candidate = candidates[input_path]
        duplicate = counts[(str(output_dir.resolve()), candidate)] > 1
        overwrites_source = (output_dir / candidate).resolve() == input_path.resolve()
        if duplicate or overwrites_source:
            legacy = derive_output_name(input_path.name, avoid={candidate})
            reason = (
                "output would overwrite source file"
                if overwrites_source
                else "output name collides with another input"
            )
            logger.info(
                "Using legacy output name {} for {} ({})",
                legacy,
                input_path.name,
                reason,
            )
            planned[input_path] = legacy
        else:
            planned[input_path] = candidate
    return planned


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
