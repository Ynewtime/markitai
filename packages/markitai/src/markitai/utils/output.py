"""Output path utilities for Markitai."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def resolve_name_conflict(
    path: Path,
    on_conflict: str,
    rename_fn: Callable[[int], Path],
) -> Path | None:
    """Resolve a filename conflict using the given strategy.

    This is the single authoritative implementation of the
    skip / overwrite / rename-with-sequential-version-number pattern
    used throughout the codebase.

    Args:
        path: The target file path that already exists.
        on_conflict: Strategy -- "skip", "overwrite", or "rename".
        rename_fn: A callable that receives a sequence number (starting at 2)
            and returns a candidate ``Path``.  Called repeatedly until the
            returned path does not yet exist on disk.

    Returns:
        Resolved path, or ``None`` if the file should be skipped.
    """
    if not path.exists():
        return path

    if on_conflict == "skip":
        return None
    if on_conflict == "overwrite":
        return path

    # rename: find next available sequence number
    seq = 2
    while True:
        candidate = rename_fn(seq)
        if not candidate.exists():
            return candidate
        seq += 1


def resolve_output_path(
    base_path: Path,
    on_conflict: str,
) -> Path | None:
    """Resolve output path based on conflict strategy.

    Args:
        base_path: The original output file path
        on_conflict: Conflict resolution strategy ("skip", "overwrite", "rename")

    Returns:
        Resolved path, or None if file should be skipped.
        For rename strategy: file.pdf.md -> file.pdf.v2.md -> file.pdf.v3.md
        For rename with .llm.md: file.pdf.llm.md -> file.pdf.v2.llm.md
    """
    # Determine the markitai suffix (.md or .llm.md) for rename
    name = base_path.name
    if name.endswith(".llm.md"):
        base_stem = name[:-7]  # Remove ".llm.md" -> "file.pdf"
        markitai_suffix = ".llm.md"
    else:
        base_stem = name[:-3]  # Remove ".md" -> "file.pdf"
        markitai_suffix = ".md"

    def _rename(seq: int) -> Path:
        return base_path.parent / f"{base_stem}.v{seq}{markitai_suffix}"

    return resolve_name_conflict(base_path, on_conflict, _rename)
