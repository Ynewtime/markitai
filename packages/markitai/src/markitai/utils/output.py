"""Output path utilities for Markitai."""

from __future__ import annotations

from pathlib import Path


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
        This ensures files sort in natural order (A-Z).
    """
    if not base_path.exists():
        return base_path

    if on_conflict == "skip":
        return None
    elif on_conflict == "overwrite":
        return base_path
    else:  # rename
        # Parse filename to insert version number before .md/.llm.md suffix
        # e.g., "file.pdf.md" -> "file.pdf.v2.md" -> "file.pdf.v3.md"
        # e.g., "file.pdf.llm.md" -> "file.pdf.v2.llm.md"
        # This ensures files sort in natural A-Z order (.md < .v2.md < .v3.md)
        name = base_path.name

        # Determine the markitai suffix (.md or .llm.md)
        if name.endswith(".llm.md"):
            base_stem = name[:-7]  # Remove ".llm.md" -> "file.pdf"
            markitai_suffix = ".llm.md"
        else:
            base_stem = name[:-3]  # Remove ".md" -> "file.pdf"
            markitai_suffix = ".md"

        # Find next available sequence number
        seq = 2
        while True:
            new_name = f"{base_stem}.v{seq}{markitai_suffix}"
            new_path = base_path.parent / new_name
            if not new_path.exists():
                return new_path
            seq += 1
