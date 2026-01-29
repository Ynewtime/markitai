"""CLI helper utilities.

This module contains utility functions used by the CLI module.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# URL pattern for detecting URLs
_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    """Check if string is a URL (http:// or https://).

    Args:
        s: String to check

    Returns:
        True if the string starts with http:// or https://
    """
    return bool(_URL_PATTERN.match(s))


def url_to_filename(url: str) -> str:
    """Generate a safe filename from URL.

    Examples:
        https://example.com/page.html -> page.html.md
        https://example.com/path/to/doc -> doc.md
        https://example.com/ -> example_com.md
        https://youtube.com/watch?v=abc -> youtube_com_watch.md

    Args:
        url: URL to convert

    Returns:
        Safe filename with .md extension
    """
    parsed = urlparse(url)

    # Try to get filename from path
    path = parsed.path.rstrip("/")
    if path:
        # Get last segment of path
        filename = path.split("/")[-1]
        if filename:
            # Sanitize for cross-platform compatibility
            filename = sanitize_filename(filename)
            return f"{filename}.md"

    # Fallback: use domain name
    domain = parsed.netloc.replace(".", "_").replace(":", "_")
    path_part = parsed.path.strip("/").replace("/", "_")[:50]  # limit length
    if path_part:
        return f"{sanitize_filename(domain)}_{sanitize_filename(path_part)}.md"
    return f"{sanitize_filename(domain)}.md"


def sanitize_filename(name: str) -> str:
    """Sanitize filename for cross-platform compatibility.

    Removes or replaces characters that are invalid on Windows/Linux/macOS.

    Args:
        name: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Characters invalid on Windows: \ / : * ? " < > |
    # Also replace other problematic characters
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    # Remove leading/trailing spaces and dots (Windows issue)
    name = name.strip(". ")
    # Limit length (255 is common max, but leave room for .md extension)
    if len(name) > 200:
        name = name[:200]
    return name or "unnamed"


def compute_task_hash(
    input_path: Path,
    output_dir: Path,
    options: dict[str, Any] | None = None,
) -> str:
    """Compute hash from task input parameters.

    Hash is based on:
    - input_path (resolved)
    - output_dir (resolved)
    - key task options (llm, ocr, etc.)

    This ensures different parameter combinations produce different hashes.

    Args:
        input_path: Input file or directory path
        output_dir: Output directory path
        options: Task options dict (llm, ocr, etc.)

    Returns:
        6-character hex hash string
    """
    # Extract key options that affect output
    key_options = {}
    if options:
        key_options = {
            k: v
            for k, v in options.items()
            if k
            in (
                "llm",
                "ocr",
                "screenshot",
                "alt",
                "desc",
            )
        }

    hash_params = {
        "input": str(input_path.resolve()),
        "output": str(output_dir.resolve()),
        "options": key_options,
    }
    hash_str = json.dumps(hash_params, sort_keys=True)
    return hashlib.md5(hash_str.encode(), usedforsecurity=False).hexdigest()[:6]


def get_report_file_path(
    output_dir: Path,
    task_hash: str,
    on_conflict: str = "rename",
) -> Path:
    """Generate report file path based on task hash.

    Format: reports/markitai.<hash>.report.json
    Respects on_conflict strategy for rename.

    Args:
        output_dir: Output directory
        task_hash: Task hash string
        on_conflict: Conflict resolution strategy

    Returns:
        Path to the report file
    """
    reports_dir = output_dir / "reports"
    base_path = reports_dir / f"markitai.{task_hash}.report.json"

    if not base_path.exists():
        return base_path

    if on_conflict == "skip":
        return base_path  # Will be handled by caller
    elif on_conflict == "overwrite":
        return base_path
    else:  # rename
        seq = 2
        while True:
            new_path = reports_dir / f"markitai.{task_hash}.v{seq}.report.json"
            if not new_path.exists():
                return new_path
            seq += 1
