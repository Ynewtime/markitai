"""Security utilities for Markit."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markit.constants import DEFAULT_JSON_INDENT

if TYPE_CHECKING:
    pass


def atomic_write_text(
    path: Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """Write text to file atomically using temp file + rename.

    This prevents partial writes and ensures file integrity even if
    the process is interrupted during write.

    Args:
        path: Target file path
        content: Text content to write
        encoding: Text encoding (default: utf-8)
    """
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{path.name}.",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(
    path: Path,
    obj: Any,
    indent: int = DEFAULT_JSON_INDENT,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON to file atomically.

    Args:
        path: Target file path
        obj: Object to serialize as JSON
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
    """
    content = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(path, content, encoding="utf-8")


async def atomic_write_text_async(
    path: Path,
    content: str,
    encoding: str = "utf-8",
) -> None:
    """Write text to file atomically using temp file + rename (async version).

    This prevents partial writes and ensures file integrity even if
    the process is interrupted during write.

    Args:
        path: Target file path
        content: Text content to write
        encoding: Text encoding (default: utf-8)
    """
    import aiofiles
    import aiofiles.os

    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{path.name}.",
        dir=parent,
    )
    try:
        # Close fd and use aiofiles for async write
        os.close(fd)
        async with aiofiles.open(tmp_path, "w", encoding=encoding) as f:
            await f.write(content)
        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        await aiofiles.os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        try:
            await aiofiles.os.remove(tmp_path)
        except OSError:
            pass
        raise


async def atomic_write_json_async(
    path: Path,
    obj: Any,
    indent: int = DEFAULT_JSON_INDENT,
    ensure_ascii: bool = False,
) -> None:
    """Write JSON to file atomically (async version).

    Args:
        path: Target file path
        obj: Object to serialize as JSON
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
    """
    content = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    await atomic_write_text_async(path, content, encoding="utf-8")


async def write_bytes_async(path: Path, data: bytes) -> None:
    """Write bytes to file asynchronously.

    Args:
        path: Target file path
        data: Bytes to write
    """
    import aiofiles

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(path, "wb") as f:
        await f.write(data)


def escape_glob_pattern(s: str) -> str:
    """Escape special glob characters in a string.

    Args:
        s: String that may contain glob special characters

    Returns:
        Escaped string safe for use in glob patterns
    """
    # Escape glob special characters: [ ] * ?
    return s.translate(
        str.maketrans(
            {
                "[": "[[]",
                "]": "[]]",
                "*": "[*]",
                "?": "[?]",
            }
        )
    )


def validate_path_within_base(path: Path, base_dir: Path) -> Path:
    """Validate that a path is within the base directory.

    Args:
        path: Path to validate
        base_dir: Base directory that path must be within

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path is outside base directory
    """
    resolved = path.resolve()
    base_resolved = base_dir.resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path traversal detected: {path} is outside {base_dir}")

    return resolved


def check_symlink_safety(path: Path, allow_symlinks: bool = False) -> None:
    """Check if a path involves symlinks at any level.

    This function checks not just the final path, but all parent directories
    to detect nested symlinks that could be used for path traversal.

    Args:
        path: Path to check
        allow_symlinks: If False, raises error on symlinks

    Raises:
        ValueError: If symlinks are not allowed and any path component is a symlink
    """
    # Check the path itself
    if path.is_symlink():
        if not allow_symlinks:
            target = path.readlink()
            raise ValueError(f"Symlink not allowed: {path} -> {target}")
        else:
            logger.warning(f"Symlink detected: {path} -> {path.readlink()}")
            return  # If symlinks allowed, no need to check further

    # Check all parent directories for nested symlinks
    if not allow_symlinks:
        checked_parts: list[Path] = []
        for part in path.parts:
            checked_parts.append(
                Path(part) if not checked_parts else checked_parts[-1] / part
            )
            current_path = checked_parts[-1]
            # Only check if path exists and is absolute enough to be meaningful
            if (
                len(checked_parts) > 1
                and current_path.exists()
                and current_path.is_symlink()
            ):
                target = current_path.readlink()
                raise ValueError(
                    f"Nested symlink not allowed: {current_path} -> {target} (in path {path})"
                )


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error message to remove sensitive information.

    Args:
        error: Exception to sanitize

    Returns:
        Sanitized error message
    """
    msg = str(error)

    # Remove absolute paths (Unix style)
    msg = re.sub(r"/[a-zA-Z0-9_\-./]+", "[PATH]", msg)

    # Remove absolute paths (Windows style)
    msg = re.sub(r"[A-Za-z]:\\[a-zA-Z0-9_\-\\. ]+", "[PATH]", msg)

    # Remove potential usernames in paths
    msg = re.sub(r"/home/[^/\s]+/", "/home/[USER]/", msg)
    msg = re.sub(r"C:\\Users\\[^\\]+\\", r"C:\\Users\\[USER]\\", msg)

    return msg


def validate_file_size(path: Path, max_size_bytes: int) -> None:
    """Validate that a file is within size limits.

    Args:
        path: Path to file
        max_size_bytes: Maximum allowed size in bytes

    Raises:
        ValueError: If file exceeds size limit
    """
    if not path.exists():
        return

    size = path.stat().st_size
    if size > max_size_bytes:
        raise ValueError(
            f"File too large: {path.name} is {size} bytes (max: {max_size_bytes} bytes)"
        )


# Re-export size limits from constants for backward compatibility
from markit.constants import MAX_DOCUMENT_SIZE as MAX_DOCUMENT_SIZE
from markit.constants import MAX_IMAGE_SIZE as MAX_IMAGE_SIZE
from markit.constants import MAX_STATE_FILE_SIZE as MAX_STATE_FILE_SIZE
from markit.constants import MAX_TOTAL_IMAGES_SIZE as MAX_TOTAL_IMAGES_SIZE
