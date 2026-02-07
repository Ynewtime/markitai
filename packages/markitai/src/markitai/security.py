"""Security utilities for Markitai."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.constants import DEFAULT_JSON_INDENT

# Windows-specific retry settings for file operations
_WINDOWS_RETRY_COUNT = 5
_WINDOWS_RETRY_DELAY = 0.05  # 50ms


def _replace_with_retry(src: str, dst: Path) -> None:
    """Replace file with retry logic for Windows file locking.

    On Windows, os.replace() can fail with PermissionError when the target
    file is briefly locked by another process (e.g., antivirus, indexer,
    or concurrent writes). This function retries the operation.

    Args:
        src: Source file path (temp file)
        dst: Destination file path
    """
    if sys.platform != "win32":
        os.replace(src, dst)
        return

    last_error: OSError | None = None
    for attempt in range(_WINDOWS_RETRY_COUNT):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last_error = e
            if attempt < _WINDOWS_RETRY_COUNT - 1:
                time.sleep(_WINDOWS_RETRY_DELAY * (attempt + 1))  # Exponential backoff

    # All retries failed
    if last_error:
        raise last_error


async def _replace_with_retry_async(src: str, dst: Path) -> None:
    """Async version of _replace_with_retry for Windows file locking.

    Args:
        src: Source file path (temp file)
        dst: Destination file path
    """
    import asyncio

    import aiofiles.os

    if sys.platform != "win32":
        await aiofiles.os.replace(src, dst)
        return

    last_error: OSError | None = None
    for attempt in range(_WINDOWS_RETRY_COUNT):
        try:
            await aiofiles.os.replace(src, dst)
            return
        except PermissionError as e:
            last_error = e
            if attempt < _WINDOWS_RETRY_COUNT - 1:
                await asyncio.sleep(_WINDOWS_RETRY_DELAY * (attempt + 1))

    # All retries failed
    if last_error:
        raise last_error


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
    fd_closed = False
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            fd_closed = True  # fdopen takes ownership of fd
            f.write(content)
        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        # On Windows, use retry logic to handle file locking
        _replace_with_retry(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        # Close fd if fdopen failed (it didn't take ownership)
        if not fd_closed:
            try:
                os.close(fd)
            except OSError:
                pass
        # On Windows, may need retry for unlink due to file locking
        for _ in range(_WINDOWS_RETRY_COUNT if sys.platform == "win32" else 1):
            try:
                os.unlink(tmp_path)
                break
            except OSError:
                if sys.platform == "win32":
                    time.sleep(_WINDOWS_RETRY_DELAY)
        raise


def atomic_write_json(
    path: Path,
    obj: Any,
    indent: int = DEFAULT_JSON_INDENT,
    ensure_ascii: bool = False,
    order_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Write JSON to file atomically.

    Args:
        path: Target file path
        obj: Object to serialize as JSON
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
        order_func: Optional function to order/transform dict before serialization
    """
    if order_func is not None and isinstance(obj, dict):
        obj = order_func(obj)
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
        # On Windows, use retry logic to handle file locking
        await _replace_with_retry_async(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        try:
            import aiofiles.os

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

    This function checks the path itself for symlinks. Parent directories are
    checked via the resolved path to avoid false positives on system-level
    symlinks (e.g. /tmp -> /private/tmp on macOS).

    Args:
        path: Path to check
        allow_symlinks: If False, raises error on symlinks

    Raises:
        ValueError: If symlinks are not allowed and the path is a symlink
    """
    # Check the path itself (unresolved, to catch user-created symlinks)
    if path.is_symlink():
        if not allow_symlinks:
            target = path.readlink()
            raise ValueError(f"Symlink not allowed: {path} -> {target}")
        else:
            logger.warning(f"Symlink detected: {path} -> {path.readlink()}")
            return  # If symlinks allowed, no need to check further

    # Check parent directories for nested symlinks using the resolved path.
    # This avoids false positives on system-level symlinks like /tmp -> /private/tmp
    # on macOS, while still catching user-created symlinks in the resolved tree.
    if not allow_symlinks:
        resolved = path.resolve()
        checked_parts: list[Path] = []
        for part in resolved.parts:
            checked_parts.append(
                Path(part) if not checked_parts else checked_parts[-1] / part
            )
            current_path = checked_parts[-1]
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

    # Remove potential usernames in paths (must run BEFORE generic path replacement)
    msg = re.sub(r"/home/[^/\s]+/", "/home/[USER]/", msg)
    msg = re.sub(r"C:\\Users\\[^\\]+\\", r"C:\\Users\\[USER]\\", msg)

    # Remove absolute paths (Unix style)
    msg = re.sub(r"/[a-zA-Z0-9_\-./]+", "[PATH]", msg)

    # Remove absolute paths (Windows style, including UNC paths)
    msg = re.sub(r"[A-Za-z]:\\[a-zA-Z0-9_\-\\. ]+", "[PATH]", msg)
    msg = re.sub(r"\\\\[a-zA-Z0-9_\-\\. ]+", "[PATH]", msg)

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
