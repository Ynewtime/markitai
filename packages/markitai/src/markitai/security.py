"""Security utilities for Markitai."""

from __future__ import annotations

import json
import os
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
    *,
    follow_symlinks: bool = False,
) -> None:
    """Write text to file atomically using temp file + rename.

    This prevents partial writes and ensures file integrity even if
    the process is interrupted during write.

    Args:
        path: Target file path
        content: Text content to write
        encoding: Text encoding (default: utf-8)
        follow_symlinks: If True, write to the resolved symlink target instead of
            replacing the symlink entry itself.
    """
    path = Path(path)
    if follow_symlinks and path.is_symlink():
        path = path.resolve()
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
    *,
    follow_symlinks: bool = False,
) -> None:
    """Write JSON to file atomically.

    Args:
        path: Target file path
        obj: Object to serialize as JSON
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
        order_func: Optional function to order/transform dict before serialization
        follow_symlinks: If True, write to the resolved symlink target instead of
            replacing the symlink entry itself.
    """
    if order_func is not None and isinstance(obj, dict):
        obj = order_func(obj)
    content = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(
        path,
        content,
        encoding="utf-8",
        follow_symlinks=follow_symlinks,
    )


async def atomic_write_text_async(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    *,
    follow_symlinks: bool = False,
) -> None:
    """Write text to file atomically using temp file + rename (async version).

    This prevents partial writes and ensures file integrity even if
    the process is interrupted during write.

    Args:
        path: Target file path
        content: Text content to write
        encoding: Text encoding (default: utf-8)
        follow_symlinks: If True, write to the resolved symlink target instead of
            replacing the symlink entry itself.
    """
    import aiofiles
    import aiofiles.os

    path = Path(path)
    if follow_symlinks and path.is_symlink():
        path = path.resolve()
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
    *,
    follow_symlinks: bool = False,
) -> None:
    """Write JSON to file atomically (async version).

    Args:
        path: Target file path
        obj: Object to serialize as JSON
        indent: JSON indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
        follow_symlinks: If True, write to the resolved symlink target instead of
            replacing the symlink entry itself.
    """
    content = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii)
    await atomic_write_text_async(
        path,
        content,
        encoding="utf-8",
        follow_symlinks=follow_symlinks,
    )


async def write_bytes_async(
    path: Path,
    data: bytes,
    *,
    follow_symlinks: bool = False,
) -> None:
    """Write bytes to file atomically using temp file + rename.

    Args:
        path: Target file path
        data: Bytes to write
        follow_symlinks: If True, write to the resolved symlink target instead of
            replacing the symlink entry itself.
    """
    import aiofiles

    path = Path(path)
    if follow_symlinks and path.is_symlink():
        path = path.resolve()
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{path.name}.",
        dir=parent,
    )
    try:
        # Close fd and reopen by path (consistent with atomic_write_text_async,
        # avoids fd leak if aiofiles.open fails)
        os.close(fd)
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(data)
        await _replace_with_retry_async(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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


# System-level symlinks that are safe to traverse (e.g. on macOS:
# /tmp -> private/tmp, /var -> private/var, /etc -> private/etc;
# /home -> var/home on some Linux distros).
_SYSTEM_SYMLINK_PATHS = frozenset({"/tmp", "/var", "/etc", "/home"})  # nosec B108 - allowlist of OS symlink roots, not temp file usage


def check_symlink_safety(path: Path, allow_symlinks: bool = False) -> None:
    """Check if a path involves symlinks at any level.

    This function checks the path itself and its parent directories for
    symlinks, using the ORIGINAL (unresolved) path — a resolved path never
    contains symlinks, so checking it would miss writes through a symlinked
    parent (e.g. `linkdir/sub`). Known system-level symlinks
    (e.g. /tmp -> /private/tmp on macOS) are still allowed to avoid
    false positives.

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

    # Check ancestors of the original path for symlinked directories.
    # absolute() (not resolve()) keeps symlinks visible for relative paths.
    if not allow_symlinks:
        for ancestor in path.absolute().parents:
            if str(ancestor) in _SYSTEM_SYMLINK_PATHS:
                continue
            if ancestor.is_symlink():
                if _is_system_symlink(ancestor):
                    continue
                target = ancestor.readlink()
                raise ValueError(
                    f"Nested symlink not allowed: {ancestor} -> {target} (in path {path})"
                )


def _is_system_symlink(link: Path) -> bool:
    """Return True for OS-managed symlinks that are safe to traverse.

    On POSIX, root-owned symlinks (e.g. /var/run -> /run, /etc/alternatives/*)
    are OS artifacts an unprivileged attacker cannot plant, so they don't
    represent the symlink-swap threat this check defends against.
    """
    if os.name != "posix":
        return False
    try:
        return link.lstat().st_uid == 0
    except OSError:
        return False


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
