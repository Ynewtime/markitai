"""File system utilities for MarkIt.

Provides safe file operations, path handling, and file discovery functions.
"""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from markit.config.constants import SUPPORTED_EXTENSIONS
from markit.utils.logging import get_logger

log = get_logger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str, max_length: int = 255) -> str:
    """Create a safe filename by removing/replacing problematic characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Safe filename
    """
    # Characters to remove or replace
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        "\0": "",
    }

    result = filename
    for old, new in replacements.items():
        result = result.replace(old, new)

    # Remove leading/trailing dots and spaces
    result = result.strip(". ")

    # Truncate if too long (preserve extension)
    if len(result) > max_length:
        stem = Path(result).stem
        suffix = Path(result).suffix
        max_stem = max_length - len(suffix)
        result = stem[:max_stem] + suffix

    return result


def get_unique_path(path: Path) -> Path:
    """Get a unique path by adding a counter suffix if path exists.

    Args:
        path: Original path

    Returns:
        Unique path that doesn't exist
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_quick_hash(file_path: Path) -> str:
    """Compute a quick hash based on file metadata.

    Useful for change detection without reading entire file.

    Args:
        file_path: Path to file

    Returns:
        Hash string based on size and mtime
    """
    stat = file_path.stat()
    data = f"{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(data.encode()).hexdigest()


def discover_files(
    directory: Path,
    recursive: bool = False,
    extensions: set[str] | None = None,
    include_pattern: str | None = None,
    exclude_pattern: str | None = None,
) -> list[Path]:
    """Discover files in a directory.

    Args:
        directory: Directory to search
        recursive: Search subdirectories
        extensions: File extensions to include (default: SUPPORTED_EXTENSIONS)
        include_pattern: Glob pattern for files to include
        exclude_pattern: Glob pattern for files to exclude

    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = set(SUPPORTED_EXTENSIONS.keys())

    files = []
    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        glob_pattern = f"{pattern}{ext}"
        for file_path in directory.glob(glob_pattern):
            if file_path.is_file():
                # Apply include pattern
                if include_pattern:
                    if not file_path.match(include_pattern):
                        continue

                # Apply exclude pattern
                if exclude_pattern:
                    if file_path.match(exclude_pattern):
                        continue

                files.append(file_path)

    # Sort for consistent ordering
    files.sort()
    return files


def iter_files(
    directory: Path,
    recursive: bool = False,
    extensions: set[str] | None = None,
) -> Iterator[Path]:
    """Iterate over files in a directory.

    Args:
        directory: Directory to search
        recursive: Search subdirectories
        extensions: File extensions to include

    Yields:
        File paths
    """
    if extensions is None:
        extensions = set(SUPPORTED_EXTENSIONS.keys())

    if recursive:
        for root, _, files in os.walk(directory):
            root_path = Path(root)
            for filename in files:
                file_path = root_path / filename
                if file_path.suffix.lower() in extensions:
                    yield file_path
    else:
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                yield file_path


def get_relative_path(file_path: Path, base_path: Path) -> Path:
    """Get relative path from base, handling edge cases.

    Args:
        file_path: File path
        base_path: Base path

    Returns:
        Relative path
    """
    try:
        return file_path.relative_to(base_path)
    except ValueError:
        # file_path is not relative to base_path
        return file_path


def copy_file_safe(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """Safely copy a file.

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Allow overwriting existing files

    Returns:
        Destination path

    Raises:
        FileExistsError: If destination exists and overwrite is False
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy file
    shutil.copy2(src, dst)
    return dst


def move_file_safe(src: Path, dst: Path, overwrite: bool = False) -> Path:
    """Safely move a file.

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Allow overwriting existing files

    Returns:
        Destination path

    Raises:
        FileExistsError: If destination exists and overwrite is False
    """
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")

    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Move file
    shutil.move(str(src), str(dst))
    return dst


@contextmanager
def atomic_write(
    file_path: Path,
    mode: str = "w",
    encoding: str | None = "utf-8",
) -> Iterator[IO[Any]]:
    """Context manager for atomic file writes.

    Writes to a temp file first, then atomically moves to target.

    Args:
        file_path: Target file path
        mode: File mode ('w' or 'wb')
        encoding: File encoding (ignored for binary mode)

    Yields:
        File handle
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".{file_path.name}.",
    )
    temp_path = Path(temp_path)

    try:
        os.close(temp_fd)

        if "b" in mode:
            with open(temp_path, mode) as f:
                yield f
        else:
            with open(temp_path, mode, encoding=encoding) as f:
                yield f

        # Atomic rename
        temp_path.replace(file_path)

    except Exception:
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
        raise


@contextmanager
def temporary_directory() -> Iterator[Path]:
    """Context manager for a temporary directory.

    Yields:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def get_file_size_human(file_path: Path) -> str:
    """Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        Human-readable size string
    """
    size = file_path.stat().st_size
    return format_size(size)


def format_size(size: int | float) -> str:
    """Format byte size as human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Human-readable size string
    """
    size_f = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} PB"


def is_hidden(path: Path) -> bool:
    """Check if a path is hidden.

    Args:
        path: Path to check

    Returns:
        True if hidden
    """
    # Unix hidden files/directories
    if path.name.startswith("."):
        return True

    # Windows hidden attribute
    try:
        import ctypes

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))  # type: ignore[attr-defined]
        if attrs != -1:
            return bool(attrs & 2)  # FILE_ATTRIBUTE_HIDDEN
    except (AttributeError, OSError):
        pass

    return False


def clean_empty_directories(directory: Path, recursive: bool = True) -> int:
    """Remove empty directories.

    Args:
        directory: Directory to clean
        recursive: Clean subdirectories recursively

    Returns:
        Number of directories removed
    """
    removed = 0

    if recursive:
        # Walk bottom-up to remove empty dirs
        for root, _dirs, _files in os.walk(directory, topdown=False):
            root_path = Path(root)
            if root_path != directory:
                try:
                    if not any(root_path.iterdir()):
                        root_path.rmdir()
                        removed += 1
                except OSError:
                    pass

    return removed
