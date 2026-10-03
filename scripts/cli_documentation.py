"""Collect the selected public Markdown guides for offline CLI archives.

Keep this whitelist explicit: maintenance records, credentials, models and
arbitrary repository files are not package documentation.
"""
from pathlib import Path
import os
import stat


DOCUMENTATION_PATHS = (
    "README.md",
    "llms.txt",
    "llms-full.txt",
    "docs/index.md",
    "docs/quickstart.md",
    "docs/cli.md",
    "docs/mcp.md",
)
MAX_DOCUMENT_BYTES = 64 * 1024
MAX_DOCUMENTATION_BYTES = 192 * 1024


def _regular(status):
    return stat.S_ISREG(status.st_mode) and not getattr(status, "st_file_attributes", 0) & 0x400


def _identity(status):
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def validate_documentation(files):
    """Provided maps must contain the complete whitelist and bounded UTF-8 bytes."""
    if set(files) != set(DOCUMENTATION_PATHS):
        raise RuntimeError("CLI documentation inventory differs from the public whitelist")
    total = 0
    for name, content in files.items():
        if not isinstance(content, bytes) or not content or len(content) > MAX_DOCUMENT_BYTES or b"\0" in content:
            raise RuntimeError(f"Invalid CLI documentation bytes: {name}")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"CLI documentation is not UTF-8: {name}") from error
        total += len(content)
    if total > MAX_DOCUMENTATION_BYTES:
        raise RuntimeError("CLI documentation exceeds the total size bound")


def cli_documentation(root):
    """Read only named guides; reject links and replacement while reading.

    `root` is the packager's source checkout. Its external ancestors are not
    traversed or subjected to user-directory policy; relative managed parents
    and leaves are checked, including Windows reparse points.
    """
    root = Path(root)
    files = {}
    for name in DOCUMENTATION_PATHS:
        path = root / name
        parents = []
        parent = path.parent
        while parent != root:
            status = parent.lstat()
            if not stat.S_ISDIR(status.st_mode) or getattr(status, "st_file_attributes", 0) & 0x400:
                raise RuntimeError(f"CLI documentation parent is not a plain directory: {name}")
            parents.append((parent, _identity(status)))
            parent = parent.parent
        before = path.lstat()
        if not _regular(before) or before.st_size > MAX_DOCUMENT_BYTES:
            raise RuntimeError(f"CLI documentation must be a bounded regular file: {name}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not _regular(opened) or _identity(opened) != _identity(before):
                raise RuntimeError(f"CLI documentation changed before reading: {name}")
            files[name] = stream.read(MAX_DOCUMENT_BYTES + 1)
            held = os.fstat(stream.fileno())
            named = path.lstat()
            if (not _regular(held) or not _regular(named)
                    or _identity(held) != _identity(before) or _identity(named) != _identity(before)):
                raise RuntimeError(f"CLI documentation changed while reading: {name}")
        for parent, expected in parents:
            if _identity(parent.lstat()) != expected:
                raise RuntimeError(f"CLI documentation parent changed while reading: {name}")
    validate_documentation(files)
    return files
