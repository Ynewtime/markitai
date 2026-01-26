"""URL list parsing module for batch URL processing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# URL pattern for validation
_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


@dataclass
class UrlEntry:
    """Represents a URL entry from a URL list file.

    Attributes:
        url: The URL to process
        output_name: Optional custom output filename (without extension)
    """

    url: str
    output_name: str | None = None


class UrlListParseError(Exception):
    """Raised when URL list file cannot be parsed."""

    pass


def is_url_list_file(path: Path) -> bool:
    """Check if path is a URL list file.

    URL list files are identified by the .urls extension.

    Args:
        path: Path to check

    Returns:
        True if the file has .urls extension
    """
    return path.suffix.lower() == ".urls"


def parse_url_list(file_path: Path) -> list[UrlEntry]:
    """Parse a URL list file.

    Supported formats:
    1. Plain text: one URL per line
       - Empty lines are ignored
       - Lines starting with # are comments
       - Lines can optionally have a custom output name after whitespace:
         https://example.com custom_name

    2. JSON array of strings:
       ["https://example1.com", "https://example2.com"]

    3. JSON array of objects:
       [
         {"url": "https://example1.com"},
         {"url": "https://example2.com", "output_name": "custom"}
       ]

    Args:
        file_path: Path to the URL list file

    Returns:
        List of UrlEntry objects

    Raises:
        UrlListParseError: If the file cannot be parsed
        FileNotFoundError: If the file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"URL list file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        return []

    # Try JSON first
    if content.startswith("["):
        return _parse_json_url_list(content, file_path)

    # Fall back to plain text
    return _parse_text_url_list(content, file_path)


def _parse_json_url_list(content: str, file_path: Path) -> list[UrlEntry]:
    """Parse JSON format URL list."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise UrlListParseError(f"Invalid JSON in {file_path}: {e}")

    if not isinstance(data, list):
        raise UrlListParseError(
            f"Expected JSON array in {file_path}, got {type(data).__name__}"
        )

    entries = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            # Simple string URL
            url = item.strip()
            if not url:
                continue
            if not _URL_PATTERN.match(url):
                logger.warning(f"Skipping invalid URL at index {i}: {url[:50]}...")
                continue
            entries.append(UrlEntry(url=url))

        elif isinstance(item, dict):
            # Object with url and optional output_name
            url = item.get("url", "").strip()
            if not url:
                logger.warning(f"Skipping entry at index {i}: missing 'url' field")
                continue
            if not _URL_PATTERN.match(url):
                logger.warning(f"Skipping invalid URL at index {i}: {url[:50]}...")
                continue

            output_name = item.get("output_name")
            if output_name:
                output_name = str(output_name).strip() or None

            entries.append(UrlEntry(url=url, output_name=output_name))

        else:
            logger.warning(
                f"Skipping entry at index {i}: expected string or object, got {type(item).__name__}"
            )

    return entries


def _parse_text_url_list(content: str, file_path: Path) -> list[UrlEntry]:
    """Parse plain text format URL list."""
    entries = []

    for line_num, line in enumerate(content.splitlines(), start=1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Split line into URL and optional output name
        parts = line.split(None, 1)  # Split on first whitespace
        url = parts[0]

        if not _URL_PATTERN.match(url):
            logger.warning(f"Skipping invalid URL at line {line_num}: {url[:50]}...")
            continue

        output_name = None
        if len(parts) > 1:
            output_name = parts[1].strip()
            # Remove quotes if present
            if (output_name.startswith('"') and output_name.endswith('"')) or (
                output_name.startswith("'") and output_name.endswith("'")
            ):
                output_name = output_name[1:-1]

        entries.append(UrlEntry(url=url, output_name=output_name or None))

    return entries


def find_url_list_files(directory: Path) -> list[Path]:
    """Find all .urls files in a directory (recursive).

    Args:
        directory: Directory to search

    Returns:
        List of paths to .urls files, sorted by path
    """
    if not directory.is_dir():
        return []

    return sorted(directory.glob("**/*.urls"))
