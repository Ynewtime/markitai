"""Office application detection utilities.

Provides unified detection for MS Office and LibreOffice across platforms.
On Windows, MS Office is preferred when available.
"""

from __future__ import annotations

import platform
import shutil
from functools import lru_cache

from loguru import logger


def _is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


@lru_cache(maxsize=1)
def has_ms_office() -> bool:
    """Detect if MS Office is installed on Windows (cached).

    Checks for PowerPoint availability via COM automation.

    Returns:
        True if MS Office is available, False otherwise.
    """
    if not _is_windows():
        return False

    try:
        import win32com.client

        # Check PowerPoint availability (most relevant for PPTX)
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        ppt.Quit()
        logger.debug("MS Office (PowerPoint) detected via COM")
        return True
    except Exception:
        logger.debug("MS Office not available via COM")
        return False


@lru_cache(maxsize=1)
def find_libreoffice() -> str | None:
    """Find LibreOffice soffice executable (cached).

    Searches PATH first, then common installation paths.

    Returns:
        Path to soffice executable, or None if not found.
    """
    # Check PATH first
    for cmd in ("soffice", "libreoffice"):
        path = shutil.which(cmd)
        if path:
            logger.debug(f"LibreOffice found in PATH: {path}")
            return path

    # Check common installation paths
    common_paths = [
        # Linux
        "/usr/bin/soffice",
        "/usr/local/bin/soffice",
        "/opt/libreoffice/program/soffice",
        # macOS
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        # Windows
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]

    for path in common_paths:
        if shutil.which(path):
            logger.debug(f"LibreOffice found at: {path}")
            return path

    logger.debug("LibreOffice not found")
    return None
