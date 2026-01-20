"""Office application detection utilities.

Provides detection for MS Office (Windows) and LibreOffice (cross-platform).
- MS Office COM: Used for legacy format conversion (.doc/.ppt) and PPTX slide rendering
- LibreOffice: Used as fallback for legacy format conversion and PDF export
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
def check_ms_powerpoint_available() -> bool:
    """Check if MS Office PowerPoint is installed (Windows only).

    Uses Windows Registry lookup for fast detection without launching PowerPoint.
    Preferred over COM-based detection for better performance.

    Returns:
        True if PowerPoint is installed, False otherwise.
    """
    if not _is_windows():
        return False

    try:
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"PowerPoint.Application")
            winreg.CloseKey(key)
            logger.debug("MS PowerPoint detected via registry")
            return True
        except OSError:
            return False
    except ImportError:
        return False


@lru_cache(maxsize=1)
def check_ms_word_available() -> bool:
    """Check if MS Office Word is installed (Windows only).

    Uses Windows Registry lookup for fast detection without launching Word.
    Preferred over COM-based detection for better performance.

    Returns:
        True if Word is installed, False otherwise.
    """
    if not _is_windows():
        return False

    try:
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application")
            winreg.CloseKey(key)
            logger.debug("MS Word detected via registry")
            return True
        except OSError:
            return False
    except ImportError:
        return False


@lru_cache(maxsize=1)
def check_ms_excel_available() -> bool:
    """Check if MS Office Excel is installed (Windows only).

    Uses Windows Registry lookup for fast detection without launching Excel.
    Preferred over COM-based detection for better performance.

    Returns:
        True if Excel is installed, False otherwise.
    """
    if not _is_windows():
        return False

    try:
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Excel.Application")
            winreg.CloseKey(key)
            logger.debug("MS Excel detected via registry")
            return True
        except OSError:
            return False
    except ImportError:
        return False


@lru_cache(maxsize=1)
def has_ms_office() -> bool:
    """Detect if MS Office PowerPoint is available via COM (Windows only).

    Used for optional high-quality PPTX slide rendering.
    Text extraction uses MarkItDown (cross-platform) and doesn't need COM.

    Note: For checking installation status, prefer `check_ms_powerpoint_available()`
    which uses registry lookup and is faster.

    Returns:
        True if PowerPoint COM is available, False otherwise.
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
