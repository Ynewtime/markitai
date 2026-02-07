"""Office application detection utilities.

Provides detection for MS Office (Windows) and LibreOffice (cross-platform).
- MS Office COM: Used for legacy format conversion (.doc/.ppt) and PPTX slide rendering
- LibreOffice: Used as fallback for legacy format conversion and PDF export
"""

from __future__ import annotations

import platform
import shutil
from functools import lru_cache
from pathlib import Path

from loguru import logger

# Common MS Office installation paths on Windows
_MS_OFFICE_PATHS = [
    # Microsoft 365 / Office 2019+ (Click-to-Run)
    r"C:\Program Files\Microsoft Office\root\Office16",
    r"C:\Program Files (x86)\Microsoft Office\root\Office16",
    # Office 2016 (MSI)
    r"C:\Program Files\Microsoft Office\Office16",
    r"C:\Program Files (x86)\Microsoft Office\Office16",
    # Office 2013
    r"C:\Program Files\Microsoft Office\Office15",
    r"C:\Program Files (x86)\Microsoft Office\Office15",
    # Office 2010
    r"C:\Program Files\Microsoft Office\Office14",
    r"C:\Program Files (x86)\Microsoft Office\Office14",
]


def _is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


def _check_office_exe_exists(app_name: str) -> bool:
    """Check if an Office application executable exists in common paths.

    Args:
        app_name: Application name without extension (e.g., "POWERPNT", "WINWORD", "EXCEL")

    Returns:
        True if the executable is found in any common path.
    """
    exe_name = f"{app_name}.EXE"
    for office_path in _MS_OFFICE_PATHS:
        exe_path = Path(office_path) / exe_name
        if exe_path.exists():
            logger.debug(f"Found {app_name} at: {exe_path}")
            return True
    return False


def _check_ms_office_app(registry_key: str, exe_name: str, display_name: str) -> bool:
    """Check if a specific MS Office application is available (Windows only).

    Detection strategy:
    1. Windows Registry lookup (fast, preferred)
    2. Direct file path check (fallback for Click-to-Run installations)

    Args:
        registry_key: Registry key name (e.g., "PowerPoint.Application")
        exe_name: Executable name without extension (e.g., "POWERPNT")
        display_name: Human-readable name for logging (e.g., "MS PowerPoint")

    Returns:
        True if the application is installed, False otherwise.
    """
    if not _is_windows():
        return False

    # Method 1: Registry lookup
    try:
        import winreg  # type: ignore[import-not-found]

        try:
            key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, registry_key)  # type: ignore[attr-defined]
            winreg.CloseKey(key)  # type: ignore[attr-defined]
            logger.debug(f"{display_name} detected via registry")
            return True
        except OSError:
            pass  # Registry key not found, try file path
    except ImportError:
        pass  # winreg not available

    # Method 2: Direct file path check (for Click-to-Run installations)
    if _check_office_exe_exists(exe_name):
        logger.debug(f"{display_name} detected via file path")
        return True

    logger.debug(f"{display_name} not found")
    return False


@lru_cache(maxsize=1)
def check_ms_powerpoint_available() -> bool:
    """Check if MS Office PowerPoint is installed (Windows only)."""
    return _check_ms_office_app("PowerPoint.Application", "POWERPNT", "MS PowerPoint")


@lru_cache(maxsize=1)
def check_ms_word_available() -> bool:
    """Check if MS Office Word is installed (Windows only)."""
    return _check_ms_office_app("Word.Application", "WINWORD", "MS Word")


@lru_cache(maxsize=1)
def check_ms_excel_available() -> bool:
    """Check if MS Office Excel is installed (Windows only)."""
    return _check_ms_office_app("Excel.Application", "EXCEL", "MS Excel")


import threading

# Thread-safe cache for has_ms_office result
_ms_office_check_lock = threading.Lock()
_ms_office_checked = False
_ms_office_available = False


def has_ms_office() -> bool:
    """Detect if MS Office PowerPoint is available via COM (Windows only).

    Used for optional high-quality PPTX slide rendering.
    Text extraction uses MarkItDown (cross-platform) and doesn't need COM.

    Note: For checking installation status, prefer `check_ms_powerpoint_available()`
    which uses registry lookup and is faster.

    Returns:
        True if PowerPoint COM is available, False otherwise.
    """
    global _ms_office_checked, _ms_office_available

    # Fast path: already checked
    if _ms_office_checked:
        return _ms_office_available

    if not _is_windows():
        _ms_office_checked = True
        _ms_office_available = False
        return False

    # Thread-safe check with proper COM initialization
    with _ms_office_check_lock:
        # Double-check after acquiring lock
        if _ms_office_checked:
            return _ms_office_available

        try:
            import pythoncom  # type: ignore[import-not-found]
            import win32com.client  # type: ignore[import-not-found]

            # Initialize COM for this thread (required in worker threads)
            pythoncom.CoInitialize()
            try:
                # Check PowerPoint availability (most relevant for PPTX)
                ppt = win32com.client.Dispatch("PowerPoint.Application")
                ppt.Quit()
                logger.debug("MS Office (PowerPoint) detected via COM")
                _ms_office_available = True
            finally:
                pythoncom.CoUninitialize()
        except Exception:
            logger.debug("MS Office not available via COM")
            _ms_office_available = False

        _ms_office_checked = True
        return _ms_office_available


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
    common_paths: list[str] = []

    if platform.system() == "Windows":
        import os

        prog_dirs = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ]
        for prog_dir in prog_dirs:
            for subdir in (
                "LibreOffice",
                "LibreOffice 7",
                "LibreOffice 24",
                "LibreOffice 25",
            ):
                common_paths.append(
                    os.path.join(prog_dir, subdir, "program", "soffice.exe")
                )
    elif platform.system() == "Darwin":
        common_paths.append("/Applications/LibreOffice.app/Contents/MacOS/soffice")

    # Linux paths (always checked as fallback)
    common_paths.extend(
        [
            "/usr/bin/soffice",
            "/usr/local/bin/soffice",
            "/opt/libreoffice/program/soffice",
        ]
    )

    for path in common_paths:
        if Path(path).exists():
            logger.debug(f"LibreOffice found at: {path}")
            return path

    logger.debug("LibreOffice not found")
    return None
