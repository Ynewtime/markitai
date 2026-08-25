"""Office application detection utilities.

Provides detection for MS Office (Windows) and LibreOffice (cross-platform),
used by PPTX slide rendering (COM preferred, LibreOffice PDF export as
fallback). Legacy .doc/.ppt conversion no longer lives here — it is handled
by the anydoc backend (converter/legacy.py, markitai[legacy] extra).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from loguru import logger


def _is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == "Windows"


import threading

# Thread-safe cache for has_ms_office result
_ms_office_check_lock = threading.Lock()
_ms_office_checked = False
_ms_office_available = False


def has_ms_office() -> bool:
    """Detect if MS Office PowerPoint is available via COM (Windows only).

    Used for optional high-quality PPTX slide rendering.
    Text extraction uses MarkItDown (cross-platform) and doesn't need COM.

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


@lru_cache(maxsize=1)
def is_libreoffice_functional() -> bool:
    """Check if LibreOffice can actually convert files (cached).

    Simply finding the ``soffice`` binary is not enough — the installation
    may be incomplete (e.g. missing import filters on minimal CI images).
    This function creates a tiny test file and attempts a real conversion.
    """
    soffice = find_libreoffice()
    if not soffice:
        return False

    try:
        with tempfile.TemporaryDirectory(prefix="lo_check_") as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test", encoding="utf-8")
            profile_url = Path(tmpdir, "profile").as_uri()
            Path(tmpdir, "profile").mkdir()
            result = subprocess.run(
                [
                    soffice,
                    "--headless",
                    f"-env:UserInstallation={profile_url}",
                    "--convert-to",
                    "html",
                    "--outdir",
                    tmpdir,
                    str(test_file),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            converted = Path(tmpdir) / "test.html"
            ok = result.returncode == 0 and converted.exists()
            if not ok:
                logger.debug(
                    "LibreOffice functional check failed: rc={}, stderr={}",
                    result.returncode,
                    result.stderr[:200],
                )
            return ok
    except Exception as exc:
        logger.debug("LibreOffice functional check error: {}", exc)
        return False
