"""Office application detection utilities.

Provides detection for MS Office (Windows) and LibreOffice (cross-platform),
used by PPTX slide rendering (COM preferred, LibreOffice PDF export as
fallback). Legacy .doc/.ppt conversion no longer lives here — it is handled
by the anydoc backend (converter/legacy.py, markitai[legacy] extra).
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from loguru import logger

from markitai.utils import osinfo


def _is_windows() -> bool:
    """Check if running on Windows."""
    return osinfo.system() == "Windows"


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


# LibreOffice per Linux package manager: (os-release IDs, label, command).
# The first family whose ID matches ID or ID_LIKE wins.
_LIBREOFFICE_LINUX: tuple[tuple[frozenset[str], str, str], ...] = (
    (
        frozenset({"debian", "ubuntu"}),
        "Debian, Ubuntu",
        "sudo apt install libreoffice",
    ),
    (
        frozenset({"fedora", "rhel", "centos"}),
        "Fedora, RHEL",
        "sudo dnf install libreoffice",
    ),
    (frozenset({"arch"}), "Arch", "sudo pacman -S libreoffice-fresh"),
    (
        frozenset({"suse", "opensuse"}),
        "openSUSE",
        "sudo zypper install libreoffice",
    ),
    (frozenset({"alpine"}), "Alpine", "sudo apk add libreoffice"),
)


def _linux_distro_ids(os_release: str | None = None) -> set[str]:
    """IDs from /etc/os-release (``ID`` plus ``ID_LIKE``), lowercased."""
    if os_release is None:
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                os_release = f.read()
        except OSError:
            return set()
    ids: set[str] = set()
    for line in os_release.splitlines():
        key, _, value = line.partition("=")
        if key.strip() in ("ID", "ID_LIKE"):
            ids.update(value.strip().strip("\"'").lower().split())
    # openSUSE reports opensuse-leap / opensuse-tumbleweed
    if any(i.startswith("opensuse") for i in ids):
        ids.add("opensuse")
    return ids


def libreoffice_linux_install_commands(
    os_release: str | None = None,
) -> list[tuple[str, str]]:
    """The command that installs LibreOffice on this Linux distribution.

    Args:
        os_release: /etc/os-release text (tests); read from disk if None.

    Returns:
        One ``("", command)`` pair for a recognised distribution, otherwise
        every ``(label, command)`` alternative.
    """
    ids = _linux_distro_ids(os_release)
    for family, _label, command in _LIBREOFFICE_LINUX:
        if ids & family:
            return [("", command)]
    return [(label, command) for _family, label, command in _LIBREOFFICE_LINUX]


#: Settings of the private LibreOffice profile: document macros off, and no
#: content a document only links to. Without the last two, a document's linked
#: pictures (a ``file:`` path on this machine, an http(s) address) are loaded
#: into the export, which becomes slide images and model input.
LIBREOFFICE_PROFILE_SETTINGS = (
    '<?xml version="1.0"?><oor:items xmlns:oor="http://openoffice.org/2001/registry">'
    '<item oor:path="/org.openoffice.Office.Common/Security/Scripting">'
    '<prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item>'
    '<item oor:path="/org.openoffice.Office.Common/Security/Scripting">'
    '<prop oor:name="BlockUntrustedRefererLinks" oor:op="fuse"><value>true</value></prop></item>'
    '<item oor:path="/org.openoffice.Office.Common/Security/Scripting">'
    '<prop oor:name="DisableActiveContent" oor:op="fuse"><value>true</value></prop></item>'
    "</oor:items>"
)


def prepare_libreoffice_profile(profile: Path) -> None:
    """Write :data:`LIBREOFFICE_PROFILE_SETTINGS` into a new, empty *profile*."""
    (profile / "user").mkdir(parents=True)
    (profile / "user" / "registrymodifications.xcu").write_text(
        LIBREOFFICE_PROFILE_SETTINGS, encoding="utf-8"
    )


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

    if osinfo.system() == "Windows":
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
    elif osinfo.system() == "Darwin":
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
