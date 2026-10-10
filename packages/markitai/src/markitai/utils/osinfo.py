"""The operating system's name, release and machine, without WMI on Windows.

Since Python 3.12, ``platform.system()``, ``release()`` and ``machine()`` run
WMI queries on Windows (``platform.uname()`` asks for the OS and the CPU).
Each is an out-of-process COM call, and in a session that may not query WMI
(an SSH logon, for one) each waits about 5 seconds before failing
(python/cpython#112278). markitai asked on every run: the converter pool's
size at import and the system proxy lookup before each fetch.

Tests simulate other systems by patching ``platform.system``; a patched
function is always called.
"""

from __future__ import annotations

import os
import platform
import sys

_PLATFORM_SYSTEM = platform.system


def system() -> str:
    """``platform.system()``: "Windows", "Darwin", "Linux", ..."""
    if sys.platform == "win32" and platform.system is _PLATFORM_SYSTEM:
        return "Windows"
    return platform.system()


def release() -> str:
    """``platform.release()``; on Windows "10" or "11" from the build number."""
    if sys.platform == "win32" and platform.system is _PLATFORM_SYSTEM:
        version = sys.getwindowsversion()
        if version.major == 10 and version.build >= 22000:
            return "11"
        return str(version.major)
    return platform.release()


def machine() -> str:
    """``platform.machine()``; on Windows this process's architecture.

    That is the emulated one for an x64 Python on ARM64 (``AMD64``), where
    ``platform.machine()`` reports the CPU.
    """
    if sys.platform == "win32" and platform.system is _PLATFORM_SYSTEM:
        return os.environ.get("PROCESSOR_ARCHITECTURE", "")
    return platform.machine()
