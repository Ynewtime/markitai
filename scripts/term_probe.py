#!/usr/bin/env python3
"""Show how markitai sees this terminal, for reports of wrapped output.

Run it with markitai's own Python, in the same window or SSH session where
the output looked wrong::

    # Windows PowerShell (markitai installed with uv tool install / setup.ps1):
    & (Join-Path (uv tool dir) "markitai\\Scripts\\python.exe") term_probe.py
    # macOS / Linux:
    "$(uv tool dir)/markitai/bin/python" term_probe.py

It prints, for stdout and stderr: whether each is a terminal, the size the
OS reports for it, what Rich (markitai's console library) decides (terminal,
width, legacy Windows console, colour system), the environment variables
that change those answers, and on Windows the console mode of each handle.
Then it prints a long warning through markitai's own helper, so you can see
whether it wraps under its text (a terminal) or stays one line (not one).
Compare the reported width with the window's real width.
"""

from __future__ import annotations

import os
import sys


def _size(fd: int) -> str:
    try:
        size = os.get_terminal_size(fd)
    except (OSError, ValueError) as e:
        return f"none ({type(e).__name__})"
    return f"{size.columns}x{size.lines}"


def _windows_console_mode(stream: object) -> str:
    if sys.platform != "win32":
        return "n/a"
    import ctypes
    import msvcrt
    from ctypes import wintypes

    try:
        handle = msvcrt.get_osfhandle(stream.fileno())  # type: ignore[attr-defined]
    except (OSError, ValueError, AttributeError) as e:
        return f"no handle ({type(e).__name__})"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    file_type = kernel32.GetFileType(wintypes.HANDLE(handle))
    kinds = {1: "disk", 2: "char (console)", 3: "pipe"}
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(wintypes.HANDLE(handle), ctypes.byref(mode)):
        return f"type {kinds.get(file_type, file_type)}, not a console"
    vt = bool(mode.value & 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    return f"type {kinds.get(file_type, file_type)}, mode 0x{mode.value:04x}, VT {vt}"


def main() -> None:
    from markitai.utils import term

    print(f"python {sys.version.split()[0]} on {sys.platform}")
    for name in (
        "TERM",
        "COLUMNS",
        "LINES",
        "WT_SESSION",
        "TERM_PROGRAM",
        "SSH_CONNECTION",
        "SSH_TTY",
        "FORCE_COLOR",
        "NO_COLOR",
        "TTY_COMPATIBLE",
        "CI",
    ):
        print(f"  {name}={os.environ.get(name)!r}")
    for label, stream, fd in (("stdout", sys.stdout, 1), ("stderr", sys.stderr, 2)):
        print(
            f"{label}: isatty={stream.isatty()} os size={_size(fd)} "
            f"console={_windows_console_mode(stream)}"
        )
    for label, console in (
        ("stdout console", term.get_console()),
        ("stderr console", term.get_stderr_console()),
    ):
        print(
            f"{label}: terminal={console.is_terminal} width={console.width} "
            f"legacy_windows={console.legacy_windows} "
            f"color_system={console.color_system} soft_wrap={console.soft_wrap}"
        )
    print("A long warning through markitai's helper, on stderr:")
    sys.stdout.flush()
    term.print_marked(
        "2 files skipped (image_only): english.png, sample.jpg; use --llm or "
        "--ocr for content extraction. A terminal wraps this under its text; "
        "a pipe or file gets it on one line.",
        mark=term.MARK_WARNING,
        console=term.get_stderr_console(),
    )


if __name__ == "__main__":
    main()
