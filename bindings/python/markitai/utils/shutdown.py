"""Compatibility with Python markitai's `markitai.utils.shutdown` module."""

import os
import sys
from typing import NoReturn


def finalize_process(code: int = 0) -> NoReturn:
    """Flush both streams and end the process with `code`, as Python markitai does.

    Python markitai's scripts ended with it to skip a native library's teardown,
    which could abort a process whose conversion had succeeded. The native
    engine has no such teardown and nothing else to release, so only the exit
    remains: through `os._exit`, like the original, so a script that relied on
    it ending the process still ends. Only for a process the script owns; a
    long-lived host decides its own exit.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass
    os._exit(code)
