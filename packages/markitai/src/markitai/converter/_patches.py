"""Compatibility patches for third-party libraries.

This module applies monkey patches to fix known issues in dependencies:
- openpyxl 3.1.x FileVersion TypeError ('bg' argument)
"""

from __future__ import annotations

import functools
from typing import Any

_patches_applied = False


def apply_openpyxl_patches() -> None:
    """Apply patches for openpyxl compatibility issues.

    Fixes: TypeError: FileVersion.__init__() got an unexpected keyword argument 'bg'

    This issue occurs in openpyxl 3.1.x when reading Excel files created by
    older versions of MS Office or converted from .xls format.
    """
    try:
        from openpyxl.workbook.properties import FileVersion
    except ImportError:
        return  # openpyxl not installed

    original_init = FileVersion.__init__

    @functools.wraps(original_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Remove unsupported keyword arguments
        unsupported_keys = ["bg"]
        for key in unsupported_keys:
            kwargs.pop(key, None)
        original_init(self, *args, **kwargs)

    # Only patch if not already patched
    if not getattr(FileVersion.__init__, "_markitai_patched", False):
        FileVersion.__init__ = patched_init
        FileVersion.__init__._markitai_patched = True  # type: ignore[attr-defined]


def apply_all_patches() -> None:
    """Apply all compatibility patches.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _patches_applied
    if _patches_applied:
        return

    apply_openpyxl_patches()

    _patches_applied = True
