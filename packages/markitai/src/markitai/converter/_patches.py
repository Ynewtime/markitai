"""Compatibility patches for third-party libraries.

This module applies monkey patches to fix known issues in dependencies:
- openpyxl 3.1.x FileVersion TypeError ('bg' argument)
- MarkItDown's Markdownify converter writing a table cell's line breaks as
  spaces (or leaving U+2028 in the row): ``apply_markitdown_cell_patches``
"""

from __future__ import annotations

import functools
from typing import Any

from markitai.utils.markitdown_import import import_markitdown

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


_CELL_RULES = ("convert_br", "convert_p", "convert_div", "convert_td", "convert_th")


def apply_markitdown_cell_patches() -> None:
    """Give MarkItDown's Markdownify converter markitai's table cell rules.

    MarkItDown renders DOCX (Mammoth), XLSX/PPTX fallbacks, EPUB chapters
    and HTML fallbacks through its own ``_CustomMarkdownify``; with these
    rules a cell's line breaks become ``<br>`` there too
    (:class:`markitai.webextract.markdownify_compat.TableCellLines`). Call
    it where MarkItDown is used: importing MarkItDown only to patch it
    would load its format detectors in every process. Idempotent.
    """
    try:
        import_markitdown()
        from markitdown.converters._markdownify import _CustomMarkdownify
    except ImportError:
        return
    from markitai.webextract.markdownify_compat import TableCellLines

    for name in _CELL_RULES:
        setattr(_CustomMarkdownify, name, TableCellLines.__dict__[name])


def apply_all_patches() -> None:
    """Apply all compatibility patches.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _patches_applied
    if _patches_applied:
        return

    apply_openpyxl_patches()

    _patches_applied = True
