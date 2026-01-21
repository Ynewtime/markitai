"""Compatibility patches for third-party libraries.

This module applies monkey patches to fix known issues in dependencies:
- openpyxl 3.1.x FileVersion TypeError ('bg' argument)
- lxml XMLSyntaxError from malformed PPTX files
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
    if not getattr(FileVersion.__init__, "_markit_patched", False):
        FileVersion.__init__ = patched_init
        FileVersion.__init__._markit_patched = True  # type: ignore[attr-defined]


def apply_pptx_patches() -> None:
    """Apply patches for python-pptx/lxml compatibility issues.

    Fixes: XMLSyntaxError from malformed PPTX files converted from PPT

    When MS Office converts .ppt to .pptx, it may produce XML with
    mismatched tags (e.g., 'rupB' vs 'bgPr'). This patch makes the
    XML parser more lenient.
    """
    # Check if lxml is available
    try:
        from lxml import etree
    except ImportError:
        return  # lxml not installed

    # Check if python-pptx is installed
    try:
        import pptx.oxml
    except ImportError:
        return  # python-pptx not installed

    # Create a lenient parser that recovers from errors
    _lenient_parser = etree.XMLParser(recover=True, remove_blank_text=True)

    original_parse_xml = pptx.oxml.parse_xml

    @functools.wraps(original_parse_xml)
    def patched_parse_xml(xml: bytes | str) -> Any:
        try:
            return original_parse_xml(xml)
        except etree.XMLSyntaxError:
            # Fallback to lenient parser
            if isinstance(xml, str):
                xml = xml.encode("utf-8")
            return etree.fromstring(xml, parser=_lenient_parser)

    if not getattr(pptx.oxml.parse_xml, "_markit_patched", False):
        pptx.oxml.parse_xml = patched_parse_xml
        pptx.oxml.parse_xml._markit_patched = True  # type: ignore[attr-defined]


def apply_all_patches() -> None:
    """Apply all compatibility patches.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _patches_applied
    if _patches_applied:
        return

    apply_openpyxl_patches()
    apply_pptx_patches()

    _patches_applied = True
