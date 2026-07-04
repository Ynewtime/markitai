from __future__ import annotations

from markitai.webextract.elements.code import normalize_code_blocks
from markitai.webextract.elements.footnotes import (
    adopt_external_footnotes,
    standardize_footnotes,
)
from markitai.webextract.elements.images import normalize_images

__all__ = [
    "adopt_external_footnotes",
    "normalize_code_blocks",
    "normalize_images",
    "standardize_footnotes",
]
