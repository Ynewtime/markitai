"""Markitai - Opinionated Markdown converter with native LLM enhancement support.

Public programmatic API (provisional — 0.x, may change in minor releases)::

    import markitai

    out = markitai.convert("report.pdf", output_dir="out/")
    print(out.markdown)

Exports are loaded lazily so ``import markitai`` stays cheap for the CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.24.0"  # single source of truth — bump before tagging vX.Y.Z

if TYPE_CHECKING:
    from markitai.api import ConversionOutput, ConversionUsage, aconvert, convert
    from markitai.config import MarkitaiConfig

__all__ = [
    "ConversionOutput",
    "ConversionUsage",
    "MarkitaiConfig",
    "__version__",
    "aconvert",
    "convert",
]


def __getattr__(name: str) -> Any:
    """Lazy exports (PEP 562): defer heavy imports until first use."""
    if name in {"ConversionOutput", "ConversionUsage", "aconvert", "convert"}:
        from markitai import api as _api

        return getattr(_api, name)
    if name == "MarkitaiConfig":
        from markitai.config import MarkitaiConfig

        return MarkitaiConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
