"""CLI commands package.

This package contains CLI command groups for Markitai.

Available command groups:
- config: Configuration management commands
- cache: Cache management commands
- doctor: System health and dependency checking command
- init: Initialize Markitai configuration

Commands are lazily loaded by MarkitaiGroup in cli/framework.py.
Direct imports (e.g., ``from markitai.cli.commands import cache``)
still work via __getattr__ but do not eagerly load all siblings.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "cache": ("markitai.cli.commands.cache", "cache"),
    "config": ("markitai.cli.commands.config", "config"),
    "doctor": ("markitai.cli.commands.doctor", "doctor"),
    "init": ("markitai.cli.commands.init", "init"),
}

__all__ = ["cache", "config", "doctor", "init"]  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    spec = _LAZY_MAP.get(name)
    if spec is not None:
        module_path, attr_name = spec
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr_name)
        globals()[name] = obj  # Cache so __getattr__ isn't called again
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
