"""CLI commands package.

This package contains CLI command groups for Markitai.

Available command groups:
- config: Configuration management commands
- cache: Cache management commands
- check_deps: Dependency checking command
"""

from __future__ import annotations

from markitai.cli.commands.cache import cache
from markitai.cli.commands.config import config
from markitai.cli.commands.deps import check_deps

__all__ = ["cache", "config", "check_deps"]
