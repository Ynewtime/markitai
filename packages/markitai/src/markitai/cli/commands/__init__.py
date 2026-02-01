"""CLI commands package.

This package contains CLI command groups for Markitai.

Available command groups:
- config: Configuration management commands
- cache: Cache management commands
- doctor: System health and dependency checking command
- check_deps: Alias for doctor (backward compatibility)
"""

from __future__ import annotations

from markitai.cli.commands.cache import cache
from markitai.cli.commands.config import config
from markitai.cli.commands.doctor import check_deps, doctor

__all__ = ["cache", "config", "doctor", "check_deps"]
