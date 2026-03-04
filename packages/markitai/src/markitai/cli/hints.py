"""Cross-platform hints for CLI commands."""

from __future__ import annotations

import sys


def get_env_set_command(var: str, value: str = "your_key") -> str:
    """Get the appropriate environment variable set command for current OS.

    Args:
        var: Environment variable name
        value: Value to set (default: "your_key")

    Returns:
        Platform-appropriate command string:
        - Windows PowerShell: $env:VAR="value"
        - Windows CMD: set VAR=value
        - Linux/macOS: export VAR=value
    """
    if sys.platform == "win32":
        # Check if running in PowerShell or CMD
        import os

        if os.environ.get("PSModulePath"):  # noqa: SIM112 - Windows uses mixed case
            return f'$env:{var}="{value}"'
        return f"set {var}={value}"
    return f"export {var}={value}"
