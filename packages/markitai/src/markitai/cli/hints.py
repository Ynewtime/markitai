"""Cross-platform hints for CLI commands."""

from __future__ import annotations


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
    # One copy, shared with the LLM layer's missing-key messages (which
    # must not import the CLI package)
    from markitai.providers.api_keys import env_set_command

    return env_set_command(var, value)
