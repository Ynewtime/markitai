"""Actionable error guidance helpers.

Principle: never tell the user "set X in config" without telling them WHERE
and HOW. Every configuration-related error should include:

- the concrete config file path (the actually-loaded one, or the default
  candidates plus a pointer to ``markitai init``),
- copy-pasteable fix commands (``markitai config set <key> <value>`` and/or
  environment variable exports),
- where to obtain credentials when applicable.

This module centralizes that formatting so all error sites stay consistent.
"""

from __future__ import annotations

import sys


def config_location_hint() -> str:
    """Return a one-line hint pointing at the config file markitai reads.

    Prefers the path of an already-loaded ConfigManager; otherwise resolves
    the same search chain the CLI uses (MARKITAI_CONFIG env var >
    ./markitai.json > ~/.markitai/config.json) without loading the file.
    """
    from markitai.config import ConfigManager, config_manager

    path = config_manager.config_path
    if path is None:
        path = ConfigManager()._resolve_config_path(None, env_override=True)
    if path is not None and path.exists():
        return f"Config file: {path}"
    return (
        "Config file: none found (searched ./markitai.json and "
        "~/.markitai/config.json) - run 'markitai init' to create one"
    )


def format_actionable_error(
    problem: str,
    steps: list[str],
    *,
    include_config_hint: bool = True,
) -> str:
    """Format a problem plus numbered fix steps into a consistent block.

    Args:
        problem: Short description of what went wrong (may be multi-line).
        steps: Fix steps; embedded newlines become indented continuation lines.
        include_config_hint: Append the config file location line (default on;
            disable for errors that are not configuration-related).

    Returns:
        Multi-line error text ready to raise/print.
    """
    lines = [problem, "", "To fix:"]
    for index, step in enumerate(steps, 1):
        first, *rest = step.splitlines() or [""]
        lines.append(f"  {index}. {first}")
        lines.extend(f"     {line}" for line in rest)
    if include_config_hint:
        lines.append("")
        lines.append(config_location_hint())
    return "\n".join(lines)


def cloudflare_credentials_error() -> str:
    """Actionable error for missing Cloudflare API credentials.

    Covers both entry points that need them: URL fetching via Browser
    Rendering (-s cloudflare) and file conversion via Workers AI toMarkdown.
    """
    return format_actionable_error(
        "Cloudflare API token and account ID required "
        "(fetch.cloudflare.api_token / fetch.cloudflare.account_id are not set).",
        [
            "Create an API token: https://dash.cloudflare.com/profile/api-tokens\n"
            "Choose 'Create Token' -> 'Create Custom Token' with permissions:\n"
            "Account / Browser Rendering / Edit  +  Account / Workers AI / Read\n"
            "(shortcut: dash.cloudflare.com -> Workers AI -> 'Use REST API' -> "
            "'Create a Workers AI API Token')",
            "Copy your Account ID: dash.cloudflare.com -> select your account ->\n"
            "the Account ID is on the account home page (right sidebar) and in\n"
            "the dashboard URL: dash.cloudflare.com/<account-id>",
            "Save both values:\n"
            "markitai config set fetch.cloudflare.api_token <token>\n"
            "markitai config set fetch.cloudflare.account_id <account-id>\n"
            "or: export CLOUDFLARE_API_TOKEN=<token> CLOUDFLARE_ACCOUNT_ID=<account-id>",
        ],
    )


def jina_api_key_hint() -> str:
    """Hint for lifting Jina Reader anonymous-access blocks / rate limits."""
    return (
        "A Jina API key lifts anonymous-access blocks and raises rate limits:\n"
        "  markitai config set fetch.jina.api_key <key>\n"
        "  or: export JINA_API_KEY=<key>\n"
        "  Get a free key at https://jina.ai/reader"
    )


def playwright_package_missing_error() -> str:
    """Actionable error when the playwright Python package is not installed."""
    return format_actionable_error(
        "The playwright fetch strategy requires the 'playwright' package, "
        "which is not installed.",
        [
            "Install the package:\n"
            "pip:      pip install 'markitai[browser]'\n"
            "uv proj:  uv add playwright\n"
            "uv tool:  uv tool install --force 'markitai[all]'",
            "Then download the browser (a separate one-time step):\n"
            "markitai doctor --fix\n"
            "(or: playwright install chromium; "
            "Linux: also 'playwright install-deps chromium')",
        ],
        include_config_hint=False,
    )


def playwright_browser_missing_error() -> str:
    """Actionable error when playwright is installed but Chromium is not.

    pip/uv only install the playwright Python package; the browser binaries
    are a separate ``playwright install chromium`` download. This is why a
    fresh ``uv tool install "markitai[all]"`` has playwright "installed" but
    no Chrome.
    """
    return format_actionable_error(
        "Playwright is installed but the Chromium browser is missing.\n"
        "(pip/uv only install the playwright Python package; the browser "
        "binaries are a separate one-time download.)",
        [
            "markitai doctor --fix   (detects and installs Chromium automatically)",
            f'or run: "{sys.executable}" -m playwright install chromium\n'
            "(targets the exact Python environment markitai runs in;\n"
            "in a uv project checkout: uv run playwright install chromium)",
            "Linux only: also run 'playwright install-deps chromium' "
            "for system libraries",
        ],
        include_config_hint=False,
    )
