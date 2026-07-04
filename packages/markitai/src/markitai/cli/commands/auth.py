from __future__ import annotations

import asyncio
import json
from typing import Any

import rich_click as click

from markitai.providers.auth import (
    AuthManager,
    AuthStatus,
    attempt_login,
    get_auth_resolution_hint,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_status(provider: str) -> AuthStatus:
    """Check auth status for a provider (sync wrapper)."""
    manager = AuthManager()
    return asyncio.run(manager.check_auth(provider, force_refresh=True))


def _status_to_payload(status: AuthStatus) -> dict[str, Any]:
    """Convert AuthStatus to a JSON-serialisable dict."""
    return {
        "provider": status.provider,
        "authenticated": status.authenticated,
        "user": status.user,
        "expires_at": status.expires_at.isoformat() if status.expires_at else None,
        "error": status.error,
        "details": status.details or {},
    }


def _display_user(status: AuthStatus) -> str:
    """Format user field for human-readable display.

    Raw values like ``subscription: max`` or ``gemini-cli`` are cleaned
    up so that the terminal output reads naturally.
    """
    user = status.user or ""
    provider = status.provider

    if provider == "claude-agent":
        # "subscription: max" → "max plan"
        # "cloud: bedrock" → "bedrock (cloud)"
        if user.startswith("subscription: "):
            return f"{user.removeprefix('subscription: ')} plan"
        if user.startswith("cloud: "):
            return f"{user.removeprefix('cloud: ')} (cloud)"
        # "user@example.com" + subscription detail → "user@example.com (max plan)"
        subscription = (status.details or {}).get("subscription")
        if isinstance(subscription, str) and subscription not in ("", "unknown"):
            return f"{user} ({subscription} plan)"
        return user

    if provider == "copilot" and user == "token":
        return "env token (GH_TOKEN)"

    if provider == "chatgpt" and user == "chatgpt":
        return "authenticated"

    if provider == "gemini-cli" and user == "gemini-cli":
        return "shared credentials"

    return user


def _print_status(provider_label: str, status: AuthStatus) -> None:
    """Print status check result, raise SystemExit(1) if unauthenticated."""
    if status.authenticated:
        click.echo(f"{provider_label} authenticated: {_display_user(status)}")
    else:
        click.echo(f"{provider_label} not authenticated: {status.error}")
        click.echo(get_auth_resolution_hint(status.provider))
        raise SystemExit(1)


def _print_login_result(provider_label: str, status: AuthStatus) -> None:
    """Print login attempt result, raise SystemExit(1) on failure."""
    if status.authenticated:
        click.echo(f"{provider_label} authenticated: {_display_user(status)}")
        click.echo(
            "Next: run 'markitai init' to add this provider to your config, "
            "then convert with --llm."
        )
    else:
        click.echo(f"{provider_label} login failed: {status.error}")
        click.echo(get_auth_resolution_hint(status.provider))
        raise SystemExit(1)


# ── Main group ───────────────────────────────────────────────────────────────


@click.group()
def auth() -> None:
    """Authentication helpers for local providers.

    Check or set up login for Claude Code, GitHub Copilot, Gemini CLI,
    and ChatGPT so markitai can use them for LLM processing (--llm)
    without API keys.

    Examples:
        markitai auth claude status     # Is Claude Code logged in?
        markitai auth claude login      # Log in via Claude Code CLI
        markitai auth gemini login      # Google OAuth login for Gemini
        markitai doctor                 # Check all providers at once
    """


# ── Gemini ───────────────────────────────────────────────────────────────────


@auth.group()
def gemini() -> None:
    """Gemini CLI authentication helpers.

    Uses Google OAuth (free tier available); credentials can be shared
    with an existing gemini CLI install or managed by markitai.

    Examples:
        markitai auth gemini status     # Show the active profile
        markitai auth gemini login      # Run OAuth login in the browser
    """


@gemini.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def gemini_status(as_json: bool) -> None:
    """Show the current Gemini authentication profile.

    Exits non-zero when not authenticated, so it is script-friendly.

    Examples:
        markitai auth gemini status         # Human-readable status
        markitai auth gemini status --json  # Machine-readable status
    """
    status = _check_status("gemini-cli")

    if as_json:
        click.echo(json.dumps(_status_to_payload(status), indent=2, ensure_ascii=False))
        return

    if status.authenticated:
        click.echo(f"Gemini authenticated: {_display_user(status)}")
        details = status.details or {}
        # Only show extra details for managed profiles (markitai-managed)
        if details.get("source") == "markitai":
            project_id = details.get("project_id")
            auth_mode = details.get("auth_mode")
            credential_path = details.get("credential_path")
            if project_id:
                click.echo(f"Project: {project_id}")
            if auth_mode:
                click.echo(f"Mode: {auth_mode}")
            if credential_path:
                click.echo(f"Profile: {credential_path}")
        return

    click.echo(f"Gemini not authenticated: {status.error}")
    click.echo(get_auth_resolution_hint("gemini-cli"))
    raise SystemExit(1)


@gemini.command("login")
@click.option(
    "--mode",
    type=click.Choice(["google-one", "code-assist"], case_sensitive=False),
    default="google-one",
    show_default=True,
    help="Project binding mode.",
)
@click.option(
    "--project-id",
    type=str,
    default=None,
    help="Explicit GCP project ID for Code Assist mode.",
)
def gemini_login(mode: str, project_id: str | None) -> None:
    """Run Gemini OAuth login and save a Markitai-managed profile.

    Opens a browser for Google OAuth. Use --mode code-assist with
    --project-id if your account requires a GCP project binding.

    Examples:
        markitai auth gemini login
        markitai auth gemini login --mode code-assist --project-id my-project
    """
    from markitai.providers.gemini_cli import GeminiCLIProvider

    provider = GeminiCLIProvider()
    record = asyncio.run(provider.alogin(mode=mode, project_id=project_id))
    AuthManager().clear_cache("gemini-cli")

    click.echo(f"Gemini authenticated: {record.email}")
    click.echo(f"Project: {record.project_id}")
    click.echo(f"Mode: {record.auth_mode}")
    click.echo(f"Profile: {record.path}")
    click.echo(
        "Next: run 'markitai init' to add this provider to your config, "
        "then convert with --llm."
    )


# ── Copilot ──────────────────────────────────────────────────────────────────


@auth.group()
def copilot() -> None:
    """Copilot CLI authentication helpers.

    Uses your GitHub Copilot subscription via the copilot CLI
    (or a GH_TOKEN/GITHUB_TOKEN env var).

    Examples:
        markitai auth copilot status    # Check current login
        markitai auth copilot login     # Log in via copilot CLI
    """


@copilot.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def copilot_status(as_json: bool) -> None:
    """Show Copilot authentication status.

    Exits non-zero when not authenticated, so it is script-friendly.

    Examples:
        markitai auth copilot status         # Human-readable status
        markitai auth copilot status --json  # Machine-readable status
    """
    status = _check_status("copilot")
    if as_json:
        click.echo(json.dumps(_status_to_payload(status), indent=2, ensure_ascii=False))
        return
    _print_status("Copilot", status)


@copilot.command("login")
def copilot_login() -> None:
    """Run Copilot CLI login (copilot login)."""
    result = asyncio.run(attempt_login("copilot"))
    _print_login_result("Copilot", result)


# ── Claude ───────────────────────────────────────────────────────────────────


@auth.group()
def claude() -> None:
    """Claude Code CLI authentication helpers.

    A logged-in Claude Code CLI lets markitai use your Claude
    subscription quota via claude-agent/ models (no ANTHROPIC_API_KEY).

    Examples:
        markitai auth claude status     # Check CLI + SDK + login state
        markitai auth claude login      # Log in via Claude Code CLI
    """


@claude.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def claude_status(as_json: bool) -> None:
    """Show Claude authentication status.

    Reports whether the Claude Code CLI is installed and logged in.
    A logged-in CLI lets markitai use your Claude subscription quota
    (no ANTHROPIC_API_KEY needed) via claude-agent/ models.

    Examples:
        markitai auth claude status         # Human-readable status
        markitai auth claude status --json  # Machine-readable status
    """
    from markitai.providers.auth import (
        _is_claude_agent_sdk_available,
        _resolve_cli_path,
    )

    status = _check_status("claude-agent")
    cli_path = _resolve_cli_path("claude")
    sdk_installed = _is_claude_agent_sdk_available()

    if as_json:
        payload = _status_to_payload(status)
        payload["cli_path"] = cli_path
        payload["sdk_installed"] = sdk_installed
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if not status.authenticated:
        _print_status("Claude", status)
        return

    click.echo(f"Claude authenticated: {_display_user(status)}")
    click.echo(f"CLI: {cli_path or 'not found in PATH (SDK bundles its own)'}")
    if sdk_installed:
        click.echo("SDK: claude-agent-sdk installed")
    else:
        click.echo(
            "SDK: claude-agent-sdk not installed — run: "
            "uv tool install 'markitai[claude-agent]' --upgrade"
        )
    click.echo(
        "\nLLM calls with claude-agent/ models use your Claude subscription "
        "quota.\nEnable via 'markitai init' (auto-detected), or add a model "
        "manually:\n"
        '  {"model_name": "default", '
        '"litellm_params": {"model": "claude-agent/sonnet"}}'
    )


@claude.command("login")
def claude_login() -> None:
    """Run Claude Code CLI login (claude auth login)."""
    result = asyncio.run(attempt_login("claude-agent"))
    _print_login_result("Claude", result)


# ── ChatGPT ──────────────────────────────────────────────────────────────────


@auth.group()
def chatgpt() -> None:
    """ChatGPT authentication helpers.

    Uses OAuth Device Code Flow; login is triggered automatically on
    the first chatgpt/ model call if needed.

    Examples:
        markitai auth chatgpt status    # Check current login
        markitai auth chatgpt login     # Trigger device-code login
    """


@chatgpt.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def chatgpt_status(as_json: bool) -> None:
    """Show ChatGPT authentication status.

    Exits non-zero when not authenticated, so it is script-friendly.

    Examples:
        markitai auth chatgpt status         # Human-readable status
        markitai auth chatgpt status --json  # Machine-readable status
    """
    status = _check_status("chatgpt")
    if as_json:
        click.echo(json.dumps(_status_to_payload(status), indent=2, ensure_ascii=False))
        return
    _print_status("ChatGPT", status)


@chatgpt.command("login")
def chatgpt_login() -> None:
    """Trigger ChatGPT Device Code Flow login."""
    result = asyncio.run(attempt_login("chatgpt"))
    if result.details and result.details.get("auto_login"):
        click.echo(
            "ChatGPT uses OAuth Device Code Flow.\n"
            "Login will be triggered automatically on first API call."
        )
    else:
        _print_login_result("ChatGPT", result)
