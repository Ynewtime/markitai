from __future__ import annotations

import asyncio
import json
from typing import Any

import click

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
    else:
        click.echo(f"{provider_label} login failed: {status.error}")
        click.echo(get_auth_resolution_hint(status.provider))
        raise SystemExit(1)


# ── Main group ───────────────────────────────────────────────────────────────


@click.group()
def auth() -> None:
    """Authentication helpers for local providers."""


# ── Gemini ───────────────────────────────────────────────────────────────────


@auth.group()
def gemini() -> None:
    """Gemini CLI authentication helpers."""


@gemini.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def gemini_status(as_json: bool) -> None:
    """Show the current Gemini authentication profile."""
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
    """Run Gemini OAuth login and save a Markitai-managed profile."""
    from markitai.providers.gemini_cli import GeminiCLIProvider

    provider = GeminiCLIProvider()
    record = asyncio.run(provider.alogin(mode=mode, project_id=project_id))
    AuthManager().clear_cache("gemini-cli")

    click.echo(f"Gemini authenticated: {record.email}")
    click.echo(f"Project: {record.project_id}")
    click.echo(f"Mode: {record.auth_mode}")
    click.echo(f"Profile: {record.path}")


# ── Copilot ──────────────────────────────────────────────────────────────────


@auth.group()
def copilot() -> None:
    """Copilot CLI authentication helpers."""


@copilot.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def copilot_status(as_json: bool) -> None:
    """Show Copilot authentication status."""
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
    """Claude Code CLI authentication helpers."""


@claude.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def claude_status(as_json: bool) -> None:
    """Show Claude authentication status."""
    status = _check_status("claude-agent")
    if as_json:
        click.echo(json.dumps(_status_to_payload(status), indent=2, ensure_ascii=False))
        return
    _print_status("Claude", status)


@claude.command("login")
def claude_login() -> None:
    """Run Claude Code CLI login (claude auth login)."""
    result = asyncio.run(attempt_login("claude-agent"))
    _print_login_result("Claude", result)


# ── ChatGPT ──────────────────────────────────────────────────────────────────


@auth.group()
def chatgpt() -> None:
    """ChatGPT authentication helpers."""


@chatgpt.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output auth status as JSON.")
def chatgpt_status(as_json: bool) -> None:
    """Show ChatGPT authentication status."""
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
