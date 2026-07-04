from __future__ import annotations

import asyncio
import json
from typing import Any

import rich_click as click

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.providers.auth import (
    AuthManager,
    AuthStatus,
    attempt_login,
    get_auth_resolution_hint,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

# provider key -> (exact login command, what it does)
_LOGIN_HINTS: dict[str, tuple[str, str]] = {
    "claude-agent": (
        "markitai auth claude login",
        "Runs 'claude auth login' to sign in with your Claude account.",
    ),
    "copilot": (
        "markitai auth copilot login",
        "Runs the GitHub Copilot CLI login flow.",
    ),
    "gemini-cli": (
        "markitai auth gemini login",
        "Opens a browser for Google OAuth login.",
    ),
    "chatgpt": (
        "markitai auth chatgpt login",
        "Starts the OAuth Device Code Flow (visit the URL, enter the code).",
    ),
}

# provider key -> what an authenticated provider enables
_USAGE_HINTS: dict[str, str] = {
    "claude-agent": (
        "LLM calls with claude-agent/ models use your Claude subscription quota."
    ),
    "copilot": ("LLM calls with copilot/ models use your GitHub Copilot subscription."),
    "gemini-cli": (
        "LLM calls with gemini-cli/ models use your Google account "
        "(free tier available)."
    ),
    "chatgpt": "LLM calls with chatgpt/ models use your ChatGPT subscription.",
}

_NEXT_INIT = "Next: markitai init   (auto-detects and enables this provider)"


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


def _render_status_card(
    provider_label: str,
    status: AuthStatus,
    *,
    checks: list[tuple[str, str]] | None = None,
    infos: list[str] | None = None,
) -> None:
    """Render the unified status card; raise SystemExit(1) if unauthenticated.

    Args:
        provider_label: Human-readable label ("Claude", "ChatGPT", ...).
        status: Authentication status to render.
        checks: Extra check lines as (kind, text) with kind "ok" or "warn".
        infos: Extra dim detail lines (bullet, no pass/fail meaning).
    """
    console = get_console()
    ui.title(f"{provider_label} Authentication")

    if status.authenticated:
        ui.success(f"Logged in: {_display_user(status)}")
    else:
        ui.error("Not logged in", detail=status.error)

    for kind, text in checks or []:
        if kind == "ok":
            ui.success(text)
        else:
            ui.warning(text)
    for line in infos or []:
        ui.info(line)

    console.print()
    if status.authenticated:
        console.print(f"  {_USAGE_HINTS[status.provider]}")
        console.print(f"  [dim]{_NEXT_INIT}[/]")
        return

    cmd, does = _LOGIN_HINTS[status.provider]
    console.print(f"  Next: [bold]{cmd}[/]")
    console.print(f"        [dim]{does}[/]")
    raise SystemExit(1)


def _print_login_result(provider_label: str, status: AuthStatus) -> None:
    """Print login attempt result, raise SystemExit(1) on failure."""
    console = get_console()
    if status.authenticated:
        ui.summary(f"{provider_label} login successful: {_display_user(status)}")
        console.print(f"  [dim]{_NEXT_INIT}[/]")
    else:
        ui.summary(f"{provider_label} login failed: {status.error}", ok=False)
        click.echo(get_auth_resolution_hint(status.provider))
        raise SystemExit(1)


# ── Main group ───────────────────────────────────────────────────────────────


# CLI subcommand name -> provider key, in overview display order
_OVERVIEW_PROVIDERS: list[tuple[str, str]] = [
    ("claude", "claude-agent"),
    ("chatgpt", "chatgpt"),
    ("copilot", "copilot"),
    ("gemini", "gemini-cli"),
]


@click.group(invoke_without_command=True)
@click.pass_context
def auth(ctx: click.Context) -> None:
    """Authentication helpers for local providers.

    Check or set up login for Claude Code, GitHub Copilot, Gemini CLI,
    and ChatGPT so markitai can use them for LLM processing (--llm)
    without API keys. Run without a subcommand to see an overview of
    all providers.

    Examples:
        markitai auth                   # Overview of all providers
        markitai auth claude status     # Is Claude Code logged in?
        markitai auth claude login      # Log in via Claude Code CLI
        markitai auth gemini login      # Google OAuth login for Gemini
        markitai doctor                 # Check all providers at once
    """
    if ctx.invoked_subcommand is not None:
        return

    console = get_console()
    ui.title("Provider Authentication")
    for name, provider in _OVERVIEW_PROVIDERS:
        status = _check_status(provider)
        if status.authenticated:
            ui.success(f"{name:<8} {_display_user(status)}")
        else:
            ui.error(f"{name:<8} not logged in")
    console.print()
    console.print("  [dim]Details: markitai auth <provider> status[/]")
    console.print("  [dim]Login:   markitai auth <provider> login[/]")


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

    infos: list[str] = []
    details = status.details or {}
    # Only show extra details for managed profiles (markitai-managed)
    if status.authenticated and details.get("source") == "markitai":
        if details.get("project_id"):
            infos.append(f"Project: {details['project_id']}")
        if details.get("auth_mode"):
            infos.append(f"Mode: {details['auth_mode']}")
        if details.get("credential_path"):
            infos.append(f"Profile: {details['credential_path']}")
    _render_status_card("Gemini", status, infos=infos)


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

    console = get_console()
    ui.summary(f"Gemini login successful: {record.email}")
    ui.info(f"Project: {record.project_id}")
    ui.info(f"Mode: {record.auth_mode}")
    ui.info(f"Profile: {record.path}")
    console.print(f"  [dim]{_NEXT_INIT}[/]")


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
    from markitai.providers.auth import _get_cli_install_cmd, _resolve_cli_path

    status = _check_status("copilot")
    if as_json:
        click.echo(json.dumps(_status_to_payload(status), indent=2, ensure_ascii=False))
        return

    checks: list[tuple[str, str]] = []
    infos: list[str] = []
    if status.authenticated:
        source = (status.details or {}).get("source")
        if source == "env":
            infos.append("Source: GH_TOKEN/GITHUB_TOKEN environment variable")
        elif source == "config":
            infos.append("Source: copilot CLI config (~/.copilot/config.json)")
    else:
        if _resolve_cli_path("copilot") is None:
            checks.append(
                (
                    "warn",
                    "copilot CLI not found — install: "
                    f"{_get_cli_install_cmd('copilot')}",
                )
            )
        infos.append(
            "Alternative: set GH_TOKEN or GITHUB_TOKEN "
            "(needs 'Copilot Requests' permission)"
        )
    _render_status_card("Copilot", status, checks=checks, infos=infos)


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
        _get_cli_install_cmd,
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

    checks: list[tuple[str, str]] = []
    if cli_path:
        checks.append(("ok", f"CLI: {cli_path}"))
    elif status.authenticated:
        checks.append(("warn", "CLI: not found in PATH (SDK bundles its own)"))
    else:
        checks.append(
            (
                "warn",
                f"CLI: claude not found — install: {_get_cli_install_cmd('claude')}",
            )
        )
    if sdk_installed:
        checks.append(("ok", "SDK: claude-agent-sdk installed"))
    else:
        checks.append(
            (
                "warn",
                "SDK: claude-agent-sdk not installed — run: "
                "uv tool install 'markitai\\[claude-agent]' --upgrade",
            )
        )
    _render_status_card("Claude", status, checks=checks)


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
    from markitai.providers.auth import can_attempt_login

    status = _check_status("chatgpt")
    if as_json:
        click.echo(json.dumps(_status_to_payload(status), indent=2, ensure_ascii=False))
        return

    checks: list[tuple[str, str]] = []
    if not status.authenticated and not can_attempt_login("chatgpt"):
        checks.append(
            ("warn", "LiteLLM chatgpt authenticator not available (update litellm)")
        )
    _render_status_card("ChatGPT", status, checks=checks)


@chatgpt.command("login")
def chatgpt_login() -> None:
    """Trigger ChatGPT OAuth Device Code Flow login.

    Requests a device code, shows the verification URL and code, then
    waits for you to finish the login in a browser. Tokens are saved
    to ~/.config/litellm/chatgpt/auth.json.
    """
    result = asyncio.run(attempt_login("chatgpt"))
    _print_login_result("ChatGPT", result)
