"""Interactive CLI mode for guided setup.

This module provides an interactive TUI for users who prefer guided
configuration over command-line flags.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import questionary

from markitai.cli.console import get_console


@dataclass
class ProviderDetectionResult:
    """Result of LLM provider auto-detection."""

    provider: str
    model: str
    authenticated: bool
    source: str  # "cli", "env", "config"


def _check_claude_auth() -> bool:
    """Check if Claude CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("claude-agent"))
        return status.authenticated
    except Exception:
        return False


def _check_copilot_auth() -> bool:
    """Check if Copilot CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("copilot"))
        return status.authenticated
    except Exception:
        return False


def detect_llm_provider() -> ProviderDetectionResult | None:
    """Auto-detect available LLM provider.

    Detection priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ANTHROPIC_API_KEY environment variable
    4. OPENAI_API_KEY environment variable
    5. GEMINI_API_KEY environment variable

    Returns:
        ProviderDetectionResult if a provider is found, None otherwise.
    """
    # 1. Check Claude CLI
    if shutil.which("claude"):
        if _check_claude_auth():
            return ProviderDetectionResult(
                provider="claude-agent",
                model="claude-agent/sonnet",
                authenticated=True,
                source="cli",
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot"):
        if _check_copilot_auth():
            return ProviderDetectionResult(
                provider="copilot",
                model="copilot/gpt-4o",
                authenticated=True,
                source="cli",
            )

    # 3. Check environment variables
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderDetectionResult(
            provider="anthropic",
            model="anthropic/claude-sonnet-4-20250514",
            authenticated=True,
            source="env",
        )

    if os.environ.get("OPENAI_API_KEY"):
        return ProviderDetectionResult(
            provider="openai",
            model="openai/gpt-4o",
            authenticated=True,
            source="env",
        )

    if os.environ.get("GEMINI_API_KEY"):
        return ProviderDetectionResult(
            provider="gemini",
            model="gemini/gemini-2.0-flash",
            authenticated=True,
            source="env",
        )

    return None


@dataclass
class InteractiveSession:
    """State for an interactive CLI session."""

    input_path: Path | str | None = None
    input_type: str = "file"  # "file", "directory", "url"
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    enable_llm: bool = False
    enable_alt: bool = False
    enable_desc: bool = False
    enable_ocr: bool = False
    enable_screenshot: bool = False
    provider_result: ProviderDetectionResult | None = None


def prompt_input_type(session: InteractiveSession) -> str:
    """Prompt user to select input type."""
    choices = [
        questionary.Choice("Single file", value="file"),
        questionary.Choice("Directory (batch)", value="directory"),
        questionary.Choice("URL", value="url"),
    ]
    result = questionary.select(
        "What would you like to convert?",
        choices=choices,
        style=questionary.Style(
            [
                ("highlighted", "bold"),
                ("selected", "fg:cyan"),
            ]
        ),
    ).ask()

    if result:
        session.input_type = result
    return result or "file"


def prompt_input_path(session: InteractiveSession) -> Path | str | None:
    """Prompt user for input path."""
    if session.input_type == "url":
        result = questionary.text(
            "Enter URL:",
            validate=lambda x: len(x) > 0 or "URL cannot be empty",
        ).ask()
    elif session.input_type == "directory":
        result = questionary.path(
            "Enter directory path:",
            only_directories=True,
        ).ask()
    else:
        result = questionary.path(
            "Enter file path:",
        ).ask()

    if result:
        if session.input_type == "url":
            session.input_path = result  # Keep as string for URLs
        else:
            session.input_path = Path(result)
        return session.input_path
    return None


def prompt_enable_llm(session: InteractiveSession) -> bool:
    """Prompt user to enable LLM enhancement."""
    # First, detect available providers
    session.provider_result = detect_llm_provider()

    if session.provider_result:
        provider_info = (
            f"Detected: {session.provider_result.provider} "
            f"({session.provider_result.source})"
        )
        get_console().print(f"[green]✓[/green] {provider_info}")

    result = questionary.confirm(
        "Enable LLM enhancement? (better formatting, metadata)",
        default=session.provider_result is not None,
    ).ask()

    if result is not None:
        session.enable_llm = result
    return session.enable_llm


def prompt_llm_options(session: InteractiveSession) -> None:
    """Prompt user for LLM-related options."""
    if not session.enable_llm:
        return

    choices = questionary.checkbox(
        "Select LLM features:",
        choices=[
            questionary.Choice(
                "Generate alt text for images", value="alt", checked=True
            ),
            questionary.Choice(
                "Generate image descriptions (JSON)", value="desc", checked=False
            ),
            questionary.Choice(
                "Enable OCR for scanned documents", value="ocr", checked=True
            ),
            questionary.Choice(
                "Take page screenshots", value="screenshot", checked=False
            ),
        ],
    ).ask()

    if choices:
        session.enable_alt = "alt" in choices
        session.enable_desc = "desc" in choices
        session.enable_ocr = "ocr" in choices
        session.enable_screenshot = "screenshot" in choices


def prompt_configure_provider(session: InteractiveSession) -> bool:
    """Prompt user to configure LLM provider if none detected."""
    if session.provider_result:
        return True

    get_console().print("[yellow]![/yellow] No LLM provider detected.")

    choices = [
        questionary.Choice("Auto-detect (Claude CLI / Copilot CLI)", value="auto"),
        questionary.Choice("Enter API key manually", value="manual"),
        questionary.Choice("Use .env file", value="env"),
        questionary.Choice("Skip for now", value="skip"),
    ]

    result = questionary.select(
        "How would you like to configure LLM?",
        choices=choices,
    ).ask()

    if result == "skip":
        session.enable_llm = False
        return False

    if result == "manual":
        return _prompt_manual_api_key(session)

    if result == "env":
        return _prompt_env_file(session)

    return False


def _prompt_manual_api_key(session: InteractiveSession) -> bool:
    """Prompt for manual API key entry."""
    provider = questionary.select(
        "Select provider:",
        choices=[
            questionary.Choice("Anthropic (Claude)", value="anthropic"),
            questionary.Choice("OpenAI (GPT-4o)", value="openai"),
            questionary.Choice("Google (Gemini)", value="gemini"),
        ],
    ).ask()

    if not provider:
        return False

    api_key = questionary.password(
        f"Enter {provider.upper()} API key:",
    ).ask()

    if not api_key:
        return False

    # Save to config
    from markitai.config import ConfigManager, LiteLLMParams, ModelConfig

    model_map = {
        "anthropic": "anthropic/claude-sonnet-4-20250514",
        "openai": "openai/gpt-4o",
        "gemini": "gemini/gemini-2.0-flash",
    }

    manager = ConfigManager()
    cfg = manager.load()
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model=model_map[provider], api_key=api_key),
        )
    ]
    manager.save()

    session.provider_result = ProviderDetectionResult(
        provider=provider,
        model=model_map[provider],
        authenticated=True,
        source="manual",
    )

    get_console().print("[green]✓[/green] API key saved to config")
    return True


def _prompt_env_file(session: InteractiveSession) -> bool:
    """Prompt for .env file location."""
    env_path = questionary.path(
        "Enter .env file path:",
        default=".env",
    ).ask()

    if not env_path or not Path(env_path).exists():
        get_console().print("[red]✗[/red] .env file not found")
        return False

    # Load .env file
    from dotenv import load_dotenv

    load_dotenv(env_path)

    # Re-detect provider
    session.provider_result = detect_llm_provider()

    if session.provider_result:
        get_console().print(
            f"[green]✓[/green] Detected: {session.provider_result.provider}"
        )
        return True

    get_console().print("[red]✗[/red] No API key found in .env file")
    return False
