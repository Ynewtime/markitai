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
from typing import Any

import questionary

from markitai.cli import ui
from markitai.cli.console import get_console


def _is_dev_mode() -> bool:
    """Check if running in the markitai project directory (dev mode).

    Mirrors the is_dev_mode() check in setup.sh.
    """
    cwd = Path.cwd()
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file() and (cwd / ".git").exists() and (cwd / "scripts").is_dir():
        try:
            return "markitai" in pyproject.read_text(encoding="utf-8")
        except OSError:
            pass
    return False


def _get_default_env_path() -> Path:
    """Return the default .env path based on dev vs installed mode.

    Dev mode:  ./.env  (project root, loaded first by main.py)
    Installed: ~/.markitai/.env  (global config dir)
    """
    if _is_dev_mode():
        return Path.cwd() / ".env"
    return Path.home() / ".markitai" / ".env"


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


def _check_chatgpt_auth() -> bool:
    """Check if ChatGPT provider is authenticated."""
    from markitai.providers.auth import _check_chatgpt_auth as check_fn

    try:
        status = check_fn()
        return status.authenticated
    except Exception:
        return False


def _check_gemini_cli_auth() -> bool:
    """Check if Gemini CLI is authenticated."""
    from markitai.providers.auth import _check_gemini_cli_auth as check_fn

    try:
        status = check_fn()
        return status.authenticated
    except Exception:
        return False


def get_active_models_from_config(
    model_list: list[dict[str, Any]],
) -> list[str]:
    """Extract active model names (weight > 0) from config model_list.

    Args:
        model_list: Raw model_list dicts from config (each has litellm_params).

    Returns:
        List of model identifiers with positive weight.
    """
    active: list[str] = []
    for entry in model_list:
        params = entry.get("litellm_params", {})
        model = params.get("model", "")
        weight = params.get("weight", 1)  # default weight is 1 (enabled)
        if model and weight > 0:
            active.append(model)
    return active


def detect_all_llm_providers() -> list[ProviderDetectionResult]:
    """Auto-detect all available LLM providers.

    Checks each provider independently and returns all that are available,
    ordered by priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ChatGPT (if authenticated via OAuth)
    4. Gemini CLI (if authenticated via OAuth)
    5. ANTHROPIC_API_KEY environment variable
    6. OPENAI_API_KEY environment variable
    7. GEMINI_API_KEY environment variable
    8. DEEPSEEK_API_KEY environment variable
    9. OPENROUTER_API_KEY environment variable

    Returns:
        List of all detected providers (may be empty).
    """
    results: list[ProviderDetectionResult] = []

    # 1. Check Claude CLI
    if shutil.which("claude"):
        if _check_claude_auth():
            results.append(
                ProviderDetectionResult(
                    provider="claude-agent",
                    model="claude-agent/sonnet",
                    authenticated=True,
                    source="cli",
                )
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot"):
        if _check_copilot_auth():
            results.append(
                ProviderDetectionResult(
                    provider="copilot",
                    model="copilot/claude-sonnet-4.5",
                    authenticated=True,
                    source="cli",
                )
            )

    # 3. Check ChatGPT (OAuth)
    if _check_chatgpt_auth():
        results.append(
            ProviderDetectionResult(
                provider="chatgpt",
                model="chatgpt/gpt-5.2",
                authenticated=True,
                source="cli",
            )
        )

    # 4. Check Gemini CLI (OAuth)
    if _check_gemini_cli_auth():
        results.append(
            ProviderDetectionResult(
                provider="gemini-cli",
                model="gemini-cli/gemini-2.5-pro",
                authenticated=True,
                source="cli",
            )
        )

    # 5-9. Check environment variables
    env_providers = [
        ("ANTHROPIC_API_KEY", "anthropic", "anthropic/claude-sonnet-4-5-20250929"),
        ("OPENAI_API_KEY", "openai", "openai/gpt-5.2"),
        ("GEMINI_API_KEY", "gemini", "gemini/gemini-2.5-flash"),
        ("DEEPSEEK_API_KEY", "deepseek", "deepseek/deepseek-chat"),
        ("OPENROUTER_API_KEY", "openrouter", "openrouter/google/gemini-2.5-flash"),
    ]
    for env_var, provider, model in env_providers:
        if os.environ.get(env_var):
            results.append(
                ProviderDetectionResult(
                    provider=provider,
                    model=model,
                    authenticated=True,
                    source="env",
                )
            )

    return results


def detect_llm_provider() -> ProviderDetectionResult | None:
    """Auto-detect the highest-priority available LLM provider.

    Returns:
        ProviderDetectionResult for the best provider, or None.
    """
    providers = detect_all_llm_providers()
    return providers[0] if providers else None


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
    active_models: list[str] = field(default_factory=list)


def _ask_or_exit(question: questionary.Question) -> Any:
    """Run a questionary prompt; raise KeyboardInterrupt on Ctrl+C / None."""
    result = question.ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def prompt_input_type(session: InteractiveSession) -> str:
    """Prompt user to select input type."""
    choices = [
        questionary.Choice("Single file", value="file"),
        questionary.Choice("Directory (batch)", value="directory"),
        questionary.Choice("URL", value="url"),
    ]
    result = _ask_or_exit(
        questionary.select(
            "What would you like to convert?",
            choices=choices,
            style=questionary.Style(
                [
                    ("highlighted", "bold"),
                    ("selected", "fg:cyan"),
                ]
            ),
        )
    )

    session.input_type = result
    return result


def prompt_input_path(session: InteractiveSession) -> Path | str:
    """Prompt user for input path."""
    if session.input_type == "url":
        result = _ask_or_exit(
            questionary.text(
                "Enter URL:",
                validate=lambda x: len(x) > 0 or "URL cannot be empty",
            )
        )
    elif session.input_type == "directory":
        result = _ask_or_exit(
            questionary.path(
                "Enter directory path:",
                only_directories=True,
            )
        )
    else:
        result = _ask_or_exit(
            questionary.path(
                "Enter file path:",
            )
        )

    path: Path | str = result if session.input_type == "url" else Path(result)
    session.input_path = path
    return path


def prompt_output_dir(session: InteractiveSession) -> Path:
    """Prompt user for output directory."""
    result = _ask_or_exit(
        questionary.path(
            "Enter output directory:",
            default=str(session.output_dir),
            only_directories=True,
        )
    )

    session.output_dir = Path(result)
    return session.output_dir


def _format_model_list(models: list[str], max_show: int = 3) -> str:
    """Format a list of model names for display.

    Shows up to *max_show* names, with a "+N more" suffix if there are extras.
    """
    shown = ", ".join(models[:max_show])
    extra = len(models) - max_show
    if extra > 0:
        shown += f" (+{extra} more)"
    return shown


def prompt_enable_llm(session: InteractiveSession) -> bool:
    """Prompt user to enable LLM enhancement."""
    console = get_console()
    has_provider = False

    # 1. Check config file for active models (weight > 0)
    from markitai.config import ConfigManager

    try:
        manager = ConfigManager()
        cfg = manager.load()
        raw_model_list = [m.model_dump() for m in cfg.llm.model_list]
        active = get_active_models_from_config(raw_model_list)
        if active:
            session.active_models = active
            has_provider = True
            console.print(
                f"[green]\u2713[/green] Configured: {_format_model_list(active)}"
            )
    except Exception:
        pass

    # 2. Fallback: auto-detect providers if no config models
    if not has_provider:
        all_providers = detect_all_llm_providers()
        session.provider_result = all_providers[0] if all_providers else None
        if all_providers:
            has_provider = True
            names = [p.provider for p in all_providers]
            console.print(
                f"[green]\u2713[/green] Detected: {_format_model_list(names)}"
            )
        else:
            console.print(
                "[yellow]![/yellow] No LLM provider detected "
                "(no CLI tools or API keys found)"
            )

    result = _ask_or_exit(
        questionary.confirm(
            "Enable LLM enhancement? (better formatting, metadata)",
            default=has_provider,
        )
    )

    session.enable_llm = result
    return session.enable_llm


def prompt_llm_options(session: InteractiveSession) -> None:
    """Prompt user for LLM-related options."""
    if not session.enable_llm:
        return

    choices = _ask_or_exit(
        questionary.checkbox(
            "Select LLM features:",
            choices=[
                questionary.Choice(
                    "Generate alt text for images", value="alt", checked=False
                ),
                questionary.Choice(
                    "Generate image descriptions (JSON)", value="desc", checked=False
                ),
                questionary.Choice(
                    "Enable OCR for scanned documents", value="ocr", checked=False
                ),
                questionary.Choice(
                    "Take page screenshots", value="screenshot", checked=False
                ),
            ],
        )
    )

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

    result = _ask_or_exit(
        questionary.select(
            "How would you like to configure LLM?",
            choices=choices,
        )
    )

    if result == "skip":
        session.enable_llm = False
        return False

    if result == "manual":
        return _prompt_manual_api_key(session)

    if result == "env":
        return _prompt_env_file(session)

    # result == "auto": run auto-detection
    detected = detect_llm_provider()
    if detected:
        session.provider_result = detected
        get_console().print(
            f"[green]\u2713[/green] Detected: {detected.provider} ({detected.source})"
        )
        return True

    get_console().print(
        "[red]\u2717[/red] No provider found. "
        "Install Claude CLI / Copilot CLI, or set an API key."
    )
    return False


def _prompt_manual_api_key(session: InteractiveSession) -> bool:
    """Prompt for manual API key entry.

    Saves API key to .env file and references it via env: prefix in config.
    """
    provider = _ask_or_exit(
        questionary.select(
            "Select provider:",
            choices=[
                questionary.Choice("Anthropic (Claude)", value="anthropic"),
                questionary.Choice("OpenAI", value="openai"),
                questionary.Choice("Google (Gemini)", value="gemini"),
                questionary.Choice("DeepSeek", value="deepseek"),
            ],
        )
    )

    api_key = _ask_or_exit(
        questionary.password(
            f"Enter {provider.upper()} API key:",
        )
    )

    model_map = {
        "anthropic": "anthropic/claude-sonnet-4-5-20250929",
        "openai": "openai/gpt-5.2",
        "gemini": "gemini/gemini-2.5-flash",
        "deepseek": "deepseek/deepseek-chat",
    }

    env_var_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }

    # Save API key to .env file (project-local in dev mode, global otherwise)
    env_var = env_var_map[provider]
    env_path = _get_default_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    _append_env_var(env_path, env_var, api_key)

    # Save config with env: reference (no plaintext key)
    from markitai.config import ConfigManager, LiteLLMParams, ModelConfig

    manager = ConfigManager()
    cfg = manager.load()
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(
                model=model_map[provider], api_key=f"env:{env_var}"
            ),
        )
    ]
    manager.save()

    session.provider_result = ProviderDetectionResult(
        provider=provider,
        model=model_map[provider],
        authenticated=True,
        source="manual",
    )

    get_console().print(f"[green]✓[/green] API key saved to {env_path}")
    get_console().print(
        f"[green]✓[/green] Config uses [cyan]env:{env_var}[/cyan] reference"
    )
    return True


def _append_env_var(env_path: Path, var_name: str, value: str) -> None:
    """Append or update an environment variable in a .env file."""
    from markitai.security import atomic_write_text

    lines: list[str] = []
    found = False

    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{var_name}="):
                lines[i] = f"{var_name}={value}"
                found = True
                break

    if not found:
        lines.append(f"{var_name}={value}")

    atomic_write_text(env_path, "\n".join(lines) + "\n")


def _prompt_env_file(session: InteractiveSession) -> bool:
    """Prompt for .env file location."""
    global_env = str(_get_default_env_path())
    env_path = _ask_or_exit(
        questionary.path(
            "Enter .env file path:",
            default=global_env,
        )
    )

    if not Path(env_path).exists():
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


def _print_loaded_paths() -> None:
    """Print detected config and .env file paths in the header."""
    from markitai.config import ConfigManager

    console = get_console()

    # Detect config file
    manager = ConfigManager()
    manager.load()
    config_path = manager.config_path
    if config_path:
        console.print(f"  [dim]Config:[/dim] {config_path}")

    # Detect .env file(s)
    env_paths = [
        Path.cwd() / ".env",
        Path.home() / ".markitai" / ".env",
    ]
    for env_p in env_paths:
        if env_p.is_file():
            console.print(f"  [dim].env:[/dim]   {env_p}")
            break

    if config_path or any(p.is_file() for p in env_paths):
        console.print()


def run_interactive() -> InteractiveSession:
    """Run the interactive CLI session.

    Returns:
        InteractiveSession with user's choices populated.
    """
    session = InteractiveSession()
    console = get_console()

    # Print header using unified UI
    console.print()
    ui.title("Markitai Interactive")
    _print_loaded_paths()
    console.print("  Answer the following questions to configure your conversion.")
    console.print()

    # 1. Input type
    prompt_input_type(session)

    # 2. Input path
    prompt_input_path(session)

    # 3. Output directory
    prompt_output_dir(session)

    # 4. LLM enablement
    prompt_enable_llm(session)

    # 5. Configure provider if needed (skip if config models or auto-detected)
    if session.enable_llm and not session.active_models and not session.provider_result:
        if not prompt_configure_provider(session):
            session.enable_llm = False

    # 6. LLM options
    if session.enable_llm:
        prompt_llm_options(session)

    # Print summary
    _print_summary(session)

    return session


def _print_summary(session: InteractiveSession) -> None:
    """Print a summary of the session configuration."""
    console = get_console()
    console.print()
    console.print("[bold]Configuration Summary:[/bold]")
    console.print(f"  Input: {session.input_path} ({session.input_type})")
    console.print(f"  Output: {session.output_dir}")
    console.print(f"  LLM: {'enabled' if session.enable_llm else 'disabled'}")

    if session.enable_llm:
        if session.active_models:
            console.print(f"  Models: {_format_model_list(session.active_models)}")
        elif session.provider_result:
            console.print(f"  Provider: {session.provider_result.provider}")
        console.print(f"  Alt text: {'yes' if session.enable_alt else 'no'}")
        console.print(f"  Descriptions: {'yes' if session.enable_desc else 'no'}")
        console.print(f"  OCR: {'yes' if session.enable_ocr else 'no'}")
        console.print(f"  Screenshots: {'yes' if session.enable_screenshot else 'no'}")
    console.print()


def session_to_cli_args(session: InteractiveSession) -> list[str]:
    """Convert InteractiveSession to CLI arguments.

    This allows the interactive session to be executed by the main CLI.
    """
    args: list[str] = []

    if session.input_path:
        args.append(str(session.input_path))

    args.extend(["-o", str(session.output_dir)])

    if session.enable_llm:
        args.append("--llm")
        if session.enable_alt:
            args.append("--alt")
        if session.enable_desc:
            args.append("--desc")
        if session.enable_ocr:
            args.append("--ocr")
        if session.enable_screenshot:
            args.append("--screenshot")
    else:
        args.append("--no-llm")

    return args
