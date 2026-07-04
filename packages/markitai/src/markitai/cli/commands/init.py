"""Init command for Markitai CLI.

One-stop setup: checks dependencies, detects LLM providers, generates config.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rich_click as click

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.hints import get_env_set_command
from markitai.config import ConfigManager
from markitai.security import atomic_write_json, atomic_write_text

console = get_console()


@click.command("init")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Quick mode, generate default config without prompts.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for configuration file.",
)
@click.option(
    "--local",
    "use_local",
    is_flag=True,
    default=False,
    help="Initialize local project configuration (./markitai.json).",
)
def init(yes: bool, output_path: Path | None, use_local: bool) -> None:
    """Initialize Markitai configuration.

    Checks dependencies, detects LLM providers, and generates config.
    Use -y for quick setup without prompts.

    Examples:
        markitai init                   # Interactive setup wizard
        markitai init -y                # Quick setup, no prompts
        markitai init --local           # Write ./markitai.json (project)
        markitai init -o ./cfg.json     # Write to a custom path
    """
    target = _resolve_output_path(output_path, use_local)
    # When no explicit path flags given, let wizard ask the user
    explicit_path = output_path is not None or use_local

    if yes:
        _quick_init(target)
    else:
        _wizard_init(target, prompt_path=not explicit_path)


def _resolve_output_path(output_path: Path | None, use_local: bool) -> Path:
    """Resolve the target config file path."""
    if output_path is not None:
        if output_path.is_dir():
            return output_path / "markitai.json"
        return output_path
    if use_local:
        return Path.cwd() / "markitai.json"
    return ConfigManager.DEFAULT_USER_CONFIG_DIR / "config.json"


def _quick_init(target: Path) -> None:
    """Quick init - detect providers, generate config, no prompts."""
    # Generate .env template if not exists
    _ensure_env_template()

    if target.exists():
        ui.info(f"Config already exists: {target} (unchanged)")
        console.print(
            "  [dim]Run [cyan]markitai init[/cyan] without -y to reconfigure, "
            "or [cyan]markitai config edit[/cyan] to adjust settings.[/dim]"
        )
        return
    providers = _detect_providers()
    config_data = _build_config(providers)
    _write_config(target, config_data)
    _print_created_summary(target, config_data)


def _print_created_summary(target: Path, config_data: dict) -> None:
    """Print the created-config summary with detected models and next steps."""
    ui.summary(f"Configuration created: {target}")
    model_list = config_data.get("llm", {}).get("model_list", [])
    for m in model_list:
        model = m.get("litellm_params", {}).get("model", "")
        console.print(f"  {ui.MARK_LINE} {model}")

    console.print()
    console.print("  [bold]Next steps:[/bold]")
    console.print("  [dim]•[/dim] Convert a file:  [cyan]markitai <file> --llm[/cyan]")
    console.print("  [dim]•[/dim] Full diagnostics: [cyan]markitai doctor[/cyan]")


def _check_playwright_dep() -> tuple[str, str, bool]:
    """Check Playwright dependency status.

    Returns:
        Tuple of (name, detail, available).
    """
    from markitai.cli.commands.doctor import get_install_hint

    try:
        from markitai.fetch_playwright import (
            is_playwright_available,
            is_playwright_browser_installed,
        )

        pw_available = is_playwright_available()
        pw_browser = is_playwright_browser_installed() if pw_available else False

        if pw_available:
            if pw_browser:
                return ("Playwright", "Chromium installed", True)
            else:
                hint = get_install_hint("playwright")
                return ("Playwright", f"browser missing ({hint})", False)
        else:
            return ("Playwright", "not installed (uv add playwright)", False)
    except Exception:
        return ("Playwright", "not installed (uv add playwright)", False)


def _check_libreoffice_dep() -> tuple[str, str, bool]:
    """Check LibreOffice dependency status.

    Returns:
        Tuple of (name, detail, available).
    """
    from markitai.cli.commands.doctor import get_install_hint
    from markitai.utils.office import find_libreoffice

    lo = find_libreoffice()
    if lo:
        return ("LibreOffice", "installed", True)
    else:
        hint = get_install_hint("libreoffice")
        return ("LibreOffice", f"not found ({hint})", False)


def _check_ffmpeg_dep() -> tuple[str, str, bool]:
    """Check FFmpeg dependency status.

    Returns:
        Tuple of (name, detail, available).
    """
    ff = shutil.which("ffmpeg")
    if ff:
        return ("FFmpeg", "installed", True)
    else:
        return ("FFmpeg", "not found (optional)", False)


def _check_rapidocr_dep() -> tuple[str, str, bool]:
    """Check RapidOCR dependency status.

    Returns:
        Tuple of (name, detail, available).
    """
    try:
        from importlib.metadata import version as get_version

        import rapidocr  # noqa: F401

        try:
            ver = get_version("rapidocr")
        except Exception:
            ver = getattr(rapidocr, "__version__", "unknown")
        return ("RapidOCR", f"v{ver}", True)
    except ImportError:
        return ("RapidOCR", "not installed (uv add rapidocr)", False)


def _check_deps() -> list[tuple[str, str, bool]]:
    """Quick dependency check (parallelized).

    Returns:
        List of (name, detail, available) tuples.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_check_playwright_dep),
            executor.submit(_check_libreoffice_dep),
            executor.submit(_check_ffmpeg_dep),
            executor.submit(_check_rapidocr_dep),
        ]
        # Collect in submission order for deterministic output
        return [f.result() for f in futures]


def _wizard_init(target: Path, *, prompt_path: bool = False) -> None:
    """Interactive wizard: check deps, detect providers, generate config."""
    ui.title("Markitai Setup")

    # Phase 1: Check dependencies
    ui.section("Dependencies")
    deps = _check_deps()
    for name, detail, available in deps:
        if available:
            ui.success(f"{name}: {detail}")
        else:
            ui.warning(f"{name}: {detail}")
    console.print()

    # Phase 2: Detect LLM providers
    ui.section("LLM Providers")
    providers = _detect_providers()
    available_count = 0

    for name, available in providers:
        if available:
            ui.success(f"{name}")
            available_count += 1
    console.print()

    if available_count == 0:
        env_template = _ensure_env_template()
        ui.warning("No LLM providers detected")
        console.print()
        console.print("  Quick setup (choose one):")
        console.print()
        console.print("  [dim]•[/dim] Claude CLI:  [cyan]claude login[/cyan]")
        console.print("  [dim]•[/dim] Copilot CLI: [cyan]copilot login[/cyan]")
        if env_template:
            console.print(
                f"  [dim]•[/dim] API Key:     edit [cyan]{env_template}[/cyan]"
            )
        else:
            env_cmd = get_env_set_command("GEMINI_API_KEY")
            console.print(f"  [dim]•[/dim] API Key:     [cyan]{env_cmd}[/cyan]")
        console.print()

    # Phase 3: Choose config location
    if prompt_path:
        global_path = ConfigManager.DEFAULT_USER_CONFIG_DIR / "config.json"
        local_path = Path.cwd() / "markitai.json"
        ui.section("Config Location")
        console.print(f"  [1] Global: {global_path}")
        console.print(f"  [2] Local:  {local_path}")
        console.print()
        choice = click.prompt("  Save to", type=click.IntRange(1, 2), default=1)
        target = global_path if choice == 1 else local_path

    # Check existing config before writing
    if target.exists():
        if not click.confirm(
            f"  Config already exists: {target}\n  Overwrite?", default=False
        ):
            console.print(f"  [dim]Kept existing config: {target}[/dim]")
            return

    # Phase 4: Generate config
    config_data = _build_config(providers)

    _write_config(target, config_data)

    _print_created_summary(target, config_data)


def _detect_providers() -> list[tuple[str, bool]]:
    """Detect available LLM providers.

    Checks local CLI providers, OAuth-authenticated providers,
    and API key environment variables.

    Returns:
        List of (provider_name, is_available) tuples.
    """
    import os

    providers: list[tuple[str, bool]] = []

    # Local CLI providers
    claude_available = shutil.which("claude") is not None
    providers.append(("Claude CLI", claude_available))

    copilot_available = shutil.which("copilot") is not None
    providers.append(("GitHub Copilot CLI", copilot_available))

    # OAuth-authenticated providers (no CLI binary needed)
    from markitai.cli.providers_detect import (
        _check_chatgpt_auth,
        _check_gemini_cli_auth,
    )

    providers.append(("ChatGPT", _check_chatgpt_auth()))
    providers.append(("Gemini CLI", _check_gemini_cli_auth()))

    # API key providers
    api_keys = [
        ("DeepSeek API", "DEEPSEEK_API_KEY"),
        ("Google Gemini API", "GEMINI_API_KEY"),
        ("OpenAI API", "OPENAI_API_KEY"),
        ("Anthropic API", "ANTHROPIC_API_KEY"),
        ("OpenRouter API", "OPENROUTER_API_KEY"),
    ]
    for name, env_var in api_keys:
        providers.append((name, bool(os.environ.get(env_var))))

    return providers


def _build_config(
    providers: list[tuple[str, bool]] | None = None,
) -> dict:
    """Build config data with all detected providers.

    Deduplication rules to avoid redundant Router entries:
    - Skip Anthropic API when Claude CLI is available (same model family)
    - Skip OpenRouter Gemini when direct Gemini API is available
    - Skip direct Gemini API when Gemini CLI is available (same backend)
    """
    model_list = []

    # Map provider display names to model configs (order = priority)
    provider_models = {
        "Claude": ("default", "claude-agent/sonnet"),
        "ChatGPT": ("default", "chatgpt/gpt-5.4"),
        "Copilot": ("default", "copilot/claude-sonnet-4.6"),
        "Gemini CLI": ("default", "gemini-cli/gemini-3.1-pro-preview"),
        "DeepSeek": ("default", "deepseek/deepseek-chat"),
        "Gemini": ("default", "gemini/gemini-3.1-flash-lite-preview"),
        "OpenAI": ("default", "openai/gpt-5.4"),
        "Anthropic": ("default", "anthropic/claude-sonnet-4-6"),
        "OpenRouter": ("default", "openrouter/google/gemini-3.1-flash-lite-preview"),
    }

    if providers:
        available_keys: set[str] = set()
        for name, available in providers:
            if not available:
                continue
            for key in provider_models:
                if key in name:
                    available_keys.add(key)
                    break

        # Deduplicate: skip redundant provider routes
        if "Claude" in available_keys and "Anthropic" in available_keys:
            available_keys.discard("Anthropic")
        if "Gemini CLI" in available_keys and "Gemini" in available_keys:
            available_keys.discard("Gemini")
        if "Gemini" in available_keys and "OpenRouter" in available_keys:
            available_keys.discard("OpenRouter")

        # Build model_list preserving provider_models order
        for key, (model_name, model_id) in provider_models.items():
            if key in available_keys:
                model_list.append(
                    {
                        "model_name": model_name,
                        "litellm_params": {"model": model_id},
                    }
                )

    config_data: dict = {
        "output": {"dir": "./output"},
    }

    if model_list:
        config_data["llm"] = {
            "enabled": False,
            "model_list": model_list,
        }

    return config_data


_ENV_TEMPLATE = """\
# Markitai — LLM API Keys
# Uncomment and fill in at least one key to enable LLM features.
# Docs: https://markitai.ynewtime.com/guide/configuration

# DEEPSEEK_API_KEY=
# GEMINI_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# OPENROUTER_API_KEY=

# Optional: Jina Reader API (URL conversion)
# JINA_API_KEY=

# Optional: Cloudflare (cloud rendering)
# CLOUDFLARE_API_TOKEN=
# CLOUDFLARE_ACCOUNT_ID=
"""


def _ensure_env_template() -> Path | None:
    """Create ~/.markitai/.env template if it doesn't exist.

    Returns:
        Path to the .env file if created, None if already exists.
    """
    env_path = ConfigManager.DEFAULT_USER_CONFIG_DIR / ".env"
    if env_path.exists():
        return None
    atomic_write_text(env_path, _ENV_TEMPLATE)
    return env_path


def _write_config(target: Path, config_data: dict) -> None:
    """Write config data to file atomically."""
    atomic_write_json(target, config_data)
