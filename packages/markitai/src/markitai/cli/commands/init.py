"""Init command for Markitai CLI.

One-stop setup: checks dependencies, detects LLM providers, generates config.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rich_click as click

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.hints import get_env_set_command
from markitai.config import ConfigManager
from markitai.constants import PROVIDER_API_KEY_ENV, PROVIDER_DEFAULT_MODELS
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
    """Quick init - detect providers, generate or update config, no prompts."""
    # Generate .env template if not exists
    _ensure_env_template()

    with ui.StageList(transient=True) as stages:
        stages.advance("detect", "Detecting LLM providers...")
        providers = _detect_providers()

    if target.exists():
        # Non-destructive: only append newly detected providers.
        _apply_config_update(target, providers)
        return

    config_data = _build_config(providers)
    _write_config(target, config_data)
    _print_created_summary(target, config_data, providers)


def _read_config_dict(target: Path) -> dict | None:
    """Read an existing config file; None when unreadable or not a JSON object."""
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _provider_prefix(model: object) -> str:
    """Provider part of a model id (``deepseek`` in ``deepseek/deepseek-flash``)."""
    text = model if isinstance(model, str) else ""
    return text.split("/", 1)[0].lower() if "/" in text else text.lower()


def _merge_new_models(existing: dict, detected: dict) -> list[str]:
    """Append detected model_list entries missing from *existing* (in place).

    Non-destructive: only appends ``llm.model_list`` entries whose provider
    has no model configured yet. Existing model entries and all other settings
    are preserved untouched.

    Args:
        existing: Parsed existing config (mutated in place).
        detected: Freshly generated config from ``_build_config()``.

    Returns:
        List of added model ids (empty when already up to date).
    """
    detected_list = detected.get("llm", {}).get("model_list", [])

    llm_value = existing.get("llm")
    llm: dict = llm_value if isinstance(llm_value, dict) else {"enabled": False}
    model_list = llm.get("model_list")
    if not isinstance(model_list, list):
        model_list = []

    # By provider, not by model id: a config that names another model of a
    # provider (its own pick, or an earlier default such as
    # claude-agent/sonnet) already has that provider's route
    have = {
        _provider_prefix(entry.get("litellm_params", {}).get("model", ""))
        for entry in model_list
        if isinstance(entry, dict)
    }
    new_entries = [
        entry
        for entry in detected_list
        if _provider_prefix(entry["litellm_params"]["model"]) not in have
    ]
    if not new_entries:
        return []

    llm["model_list"] = model_list + new_entries
    existing["llm"] = llm
    return [entry["litellm_params"]["model"] for entry in new_entries]


def _apply_config_update(target: Path, providers: list[tuple[str, bool]]) -> None:
    """Non-destructively add newly detected providers to an existing config."""
    existing = _read_config_dict(target)
    if existing is None:
        ui.warning(f"Existing config could not be parsed: {target}")
        console.print(
            "  [dim]Fix it manually, or rerun [cyan]markitai init[/cyan] "
            "and choose Overwrite.[/dim]"
        )
        return

    added = _merge_new_models(existing, _build_config(providers))
    if not added:
        ui.summary(f"Config already up to date: {target}")
        console.print("  [dim]No newly detected providers to add.[/dim]")
        _print_skipped(providers)
        return

    _write_config(target, existing)
    ui.summary(f"Configuration updated: {target}")
    console.print("  [dim]Added newly detected providers:[/dim]")
    for model in added:
        ui.print_marked(model, mark=ui.MARK_LINE, console=console)
    _print_skipped(providers)


def _prompt_existing_config_action(target: Path) -> str:
    """Ask what to do with an existing config: update, overwrite, or keep."""
    console.print(f"  Config already exists: {target}")
    console.print()
    console.print(
        "  [1] Update:    add newly detected providers (keeps everything else)"
    )
    console.print("  [2] Overwrite: replace with a freshly generated config")
    console.print("  [3] Keep:      leave the file unchanged")
    console.print()
    choice = click.prompt("  Choose", type=click.IntRange(1, 3), default=3)
    return {1: "update", 2: "overwrite", 3: "keep"}[choice]


def _print_created_summary(
    target: Path,
    config_data: dict,
    providers: list[tuple[str, bool]] | None = None,
) -> None:
    """Print the created-config summary with detected models and next steps."""
    ui.summary(f"Configuration created: {target}")
    model_list = config_data.get("llm", {}).get("model_list", [])
    for m in model_list:
        model = m.get("litellm_params", {}).get("model", "")
        ui.print_marked(model, mark=ui.MARK_LINE, console=console)
    _print_skipped(providers)

    console.print()
    ui.section("Next steps", console=console)
    console.print(
        f"  [dim]{ui.MARK_INFO}[/dim] Convert a file:   [cyan]markitai <file> --llm[/cyan]"
    )
    console.print(
        f"  [dim]{ui.MARK_INFO}[/dim] Full diagnostics: [cyan]markitai doctor[/cyan]"
    )


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
            return ("Playwright", f"not installed ({_browser_hint()})", False)
    except Exception:
        return ("Playwright", f"not installed ({_browser_hint()})", False)


def _browser_hint() -> str:
    from markitai.utils.errors import extra_install_command

    return extra_install_command("browser")


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


def _check_rapidocr_dep() -> tuple[str, str, bool]:
    """Check the optional OCR backend's status.

    Returns:
        Tuple of (name, detail, available).
    """
    from markitai.ocr import OCR_INSTALL_HINT, is_ocr_available

    if not is_ocr_available():
        return ("RapidOCR (--ocr)", f"not installed — {OCR_INSTALL_HINT}", False)

    try:
        from importlib.metadata import version as get_version

        ver = get_version("rapidocr")
    except Exception:
        ver = "unknown"
    return ("RapidOCR (--ocr)", f"v{ver}", True)


def _check_deps() -> list[tuple[str, str, bool]]:
    """Quick dependency check (parallelized).

    Returns:
        List of (name, detail, available) tuples.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_check_playwright_dep),
            executor.submit(_check_libreoffice_dep),
            executor.submit(_check_rapidocr_dep),
        ]
        # Collect in submission order for deterministic output
        return [f.result() for f in futures]


def _wizard_init(target: Path, *, prompt_path: bool = False) -> None:
    """Interactive wizard: check deps, detect providers, generate config."""
    ui.title("Markitai Setup")

    # Run independent checks concurrently (dependency probes and provider
    # detection don't depend on each other); a transient stderr spinner
    # gives feedback until the first sections render.
    stages = ui.StageList(transient=True)
    stages.start()
    try:
        stages.advance("init", "Checking dependencies...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            deps_future = executor.submit(_check_deps)
            providers_future = executor.submit(_detect_providers)
            deps = deps_future.result()
            stages.update_text("Detecting LLM providers...")
            providers = providers_future.result()
    finally:
        stages.stop()

    # Phase 1: Check dependencies
    ui.section("Dependencies")
    for name, detail, available in deps:
        if available:
            ui.success(f"{name}: {detail}")
        else:
            ui.warning(f"{name}: {detail}")
    console.print()

    # Phase 2: Detected LLM providers
    ui.section("LLM Providers")
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
            env_cmd = get_env_set_command(PROVIDER_API_KEY_ENV["gemini"])
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
        action = _prompt_existing_config_action(target)
        if action == "keep":
            console.print(f"  [dim]Kept existing config: {target}[/dim]")
            return
        if action == "update":
            _apply_config_update(target, providers)
            return
        # "overwrite" falls through to generating a fresh config

    # Phase 4: Generate config
    config_data = _build_config(providers)

    _write_config(target, config_data)

    _print_created_summary(target, config_data, providers)


def _detect_providers() -> list[tuple[str, bool]]:
    """Adapt the shared provider detector to the init wizard's status rows."""
    from markitai.providers.detect import detect_all_providers

    detected = {result.provider for result in detect_all_providers()}
    provider_rows = (
        ("Claude CLI", "claude-agent"),
        ("GitHub Copilot CLI", "copilot"),
        ("ChatGPT", "chatgpt"),
        ("DeepSeek API", "deepseek"),
        ("Google Gemini API", "gemini"),
        ("OpenAI API", "openai"),
        ("Anthropic API", "anthropic"),
        ("OpenRouter API", "openrouter"),
    )
    return [(label, provider in detected) for label, provider in provider_rows]


# Display name -> provider id, in priority order; the model each one
# defaults to comes from the single PROVIDER_DEFAULT_MODELS table.
_PROVIDER_IDS = {
    "Claude": "claude-agent",
    "ChatGPT": "chatgpt",
    "Copilot": "copilot",
    "DeepSeek": "deepseek",
    "Gemini": "gemini",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
    "OpenRouter": "openrouter",
}


def _plan_providers(
    providers: list[tuple[str, bool]] | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Pick the providers a new config uses, and say why others are left out.

    Deduplication rules to avoid redundant Router entries (each would split
    the requests with a route to the same model):
    - Skip the Anthropic API when Claude Code is signed in to a subscription:
      both default to Claude Haiku, and the subscription has no per-token
      charge
    - Skip OpenRouter, whose default is Claude Haiku too, when the Anthropic
      API or Claude Code is available

    Returns:
        ``(provider ids in priority order, [(key variable, reason), ...])``.
    """
    available: set[str] = set()
    for name, ok in providers or []:
        if not ok:
            continue
        for key in _PROVIDER_IDS:
            if key in name:
                available.add(key)
                break

    skipped: list[tuple[str, str]] = []
    if "Claude" in available and "Anthropic" in available:
        available.discard("Anthropic")
        skipped.append(
            (
                PROVIDER_API_KEY_ENV["anthropic"],
                "Claude Code serves the same model (Claude Haiku) on your "
                "subscription, with no per-token charge",
            )
        )
    if "OpenRouter" in available and available & {"Claude", "Anthropic"}:
        available.discard("OpenRouter")
        route = "Claude Code" if "Claude" in available else "the Anthropic API key"
        skipped.append(
            (
                PROVIDER_API_KEY_ENV["openrouter"],
                f"its default model (Claude Haiku) is already served by {route}",
            )
        )
    chosen = [pid for key, pid in _PROVIDER_IDS.items() if key in available]
    return chosen, skipped


def _claude_code_key_only_note() -> str | None:
    """Why a Claude Code CLI on PATH was not offered, when the key is why.

    Signed in with nothing but ANTHROPIC_API_KEY, Claude Code bills that key
    per token through a CLI process per request; ``anthropic/`` uses the
    same key directly.
    """
    import os
    import shutil

    from markitai.providers.auth import claude_code_subscription_login

    if not shutil.which("claude") or not os.environ.get(
        PROVIDER_API_KEY_ENV["anthropic"]
    ):
        return None
    if claude_code_subscription_login() or any(
        os.environ.get(v)
        for v in (
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        )
    ):
        return None
    return (
        "Claude Code is not used: it is signed in only through "
        "ANTHROPIC_API_KEY, which anthropic/ models use directly "
        "(run 'claude login' to use a Claude subscription)"
    )


def _skipped_notes(providers: list[tuple[str, bool]] | None) -> list[str]:
    """One line per detected credential the config leaves out, with why."""
    _chosen, skipped = _plan_providers(providers)
    notes = [f"{env} is not used: {reason}" for env, reason in skipped]
    claude_note = _claude_code_key_only_note()
    if claude_note:
        notes.append(claude_note)
    return notes


def _print_skipped(providers: list[tuple[str, bool]] | None) -> None:
    for note in _skipped_notes(providers):
        ui.info(note)


def _build_config(
    providers: list[tuple[str, bool]] | None = None,
) -> dict:
    """Build config data with all detected providers (see _plan_providers)."""
    chosen, _skipped = _plan_providers(providers)
    model_list = [
        {
            "model_name": "default",
            "litellm_params": {"model": PROVIDER_DEFAULT_MODELS[provider_id]},
        }
        for provider_id in chosen
    ]

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
# Docs: https://markitai.dev/guide/configuration

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
    atomic_write_text(env_path, _ENV_TEMPLATE, private=True)
    return env_path


def _write_config(target: Path, config_data: dict) -> None:
    """Write config data to file atomically (owner-only: it may hold API keys)."""
    atomic_write_json(target, config_data, private=True)
