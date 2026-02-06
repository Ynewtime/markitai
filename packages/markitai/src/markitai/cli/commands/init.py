"""Init command for Markitai CLI.

One-stop setup: checks dependencies, detects LLM providers, generates config.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.hints import get_env_set_command
from markitai.config import ConfigManager

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
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Initialize global configuration (~/.markitai/config.json).",
)
def init(yes: bool, output_path: Path | None, use_global: bool) -> None:
    """Initialize Markitai configuration.

    Checks dependencies, detects LLM providers, and generates config.
    Use -y for quick setup without prompts.
    """
    target = _resolve_output_path(output_path, use_global)

    if yes:
        _quick_init(target)
    else:
        _wizard_init(target)


def _resolve_output_path(output_path: Path | None, use_global: bool) -> Path:
    """Resolve the target config file path."""
    if output_path is not None:
        if output_path.is_dir():
            return output_path / "markitai.json"
        return output_path
    if use_global:
        return ConfigManager.DEFAULT_USER_CONFIG_DIR / "config.json"
    return Path.cwd() / "markitai.json"


def _quick_init(target: Path) -> None:
    """Quick init - detect providers, generate config, no prompts."""
    if target.exists():
        ui.warning(f"Config already exists: {target}")
        console.print(f"  {ui.MARK_LINE} Use 'markitai config set' to modify")
        return
    providers = _detect_providers()
    config_data = _build_config(providers)
    _write_config(target, config_data)
    ui.success(f"Configuration created: {target}")


def _check_deps() -> list[tuple[str, str, bool]]:
    """Quick dependency check.

    Returns:
        List of (name, detail, available) tuples.
    """
    from markitai.cli.commands.doctor import get_install_hint

    results: list[tuple[str, str, bool]] = []

    # Playwright
    try:
        from markitai.fetch_playwright import (
            is_playwright_available,
            is_playwright_browser_installed,
        )

        pw_available = is_playwright_available()
        pw_browser = is_playwright_browser_installed() if pw_available else False

        if pw_available:
            if pw_browser:
                results.append(("Playwright", "Chromium installed", True))
            else:
                hint = get_install_hint("playwright")
                results.append(("Playwright", f"browser missing ({hint})", False))
        else:
            results.append(("Playwright", "not installed (uv add playwright)", False))
    except Exception:
        results.append(("Playwright", "not installed (uv add playwright)", False))

    # LibreOffice
    lo = shutil.which("soffice") or shutil.which("libreoffice")
    if lo:
        results.append(("LibreOffice", lo, True))
    else:
        hint = get_install_hint("libreoffice")
        results.append(("LibreOffice", f"not found ({hint})", False))

    # FFmpeg
    ff = shutil.which("ffmpeg")
    if ff:
        results.append(("FFmpeg", ff, True))
    else:
        results.append(("FFmpeg", "not found (optional)", False))

    # RapidOCR
    try:
        import rapidocr

        ver = getattr(rapidocr, "__version__", "?")
        results.append(("RapidOCR", f"v{ver}", True))
    except ImportError:
        results.append(("RapidOCR", "not installed (uv add rapidocr)", False))

    return results


def _wizard_init(target: Path) -> None:
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
        else:
            console.print(f"  [dim]{ui.MARK_INFO} {name}: not found[/dim]")
    console.print()

    if available_count == 0:
        env_cmd = get_env_set_command("GEMINI_API_KEY")
        ui.warning("No LLM providers detected")
        console.print()
        console.print("  Quick setup (choose one):")
        console.print()
        console.print("  [dim]•[/dim] Claude CLI:  [cyan]claude login[/cyan]")
        console.print("  [dim]•[/dim] Copilot CLI: [cyan]copilot auth login[/cyan]")
        console.print(f"  [dim]•[/dim] API Key:     [cyan]{env_cmd}[/cyan]")
        console.print()
        console.print("  Run 'markitai init' again after setup.")
        return

    # Check existing config before writing
    if target.exists():
        if not click.confirm(
            f"  Config already exists: {target}\n  Overwrite?", default=False
        ):
            return

    # Phase 3: Generate config
    config_data = _build_config(providers)
    _write_config(target, config_data)

    ui.summary(f"Configuration created: {target}")
    model_list = config_data.get("llm", {}).get("model_list", [])
    for m in model_list:
        model = m.get("litellm_params", {}).get("model", "")
        console.print(f"  {ui.MARK_LINE} {m.get('model_name', '')}: {model}")

    # Hint: run doctor for full diagnostics
    console.print()
    ui.step("Run 'markitai doctor' for full diagnostics")


def _detect_providers() -> list[tuple[str, bool]]:
    """Detect available LLM providers.

    Returns:
        List of (provider_name, is_available) tuples.
    """
    providers: list[tuple[str, bool]] = []

    claude_available = shutil.which("claude") is not None
    providers.append(("Claude CLI", claude_available))

    copilot_available = shutil.which("copilot") is not None
    providers.append(("GitHub Copilot CLI", copilot_available))

    return providers


def _build_config(
    providers: list[tuple[str, bool]] | None = None,
) -> dict:
    """Build config data with all detected providers."""
    model_list = []

    if providers:
        for name, available in providers:
            if not available:
                continue
            if "Claude" in name:
                model_list.append(
                    {
                        "model_name": "default",
                        "litellm_params": {"model": "claude-agent/haiku"},
                    }
                )
            elif "Copilot" in name:
                model_list.append(
                    {
                        "model_name": "copilot",
                        "litellm_params": {"model": "copilot/claude-sonnet-4.5"},
                    }
                )

    config_data: dict = {
        "output": {"dir": "./output"},
        "image": {"compress": True, "quality": 75},
    }

    if model_list:
        config_data["llm"] = {
            "enabled": False,
            "model_list": model_list,
        }

    return config_data


def _write_config(target: Path, config_data: dict) -> None:
    """Write config data to file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
        f.write("\n")
