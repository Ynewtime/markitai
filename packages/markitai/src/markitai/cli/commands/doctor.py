"""Doctor CLI command for system health checking.

This module provides the doctor command for verifying all optional dependencies,
authentication status, and overall system health.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from typing import Any

import click

# Cross-platform installation hints
INSTALL_HINTS: dict[str, dict[str, str]] = {
    "libreoffice": {
        "darwin": "brew install --cask libreoffice",
        "linux": "sudo apt install libreoffice  # Ubuntu/Debian\nsudo dnf install libreoffice  # Fedora\nsudo pacman -S libreoffice-fresh  # Arch",
        "win32": "winget install LibreOffice.LibreOffice",
    },
    "ffmpeg": {
        "darwin": "brew install ffmpeg",
        "linux": "sudo apt install ffmpeg  # Ubuntu/Debian\nsudo dnf install ffmpeg  # Fedora\nsudo pacman -S ffmpeg  # Arch",
        "win32": "winget install FFmpeg.FFmpeg\n# Or: scoop install ffmpeg\n# Or: choco install ffmpeg",
    },
    "playwright": {
        "darwin": "uv run playwright install chromium",
        "linux": "uv run playwright install chromium && uv run playwright install-deps chromium",
        "win32": "uv run playwright install chromium",
    },
    "claude-cli": {
        "darwin": "curl -fsSL https://claude.ai/install.sh | bash",
        "linux": "curl -fsSL https://claude.ai/install.sh | bash",
        "win32": "irm https://claude.ai/install.ps1 | iex",
    },
    "copilot-cli": {
        "darwin": "curl -fsSL https://gh.io/copilot-install | bash",
        "linux": "curl -fsSL https://gh.io/copilot-install | bash",
        "win32": "winget install GitHub.Copilot",
    },
}


def get_install_hint(component: str, platform: str | None = None) -> str:
    """Get platform-specific installation hint for a component.

    Args:
        component: Component name (e.g., "libreoffice", "ffmpeg")
        platform: Target platform. If None, uses current sys.platform.

    Returns:
        Installation command(s) for the platform.
    """
    if platform is None:
        platform = sys.platform

    # Normalize platform
    if platform.startswith("linux"):
        platform = "linux"

    hints = INSTALL_HINTS.get(component, {})
    return hints.get(
        platform, hints.get("linux", f"Install {component} using your package manager")
    )


def _install_component(component: str) -> bool:
    """Attempt to install a missing component.

    Args:
        component: Component name to install.

    Returns:
        True if installation succeeded.
    """
    console = get_console()
    hint = get_install_hint(component)

    console.print(f"[yellow]Installing {component}...[/yellow]")

    # Only auto-install safe components
    safe_components = {"playwright"}

    if component not in safe_components:
        console.print("[yellow]Please install manually:[/yellow]")
        console.print(f"  {hint}")
        return False

    if component == "playwright":
        try:
            # Use uv to run playwright install
            result = subprocess.run(
                ["uv", "run", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                console.print(f"[green]✓[/green] {component} installed")
                return True
            else:
                console.print(f"[red]✗[/red] Failed: {result.stderr}")
                return False
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            return False

    return False


from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.i18n import t
from markitai.config import ConfigManager
from markitai.providers.auth import AuthManager, get_auth_resolution_hint

console = get_console()


def _check_copilot_auth() -> dict[str, str]:
    """Check Copilot authentication status.

    Returns:
        Result dict with status, message, install_hint
    """
    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("copilot"))
        if status.authenticated:
            user_info = f" ({status.user})" if status.user else ""
            return {
                "name": "Copilot Auth",
                "description": "GitHub Copilot authentication status",
                "status": "ok",
                "message": f"Authenticated{user_info}",
                "install_hint": "",
            }
        else:
            return {
                "name": "Copilot Auth",
                "description": "GitHub Copilot authentication status",
                "status": "error",
                "message": status.error or "Not authenticated",
                "install_hint": get_auth_resolution_hint("copilot"),
            }
    except Exception as e:
        return {
            "name": "Copilot Auth",
            "description": "GitHub Copilot authentication status",
            "status": "error",
            "message": f"Failed to check auth: {e}",
            "install_hint": get_auth_resolution_hint("copilot"),
        }


def _check_claude_auth() -> dict[str, str]:
    """Check Claude Agent authentication status.

    Returns:
        Result dict with status, message, install_hint
    """
    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("claude-agent"))
        if status.authenticated:
            return {
                "name": "Claude Agent Auth",
                "description": "Claude Code CLI authentication status",
                "status": "ok",
                "message": "Authenticated (claude doctor passed)",
                "install_hint": "",
            }
        else:
            return {
                "name": "Claude Agent Auth",
                "description": "Claude Code CLI authentication status",
                "status": "error",
                "message": status.error or "Not authenticated",
                "install_hint": get_auth_resolution_hint("claude-agent"),
            }
    except Exception as e:
        return {
            "name": "Claude Agent Auth",
            "description": "Claude Code CLI authentication status",
            "status": "error",
            "message": f"Failed to check auth: {e}",
            "install_hint": get_auth_resolution_hint("claude-agent"),
        }


def _doctor_impl(as_json: bool, fix: bool = False) -> None:
    """Implementation of the doctor command.

    Args:
        as_json: Output results as JSON.
        fix: Attempt to install missing components.
    """
    from markitai.fetch_playwright import (
        clear_browser_cache,
        is_playwright_available,
        is_playwright_browser_installed,
    )

    manager = ConfigManager()
    cfg = manager.load()

    results: dict[str, dict[str, Any]] = {}

    # 1. Check Playwright
    clear_browser_cache()  # Clear cache for fresh check
    if is_playwright_available():
        if is_playwright_browser_installed(use_cache=False):
            results["playwright"] = {
                "name": "Playwright",
                "description": "Browser automation for dynamic URLs",
                "status": "ok",
                "message": "Playwright and Chromium browser installed",
                "install_hint": "",
            }
        else:
            results["playwright"] = {
                "name": "Playwright",
                "description": "Browser automation for dynamic URLs",
                "status": "warning",
                "message": "Playwright installed but browser not found",
                "install_hint": get_install_hint("playwright"),
            }
    else:
        results["playwright"] = {
            "name": "Playwright",
            "description": "Browser automation for dynamic URLs",
            "status": "missing",
            "message": "Playwright not installed",
            "install_hint": f"uv add playwright && {get_install_hint('playwright')}",
        }

    # 2. Check LibreOffice
    soffice_path = shutil.which("soffice") or shutil.which("libreoffice")

    # Check common installation paths if not in PATH
    if not soffice_path:
        import os
        import sys

        common_paths: list[str] = []
        if sys.platform == "win32":
            # Windows: Check Program Files
            for prog_dir in [
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            ]:
                common_paths.extend(
                    [
                        os.path.join(prog_dir, "LibreOffice", "program", "soffice.exe"),
                        os.path.join(
                            prog_dir, "LibreOffice 7", "program", "soffice.exe"
                        ),
                        os.path.join(
                            prog_dir, "LibreOffice 24", "program", "soffice.exe"
                        ),
                    ]
                )
        elif sys.platform == "darwin":
            # macOS: Check Applications
            common_paths.append("/Applications/LibreOffice.app/Contents/MacOS/soffice")

        for path in common_paths:
            if os.path.isfile(path):
                soffice_path = path
                break

    if soffice_path:
        # LibreOffice found - just report the path without running --version
        # Running soffice --version can hang on Windows in non-interactive mode
        results["libreoffice"] = {
            "name": "LibreOffice",
            "description": "Office document conversion (doc, docx, xls, xlsx, ppt, pptx)",
            "status": "ok",
            "message": f"Found at {soffice_path}",
            "install_hint": "",
        }
    else:
        results["libreoffice"] = {
            "name": "LibreOffice",
            "description": "Office document conversion (doc, docx, xls, xlsx, ppt, pptx)",
            "status": "missing",
            "message": "soffice/libreoffice command not found",
            "install_hint": get_install_hint("libreoffice"),
        }

    # 3. Check FFmpeg (for audio/video processing)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = (
                proc.stdout.strip().split("\n")[0]
                if proc.returncode == 0
                else "unknown"
            )
            results["ffmpeg"] = {
                "name": "FFmpeg",
                "description": "Audio/video file processing (mp3, mp4, wav, etc.)",
                "status": "ok",
                "message": f"Found at {ffmpeg_path} ({version})",
                "install_hint": "",
            }
        except Exception as e:
            results["ffmpeg"] = {
                "name": "FFmpeg",
                "description": "Audio/video file processing (mp3, mp4, wav, etc.)",
                "status": "error",
                "message": f"Found but failed to run: {e}",
                "install_hint": "Reinstall FFmpeg",
            }
    else:
        results["ffmpeg"] = {
            "name": "FFmpeg",
            "description": "Audio/video file processing (mp3, mp4, wav, etc.)",
            "status": "missing",
            "message": "ffmpeg command not found",
            "install_hint": get_install_hint("ffmpeg"),
        }

    # 4. Check RapidOCR (Python OCR library with built-in models)
    try:
        from importlib.metadata import version as get_version

        import rapidocr

        try:
            rapidocr_version = get_version("rapidocr")
        except Exception:
            rapidocr_version = getattr(rapidocr, "__version__", "unknown")

        # Get configured language
        configured_lang = cfg.ocr.lang if cfg.ocr else "en"
        # RapidOCR supported languages
        supported_langs = {
            "zh",
            "ch",
            "en",
            "ja",
            "japan",
            "ko",
            "korean",
            "ar",
            "arabic",
            "th",
            "latin",
        }
        lang_display = {
            "zh": "Chinese",
            "ch": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "japan": "Japanese",
            "ko": "Korean",
            "korean": "Korean",
            "ar": "Arabic",
            "arabic": "Arabic",
            "th": "Thai",
            "latin": "Latin",
        }

        if configured_lang.lower() in supported_langs:
            results["rapidocr"] = {
                "name": "RapidOCR",
                "description": "OCR for scanned documents (built-in models)",
                "status": "ok",
                "message": f"v{rapidocr_version}, lang: {configured_lang} ({lang_display.get(configured_lang.lower(), configured_lang)})",
                "install_hint": "",
            }
        else:
            results["rapidocr"] = {
                "name": "RapidOCR",
                "description": "OCR for scanned documents (built-in models)",
                "status": "warning",
                "message": f"v{rapidocr_version}, unknown lang '{configured_lang}' (supported: {', '.join(sorted(supported_langs))})",
                "install_hint": "Set ocr.lang to one of: zh, en, ja, ko, ar, th, latin",
            }
    except ImportError:
        results["rapidocr"] = {
            "name": "RapidOCR",
            "description": "OCR for scanned documents (built-in models)",
            "status": "missing",
            "message": "RapidOCR not installed",
            "install_hint": "uv add rapidocr (included in markitai dependencies)",
        }

    # 5. Check LLM API configuration (check model_list for configured models)
    configured_models = cfg.llm.model_list if cfg.llm.model_list else []
    if configured_models:
        # Find first model with api_key to determine provider
        first_model = configured_models[0].litellm_params.model
        provider = first_model.split("/")[0] if "/" in first_model else "openai"
        results["llm-api"] = {
            "name": f"LLM API ({provider})",
            "description": "Content enhancement and image analysis",
            "status": "ok",
            "message": f"{len(configured_models)} model(s) configured",
            "install_hint": "",
        }
    else:
        results["llm-api"] = {
            "name": "LLM API",
            "description": "Content enhancement and image analysis",
            "status": "missing",
            "message": "No models configured in llm.model_list",
            "install_hint": "Configure llm.model_list in markitai.json",
        }

    # 5a. Check local provider SDKs if configured
    uses_claude_agent = any(
        m.litellm_params.model.startswith("claude-agent/") for m in configured_models
    )
    uses_copilot = any(
        m.litellm_params.model.startswith("copilot/") for m in configured_models
    )

    if uses_claude_agent:
        try:
            import importlib.util

            if importlib.util.find_spec("claude_agent_sdk") is not None:
                # Check if CLI is available
                claude_cli = shutil.which("claude")
                if claude_cli:
                    results["claude-agent-sdk"] = {
                        "name": "Claude Agent SDK",
                        "description": "Claude Code CLI integration",
                        "status": "ok",
                        "message": f"SDK installed, CLI at {claude_cli}",
                        "install_hint": "",
                    }
                else:
                    results["claude-agent-sdk"] = {
                        "name": "Claude Agent SDK",
                        "description": "Claude Code CLI integration",
                        "status": "warning",
                        "message": "SDK installed but 'claude' CLI not found",
                        "install_hint": f"Install Claude Code CLI: {get_install_hint('claude-cli')}",
                    }
            else:
                results["claude-agent-sdk"] = {
                    "name": "Claude Agent SDK",
                    "description": "Claude Code CLI integration",
                    "status": "missing",
                    "message": "claude-agent-sdk not installed",
                    "install_hint": f"uv add claude-agent-sdk && {get_install_hint('claude-cli')}",
                }
        except Exception as e:
            results["claude-agent-sdk"] = {
                "name": "Claude Agent SDK",
                "description": "Claude Code CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": "uv add claude-agent-sdk",
            }

        # 5b. Check Claude Agent authentication status
        results["claude-agent-auth"] = _check_claude_auth()

    if uses_copilot:
        try:
            import importlib.util

            if importlib.util.find_spec("copilot") is not None:
                # Check if CLI is available
                copilot_cli = shutil.which("copilot")
                if copilot_cli:
                    results["copilot-sdk"] = {
                        "name": "GitHub Copilot SDK",
                        "description": "GitHub Copilot CLI integration",
                        "status": "ok",
                        "message": f"SDK installed, CLI at {copilot_cli}",
                        "install_hint": "",
                    }
                else:
                    results["copilot-sdk"] = {
                        "name": "GitHub Copilot SDK",
                        "description": "GitHub Copilot CLI integration",
                        "status": "warning",
                        "message": "SDK installed but 'copilot' CLI not found",
                        "install_hint": f"Install Copilot CLI: {get_install_hint('copilot-cli')}",
                    }
            else:
                results["copilot-sdk"] = {
                    "name": "GitHub Copilot SDK",
                    "description": "GitHub Copilot CLI integration",
                    "status": "missing",
                    "message": "github-copilot-sdk not installed",
                    "install_hint": f"uv add github-copilot-sdk && {get_install_hint('copilot-cli')}",
                }
        except Exception as e:
            results["copilot-sdk"] = {
                "name": "GitHub Copilot SDK",
                "description": "GitHub Copilot CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": "uv add github-copilot-sdk",
            }

        # 5c. Check Copilot authentication status
        results["copilot-auth"] = _check_copilot_auth()

    # 6. Check vision model configuration (auto-detect from litellm or config override)
    from markitai.llm import get_model_info_cached

    def is_vision_model(model_config: Any) -> bool:
        """Check if model supports vision (config override, local providers, or auto-detect)."""
        # Config override takes priority
        if (
            model_config.model_info
            and model_config.model_info.supports_vision is not None
        ):
            return model_config.model_info.supports_vision

        # Local providers (claude-agent/, copilot/) always support vision
        from markitai.providers import is_local_provider_model

        if is_local_provider_model(model_config.litellm_params.model):
            return True

        # Auto-detect from litellm
        info = get_model_info_cached(model_config.litellm_params.model)
        return info.get("supports_vision", False)

    vision_models = [m for m in configured_models if is_vision_model(m)]
    if vision_models:
        vision_model_names = [m.litellm_params.model for m in vision_models]
        display_names = ", ".join(vision_model_names[:2])
        if len(vision_model_names) > 2:
            display_names += f" (+{len(vision_model_names) - 2} more)"
        results["vision-model"] = {
            "name": "Vision Model",
            "description": "Image analysis (alt text, descriptions)",
            "status": "ok",
            "message": f"{len(vision_models)} detected: {display_names}",
            "models": vision_model_names,  # Structured list for JSON output
            "install_hint": "",
        }
    else:
        results["vision-model"] = {
            "name": "Vision Model",
            "description": "Image analysis (alt text, descriptions)",
            "status": "warning",
            "message": "No vision model detected (auto-detect or set model_info.supports_vision)",
            "models": [],  # Empty list for JSON output
            "install_hint": "Use vision-capable models like gemini-*, gpt-4o, claude-*",
        }

    # Output results
    if as_json:
        # Use click.echo for raw JSON (avoid Rich formatting which breaks JSON)
        click.echo(json.dumps(results, indent=2))
        return

    # Unified UI output
    ui.title(t("doctor.title"))

    # Define dependency groups
    required_deps = ["playwright", "libreoffice", "rapidocr"]
    optional_deps = ["ffmpeg"]
    llm_keys = ["llm-api", "vision-model", "claude-agent-sdk", "copilot-sdk"]
    auth_keys = ["claude-agent-auth", "copilot-auth"]

    # Required dependencies
    ui.section(t("doctor.required"))
    passed = 0
    for key in required_deps:
        if key not in results:
            continue
        info = results[key]
        if info["status"] == "ok":
            ui.success(f"{info['name']}: {info['message']}")
            passed += 1
        elif info["status"] == "warning":
            ui.warning(info["name"], detail=info["message"])
        else:
            ui.error(info["name"], detail=info["message"])
    console.print()

    # Optional dependencies
    optional_missing = 0
    has_optional = any(k in results for k in optional_deps)
    if has_optional:
        ui.section(t("doctor.optional"))
        for key in optional_deps:
            if key not in results:
                continue
            info = results[key]
            if info["status"] == "ok":
                ui.success(info["name"])
            else:
                ui.warning(
                    f"{info['name']}（{t('missing')}）",
                    detail=info.get("install_hint", ""),
                )
                optional_missing += 1
        console.print()

    # LLM status
    has_llm = any(k in results for k in llm_keys)
    if has_llm:
        ui.section("LLM")
        for key in llm_keys:
            if key not in results:
                continue
            info = results[key]
            if info["status"] == "ok":
                ui.success(f"{info['name']}: {info['message']}")
            elif info["status"] == "warning":
                ui.warning(f"{info['name']}: {info['message']}")
            else:
                ui.error(f"{info['name']}: {info['message']}")
        console.print()

    # Authentication status
    has_auth = any(k in results for k in auth_keys)
    if has_auth:
        ui.section(t("doctor.auth"))
        for key in auth_keys:
            if key not in results:
                continue
            info = results[key]
            if info["status"] == "ok":
                ui.success(f"{info['name']}: {info['message']}")
            else:
                ui.error(f"{info['name']}: {info['message']}")
        console.print()

    # Summary
    all_required_ok = all(
        results.get(k, {}).get("status") == "ok" for k in required_deps if k in results
    )
    if optional_missing == 0 and all_required_ok:
        ui.summary(t("doctor.all_good"))
    else:
        ui.summary(t("doctor.summary", passed=passed, optional=optional_missing))

    # Installation hints (simplified format)
    hints = [
        (info["name"], info["install_hint"])
        for info in results.values()
        if info["status"] in ("missing", "error") and info.get("install_hint")
    ]

    if hints:
        console.print()
        console.print(f"[yellow]{t('doctor.fix_hint')}[/yellow]")
        for name, hint in hints:
            console.print(f"  [dim]\u2022[/dim] {name}: {hint}")

    # Attempt to fix missing components if --fix flag is set
    if fix and not as_json:
        missing = [
            key
            for key, info in results.items()
            if info["status"] in ("missing", "warning")
        ]
        if missing:
            console.print()
            console.print("[bold]Attempting to fix missing components...[/bold]")
            for component in missing:
                _install_component(component)


@click.command("doctor")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
@click.option(
    "--fix",
    is_flag=True,
    help="Attempt to install missing components.",
)
def doctor(as_json: bool, fix: bool) -> None:
    """Check system health, dependencies, and authentication status.

    This command helps diagnose setup issues by verifying:
    - Playwright (for dynamic URL fetching)
    - LibreOffice (for Office document conversion)
    - RapidOCR (for scanned document processing)
    - LLM API configuration (for content enhancement)
    - Authentication status for local providers (Claude Agent, Copilot)
    """
    _doctor_impl(as_json, fix=fix)
