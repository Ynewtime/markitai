"""Dependency checking CLI command.

This module provides the check-deps command for verifying
all optional dependencies and their status.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from markitai.config import ConfigManager

console = Console()


@click.command("check-deps")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
def check_deps(as_json: bool) -> None:
    """Check all optional dependencies and their status.

    This command helps diagnose setup issues by verifying:
    - agent-browser (for dynamic URL fetching)
    - LibreOffice (for Office document conversion)
    - RapidOCR (for scanned document processing)
    - LLM API configuration (for content enhancement)
    """
    from markitai.fetch import verify_agent_browser_ready

    manager = ConfigManager()
    cfg = manager.load()

    results: dict[str, dict[str, Any]] = {}

    # 1. Check agent-browser
    is_ready, message = verify_agent_browser_ready(use_cache=False)
    results["agent-browser"] = {
        "name": "agent-browser",
        "description": "Browser automation for dynamic URLs",
        "status": "ok" if is_ready else "missing",
        "message": message,
        "install_hint": "pnpm add -g agent-browser && agent-browser install",
    }

    # 2. Check LibreOffice
    soffice_path = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice_path:
        try:
            proc = subprocess.run(
                [soffice_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = (
                proc.stdout.strip().split("\n")[0]
                if proc.returncode == 0
                else "unknown"
            )
            results["libreoffice"] = {
                "name": "LibreOffice",
                "description": "Office document conversion (doc, docx, xls, xlsx, ppt, pptx)",
                "status": "ok",
                "message": f"Found at {soffice_path} ({version})",
                "install_hint": "",
            }
        except Exception as e:
            results["libreoffice"] = {
                "name": "LibreOffice",
                "description": "Office document conversion (doc, docx, xls, xlsx, ppt, pptx)",
                "status": "error",
                "message": f"Found but failed to run: {e}",
                "install_hint": "Reinstall LibreOffice",
            }
    else:
        results["libreoffice"] = {
            "name": "LibreOffice",
            "description": "Office document conversion (doc, docx, xls, xlsx, ppt, pptx)",
            "status": "missing",
            "message": "soffice/libreoffice command not found",
            "install_hint": "apt install libreoffice (Linux) / brew install libreoffice (macOS)",
        }

    # 3. Check RapidOCR (Python OCR library with built-in models)
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
            "install_hint": "pip install rapidocr (included in markitai dependencies)",
        }

    # 4. Check LLM API configuration (check model_list for configured models)
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

    # 4a. Check local provider SDKs if configured
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
                        "install_hint": "Install Claude Code CLI: pnpm add -g @anthropic-ai/claude-code",
                    }
            else:
                results["claude-agent-sdk"] = {
                    "name": "Claude Agent SDK",
                    "description": "Claude Code CLI integration",
                    "status": "missing",
                    "message": "claude-agent-sdk not installed",
                    "install_hint": "pip install claude-agent-sdk",
                }
        except Exception as e:
            results["claude-agent-sdk"] = {
                "name": "Claude Agent SDK",
                "description": "Claude Code CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": "pip install claude-agent-sdk",
            }

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
                        "install_hint": "Install Copilot CLI: pnpm add -g @github/copilot",
                    }
            else:
                results["copilot-sdk"] = {
                    "name": "GitHub Copilot SDK",
                    "description": "GitHub Copilot CLI integration",
                    "status": "missing",
                    "message": "github-copilot-sdk not installed",
                    "install_hint": "pip install github-copilot-sdk",
                }
        except Exception as e:
            results["copilot-sdk"] = {
                "name": "GitHub Copilot SDK",
                "description": "GitHub Copilot CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": "pip install github-copilot-sdk",
            }

    # 5. Check vision model configuration (auto-detect from litellm or config override)
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

    # Rich table output
    table = Table(title="Dependency Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Description")
    table.add_column("Details")

    status_icons = {
        "ok": "[green]✓[/green]",
        "warning": "[yellow]⚠[/yellow]",
        "missing": "[red]✗[/red]",
        "error": "[red]![/red]",
    }

    for _key, info in results.items():
        status_icon = status_icons.get(info["status"], "?")
        table.add_row(
            info["name"],
            status_icon,
            info["description"],
            info["message"],
        )

    console.print(table)
    console.print()

    # Show install hints for missing/error items
    hints = [
        (info["name"], info["install_hint"])
        for info in results.values()
        if info["status"] in ("missing", "error") and info["install_hint"]
    ]

    if hints:
        hint_text = "\n".join([f"  • {name}: {hint}" for name, hint in hints])
        console.print(
            Panel(
                f"[yellow]To fix missing dependencies:[/yellow]\n{hint_text}",
                title="Installation Hints",
                border_style="yellow",
            )
        )
    else:
        console.print("[green]All dependencies are properly configured![/green]")
