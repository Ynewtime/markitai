"""Doctor CLI command for system health checking.

This module provides the doctor command for verifying the core dependency,
optional capabilities, authentication status, and overall system health.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from typing import Any

import rich_click as click

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
        "darwin": "python -m playwright install chromium",
        "linux": "python -m playwright install chromium  # system libraries may also require: python -m playwright install-deps chromium",
        "win32": "python -m playwright install chromium",
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


# Components for which --fix has a safe, scoped repair path. Keys not in this
# mapping are reported with manual hints and never passed to an installer.
FIXABLE_COMPONENTS: dict[str, bool] = {"playwright": True}

_PLAYWRIGHT_LAUNCH_SMOKE = (
    "from playwright.sync_api import sync_playwright\n"
    "with sync_playwright() as p:\n"
    "    browser = p.chromium.launch(headless=True)\n"
    "    browser.close()\n"
)


def _compact_runtime_error(detail: str, *, limit: int = 1200) -> str:
    """Keep actionable launch diagnostics without dumping a full traceback."""
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    critical_markers = (
        "fatal",
        "missing",
        "not found",
        "permission denied",
        "timed out",
    )
    critical = [
        line for line in lines if any(m in line.lower() for m in critical_markers)
    ]
    errors = [line for line in lines if "error" in line.lower()]
    selected = critical[-2:] or errors[-2:] or lines[-2:]
    compact = " | ".join(selected) or "unknown launch failure"
    if len(compact) > limit:
        compact = f"…{compact[-limit:]}"
    return compact


def _playwright_package_install_hint() -> str:
    """Return install commands that preserve tool-environment isolation."""
    return (
        "uv tool install 'markitai[browser]' --force\n"
        "# Or: pipx install 'markitai[browser]' --force"
    )


def _smoke_test_playwright_browser() -> tuple[bool, str]:
    """Launch and close Chromium in an isolated subprocess with a timeout."""
    try:
        with TemporaryDirectory(prefix="markitai-doctor-smoke-") as workdir:
            result = subprocess.run(
                [sys.executable, "-c", _PLAYWRIGHT_LAUNCH_SMOKE],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=workdir,
            )
    except subprocess.TimeoutExpired:
        return False, "headless launch timed out after 30 seconds"
    except Exception as e:
        return False, f"headless launch error: {e}"

    if result.returncode == 0:
        return True, ""

    detail = _compact_runtime_error(result.stderr or result.stdout)
    if detail == "unknown launch failure":
        detail = f"exit code {result.returncode}"
    if sys.platform.startswith("linux"):
        detail += (
            "; install system libraries with "
            f"'{sys.executable} -m playwright install-deps chromium'"
        )
    return False, f"headless launch failed: {detail}"


def _has_errors(results: dict[str, dict[str, Any]]) -> bool:
    """Check whether any check result has error or missing status.

    This covers all categories: required deps, optional capabilities, LLM,
    auth, and vision.

    Args:
        results: The full results dict from doctor checks.

    Returns:
        True if any result has 'error' or 'missing' status.
    """
    return any(info.get("status") in ("error", "missing") for info in results.values())


def _install_component(component: str, *, package_missing: bool = False) -> bool:
    """Attempt to install a missing component.

    Args:
        component: Component name to install.
        package_missing: For Playwright, whether the Python package itself is
            missing rather than only its Chromium browser.

    Returns:
        True if installation succeeded.
    """
    console = get_console()
    hint = get_install_hint(component)

    # Only auto-install safe components
    safe_components = {"playwright"}

    if component not in safe_components:
        console.print(f"[yellow]{t('doctor.manual_install')}[/yellow]")
        console.print(f"  {hint}")
        return False

    if component == "playwright":
        try:
            # Installing a Python package into the caller's current project is
            # unsafe. Tell tool users how to replace their isolated install.
            if package_missing:
                console.print(
                    f"[red]\u2717[/red] {t('doctor.playwright_package_manual')}"
                )
                console.print(_playwright_package_install_hint())
                return False

            # Use the same interpreter that imports Markitai/Playwright, and
            # isolate cwd so no project metadata can be discovered or changed.
            console.print(
                f"[yellow]{t('doctor.fix_installing', component=component)}[/yellow]"
            )
            with TemporaryDirectory(prefix="markitai-doctor-") as workdir:
                result = subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=workdir,
                )
            if result.returncode == 0:
                return True
            else:
                console.print(
                    f"[red]\u2717[/red] "
                    f"{t('doctor.fix_failed', detail=result.stderr.strip())}"
                )
                return False
        except Exception as e:
            console.print(f"[red]\u2717[/red] {t('doctor.fix_error', detail=str(e))}")
            return False

    return False


from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.i18n import t
from markitai.config import ConfigManager
from markitai.providers.auth import AuthManager, get_auth_resolution_hint

console = get_console()


def _check_chatgpt_auth() -> dict[str, str]:
    """Check ChatGPT authentication status.

    Returns:
        Result dict with status, message, install_hint
    """
    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("chatgpt"))
        if status.authenticated:
            return {
                "name": "ChatGPT Auth",
                "description": "ChatGPT OAuth authentication status",
                "status": "ok",
                "message": "Authenticated",
                "install_hint": "",
            }
        else:
            return {
                "name": "ChatGPT Auth",
                "description": "ChatGPT OAuth authentication status",
                "status": "error",
                "message": status.error or "Not authenticated",
                "install_hint": get_auth_resolution_hint("chatgpt"),
            }
    except Exception as e:
        return {
            "name": "ChatGPT Auth",
            "description": "ChatGPT OAuth authentication status",
            "status": "error",
            "message": f"Failed to check auth: {e}",
            "install_hint": get_auth_resolution_hint("chatgpt"),
        }


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


def _check_playwright(*, check_runtime: bool = True) -> dict[str, Any]:
    """Check Playwright installation status.

    Returns:
        Result dict with name, description, status, message, install_hint.
    """
    from markitai.fetch_playwright import (
        is_playwright_available,
        is_playwright_browser_installed,
    )

    if is_playwright_available():
        if is_playwright_browser_installed(use_cache=False):
            if check_runtime:
                runtime_ok, runtime_detail = _smoke_test_playwright_browser()
                if not runtime_ok:
                    return {
                        "name": "Playwright",
                        "description": "Browser automation for dynamic URLs",
                        "status": "warning",
                        "message": f"Chromium installed but unusable: {runtime_detail}",
                        "install_hint": get_install_hint("playwright"),
                    }
            return {
                "name": "Playwright",
                "description": "Browser automation for dynamic URLs",
                "status": "ok",
                "message": "Chromium installed",
                "install_hint": "",
            }
        else:
            return {
                "name": "Playwright",
                "description": "Browser automation for dynamic URLs",
                "status": "warning",
                "message": "Playwright installed but browser not found",
                "install_hint": get_install_hint("playwright"),
            }
    else:
        return {
            "name": "Playwright",
            "description": "Browser automation for dynamic URLs",
            "status": "missing",
            "message": "Playwright not installed",
            "install_hint": _playwright_package_install_hint(),
        }


def _check_libreoffice(macos_fallback: bool = True) -> dict[str, Any]:
    """Check LibreOffice installation status.

    Args:
        macos_fallback: Whether configured MS Office fallback may be used.

    Returns:
        Result dict with name, description, status, message, install_hint.
    """
    from markitai.utils.office import find_libreoffice

    soffice_path = find_libreoffice()

    if soffice_path:
        # LibreOffice found - report "installed" (path available in --json output)
        # Running soffice --version can hang on Windows in non-interactive mode
        return {
            "name": "LibreOffice",
            "description": "Legacy Office conversion (doc, ppt) and PPTX slide rendering",
            "status": "ok",
            "message": "installed",
            "path": soffice_path,
            "install_hint": "",
        }
    else:
        # macOS: installed MS Office apps take over legacy conversion and
        # PPTX PDF export via AppleScript (see utils/office_mac.py)
        if sys.platform == "darwin" and macos_fallback:
            from markitai.utils import office_mac

            office_apps = [
                app.removeprefix("Microsoft ")
                for app in ("Microsoft Word", "Microsoft PowerPoint")
                if office_mac.find_ms_office_app(app)
            ]
            if office_apps:
                return {
                    "name": "LibreOffice",
                    "description": "Legacy Office conversion (doc, ppt) and PPTX slide rendering",
                    "status": "warning",
                    "message": (
                        "not found; MS Office fallback available "
                        f"({', '.join(office_apps)} via AppleScript, "
                        "needs one-time Automation permission)"
                    ),
                    "install_hint": get_install_hint("libreoffice"),
                }
        return {
            "name": "LibreOffice",
            "description": "Legacy Office conversion (doc, ppt) and PPTX slide rendering",
            "status": "missing",
            "message": "soffice/libreoffice command not found",
            "install_hint": get_install_hint("libreoffice"),
        }


def _check_ffmpeg() -> dict[str, Any]:
    """Check FFmpeg installation status.

    Returns:
        Result dict with name, description, status, message, install_hint.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Extract version: "ffmpeg version 8.0.1 ..." → "v8.0.1"
            version = "unknown"
            if proc.returncode == 0:
                parts = proc.stdout.strip().split()
                if len(parts) >= 3 and parts[0] == "ffmpeg":
                    version = f"v{parts[2]}"
                else:
                    version = parts[0] if parts else "unknown"
            return {
                "name": "FFmpeg",
                "description": "Audio/video file processing (mp3, mp4, wav, etc.)",
                "status": "ok",
                "message": version,
                "path": ffmpeg_path,
                "install_hint": "",
            }
        except Exception as e:
            return {
                "name": "FFmpeg",
                "description": "Audio/video file processing (mp3, mp4, wav, etc.)",
                "status": "error",
                "message": f"Found but failed to run: {e}",
                "install_hint": "Reinstall FFmpeg",
            }
    else:
        return {
            "name": "FFmpeg",
            "description": "Audio/video file processing (mp3, mp4, wav, etc.)",
            "status": "missing",
            "message": "ffmpeg command not found",
            "install_hint": get_install_hint("ffmpeg"),
        }


def _check_rapidocr(cfg: Any) -> dict[str, Any]:
    """Check RapidOCR installation status.

    Args:
        cfg: Configuration object with OCR language settings.

    Returns:
        Result dict with name, description, status, message, install_hint.
    """
    try:
        import importlib.util
        from importlib.metadata import version as get_version

        # Probe via metadata only — importing rapidocr pulls in cv2, whose
        # 119MB dylib pays a one-time ~25s dyld signature validation on a
        # fresh install (the "first doctor run is slow" root cause).
        # sys.modules check first: tests inject a mock module there
        if (
            "rapidocr" not in sys.modules
            and importlib.util.find_spec("rapidocr") is None
        ):
            raise ImportError("rapidocr not installed")

        try:
            rapidocr_version = get_version("rapidocr")
        except Exception:
            rapidocr_version = "unknown"

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
            return {
                "name": "RapidOCR",
                "description": "OCR for scanned documents (built-in models)",
                "status": "ok",
                "message": f"v{rapidocr_version}, lang: {configured_lang} ({lang_display.get(configured_lang.lower(), configured_lang)})",
                "install_hint": "",
            }
        else:
            return {
                "name": "RapidOCR",
                "description": "OCR for scanned documents (built-in models)",
                "status": "warning",
                "message": f"v{rapidocr_version}, unknown lang '{configured_lang}' (supported: {', '.join(sorted(supported_langs))})",
                "install_hint": "Set ocr.lang to one of: zh, en, ja, ko, ar, th, latin",
            }
    except ImportError:
        return {
            "name": "RapidOCR",
            "description": "OCR for scanned documents (built-in models)",
            "status": "missing",
            "message": "RapidOCR not installed",
            "install_hint": "uv add rapidocr (included in markitai dependencies)",
        }


def _is_active_model(model_config: Any) -> bool:
    """Return whether a model config is active for routing-dependent checks."""
    litellm_params = getattr(model_config, "litellm_params", None)
    weight = getattr(litellm_params, "weight", 1)
    return not isinstance(weight, int | float) or weight > 0


def _doctor_impl(as_json: bool, fix: bool = False) -> None:
    """Implementation of the doctor command.

    Args:
        as_json: Output results as JSON.
        fix: Attempt to install missing components.
    """
    from markitai.fetch_playwright import clear_browser_cache

    manager = ConfigManager()
    cfg = manager.load()

    results: dict[str, dict[str, Any]] = {}

    # Clear Playwright browser cache before parallel checks
    clear_browser_cache()

    # Run independent system tool checks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_playwright = executor.submit(_check_playwright)
        future_libreoffice = executor.submit(
            _check_libreoffice, cfg.office.macos_fallback
        )
        future_ffmpeg = executor.submit(_check_ffmpeg)
        future_rapidocr = executor.submit(_check_rapidocr, cfg)

    # Collect results in deterministic order
    results["playwright"] = future_playwright.result()
    results["libreoffice"] = future_libreoffice.result()
    results["ffmpeg"] = future_ffmpeg.result()
    results["rapidocr"] = future_rapidocr.result()

    # 5. Check LLM API configuration (check model_list for configured models)
    configured_models = cfg.llm.model_list if cfg.llm.model_list else []
    active_models = [m for m in configured_models if _is_active_model(m)]
    local_prefixes = ("claude-agent/", "copilot/", "chatgpt/")
    active_api_models = [
        m
        for m in active_models
        if not any(m.litellm_params.model.startswith(p) for p in local_prefixes)
    ]
    missing_llm_env_vars: set[str] = set()
    for model in active_api_models:
        for field in ("api_key", "api_base"):
            value = getattr(model.litellm_params, field, None)
            if isinstance(value, str) and value.startswith("env:"):
                env_name = value[4:]
                if env_name and os.environ.get(env_name) is None:
                    missing_llm_env_vars.add(env_name)

    if configured_models:
        api_providers = {
            (
                m.litellm_params.model.split("/")[0]
                if "/" in m.litellm_params.model
                else "openai"
            )
            for m in active_api_models
        }
        if not active_models:
            status = "warning"
            message = "Models are configured, but all have weight 0 (disabled)"
            install_hint = "Set weight > 0 on at least one model to enable LLM use"
        elif missing_llm_env_vars:
            missing = ", ".join(sorted(missing_llm_env_vars))
            status = "error"
            message = (
                f"Missing environment variable(s) used by active models: {missing}"
            )
            install_hint = f"Set the missing environment variable(s): {missing}"
        else:
            status = "ok"
            if api_providers:
                providers_str = ", ".join(sorted(api_providers))
                message = (
                    f"{len(active_models)} active model(s), "
                    f"API provider(s): {providers_str}"
                )
            else:
                message = f"{len(active_models)} active model(s) configured"
            install_hint = ""

        results["llm-api"] = {
            "name": "LLM API",
            "description": "Content enhancement and image analysis",
            "status": status,
            "message": message,
            "install_hint": install_hint,
        }
    else:
        # Point at the actually-loaded config file (matches the header line),
        # not a hardcoded generic filename.
        if manager.config_path is not None:
            llm_hint = f"Configure llm.model_list in {manager.config_path}"
        else:
            llm_hint = (
                "Configure llm.model_list (no config file found; run "
                "'markitai init' to create ~/.markitai/config.json "
                "or ./markitai.json)"
            )
        results["llm-api"] = {
            "name": "LLM API",
            "description": "Content enhancement and image analysis",
            "status": "missing",
            "message": "No models configured in llm.model_list",
            "install_hint": llm_hint,
        }

    # 5a. Check local provider SDKs if configured
    uses_claude_agent = any(
        m.litellm_params.model.startswith("claude-agent/")
        for m in configured_models
        if _is_active_model(m)
    )
    uses_copilot = any(
        m.litellm_params.model.startswith("copilot/")
        for m in configured_models
        if _is_active_model(m)
    )
    uses_chatgpt = any(
        m.litellm_params.model.startswith("chatgpt/")
        for m in configured_models
        if _is_active_model(m)
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
                        "message": "SDK + CLI installed",
                        "path": claude_cli,
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
                    "install_hint": "curl -fsSL https://markitai.dev/setup.sh | sh  # or: uv tool install 'markitai[claude-agent]' --upgrade",
                }
        except Exception as e:
            results["claude-agent-sdk"] = {
                "name": "Claude Agent SDK",
                "description": "Claude Code CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": "curl -fsSL https://markitai.dev/setup.sh | sh  # or: uv tool install 'markitai[claude-agent]' --upgrade",
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
                        "message": "SDK + CLI installed",
                        "path": copilot_cli,
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
                    "install_hint": "curl -fsSL https://markitai.dev/setup.sh | sh  # or: uv tool install 'markitai[copilot]' --upgrade",
                }
        except Exception as e:
            results["copilot-sdk"] = {
                "name": "GitHub Copilot SDK",
                "description": "GitHub Copilot CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": "curl -fsSL https://markitai.dev/setup.sh | sh  # or: uv tool install 'markitai[copilot]' --upgrade",
            }

        # 5c. Check Copilot authentication status
        results["copilot-auth"] = _check_copilot_auth()

    if uses_chatgpt:
        results["chatgpt-auth"] = _check_chatgpt_auth()

    # 6. Check vision model configuration (auto-detect from litellm or config override)
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

        # Auto-detect from litellm (deferred import: pulls litellm, ~0.5s)
        from markitai.llm import get_model_info_cached

        info = get_model_info_cached(model_config.litellm_params.model)
        return info.get("supports_vision", False)

    vision_models = [m for m in active_models if is_vision_model(m)]
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
            "install_hint": "Use vision-capable models like gemini-*, gpt-5.4, claude-*",
        }

    # Define dependency groups
    # RapidOCR ships as a core dependency. The other tools unlock specific
    # input types, so their absence should not make the base installation
    # unhealthy.
    required_deps = ["rapidocr"]
    required_checks = set(required_deps)
    fetch_strategy = getattr(getattr(cfg, "fetch", None), "strategy", "auto")
    screenshot_enabled = (
        getattr(getattr(cfg, "screenshot", None), "enabled", False) is True
    )
    uses_playwright = fetch_strategy == "playwright" or screenshot_enabled
    if uses_playwright:
        required_checks.add("playwright")
    if active_api_models:
        required_checks.add("llm-api")
    if uses_claude_agent:
        required_checks.update({"claude-agent-sdk", "claude-agent-auth"})
    if uses_copilot:
        required_checks.update({"copilot-sdk", "copilot-auth"})
    if uses_chatgpt:
        required_checks.add("chatgpt-auth")

    def _is_blocking_failure(key: str, info: dict[str, Any]) -> bool:
        if key not in required_checks:
            return False
        status = info.get("status")
        if status in {"missing", "error"}:
            return True
        # A configured capability/provider warning means the requested
        # workflow is not ready. RapidOCR's unknown-language warning is a
        # non-blocking configuration warning because the package is present.
        return status == "warning" and key != "rapidocr"

    def _required_failed_count() -> int:
        return sum(
            1
            for key in required_checks
            if key in results and _is_blocking_failure(key, results[key])
        )

    # Output results
    if as_json:
        # Use click.echo for raw JSON (avoid Rich formatting which breaks JSON)
        click.echo(json.dumps(results, indent=2))
        if _required_failed_count():
            raise SystemExit(1)
        return

    # Unified UI output
    ui.title(t("doctor.title"))
    optional_deps = ["playwright", "libreoffice", "ffmpeg"]
    llm_keys = ["llm-api", "vision-model", "claude-agent-sdk", "copilot-sdk"]
    auth_keys = ["claude-agent-auth", "copilot-auth", "chatgpt-auth"]

    # Config source (which config file was actually loaded)
    if manager.config_path is not None:
        ui.info(t("doctor.config_source", path=str(manager.config_path)))
    else:
        ui.info(t("doctor.config_source", path=t("doctor.config_defaults")))

    def render_item(info: dict[str, Any], *, blocking: bool = False) -> None:
        """Render one check result inline: '<glyph> Name: message'."""
        line = f"{info['name']}: {info['message']}"
        if info["status"] == "ok":
            ui.success(line)
        elif blocking:
            ui.error(line)
        else:
            ui.warning(line)

    def render_section(header: str, keys: list[str]) -> None:
        """Render a section preceded by exactly one blank line."""
        console.print()
        ui.section(header)
        for key in keys:
            if key in results:
                render_item(
                    results[key],
                    blocking=_is_blocking_failure(key, results[key]),
                )

    # Required dependencies
    render_section(t("doctor.required"), required_deps)

    # Optional capabilities
    if any(k in results for k in optional_deps):
        render_section(t("doctor.optional"), optional_deps)

    # LLM status
    if any(k in results for k in llm_keys):
        render_section("LLM", llm_keys)

    # Authentication status
    if any(k in results for k in auth_keys):
        render_section(t("doctor.auth"), auth_keys)

    # Attempt to fix missing components if --fix flag is set
    # Only fixable components (defined in FIXABLE_COMPONENTS) are attempted.
    # Non-fixable items (auth, vision-model, llm-api) are skipped.
    repair_failed = False
    if fix and not as_json:
        fixable = [
            key
            for key, info in results.items()
            if info["status"] in ("missing", "warning") and key in FIXABLE_COMPONENTS
        ]
        if fixable:
            console.print()
            console.print(f"[bold]{t('doctor.fix_attempting')}[/bold]")
            for component in fixable:
                # For playwright, detect whether the package itself is missing
                pkg_missing = (
                    component == "playwright"
                    and results[component].get("status") == "missing"
                )
                if not _install_component(component, package_missing=pkg_missing):
                    repair_failed = True
                    continue

                # A successful installer exit only means the command ran. The
                # capability itself must be checked again before --fix can
                # report success.
                if component == "playwright":
                    verification = _check_playwright(check_runtime=False)
                    results[component] = verification
                    if verification.get("status") != "ok":
                        ui.error(
                            t(
                                "doctor.fix_verification_failed",
                                component=verification["name"],
                                detail=verification["message"],
                            )
                        )
                        repair_failed = True
                    else:
                        runtime_ok, runtime_detail = _smoke_test_playwright_browser()
                        if not runtime_ok:
                            verification["status"] = "warning"
                            verification["message"] = (
                                f"Chromium installed but unusable: {runtime_detail}"
                            )
                            verification["install_hint"] = get_install_hint(
                                "playwright"
                            )
                            ui.error(
                                t(
                                    "doctor.fix_verification_failed",
                                    component=verification["name"],
                                    detail=runtime_detail,
                                )
                            )
                            repair_failed = True
                        else:
                            ui.success(
                                t(
                                    "doctor.fix_success",
                                    component=verification["name"],
                                )
                            )

    # Render hints and the final outcome only after any requested repair has
    # been rechecked, so one invocation cannot end with a stale pre-fix status.
    hints = [
        (info["name"], info["install_hint"])
        for info in results.values()
        if info["status"] in ("missing", "error", "warning")
        and info.get("install_hint")
    ]
    if hints:
        console.print()
        console.print(f"[yellow]{t('doctor.fix_hint')}[/yellow]")
        for name, hint in hints:
            console.print(f"  [dim]\u2022[/dim] {name}: {hint}")

    passed = sum(
        1 for key in required_checks if results.get(key, {}).get("status") == "ok"
    )
    required_failed = _required_failed_count()
    degraded_checks = sum(
        1
        for key, info in results.items()
        if info.get("status") != "ok" and not _is_blocking_failure(key, info)
    )
    if required_failed:
        ui.summary(
            t(
                "doctor.summary_failed",
                failed=required_failed,
                passed=passed,
                degraded=degraded_checks,
            ),
            ok=False,
        )
    elif repair_failed:
        ui.summary(
            t("doctor.summary_repair_failed", degraded=degraded_checks), ok=False
        )
    elif degraded_checks:
        ui.summary(
            t("doctor.summary", passed=passed, degraded=degraded_checks), ok=None
        )
    else:
        ui.summary(t("doctor.all_good"))

    # Scripts rely on the exit code: --fix must not turn an unresolved core
    # dependency into a successful health check.
    if required_failed or repair_failed:
        raise SystemExit(1)


def suggest_extras() -> list[str]:
    """Return recommended pip extras for installation.

    Always includes all extras whose dependencies are standard PyPI
    packages.  Only conditionally includes extras that depend on SDKs
    which may not be publicly available on PyPI.

    The result is a *stable* list (alphabetical) that the install
    scripts can feed directly into
    ``uv tool install markitai[browser,extra-fetch,...]``.

    This is the **single source of truth** — install scripts should call
    ``markitai doctor --suggest-extras`` instead of reimplementing
    detection logic in shell.
    """
    extras: set[str] = set()

    # --- Always-include extras (pure Python packages from PyPI) ---
    extras.add("browser")  # playwright
    extras.add("extra-fetch")  # curl-cffi
    extras.add("kreuzberg")  # kreuzberg
    extras.add("svg")  # cairosvg (pip install succeeds; runtime detects missing lib)
    extras.add("heif")  # pillow-heif (HEIC/HEIF/AVIF input decoding)

    # --- Conditional extras (SDK may not be on PyPI) ---
    # claude-agent — requires claude-agent-sdk
    if shutil.which("claude"):
        extras.add("claude-agent")

    # copilot — requires github-copilot-sdk
    if shutil.which("copilot"):
        extras.add("copilot")

    return sorted(extras)


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
    help="Attempt safe automatic repairs (currently Chromium only).",
)
@click.option(
    "--suggest-extras",
    "suggest",
    is_flag=True,
    help="Output recommended pip extras for the current environment (machine-readable).",
)
def doctor(as_json: bool, fix: bool, suggest: bool) -> None:
    """Check system health, dependencies, and authentication status.

    This command helps diagnose setup issues by verifying:

        Core requirement:
        - RapidOCR (for scanned document processing)

        Optional capabilities:
        - Playwright (for dynamic URL fetching)
        - LibreOffice (for Office document conversion)
        - FFmpeg (for audio/video processing)
        - LLM API configuration (for content enhancement)
        - Auth status for local providers (Claude, Copilot, ChatGPT)

    Exits non-zero when RapidOCR is missing; a configured Playwright workflow
    cannot launch; an active API model references a missing environment
    variable; an actively configured local provider cannot load/authenticate;
    or a requested automatic repair fails, so it can be used in scripts and CI.

    Examples:
        markitai doctor                 # Full health report
        markitai doctor --fix           # Try to install what's missing
        markitai doctor --json          # Machine-readable report
    """
    if as_json and fix:
        raise click.UsageError("--json and --fix cannot be used together")
    if suggest:
        click.echo(",".join(suggest_extras()))
        return
    _doctor_impl(as_json, fix=fix)
