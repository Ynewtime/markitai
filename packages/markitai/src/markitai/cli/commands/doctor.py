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

# Cross-platform installation hints for system tools. LibreOffice on Linux
# and Playwright's browser are resolved per machine (distro, interpreter):
# see get_install_commands.
INSTALL_HINTS: dict[str, dict[str, str]] = {
    "libreoffice": {
        "darwin": "brew install --cask libreoffice",
        "win32": "winget install TheDocumentFoundation.LibreOffice",
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


def _normalize_platform(platform: str | None) -> str:
    platform = sys.platform if platform is None else platform
    return "linux" if platform.startswith("linux") else platform


def get_install_commands(
    component: str,
    platform: str | None = None,
    *,
    os_release: str | None = None,
) -> list[tuple[str, str]]:
    """Commands that install a component on this machine.

    Args:
        component: Component name (e.g., "libreoffice", "playwright").
        platform: Target platform. If None, uses current sys.platform.
        os_release: /etc/os-release text (tests); read from disk if None.

    Returns:
        ``(label, command)`` pairs. One pair with an empty label when the
        command is known for this machine; several labelled alternatives
        when it depends on a Linux distribution that was not recognised.
    """
    platform = _normalize_platform(platform)
    if component == "playwright":
        return [("", playwright_install_command(platform))]
    if component == "libreoffice" and platform == "linux":
        from markitai.utils.office import libreoffice_linux_install_commands

        return libreoffice_linux_install_commands(os_release)
    hints = INSTALL_HINTS.get(component, {})
    hint = hints.get(platform, hints.get("linux"))
    if hint is None:
        return [("", f"Install {component} using your package manager")]
    return [("", hint)]


def get_install_hint(
    component: str, platform: str | None = None, *, os_release: str | None = None
) -> str:
    """Get the installation hint for a component as one line.

    Args:
        component: Component name (e.g., "libreoffice", "playwright")
        platform: Target platform. If None, uses current sys.platform.
        os_release: /etc/os-release text (tests); read from disk if None.

    Returns:
        The command, or labelled alternatives joined with "; " (for --json
        and one-line messages; the terminal report lists them one per line).
    """
    return "; ".join(
        f"{command} ({label})" if label else command
        for label, command in get_install_commands(
            component, platform, os_release=os_release
        )
    )


def _install_fields(component: str) -> dict[str, Any]:
    """``install_hint`` (one line) plus ``install_commands`` (one per line).

    The terminal report prints each command on its own line so it can be
    copied; ``install_hint`` keeps the one-line form --json always had.
    """
    commands = get_install_commands(component)
    return {
        "install_hint": get_install_hint(component),
        "install_commands": [
            f"{command}  # {label}" if label else command for label, command in commands
        ],
    }


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _them(items: list[str]) -> str:
    return "it" if len(items) == 1 else "them"


def _render_hint(info: dict[str, Any]) -> None:
    """One check's fix: commands on their own lines, prose with a bullet.

    Hints were printed as one raw string, so a multi-line one (an install
    command per distribution, the login instructions) lost its bullet and
    indentation from the second line on.
    """
    commands = info.get("install_commands")
    if commands:
        ui.print_marked(f"{info['name']}:", mark=ui.MARK_INFO)
        for command in commands:
            ui.print_command(command)
        return
    lines = [line for line in str(info["install_hint"]).splitlines() if line.strip()]
    ui.print_marked(f"{info['name']}: {lines[0]}", mark=ui.MARK_INFO)
    for line in lines[1:]:
        if line.startswith("  "):
            ui.print_command(line.strip(), indent=6)
        else:
            ui.print_marked(line, mark="", indent=4)


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
    return extra_install_command("browser")


# A first launch right after installing can take long: on Windows,
# Defender scans the new chrome.exe (the Windows machine: over 30 s, then 0.7 s on
# every later launch). One retry after the first timeout catches that.
_SMOKE_TIMEOUTS_S = (60, 30)
# Detail returned when every attempt timed out: slow, not broken
SMOKE_SLOW = "slow"


def _smoke_test_playwright_browser() -> tuple[bool, str]:
    """Launch and close Chromium in an isolated subprocess with a timeout.

    Returns:
        ``(True, "")`` on success; ``(False, SMOKE_SLOW)`` when every attempt
        timed out; ``(False, detail)`` when the launch failed.
    """
    result = None
    for timeout in _SMOKE_TIMEOUTS_S:
        try:
            with TemporaryDirectory(prefix="markitai-doctor-smoke-") as workdir:
                result = subprocess.run(
                    [sys.executable, "-c", _PLAYWRIGHT_LAUNCH_SMOKE],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                )
            break
        except subprocess.TimeoutExpired:
            continue
        except Exception as e:
            return False, f"headless launch error: {e}"
    if result is None:
        return False, SMOKE_SLOW

    if result.returncode == 0:
        return True, ""

    detail = _compact_runtime_error(result.stderr or result.stdout)
    if detail == "unknown launch failure":
        detail = f"exit code {result.returncode}"
    if sys.platform.startswith("linux"):
        detail += (
            "; install system libraries with "
            f"'{markitai_python_command()} -m playwright install-deps chromium'"
        )
    return False, f"headless launch failed: {detail}"


def _playwright_runtime_problem(detail: str) -> dict[str, Any]:
    """Doctor fields for a Chromium that is installed but did not launch.

    The terminal report and --json carry the same message.
    """
    if detail == SMOKE_SLOW:
        seconds = sum(_SMOKE_TIMEOUTS_S)
        return {
            "message": (
                f"Chromium is installed but slow to start (no headless launch "
                f"within {seconds} s); a first launch after installing can be "
                "that slow while antivirus scans it, so run doctor again"
            ),
            "install_hint": "Run 'markitai doctor' again in a minute",
            "install_commands": ["markitai doctor"],
            "slow_start": True,
            "summary_name": "Playwright (slow to start)",
            # Installed and launching, just slowly the first time: a
            # configured Playwright workflow is not blocked by it
            "blocking": False,
        }
    return {
        "message": f"Chromium is installed but cannot launch: {detail}",
        **_install_fields("playwright"),
    }


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

    # Only auto-install safe components
    safe_components = {"playwright"}

    if component not in safe_components:
        ui.warning(t("doctor.manual_install"), console=console)
        for label, command in get_install_commands(component):
            ui.print_command(
                f"{command}  # {label}" if label else command, console=console
            )
        return False

    if component == "playwright":
        try:
            # Installing a Python package into the caller's current project is
            # unsafe. Tell tool users how to replace their isolated install.
            if package_missing:
                ui.error(t("doctor.playwright_package_manual"), console=console)
                # Printed as plain text: rich would read the `[browser]` in
                # an extras name as a style tag and drop it, leaving the user
                # a command that reinstalls exactly what they already have.
                ui.print_command(_playwright_package_install_hint(), console=console)
                return False

            # Use the same interpreter that imports Markitai/Playwright, and
            # isolate cwd so no project metadata can be discovered or changed.
            ui.print_marked(
                t("doctor.fix_installing", component=component),
                mark=ui.MARK_ARROW,
                console=console,
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
                ui.error(
                    t("doctor.fix_failed", detail=result.stderr.strip()),
                    console=console,
                )
                return False
        except Exception as e:
            ui.error(t("doctor.fix_error", detail=str(e)), console=console)
            return False

    return False


from markitai.cli import ui
from markitai.cli.console import get_console
from markitai.cli.framework import root_config_options
from markitai.cli.hints import get_env_set_command
from markitai.cli.i18n import t
from markitai.config import ConfigManager
from markitai.constants import (
    PROVIDER_API_KEY_ENV,
    PROVIDER_API_KEY_ENV_NAMES,
    PROVIDER_DEFAULT_MODELS,
)
from markitai.providers.auth import AuthManager, get_auth_resolution_hint
from markitai.providers.discovery import provider_label
from markitai.utils.errors import extra_install_command
from markitai.utils.guidance import markitai_python_command, playwright_install_command

console = get_console()


def _check_provider_auth(
    provider: str,
    name: str,
    description: str,
    *,
    ok_message: str = "Authenticated",
    show_user: bool = False,
) -> dict[str, str]:
    """Check one provider's authentication status.

    Args:
        provider: Provider key understood by ``AuthManager.check_auth``.
        name: Display name for the check row.
        description: Description for the check row.
        ok_message: Message used when authenticated.
        show_user: Append the authenticated user in parentheses when known.

    Returns:
        Result dict with status, message, install_hint
    """
    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth(provider))
        if status.authenticated:
            user_info = f" ({status.user})" if show_user and status.user else ""
            return {
                "name": name,
                "description": description,
                "status": "ok",
                "message": f"{ok_message}{user_info}",
                "install_hint": "",
            }
        message = status.error or "Not authenticated"
    except Exception as e:
        message = f"Failed to check auth: {e}"
    return {
        "name": name,
        "description": description,
        "status": "error",
        "message": message,
        "install_hint": get_auth_resolution_hint(provider),
    }


def _check_chatgpt_auth() -> dict[str, str]:
    """Check ChatGPT authentication status."""
    return _check_provider_auth(
        "chatgpt",
        "ChatGPT Auth",
        "ChatGPT OAuth authentication status",
    )


def _check_copilot_auth() -> dict[str, str]:
    """Check Copilot authentication status."""
    return _check_provider_auth(
        "copilot",
        "Copilot Auth",
        "GitHub Copilot authentication status",
        show_user=True,
    )


def _check_claude_auth() -> dict[str, str]:
    """Check Claude Agent authentication status, and what it bills."""
    result = _check_provider_auth(
        "claude-agent",
        "Claude Agent Auth",
        "Claude Code CLI authentication status",
    )
    if result["status"] != "ok":
        return result
    try:
        status = asyncio.run(AuthManager().check_auth("claude-agent"))
    except Exception:
        return result
    details = status.details or {}
    if details.get("source") == "anthropic-api-key":
        # Works, billed per token to the key: same as anthropic/ models,
        # plus a CLI process per request
        result.update(
            status="warning",
            message=(
                "not signed in; Claude Code would use ANTHROPIC_API_KEY, "
                "billed per token"
            ),
            install_hint=(
                "Run 'claude login' to use a Claude subscription, or use "
                "anthropic/claude-haiku-5-5 (the same key, directly)"
            ),
        )
    elif details.get("billing") == "cloud":
        result["message"] = f"signed in ({status.user})"
    elif details.get("billing") == "subscription":
        plan = details.get("subscription")
        plan_str = f" ({plan})" if plan and plan != "unknown" else ""
        result["message"] = f"signed in with a Claude subscription{plan_str}"
        if os.environ.get(PROVIDER_API_KEY_ENV["anthropic"]):
            result["message"] += ", used instead of ANTHROPIC_API_KEY"
    else:
        result["message"] = f"signed in ({status.user}), billed per token"
    return result


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
                        **_playwright_runtime_problem(runtime_detail),
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
                "message": "installed, but its Chromium browser is not downloaded",
                **_install_fields("playwright"),
            }
    else:
        return {
            "name": "Playwright",
            "description": "Browser automation for dynamic URLs",
            "status": "missing",
            "message": "not installed (needed for JavaScript-heavy pages)",
            "install_hint": _playwright_package_install_hint(),
            "install_commands": [_playwright_package_install_hint()],
            "install_extra": "browser",
        }


def _check_serve() -> dict[str, Any]:
    """Check serve extra (web UI server) installation status.

    Returns:
        Result dict with name, description, status, message, install_hint.
    """
    from markitai.serve import is_serve_available

    if is_serve_available():
        return {
            "name": "Serve",
            "description": "Web UI server (markitai serve)",
            "status": "ok",
            "message": "fastapi + uvicorn installed",
            "install_hint": "",
        }
    return {
        "name": "Serve",
        "description": "Web UI server (markitai serve)",
        "status": "missing",
        "message": "serve extra not installed",
        "install_hint": extra_install_command("serve"),
        "install_commands": [extra_install_command("serve")],
        "install_extra": "serve",
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
            "description": "PPTX slide rendering",
            "status": "ok",
            "message": "installed",
            "path": soffice_path,
            "install_hint": "",
        }
    else:
        # macOS: an installed Microsoft PowerPoint takes over PPTX PDF export
        # via AppleScript (see utils/office_mac.py)
        if sys.platform == "darwin" and macos_fallback:
            from markitai.utils import office_mac

            if office_mac.powerpoint_available():
                if not office_mac.staging_container_writable():
                    # Installed, but macOS refuses the staging write, so the
                    # fallback would fail on every PPTX.
                    return {
                        "name": "LibreOffice",
                        "description": "PPTX slide rendering",
                        "status": "warning",
                        "message": (
                            "not found; PowerPoint fallback blocked: "
                            f"{office_mac.CONTAINER_BLOCKED_HINT}"
                        ),
                        **_install_fields("libreoffice"),
                    }
                return {
                    "name": "LibreOffice",
                    "description": "PPTX slide rendering",
                    "status": "warning",
                    "message": (
                        "not found; PowerPoint fallback available "
                        "(via AppleScript, needs one-time Automation permission)"
                    ),
                    **_install_fields("libreoffice"),
                }
        return {
            "name": "LibreOffice",
            "description": "PPTX slide rendering",
            "status": "missing",
            "message": "not found (needed to render PPTX slides as images)",
            **_install_fields("libreoffice"),
        }


def _check_anydoc() -> dict[str, Any]:
    """Check the optional legacy Office backend (``legacy`` extra).

    anydoc moved legacy .doc/.ppt conversion out of Office automation into
    the ``legacy`` extra, so "missing" means "capability not enabled",
    never "installation broken". Probed via metadata only.

    Returns:
        Result dict with name, description, status, optional, message,
        install_hint.
    """
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as get_version

        try:
            anydoc_version = get_version("firecrawl-anydoc")
        except PackageNotFoundError:
            raise ImportError("firecrawl-anydoc not installed") from None

        return {
            "name": "anydoc",
            "description": "Legacy Office conversion (.doc/.ppt)",
            "status": "ok",
            "optional": True,
            "message": f"v{anydoc_version}",
            "install_hint": "",
        }
    except ImportError:
        return {
            "name": "anydoc",
            "description": "Legacy Office conversion (.doc/.ppt)",
            "status": "missing",
            "optional": True,
            "message": "not installed (needed for legacy .doc and .ppt files)",
            "install_hint": extra_install_command("legacy"),
            "install_commands": [extra_install_command("legacy")],
            "install_extra": "legacy",
        }


def _check_rapidocr(cfg: Any) -> dict[str, Any]:
    """Check the optional OCR backend's installation status.

    OCR moved out of the core install into the ``ocr`` extra, so "missing"
    here means "capability not enabled", never "installation broken". The
    ``optional`` flag keeps that distinction machine-readable for --json
    consumers as well as for the exit-code logic.

    Args:
        cfg: Configuration object with OCR language settings.

    Returns:
        Result dict with name, description, status, optional, message,
        install_hint.
    """
    from markitai.ocr import OCR_INSTALL_HINT, is_ocr_available

    try:
        from importlib.metadata import version as get_version

        # Probe via metadata only — importing rapidocr pulls in cv2, whose
        # 119MB dylib pays a one-time ~25s dyld signature validation on a
        # fresh install (the "first doctor run is slow" root cause).
        if not is_ocr_available():
            raise ImportError("rapidocr not installed")

        try:
            rapidocr_version = get_version("rapidocr")
        except Exception:
            rapidocr_version = "unknown"

        # Get configured language; the same validation --ocr applies
        from markitai.ocr import OCRLanguageError, resolve_ocr_language

        configured_lang = cfg.ocr.lang if cfg.ocr else "en"
        lang_display = {
            "ch": "Chinese",
            "chinese_cht": "Traditional Chinese",
            "en": "English",
            "japan": "Japanese",
            "korean": "Korean",
            "arabic": "Arabic",
            "th": "Thai",
            "latin": "Latin",
        }

        try:
            rec_lang, needs_v5 = resolve_ocr_language(configured_lang)
        except OCRLanguageError as e:
            return {
                "name": "RapidOCR",
                "description": "OCR for scanned documents (--ocr)",
                "status": "warning",
                "optional": True,
                "message": f"v{rapidocr_version}, {e}",
                "install_hint": "Set ocr.lang to one of: en, zh, ja, ko, ar, th, latin",
            }
        if needs_v5:
            reads = lang_display.get(rec_lang, rec_lang)
        else:
            # PP-OCRv6 has one multilingual recognizer: ocr.lang en, zh and
            # ja load the same model, so "lang: en (English)" read as if
            # Chinese and Japanese scans would not work
            reads = "Chinese, English, Japanese and Latin-script"
        return {
            "name": "RapidOCR",
            "description": "OCR for scanned documents (--ocr)",
            "status": "ok",
            "optional": True,
            "message": (
                f"v{rapidocr_version}, reads {reads} text (ocr.lang: {configured_lang})"
            ),
            "install_hint": "",
        }
    except ImportError:
        return {
            "name": "RapidOCR",
            "description": "OCR for scanned documents (--ocr)",
            "status": "missing",
            "optional": True,
            "message": "not installed: --ocr unavailable (everything else works)",
            "install_hint": OCR_INSTALL_HINT,
            "install_commands": [OCR_INSTALL_HINT],
            "install_extra": "ocr",
        }


def _is_active_model(model_config: Any) -> bool:
    """Return whether a model config is active for routing-dependent checks."""
    litellm_params = getattr(model_config, "litellm_params", None)
    weight = getattr(litellm_params, "weight", 1)
    return not isinstance(weight, int | float) or weight > 0


def _provider_of(model: str) -> str:
    """Provider prefix of a model id (``openai`` when it has none)."""
    return model.split("/", 1)[0].lower() if "/" in model else "openai"


def _ordered_providers(providers: set[str]) -> list[str]:
    """Providers in markitai's own order (subscriptions, then API keys)."""
    order = list(PROVIDER_DEFAULT_MODELS)
    return sorted(providers, key=lambda p: (order.index(p) if p in order else 99, p))


def _doctor_impl(as_json: bool, fix: bool = False) -> None:
    """Implementation of the doctor command.

    Args:
        as_json: Output results as JSON.
        fix: Attempt to install missing components.
    """
    from markitai.fetch_playwright import clear_browser_cache

    manager = ConfigManager()
    config_path, overrides = root_config_options()
    cfg = manager.load(config_path=config_path, overrides=overrides)

    results: dict[str, dict[str, Any]] = {}

    # Clear Playwright browser cache before parallel checks
    clear_browser_cache()

    # Run independent system tool checks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_playwright = executor.submit(_check_playwright)
        future_libreoffice = executor.submit(
            _check_libreoffice, cfg.office.macos_fallback
        )
        future_rapidocr = executor.submit(_check_rapidocr, cfg)
        future_anydoc = executor.submit(_check_anydoc)

    # Collect results in deterministic order
    results["playwright"] = future_playwright.result()
    results["libreoffice"] = future_libreoffice.result()
    results["rapidocr"] = future_rapidocr.result()
    results["anydoc"] = future_anydoc.result()
    results["serve"] = _check_serve()

    # 5. Check LLM API configuration (check model_list for configured models)
    configured_models = cfg.llm.model_list if cfg.llm.model_list else []
    active_models = [m for m in configured_models if _is_active_model(m)]
    local_prefixes = ("claude-agent/", "copilot/", "chatgpt/")
    active_api_models = [
        m
        for m in active_models
        if not any(m.litellm_params.model.startswith(p) for p in local_prefixes)
    ]
    # The run's own check (markitai.providers.api_keys): a model doctor
    # passes is a model the run sends to
    from markitai.providers.api_keys import missing_api_key

    missing_llm_env_vars: set[str] = set()
    left_out = 0
    for model in active_api_models:
        lacking = missing_api_key(model.litellm_params)
        if lacking is not None:
            missing_llm_env_vars.add(lacking.env_var)
            left_out += 1
    llm_extra: dict[str, Any] = {}

    if configured_models:
        used_providers = {_provider_of(m.litellm_params.model) for m in active_models}
        if not active_models:
            status = "warning"
            message = "Models are configured, but all have weight 0 (disabled)"
            install_hint = "Set weight > 0 on at least one model to enable LLM use"
        elif missing_llm_env_vars:
            names = sorted(missing_llm_env_vars)
            missing = ", ".join(names)
            usable = len(active_models) - left_out
            if usable:
                # The run leaves these models out and uses the others
                # (markitai.providers.api_keys), so it is not blocked
                status = "warning"
                message = (
                    f"not set: {missing}; {_count(left_out, 'model')} left out, "
                    f"{usable} still used"
                )
                llm_extra["blocking"] = False
                llm_extra["summary_name"] = _count(left_out, "LLM model")
            else:
                status = "error"
                message = (
                    f"not set: {missing}; no model can be used without {_them(names)}"
                )
            commands = [get_env_set_command(name) for name in names]
            install_hint = "; ".join(commands) + " (or add to ~/.markitai/.env)"
            llm_extra["install_commands"] = [
                *commands,
                f"# or add {_them(names)} to ~/.markitai/.env",
            ]
        else:
            status = "ok"
            labels = ", ".join(
                "Claude Code" if p == "claude-agent" else provider_label(p)
                for p in _ordered_providers(used_providers)
            )
            message = f"{len(active_models)} active model(s): {labels}"
            install_hint = ""

        results["llm-api"] = {
            "name": "LLM API",
            "description": "Content enhancement and image analysis",
            "status": status,
            "message": message,
            "install_hint": install_hint,
            **llm_extra,
            # Provider keys set in the environment that no active model uses
            # (markitai init leaves out a key whose default model another
            # configured route already serves)
            "unused_api_keys": [
                name
                for provider, names in PROVIDER_API_KEY_ENV_NAMES.items()
                if provider not in used_providers
                for name in names
                if os.environ.get(name)
            ],
        }
    else:
        # Point at the actually-loaded config file (matches the header line),
        # not a hardcoded generic filename.
        detected_env = [
            env for env in PROVIDER_API_KEY_ENV.values() if os.environ.get(env)
        ]
        # markitai init writes the detected providers into the config file,
        # updating one that exists, so it is the fix either way
        target = (
            f" (it adds them to {manager.config_path})"
            if manager.config_path is not None
            else ""
        )
        if detected_env:
            llm_hint = f"Run 'markitai init' to configure models for {', '.join(detected_env)}{target}"
        else:
            llm_hint = (
                "Set a provider API key (DEEPSEEK_API_KEY, GEMINI_API_KEY, ...) "
                f"or sign in to Claude Code, then run 'markitai init'{target}; "
                "or configure llm.model_list by hand"
            )
        results["llm-api"] = {
            "name": "LLM API",
            "description": "Content enhancement and image analysis",
            "status": "missing",
            "message": (
                "no models in llm.model_list; --llm picks from the keys it "
                f"finds ({', '.join(detected_env)})"
                if detected_env
                else "no models in llm.model_list and no provider key set"
            ),
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
                    "install_hint": f"curl -fsSL https://markitai.dev/setup.sh | sh  # or: {extra_install_command('claude-agent')}",
                }
        except Exception as e:
            results["claude-agent-sdk"] = {
                "name": "Claude Agent SDK",
                "description": "Claude Code CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": f"curl -fsSL https://markitai.dev/setup.sh | sh  # or: {extra_install_command('claude-agent')}",
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
                    "install_hint": f"curl -fsSL https://markitai.dev/setup.sh | sh  # or: {extra_install_command('copilot')}",
                }
        except Exception as e:
            results["copilot-sdk"] = {
                "name": "GitHub Copilot SDK",
                "description": "GitHub Copilot CLI integration",
                "status": "error",
                "message": f"Check failed: {e}",
                "install_hint": f"curl -fsSL https://markitai.dev/setup.sh | sh  # or: {extra_install_command('copilot')}",
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
            "message": (
                "none of the active models reads images "
                "(or set model_info.supports_vision)"
                if active_models
                else "needs an LLM model (see LLM API)"
            ),
            "models": [],  # Empty list for JSON output
            "install_hint": (
                "Use vision-capable models like gemini-*, gpt-6*, claude-*"
                if active_models
                else ""
            ),
        }

    # 6b. VLM OCR (--ocr --llm): the OCR path that sends page images to the
    # vision model instead of RapidOCR. Distinct from "vision-model" above
    # (which powers alt text / descriptions): this row is specifically about
    # the OCR capability behind `--ocr --llm`, and it is optional — no vision
    # model means that path is unavailable, not that the install is broken.
    if vision_models:
        vlm_names = [m.litellm_params.model for m in vision_models]
        results["vlm-ocr"] = {
            "name": "VLM OCR",
            "description": "OCR for scanned documents (--ocr --llm)",
            "status": "ok",
            "optional": True,
            "message": (
                f"--ocr --llm sends page images to {_count(len(vlm_names), 'vision model')} "
                "(MARKITAI_NO_VLM_OCR=1 keeps RapidOCR)"
            ),
            "models": vlm_names,  # Structured list for JSON output
            "install_hint": "",
        }
    else:
        results["vlm-ocr"] = {
            "name": "VLM OCR",
            "description": "OCR for scanned documents (--ocr --llm)",
            "status": "warning",
            "optional": True,
            "message": "unavailable: needs a vision model (see Vision Model)",
            "install_hint": "",
        }

    # Define dependency groups.
    # Nothing is unconditionally required any more: OCR moved into the `ocr`
    # extra, so a bare install with no OCR backend is a healthy install, not a
    # broken one. What remains "required" is whatever the *user's own config*
    # asks for — a configured Playwright workflow, an active API model, a
    # local provider — because those are promises the installation is failing
    # to keep.
    required_checks: set[str] = set()
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
        if key not in required_checks or info.get("blocking") is False:
            return False
        status = info.get("status")
        if status in {"missing", "error"}:
            return True
        # A configured capability/provider warning means the requested
        # workflow is not ready.
        return status == "warning"

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
    optional_deps = [
        "rapidocr",
        "vlm-ocr",
        "playwright",
        "libreoffice",
        "anydoc",
        "serve",
    ]
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
        unused = info.get("unused_api_keys")
        if unused:
            ui.info(
                f"Not used: {', '.join(unused)} "
                f"(set, but no model in llm.model_list uses {_them(unused)})"
            )

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
    repair_failed_names: list[str] = []
    if fix and not as_json:
        fixable = [
            key
            for key, info in results.items()
            if info["status"] in ("missing", "warning") and key in FIXABLE_COMPONENTS
        ]
        if fixable:
            console.print()
            ui.section(t("doctor.fix_attempting"))
            for component in fixable:
                # For playwright, detect whether the package itself is missing
                pkg_missing = (
                    component == "playwright"
                    and results[component].get("status") == "missing"
                )
                if not _install_component(component, package_missing=pkg_missing):
                    repair_failed_names.append(results[component]["name"])
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
                        repair_failed_names.append(results[component]["name"])
                    else:
                        runtime_ok, runtime_detail = _smoke_test_playwright_browser()
                        if not runtime_ok:
                            verification["status"] = "warning"
                            verification.update(
                                _playwright_runtime_problem(runtime_detail)
                            )
                            if runtime_detail == SMOKE_SLOW:
                                # Downloaded; only its first launch was slow
                                ui.warning(verification["message"])
                            else:
                                ui.error(
                                    t(
                                        "doctor.fix_verification_failed",
                                        component=verification["name"],
                                        detail=runtime_detail,
                                    )
                                )
                                repair_failed_names.append(results[component]["name"])
                        else:
                            ui.success(
                                t(
                                    "doctor.fix_success",
                                    component=verification["name"],
                                )
                            )

    # Render hints and the final outcome only after any requested repair has
    # been rechecked, so one invocation cannot end with a stale pre-fix status.
    # One line per distinct fix: two checks with the same hint (Vision Model
    # and VLM OCR both asking for a vision-capable model) share it, and so do
    # missing extras: each hint reinstalls markitai, so a plain install (a
    # non-interactive setup.ps1) got four reinstalls where one does
    by_fix: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    extras: list[str] = []
    for info in results.values():
        if info["status"] not in ("missing", "error", "warning") or not info.get(
            "install_hint"
        ):
            continue
        remedy = (str(info["install_hint"]), tuple(info.get("install_commands") or ()))
        if info.get("install_extra"):
            extras.append(info["install_extra"])
            remedy = ("extras", ())
        if remedy in by_fix:
            by_fix[remedy]["name"] += f", {info['name']}"
        else:
            by_fix[remedy] = {**info}
    if len(extras) > 1:
        command = extra_install_command(*extras)
        by_fix[("extras", ())].update(install_hint=command, install_commands=[command])
    hint_items = list(by_fix.values())
    required_failed = _required_failed_count()
    if hint_items:
        console.print()
        ui.section(t("doctor.fix_hint" if required_failed else "doctor.enable_hint"))
        for info in hint_items:
            _render_hint(info)

    failed_names = [
        info["name"] for key, info in results.items() if _is_blocking_failure(key, info)
    ]
    # Shown and counted alike: a check the report does not show is not
    # listed as missing either
    shown = set(optional_deps + llm_keys + auth_keys)
    missing_names = [
        info.get("summary_name", info["name"])
        for key, info in results.items()
        if key in shown
        and info.get("status") != "ok"
        and not _is_blocking_failure(key, info)
    ]
    if required_failed:
        ui.summary(t("doctor.not_ready", names=", ".join(failed_names)), ok=False)
    elif repair_failed_names:
        ui.summary(
            t("doctor.summary_repair_failed", names=", ".join(repair_failed_names)),
            ok=False,
        )
    elif missing_names:
        ui.summary(t("doctor.ready_missing", names=", ".join(missing_names)))
    else:
        ui.summary(t("doctor.all_good"))

    # Scripts rely on the exit code: --fix must not turn an unresolved core
    # dependency into a successful health check.
    if required_failed or repair_failed_names:
        raise SystemExit(1)


# Extras whose SDK may not be installable, so they are offered only when the
# matching CLI is already on PATH. Everything else markitai declares is a
# plain PyPI package and is always offered.
_SDK_EXTRA_BINARIES = {"claude-agent": "claude", "copilot": "copilot"}


def suggest_extras() -> list[str]:
    """Return recommended pip extras for installation.

    Read from the package's own declared extras rather than a hand-kept
    list: that list had gone stale, silently offering neither ``legacy``
    (legacy Office conversion) nor ``mcp`` (the MCP server) nor ``serve``
    (the Web UI) long after they existed, so the guided installer promised
    a batteries-included setup and did not deliver one.

    Everything markitai declares is offered, except extras whose SDK may
    not be on PyPI — those are added only when their CLI is already
    installed.

    The result is a *stable* list (alphabetical) that the install scripts
    can feed directly into ``uv tool install markitai[browser,...]``.

    This is the **single source of truth** — install scripts should call
    ``markitai doctor --suggest-extras`` instead of reimplementing
    detection logic in shell.
    """
    import importlib.metadata

    try:
        declared = set(
            importlib.metadata.metadata("markitai").get_all("Provides-Extra") or []
        )
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        declared = set()
    # "all" is the union, not something to install alongside its own members.
    extras = {name for name in declared - {"all"} if name not in _SDK_EXTRA_BINARIES}
    extras |= {
        name
        for name, binary in _SDK_EXTRA_BINARIES.items()
        if name in declared and shutil.which(binary)
    }
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

        Optional capabilities:
        - RapidOCR (for scanned document processing, `markitai[ocr]`)
        - Playwright (for dynamic URL fetching)
        - LibreOffice (for Office document conversion)
        - LLM API configuration (for content enhancement)
        - Auth status for local providers (Claude, Copilot, ChatGPT)

    A missing optional capability is reported, not failed: the base install is
    healthy without any of them. Exits non-zero only when the *configuration*
    asks for something the machine cannot deliver — a configured Playwright
    workflow that cannot launch; an active API model referencing a missing
    environment variable; an actively configured local provider that cannot
    load/authenticate; or a requested automatic repair that fails — so it can
    be used in scripts and CI.

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
