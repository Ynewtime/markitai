"""Validation helpers for CLI processors.

This module contains validation functions for checking
configuration and environment before processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    pass

console = Console()


def check_vision_model_config(
    cfg: Any, console: Console, verbose: bool = False
) -> None:
    """Check vision model configuration when image analysis is enabled.

    Args:
        cfg: Configuration object
        console: Rich console for output
        verbose: Whether to show extra details
    """
    # Always check for unsupported Copilot models (GPT-5 series, o1/o3)
    # This applies to all LLM scenarios, not just vision
    if cfg.llm.enabled and cfg.llm.model_list:
        _check_copilot_unsupported_models(cfg.llm.model_list, console)

    # Only check vision-specific config if image analysis is enabled
    if not (cfg.image.alt_enabled or cfg.image.desc_enabled):
        return

    # Check if LLM is enabled
    if not cfg.llm.enabled:
        warning_text = (
            "[yellow]⚠ Image analysis (--alt/--desc) requires LLM to be enabled.[/yellow]\n\n"
            "[dim]Image alt text and descriptions will be skipped without LLM.[/dim]\n\n"
            "To enable LLM processing:\n"
            "  [cyan]markitai --llm ...[/cyan]  or use [cyan]--preset rich/standard[/cyan]"
        )
        console.print(Panel(warning_text, title="LLM Required", border_style="yellow"))
        return

    # Check if vision-capable models are configured (auto-detect from litellm)
    from markitai.llm import get_model_info_cached
    from markitai.providers import is_local_provider_model

    def is_vision_model(model_config: Any) -> bool:
        """Check if model supports vision (config override or auto-detect)."""
        model_id = model_config.litellm_params.model

        # Config override takes priority
        if (
            model_config.model_info
            and model_config.model_info.supports_vision is not None
        ):
            return model_config.model_info.supports_vision

        # Local providers (claude-agent/, copilot/) always support vision
        if is_local_provider_model(model_id):
            return True

        # Auto-detect from litellm
        info = get_model_info_cached(model_id)
        return info.get("supports_vision", False)

    vision_models = [m for m in cfg.llm.model_list if is_vision_model(m)]

    if not vision_models and cfg.llm.model_list:
        # List configured models
        configured_models = ", ".join(
            [m.litellm_params.model for m in cfg.llm.model_list[:3]]
        )
        if len(cfg.llm.model_list) > 3:
            configured_models += f" (+{len(cfg.llm.model_list) - 3} more)"

        warning_text = (
            "[yellow]⚠ No vision-capable models detected.[/yellow]\n\n"
            f"[dim]Current models: {configured_models}[/dim]\n"
            "[dim]Vision models are auto-detected from litellm. "
            "Add `supports_vision: true` in config to override.[/dim]"
        )
        console.print(
            Panel(warning_text, title="Vision Model Recommended", border_style="yellow")
        )
    elif verbose and vision_models:
        # In verbose mode, show which vision models are configured
        model_names = [m.litellm_params.model for m in vision_models]
        count = len(model_names)
        if count <= 3:
            logger.debug(
                f"Vision models configured: {count} ({', '.join(model_names)})"
            )
        else:
            preview = ", ".join(model_names[:3])
            logger.debug(f"Vision models configured: {count} ({preview}, ...)")


def _check_copilot_unsupported_models(model_list: list[Any], console: Console) -> None:
    """Check for Copilot models that are known to have compatibility issues.

    GPT-5 series and o1/o3 models are not supported by Copilot SDK because
    they require 'max_completion_tokens' instead of 'max_tokens'.

    Args:
        model_list: List of model configurations
        console: Rich console for output
    """
    # Unsupported model patterns for Copilot
    unsupported_patterns = (
        "copilot/gpt-5",
        "copilot/o1",
        "copilot/o3",
    )

    unsupported_models = []
    for m in model_list:
        model_id = m.litellm_params.model
        if any(model_id.startswith(p) for p in unsupported_patterns):
            # Only warn if weight > 0 (model is actually enabled)
            weight = getattr(m.litellm_params, "weight", 1)
            if weight > 0:
                unsupported_models.append(model_id)

    if unsupported_models:
        model_list_str = ", ".join(unsupported_models)
        warning_text = (
            f"[yellow]⚠ Unsupported Copilot models detected: {model_list_str}[/yellow]\n\n"
            "[dim]GPT-5/o1/o3 series models are not compatible with Copilot SDK.\n"
            "The SDK uses 'max_tokens' but these models require 'max_completion_tokens'.\n"
            "Requests to these models will fail with 400 errors.[/dim]\n\n"
            "Solutions:\n"
            "  1. Set [cyan]weight: 0[/cyan] for these models in config\n"
            "  2. Use [cyan]openrouter/openai/gpt-5.2[/cyan] instead (direct API)\n"
            "  3. Use other Copilot models like [cyan]copilot/claude-sonnet-4.5[/cyan]"
        )
        console.print(
            Panel(
                warning_text, title="Copilot Model Compatibility", border_style="yellow"
            )
        )


def check_agent_browser_for_urls(cfg: Any, console: Console) -> None:
    """Check agent-browser availability and warn if not ready for URL processing.

    Args:
        cfg: Configuration object
        console: Rich console for output
    """
    from markitai.fetch import FetchStrategy, verify_agent_browser_ready

    # Only check if strategy might use browser
    strategy = (
        cfg.fetch.strategy if hasattr(cfg.fetch, "strategy") else FetchStrategy.AUTO
    )
    if strategy == FetchStrategy.STATIC or strategy == FetchStrategy.JINA:
        return  # No browser needed

    # Get command from config
    command = "agent-browser"
    if hasattr(cfg, "agent_browser") and hasattr(cfg.agent_browser, "command"):
        command = cfg.agent_browser.command

    is_ready, message = verify_agent_browser_ready(command, use_cache=True)

    if not is_ready:
        warning_text = (
            f"[yellow]{message}[/yellow]\n\n"
            "[dim]URL processing will fall back to static fetch strategy.\n"
            "For JavaScript-rendered pages (Twitter/X, etc.), browser support is recommended.\n\n"
            "To install browser support:[/dim]\n"
            "  [cyan]agent-browser install[/cyan]  [dim]or[/dim]  [cyan]npx playwright install chromium[/cyan]"
        )
        console.print(
            Panel(warning_text, title="Browser Not Available", border_style="yellow")
        )


def warn_case_sensitivity_mismatches(
    files: list[Path],
    input_dir: Path,
    patterns: list[str],
) -> None:
    """Warn about files that would match patterns if case-insensitive.

    This helps users catch cases where e.g., '*.jpg' doesn't match 'IMAGE.JPG'
    because pattern matching is case-sensitive on most platforms.

    Args:
        files: List of files discovered for processing
        input_dir: Base input directory for relative path calculation
        patterns: List of --no-cache-for patterns
    """
    import fnmatch

    # Collect potential case mismatches
    mismatches: list[tuple[str, str]] = []  # (file_path, pattern)

    for f in files:
        try:
            rel_path = f.relative_to(input_dir).as_posix()
        except ValueError:
            rel_path = f.name

        for pattern in patterns:
            # Normalize pattern
            norm_pattern = pattern.replace("\\", "/")

            # Check if it would match case-insensitively but not case-sensitively
            if not fnmatch.fnmatch(rel_path, norm_pattern):
                if fnmatch.fnmatch(rel_path.lower(), norm_pattern.lower()):
                    mismatches.append((rel_path, pattern))

    if mismatches:
        # Group by pattern for cleaner output
        by_pattern: dict[str, list[str]] = {}
        for file_path, pattern in mismatches:
            by_pattern.setdefault(pattern, []).append(file_path)

        # Log warning
        logger.warning(
            f"[Cache] Case-sensitivity: {len(mismatches)} file(s) would match "
            "--no-cache-for patterns if case-insensitive"
        )

        # Show details in console
        console.print(
            f"[yellow]Warning: {len(mismatches)} file(s) have case mismatches "
            "with --no-cache-for patterns[/yellow]"
        )
        for pattern, file_paths in by_pattern.items():
            console.print(f"  Pattern: [cyan]{pattern}[/cyan]")
            for fp in file_paths[:3]:  # Show max 3 examples
                console.print(f"    - {fp}")
            if len(file_paths) > 3:
                console.print(f"    ... and {len(file_paths) - 3} more")
        console.print(
            "[dim]Hint: Pattern matching is case-sensitive. "
            "Use exact case or patterns like '*.[jJ][pP][gG]'[/dim]"
        )


# Backward compatibility aliases
_check_vision_model_config = check_vision_model_config
_check_agent_browser_for_urls = check_agent_browser_for_urls
_warn_case_sensitivity_mismatches = warn_case_sensitivity_mismatches
