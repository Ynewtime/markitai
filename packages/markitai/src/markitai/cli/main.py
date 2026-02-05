"""Command-line interface for Markitai."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    # Set UTF-8 mode for Windows console
    # sys.stdout/stderr are actually io.TextIOWrapper which has reconfigure()
    # but typed as TextIO for compatibility. hasattr check ensures safety.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Suppress noisy messages before imports
# Note: Most warning filters are now centralized in logging_config.setup_logging()
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

import click
from dotenv import load_dotenv

# Load .env file from current directory and parent directories
load_dotenv()

from click import Context
from loguru import logger

from markitai.cli.console import get_console, get_stderr_console
from markitai.cli.framework import MarkitaiGroup
from markitai.cli.logging_config import (
    print_version,
    setup_logging,
)

# Import processors from refactored modules
from markitai.cli.processors import (
    process_batch,
    process_single_file,
    process_url,
    process_url_batch,
)
from markitai.cli.processors.validators import (
    check_vision_model_config as _check_vision_model_config,
)
from markitai.config import ConfigManager

# Import utilities from refactored modules
from markitai.utils.cli_helpers import (
    is_url,
)
from markitai.utils.executor import shutdown_converter_executor

console = get_console()
# Separate stderr console for status/progress (doesn't mix with stdout output)
stderr_console = get_stderr_console()


# =============================================================================
# Main CLI app
# =============================================================================


def run_interactive_mode(ctx: click.Context) -> None:
    """Run interactive mode and execute with gathered options."""
    from markitai.cli.interactive import run_interactive, session_to_cli_args

    try:
        session = run_interactive()

        # Ask for confirmation before executing
        import questionary

        if questionary.confirm(
            "Execute conversion with these settings?", default=True
        ).ask():
            # Re-invoke the CLI with the gathered arguments
            args = session_to_cli_args(session)
            # Use sys.argv[0] which is the actual invoked script/executable
            # This works for both installed (markitai) and development (uv run markitai)
            import subprocess

            subprocess.run([sys.argv[0]] + args)
        else:
            click.echo("Cancelled.")
        ctx.exit(0)
    except (KeyboardInterrupt, EOFError):
        click.echo("\nCancelled.")
        ctx.exit(0)


@click.group(
    cls=MarkitaiGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory. If not specified, output to stdout.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration file.",
)
@click.option(
    "--preset",
    "-p",
    type=click.Choice(["rich", "standard", "minimal"], case_sensitive=False),
    default=None,
    help="Use a preset configuration (rich/standard/minimal).",
)
@click.option(
    "--llm/--no-llm",
    default=None,
    help="Enable/disable LLM processing.",
)
@click.option(
    "--alt/--no-alt",
    default=None,
    help="Enable/disable alt text generation for images (requires --llm).",
)
@click.option(
    "--desc/--no-desc",
    default=None,
    help="Enable/disable JSON description file for images (requires --llm).",
)
@click.option(
    "--ocr/--no-ocr",
    default=None,
    help="Enable/disable OCR for scanned documents (uses RapidOCR).",
)
@click.option(
    "--screenshot/--no-screenshot",
    default=None,
    help="Enable/disable screenshots (PDF/PPTX pages; full-page for URLs).",
)
@click.option(
    "--screenshot-only",
    is_flag=True,
    help="Capture screenshots only. Without --llm: saves only screenshots. "
    "With --llm: LLM extracts content purely from screenshots.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume interrupted batch processing.",
)
@click.option(
    "--no-compress",
    is_flag=True,
    help="Disable image compression.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable LLM result caching (force fresh API calls).",
)
@click.option(
    "--no-cache-for",
    type=str,
    default=None,
    help="Disable cache for specific files/patterns (comma-separated, supports glob). "
    "E.g., 'file.pdf', '*.docx', '**/reports/*.pdf'.",
)
@click.option(
    "--llm-concurrency",
    type=click.IntRange(min=1),
    default=None,
    help="Number of concurrent LLM requests (default from config).",
)
@click.option(
    "--batch-concurrency",
    "-j",
    type=click.IntRange(min=1),
    default=None,
    help="Number of concurrent batch tasks (default from config).",
)
@click.option(
    "--url-concurrency",
    type=click.IntRange(min=1),
    default=None,
    help="Number of concurrent URL fetches (default from config, separate from file processing).",
)
@click.option(
    "--playwright",
    "use_playwright",
    is_flag=True,
    help="Force browser rendering for URLs via Playwright.",
)
@click.option(
    "--jina",
    "use_jina",
    is_flag=True,
    help="Force Jina Reader API for URL fetching.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress and info messages, only show errors.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview conversion without writing files.",
)
@click.option(
    "--interactive",
    "-I",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=lambda ctx, _param, value: run_interactive_mode(ctx) if value else None,
    help="Enter interactive mode for guided setup.",
)
@click.option(
    "--version",
    "-v",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Show version and exit.",
)
@click.pass_context
def app(
    ctx: Context,
    output: Path | None,
    config_path: Path | None,
    preset: str | None,
    llm: bool | None,
    alt: bool | None,
    desc: bool | None,
    ocr: bool | None,
    screenshot: bool | None,
    screenshot_only: bool,
    resume: bool,
    no_compress: bool,
    no_cache: bool,
    no_cache_for: str | None,
    batch_concurrency: int | None,
    url_concurrency: int | None,
    llm_concurrency: int | None,
    use_playwright: bool,
    use_jina: bool,
    verbose: bool,
    quiet: bool,
    dry_run: bool,
) -> None:
    """Markitai - Opinionated Markdown converter with native LLM enhancement support.

    Convert various document formats and URLs to Markdown with optional
    LLM-powered enhancement for format optimization and image analysis.

    \b
    Presets:
        rich     - LLM + alt + desc + screenshot (complex documents)
        standard - LLM + alt + desc (normal documents)
        minimal  - No enhancement (just convert)

    \b
    Examples:
        markitai document.docx                      # Convert single file
        markitai https://example.com/page           # Convert web page
        markitai urls.urls -o ./output/             # Batch URL processing
        markitai https://youtube.com/watch?v=abc    # Convert YouTube video
        markitai document.pdf --preset rich         # Use rich preset
        markitai document.pdf --preset rich --ocr   # Rich + OCR for scans
        markitai document.pdf --preset rich --no-desc  # Rich without desc
        markitai ./docs/ -o ./output/ --resume      # Batch conversion
        markitai config list                        # Show configuration
    """
    # If subcommand is invoked, let it handle
    if ctx.invoked_subcommand is not None:
        return

    # Get input path from context (set by MarkitaiGroup.parse_args)
    ctx.ensure_object(dict)
    input_path_str = ctx.obj.get("_input_path")

    if not input_path_str:
        click.echo(ctx.get_help())
        ctx.exit(0)

    # Check if input is a URL
    is_url_input = is_url(input_path_str)

    # Initialize URL list mode variables
    url_entries: list = []
    is_url_list_mode = False
    input_path: Path | None = None

    # For file/directory inputs, validate existence and check for .urls file
    if not is_url_input:
        input_path = Path(input_path_str)
        if not input_path.exists():
            console.print(f"[red]Error: Path '{input_path}' does not exist.[/red]")
            ctx.exit(1)

        # Auto-detect .urls file
        if input_path.is_file() and input_path.suffix == ".urls":
            from markitai.urls import UrlListParseError, parse_url_list

            try:
                url_entries = parse_url_list(input_path)
            except UrlListParseError as e:
                console.print(f"[red]Error parsing URL list: {e}[/red]")
                ctx.exit(1)

            if not url_entries:
                console.print(f"[yellow]No valid URLs found in {input_path}[/yellow]")
                ctx.exit(0)

            is_url_list_mode = True
            input_path = None  # Clear input_path for URL list mode

    # Load configuration first
    config_manager = ConfigManager()
    cfg = config_manager.load(config_path=config_path)

    # Determine if we're in single file/URL mode (not batch)
    # Single file/URL mode: quiet console unless --verbose is specified
    # URL list mode is batch mode
    is_single_mode = (
        is_url_input or (input_path is not None and input_path.is_file())
    ) and not is_url_list_mode
    # Enable quiet mode if: explicitly requested via --quiet, or in single mode without --verbose
    quiet_console = quiet or (is_single_mode and not verbose)

    # Setup logging with configuration
    console_handler_id, log_file_path = setup_logging(
        verbose=verbose,
        log_dir=cfg.log.dir,
        log_level=cfg.log.level,
        log_format=cfg.log.format,
        rotation=cfg.log.rotation,
        retention=cfg.log.retention,
        quiet=quiet_console,
    )

    # Log configuration status after logging is set up
    if config_manager.config_path:
        logger.info(f"[Config] Loaded from: {config_manager.config_path}")
    else:
        logger.warning("[Config] No config file found, using defaults")

    # Warn if LLM is enabled but no models configured
    if cfg.llm.enabled and not cfg.llm.model_list:
        logger.warning(
            "[Config] LLM enabled but no models configured. "
            "Add models to llm.model_list in config file or specify -c <config_path>"
        )
    elif cfg.llm.enabled and cfg.llm.model_list:
        model_names = [m.litellm_params.model for m in cfg.llm.model_list]
        unique_models = set(model_names)
        logger.debug(
            f"[Config] LLM models configured: {len(model_names)} entries, "
            f"{len(unique_models)} unique models"
        )

    # Store handler ID, log file path and verbose in context for batch processing
    ctx.obj["_console_handler_id"] = console_handler_id
    ctx.obj["_log_file_path"] = log_file_path
    ctx.obj["_verbose"] = verbose

    # Apply preset first (if specified)
    from markitai.config import get_preset

    if preset:
        preset_config = get_preset(preset, cfg)
        if preset_config:
            # Apply preset values as base
            cfg.llm.enabled = preset_config.llm
            cfg.image.alt_enabled = preset_config.alt
            cfg.image.desc_enabled = preset_config.desc
            cfg.ocr.enabled = preset_config.ocr
            cfg.screenshot.enabled = preset_config.screenshot
            logger.debug(f"Applied preset: {preset}")
        else:
            console.print(f"[yellow]Warning: Unknown preset '{preset}'[/yellow]")

    # Override with explicit CLI options (--flag or --no-flag)
    # None means not specified, so we don't override
    if llm is not None:
        cfg.llm.enabled = llm
    if alt is not None:
        cfg.image.alt_enabled = alt
    if desc is not None:
        cfg.image.desc_enabled = desc
    if ocr is not None:
        cfg.ocr.enabled = ocr
    if screenshot is not None:
        cfg.screenshot.enabled = screenshot
    if screenshot_only:
        # screenshot_only enables screenshot capture but NOT implicitly LLM
        # --screenshot-only alone: just capture screenshots (no .md output)
        # --llm --screenshot-only: capture + LLM extraction
        cfg.screenshot.screenshot_only = True
        cfg.screenshot.enabled = True  # Implicitly enable screenshot
    if no_compress:
        cfg.image.compress = False
    if no_cache:
        cfg.cache.no_cache = True
    if no_cache_for:
        # Parse comma-separated patterns
        cfg.cache.no_cache_patterns = [
            p.strip() for p in no_cache_for.split(",") if p.strip()
        ]
    if batch_concurrency is not None:
        cfg.batch.concurrency = batch_concurrency
    if url_concurrency is not None:
        cfg.batch.url_concurrency = url_concurrency
    if llm_concurrency is not None:
        cfg.llm.concurrency = llm_concurrency

    # Validate vision model configuration if image analysis is enabled
    _check_vision_model_config(cfg, console, verbose)

    # Validate local provider dependencies (claude-agent, copilot)
    if cfg.llm.model_list:
        from markitai.providers import (
            check_deprecated_models,
            validate_local_provider_deps,
        )

        models = [m.litellm_params.model for m in cfg.llm.model_list]
        dep_warnings = validate_local_provider_deps(models)
        deprecation_warnings = check_deprecated_models(models)
        all_warnings = dep_warnings + deprecation_warnings
        if all_warnings:
            for warning in all_warnings:
                console.print(f"[yellow]{warning}[/yellow]")
            console.print()

    # Validate fetch strategy flags (mutually exclusive)
    if use_playwright and use_jina:
        console.print(
            "[red]Error: --playwright and --jina are mutually exclusive.[/red]"
        )
        ctx.exit(1)

    # Determine fetch strategy
    from markitai.fetch import FetchStrategy

    if use_playwright:
        fetch_strategy = FetchStrategy.PLAYWRIGHT
        explicit_fetch_strategy = True
    elif use_jina:
        fetch_strategy = FetchStrategy.JINA
        explicit_fetch_strategy = True
    else:
        # Use config default or auto
        fetch_strategy = FetchStrategy(cfg.fetch.strategy)
        explicit_fetch_strategy = False

    # Log input info
    if is_url_list_mode:
        logger.debug(f"Processing URL list: {len(url_entries)} URLs")
    elif is_url_input:
        logger.debug(f"Processing URL: {input_path_str}")
    else:
        assert input_path is not None  # Already validated above
        logger.debug(f"Processing: {input_path.resolve()}")
    if output:
        logger.debug(f"Output directory: {output.resolve()}")

    async def run_workflow() -> None:
        # URL list batch mode - requires -o
        if is_url_list_mode:
            if output is None:
                console.print(
                    "[red]Error: URL list mode requires -o/--output directory.[/red]"
                )
                ctx.exit(1)
            await process_url_batch(
                url_entries,
                output,
                cfg,
                dry_run,
                verbose,
                log_file_path,
                concurrency=cfg.batch.url_concurrency,
                fetch_strategy=fetch_strategy,
                explicit_fetch_strategy=explicit_fetch_strategy,
            )
            return

        # Single URL mode - requires -o
        if is_url_input:
            if output is None:
                console.print(
                    "[red]Error: URL mode requires -o/--output directory.[/red]"
                )
                ctx.exit(1)
            assert input_path_str is not None  # Guaranteed when is_url_input is True
            await process_url(
                input_path_str,
                output,
                cfg,
                dry_run,
                verbose,
                log_file_path,
                fetch_strategy=fetch_strategy,
                explicit_fetch_strategy=explicit_fetch_strategy,
            )
            return

        # File/directory mode
        assert input_path is not None  # Already validated above

        # Check if input is directory (batch mode) - requires -o
        if input_path.is_dir():
            if output is None:
                console.print(
                    "[red]Error: Batch mode requires -o/--output directory.[/red]"
                )
                ctx.exit(1)
            await process_batch(
                input_path,
                output,
                cfg,
                resume,
                dry_run,
                verbose=verbose,
                console_handler_id=console_handler_id,
                log_file_path=log_file_path,
                fetch_strategy=fetch_strategy,
                explicit_fetch_strategy=explicit_fetch_strategy,
            )
            return

        # Single file mode - output is optional (None means stdout)
        await process_single_file(
            input_path,
            output,
            cfg,
            dry_run,
            log_file_path,
            verbose=verbose,
            quiet=quiet,
        )

    async def run_workflow_with_cleanup() -> None:
        """Run workflow with explicit resource cleanup on exit."""
        from markitai.fetch import close_shared_clients

        try:
            await run_workflow()
        finally:
            # Cleanup shared resources
            await close_shared_clients()  # Close httpx.AsyncClient for Jina
            shutdown_converter_executor()  # Shutdown ThreadPoolExecutor

            # Close LiteLLM's aiohttp sessions to prevent "Unclosed connection" warning
            try:
                from litellm.llms.custom_httpx.async_client_cleanup import (
                    close_litellm_async_clients,
                )

                await close_litellm_async_clients()
            except Exception:
                pass  # Ignore cleanup errors

    asyncio.run(run_workflow_with_cleanup())


# Register commands from commands/ modules
from markitai.cli.commands.cache import cache
from markitai.cli.commands.config import config
from markitai.cli.commands.doctor import check_deps, doctor

app.add_command(cache)
app.add_command(config)

app.add_command(doctor)
app.add_command(check_deps)


# =============================================================================
if __name__ == "__main__":
    app()
