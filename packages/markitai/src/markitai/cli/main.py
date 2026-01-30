"""Command-line interface for Markitai."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    # Set UTF-8 mode for Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress noisy messages before imports
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Suppress litellm async client cleanup warning (harmless, occurs at exit)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'close_litellm_async_clients' was never awaited",
    category=RuntimeWarning,
)

import click
from dotenv import load_dotenv

# Load .env file from current directory and parent directories
load_dotenv()

from click import Context
from loguru import logger
from rich.console import Console
from rich.syntax import Syntax

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
from markitai.config import ConfigManager, MarkitaiConfig

# Import utilities from refactored modules
from markitai.utils.cli_helpers import (
    is_url,
)
from markitai.utils.executor import shutdown_converter_executor

console = Console()
# Separate stderr console for status/progress (doesn't mix with stdout output)
stderr_console = Console(stderr=True)


# =============================================================================
# Main CLI app
# =============================================================================


@click.group(
    cls=MarkitaiGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory.",
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
    "--agent-browser",
    "use_agent_browser",
    is_flag=True,
    help="Force browser rendering for URLs via agent-browser.",
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
    output: Path,
    config_path: Path | None,
    preset: str | None,
    llm: bool | None,
    alt: bool | None,
    desc: bool | None,
    ocr: bool | None,
    screenshot: bool | None,
    resume: bool,
    no_compress: bool,
    no_cache: bool,
    no_cache_for: str | None,
    batch_concurrency: int | None,
    url_concurrency: int | None,
    llm_concurrency: int | None,
    use_agent_browser: bool,
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
    if use_agent_browser and use_jina:
        console.print(
            "[red]Error: --agent-browser and --jina are mutually exclusive.[/red]"
        )
        ctx.exit(1)

    # Determine fetch strategy
    from markitai.fetch import FetchStrategy

    if use_agent_browser:
        fetch_strategy = FetchStrategy.BROWSER
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
    logger.debug(f"Output directory: {output.resolve()}")

    async def run_workflow() -> None:
        # URL list batch mode
        if is_url_list_mode:
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

        # Single URL mode
        if is_url_input:
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

        # Check if input is directory (batch mode)
        if input_path.is_dir():
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

        # Single file mode
        await process_single_file(
            input_path, output, cfg, dry_run, log_file_path, verbose
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


# =============================================================================
# Config subcommands
# =============================================================================


@app.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("list")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "yaml", "table"], case_sensitive=False),
    default="json",
    help="Output format (json, yaml, or table).",
)
def config_list(output_format: str) -> None:
    """Show current effective configuration."""
    from rich.table import Table

    manager = ConfigManager()
    cfg = manager.load()

    config_dict = cfg.model_dump(mode="json", exclude_none=True)

    if output_format == "json":
        config_json = json.dumps(config_dict, indent=2, ensure_ascii=False)
        syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
    elif output_format == "yaml":
        try:
            import yaml

            config_yaml = yaml.dump(
                config_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
            syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False)
            console.print(syntax)
        except ImportError:
            console.print("[red]YAML output requires PyYAML: pip install pyyaml[/red]")
            raise SystemExit(1)
    elif output_format == "table":
        table = Table(title="Markitai Configuration", show_header=True)
        table.add_column("Section", style="cyan")
        table.add_column("Key", style="green")
        table.add_column("Value", style="white")

        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    table.add_row(section, key, str(value))
            else:
                table.add_row("", section, str(values))

        console.print(table)


@config.command("path")
def config_path_cmd() -> None:
    """Show configuration file paths."""
    manager = ConfigManager()
    manager.load()

    console.print("[bold]Configuration file search order:[/bold]")
    console.print("  1. --config CLI argument")
    console.print("  2. MARKITAI_CONFIG environment variable")
    console.print("  3. ./markitai.json (current directory)")
    console.print(f"  4. {manager.DEFAULT_USER_CONFIG_DIR / 'config.json'}")
    console.print()

    if manager.config_path:
        console.print(f"[green]Currently using:[/green] {manager.config_path}")
    else:
        console.print(
            "[yellow]Using default configuration (no config file found)[/yellow]"
        )


@config.command("init")
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for configuration file.",
)
def config_init(output_path: Path | None) -> None:
    """Initialize a configuration file with defaults."""
    manager = ConfigManager()

    if output_path is None:
        output_path = manager.DEFAULT_USER_CONFIG_DIR / "config.json"
    elif output_path.is_dir():
        # User passed a directory, append default filename
        output_path = output_path / "markitai.json"

    # Check if file exists (not directory)
    if output_path.exists() and output_path.is_file():
        if not click.confirm(f"{output_path} already exists. Overwrite?"):
            raise click.Abort()

    # Save minimal template config (essential fields only)
    saved_path = manager.save(output_path, minimal=True)
    console.print(f"[green]Configuration file created:[/green] {saved_path}")
    console.print("\nEdit this file to customize your settings.")
    console.print(
        "[dim]Note: max_tokens, supports_vision are auto-detected from litellm.[/dim]"
    )
    console.print("Run 'markitai config list' to see the current configuration.")


@config.command("validate")
@click.argument(
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
def config_validate(config_file: Path | None) -> None:
    """Validate a configuration file."""
    manager = ConfigManager()

    try:
        manager.load(config_path=config_file)

        console.print("[green]Configuration is valid![/green]")

        if manager.config_path:
            console.print(f"[dim]Validated: {manager.config_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise SystemExit(2)


@config.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a configuration value."""
    manager = ConfigManager()
    manager.load()

    value = manager.get(key)
    if value is None:
        console.print(f"[yellow]Key not found:[/yellow] {key}")
        raise SystemExit(1)

    # Format output
    if isinstance(value, (dict, list)):
        console.print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        console.print(str(value))


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a configuration value."""
    from pydantic import ValidationError

    manager = ConfigManager()
    manager.load()

    # Parse value
    parsed_value: bool | int | float | str
    if value.lower() in ("true", "false"):
        parsed_value = value.lower() == "true"
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value

    # Store old config for rollback on validation failure
    old_config_dict = manager.config.model_dump()

    try:
        manager.set(key, parsed_value)

        # Validate the entire config using Pydantic
        try:
            MarkitaiConfig.model_validate(manager.config.model_dump())
        except ValidationError as ve:
            # Rollback to old config
            manager._config = MarkitaiConfig.model_validate(old_config_dict)
            # Format validation errors nicely
            errors = []
            for err in ve.errors():
                loc = ".".join(str(x) for x in err["loc"])
                msg = err["msg"]
                errors.append(f"  {loc}: {msg}")
            console.print(f"[red]Invalid value for '{key}':[/red]")
            for error in errors:
                console.print(f"[red]{error}[/red]")
            raise SystemExit(1)

        manager.save()
        console.print(f"[green]Set {key} = {parsed_value}[/green]")

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error setting value:[/red] {e}")
        raise SystemExit(1)


# =============================================================================
# Cache subcommands
# =============================================================================


@app.group()
def cache() -> None:
    """Cache management commands."""
    pass


@cache.command("stats")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed cache entries and model breakdown.",
)
@click.option(
    "--limit",
    default=20,
    type=int,
    help="Number of entries to show in verbose mode (default: 20).",
)
def cache_stats(as_json: bool, verbose: bool, limit: int) -> None:
    """Show cache statistics."""
    from rich.table import Table

    from markitai.constants import DEFAULT_CACHE_DB_FILENAME
    from markitai.llm import SQLiteCache

    def format_size(size_bytes: int) -> str:
        """Format size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"

    def print_verbose_details(
        cache_obj: SQLiteCache, cache_name: str, limit: int, as_json: bool
    ) -> dict[str, Any]:
        """Collect and optionally print verbose cache details."""
        by_model = cache_obj.stats_by_model()
        entries = cache_obj.list_entries(limit)

        if not as_json:
            # Print By Model table
            if by_model:
                model_table = Table(title=f"{cache_name} - By Model")
                model_table.add_column("Model", style="cyan")
                model_table.add_column("Entries", justify="right")
                model_table.add_column("Size", justify="right")
                for model, data in by_model.items():
                    model_table.add_row(
                        model, str(data["count"]), format_size(data["size_bytes"])
                    )
                console.print(model_table)
                console.print()

            # Print Recent Entries table
            if entries:
                entry_table = Table(title=f"{cache_name} - Recent Entries")
                entry_table.add_column("Key", style="dim", max_width=18)
                entry_table.add_column("Model", max_width=30)
                entry_table.add_column("Size", justify="right")
                entry_table.add_column("Preview", max_width=40)
                for entry in entries:
                    key_display = (
                        entry["key"][:16] + "..."
                        if len(entry["key"]) > 16
                        else entry["key"]
                    )
                    entry_table.add_row(
                        key_display,
                        entry["model"],
                        format_size(entry["size_bytes"]),
                        entry["preview"],
                    )
                console.print(entry_table)

        return {"by_model": by_model, "entries": entries}

    manager = ConfigManager()
    cfg = manager.load()

    stats_data: dict[str, Any] = {
        "cache": None,
        "enabled": cfg.cache.enabled,
    }

    # Check global cache
    global_cache: SQLiteCache | None = None
    global_cache_path = (
        Path(cfg.cache.global_dir).expanduser() / DEFAULT_CACHE_DB_FILENAME
    )
    if global_cache_path.exists():
        try:
            global_cache = SQLiteCache(global_cache_path, cfg.cache.max_size_bytes)
            stats_data["cache"] = global_cache.stats()
        except Exception as e:
            stats_data["cache"] = {"error": str(e)}

    # Collect verbose data if needed
    if (
        verbose
        and global_cache
        and stats_data["cache"]
        and "error" not in stats_data["cache"]
    ):
        verbose_data = print_verbose_details(global_cache, "Cache", limit, as_json)
        stats_data["cache"]["by_model"] = verbose_data["by_model"]
        stats_data["cache"]["entries"] = verbose_data["entries"]

    if as_json:
        # Use soft_wrap=True to prevent rich from breaking long lines
        console.print(
            json.dumps(stats_data, indent=2, ensure_ascii=False), soft_wrap=True
        )
    else:
        console.print("[bold]Cache Statistics[/bold]")
        console.print(f"Enabled: {cfg.cache.enabled}")
        console.print()

        if stats_data["cache"]:
            c = stats_data["cache"]
            if "error" in c:
                console.print(f"[red]Cache error:[/red] {c['error']}")
            else:
                console.print(f"  Path: {c['db_path']}")
                console.print(f"  Entries: {c['count']}")
                console.print(f"  Size: {c['size_mb']} MB / {c['max_size_mb']} MB")
        else:
            console.print("[dim]No cache found[/dim]")


@cache.command("clear")
@click.option(
    "--include-spa-domains",
    is_flag=True,
    help="Also clear learned SPA domains.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def cache_clear(include_spa_domains: bool, yes: bool) -> None:
    """Clear cache entries."""
    from markitai.constants import DEFAULT_CACHE_DB_FILENAME
    from markitai.llm import SQLiteCache

    manager = ConfigManager()
    cfg = manager.load()

    # Confirm if not --yes
    if not yes:
        desc = "global cache (~/.markitai)"
        if include_spa_domains:
            desc += " + learned SPA domains"
        if not click.confirm(f"Clear {desc}?"):
            console.print("[yellow]Aborted[/yellow]")
            return

    result = {"cache": 0, "spa_domains": 0}

    # Clear global cache
    global_cache_path = (
        Path(cfg.cache.global_dir).expanduser() / DEFAULT_CACHE_DB_FILENAME
    )
    if global_cache_path.exists():
        try:
            global_cache = SQLiteCache(global_cache_path, cfg.cache.max_size_bytes)
            result["cache"] = global_cache.clear()
        except Exception as e:
            console.print(f"[red]Failed to clear cache:[/red] {e}")

    # Clear SPA domains if requested
    if include_spa_domains:
        from markitai.fetch import get_spa_domain_cache

        try:
            spa_cache = get_spa_domain_cache()
            result["spa_domains"] = spa_cache.clear()
        except Exception as e:
            console.print(f"[red]Failed to clear SPA domains:[/red] {e}")

    # Report results
    if result["cache"] > 0 or result["spa_domains"] > 0:
        console.print(f"[green]Cleared {result['cache']} cache entries[/green]")
        if result["spa_domains"] > 0:
            console.print(f"  SPA domains: {result['spa_domains']}")
    else:
        console.print("[dim]No cache entries to clear[/dim]")


@cache.command("spa-domains")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear all learned SPA domains.",
)
def cache_spa_domains(as_json: bool, clear: bool) -> None:
    """View or manage learned SPA domains.

    Shows domains that were automatically detected as requiring browser
    rendering (JavaScript-heavy sites). These domains will use browser
    strategy directly on future requests, avoiding wasted static fetch attempts.
    """
    from rich.table import Table

    from markitai.fetch import get_spa_domain_cache

    spa_cache = get_spa_domain_cache()

    if clear:
        count = spa_cache.clear()
        if as_json:
            console.print(json.dumps({"cleared": count}))
        else:
            console.print(f"[green]Cleared {count} learned SPA domains[/green]")
        return

    domains = spa_cache.list_domains()

    if as_json:
        console.print(json.dumps(domains, indent=2, ensure_ascii=False), soft_wrap=True)
        return

    if not domains:
        console.print("[dim]No learned SPA domains yet[/dim]")
        console.print(
            "\n[dim]Domains are learned automatically when static fetch "
            "detects JavaScript requirement.[/dim]"
        )
        return

    console.print(f"[bold]Learned SPA Domains[/bold] ({len(domains)} total)\n")

    table = Table()
    table.add_column("Domain", style="cyan")
    table.add_column("Hits", justify="right")
    table.add_column("Learned At", style="dim")
    table.add_column("Last Hit", style="dim")
    table.add_column("Status")

    for d in domains:
        status = "[red]Expired[/red]" if d.get("expired") else "[green]Active[/green]"
        learned_at = d.get("learned_at", "")[:10] if d.get("learned_at") else "-"
        last_hit = d.get("last_hit", "")[:10] if d.get("last_hit") else "-"
        table.add_row(
            d["domain"],
            str(d.get("hits", 0)),
            learned_at,
            last_hit,
            status,
        )

    console.print(table)
    console.print(
        "\n[dim]Tip: Use --clear to reset learned domains, "
        "or configure fallback_patterns in config file for permanent rules.[/dim]"
    )


# Register check-deps command from deps.py
from markitai.cli.commands.deps import check_deps

app.add_command(check_deps)


# =============================================================================
if __name__ == "__main__":
    app()
