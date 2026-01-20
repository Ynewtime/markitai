"""Command-line interface for Markit."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markit.llm import ImageAnalysis, LLMProcessor

# Suppress noisy messages before imports
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import click
from dotenv import load_dotenv

# Load .env file from current directory and parent directories
load_dotenv()

from click import Context
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from markit import __version__
from markit.config import ConfigManager, MarkitConfig
from markit.constants import DEFAULT_MAX_IMAGES_PER_BATCH, MAX_DOCUMENT_SIZE
from markit.converter import FileFormat, detect_format, get_converter
from markit.converter.base import EXTENSION_MAP
from markit.image import ImageProcessor
from markit.security import (
    atomic_write_json,
    atomic_write_text,
    escape_glob_pattern,
    validate_file_size,
)
from markit.workflow.helpers import (
    add_basic_frontmatter as _add_basic_frontmatter,
)
from markit.workflow.helpers import (
    detect_language as _detect_language,
)
from markit.workflow.helpers import (
    merge_llm_usage as _merge_llm_usage,
)
from markit.workflow.helpers import (
    write_assets_desc_json,
)
from markit.workflow.single import ImageAnalysisResult

console = Console()


def resolve_output_path(
    base_path: Path,
    on_conflict: str,
) -> Path | None:
    """Resolve output path based on conflict strategy.

    Args:
        base_path: The original output file path
        on_conflict: Conflict resolution strategy ("skip", "overwrite", "rename")

    Returns:
        Resolved path, or None if file should be skipped.
        For rename strategy: file.pdf.md -> file.pdf.v2.md -> file.pdf.v3.md
        For rename with .llm.md: file.pdf.llm.md -> file.pdf.v2.llm.md
        This ensures files sort in natural order (A-Z).
    """
    if not base_path.exists():
        return base_path

    if on_conflict == "skip":
        return None
    elif on_conflict == "overwrite":
        return base_path
    else:  # rename
        # Parse filename to insert version number before .md/.llm.md suffix
        # e.g., "file.pdf.md" -> "file.pdf.v2.md" -> "file.pdf.v3.md"
        # e.g., "file.pdf.llm.md" -> "file.pdf.v2.llm.md"
        # This ensures files sort in natural A-Z order (.md < .v2.md < .v3.md)
        name = base_path.name

        # Determine the markit suffix (.md or .llm.md)
        if name.endswith(".llm.md"):
            base_stem = name[:-7]  # Remove ".llm.md" -> "file.pdf"
            markit_suffix = ".llm.md"
        else:
            base_stem = name[:-3]  # Remove ".md" -> "file.pdf"
            markit_suffix = ".md"

        # Find next available sequence number
        seq = 2
        while True:
            new_name = f"{base_stem}.v{seq}{markit_suffix}"
            new_path = base_path.parent / new_name
            if not new_path.exists():
                return new_path
            seq += 1
            if seq > 9999:  # Safety limit
                raise RuntimeError(f"Too many conflicting files for {base_path}")


def compute_task_hash(
    input_path: Path,
    output_dir: Path,
    options: dict[str, Any] | None = None,
) -> str:
    """Compute hash from task input parameters.

    Hash is based on:
    - input_path (resolved)
    - output_dir (resolved)
    - key task options (llm_enabled, ocr_enabled, etc.)

    This ensures different parameter combinations produce different hashes.

    Args:
        input_path: Input file or directory path
        output_dir: Output directory path
        options: Task options dict (llm_enabled, ocr_enabled, etc.)

    Returns:
        6-character hex hash string
    """
    import hashlib

    # Extract key options that affect output
    key_options = {}
    if options:
        key_options = {
            k: v
            for k, v in options.items()
            if k
            in (
                "llm_enabled",
                "ocr_enabled",
                "screenshot_enabled",
                "image_alt_enabled",
                "image_desc_enabled",
            )
        }

    hash_params = {
        "input": str(input_path.resolve()),
        "output": str(output_dir.resolve()),
        "options": key_options,
    }
    hash_str = json.dumps(hash_params, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()[:6]


def get_report_file_path(
    output_dir: Path,
    task_hash: str,
    on_conflict: str = "rename",
) -> Path:
    """Generate report file path based on task hash.

    Format: reports/markit.<hash>.report.json
    Respects on_conflict strategy for rename.

    Args:
        output_dir: Output directory
        task_hash: Task hash string
        on_conflict: Conflict resolution strategy

    Returns:
        Path to the report file
    """
    reports_dir = output_dir / "reports"
    base_path = reports_dir / f"markit.{task_hash}.report.json"

    if not base_path.exists():
        return base_path

    if on_conflict == "skip":
        return base_path  # Will be handled by caller
    elif on_conflict == "overwrite":
        return base_path
    else:  # rename
        seq = 2
        while True:
            new_path = reports_dir / f"markit.{task_hash}.v{seq}.report.json"
            if not new_path.exists():
                return new_path
            seq += 1


# =============================================================================
# Custom CLI Group
# =============================================================================


class MarkitGroup(click.Group):
    """Custom Group that supports main command with arguments and subcommands.

    This allows:
        markit document.docx --llm          # Convert file (main command)
        markit config list                   # Subcommand
    """

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        """Parse arguments, detecting if first arg is a subcommand or file path."""
        # If first non-option arg is a known subcommand, let Group handle it
        if args:
            for i, arg in enumerate(args):
                if arg.startswith("-"):
                    continue
                # First positional argument
                if arg in self.commands:
                    # It's a subcommand
                    break
                else:
                    # It's a file path - store for later use
                    ctx.ensure_object(dict)
                    ctx.obj["_input_path"] = arg
                    # Remove from args so Group doesn't treat it as subcommand
                    args = args[:i] + args[i + 1 :]
                break

        return super().parse_args(ctx, args)

    def format_usage(
        self,
        ctx: Context,
        formatter: click.HelpFormatter,
    ) -> None:
        """Custom usage line to show INPUT_PATH argument."""
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] INPUT_PATH [COMMAND]",
        )

    def format_help(self, ctx: Context, formatter: click.HelpFormatter) -> None:
        """Custom help formatting to show INPUT_PATH argument."""
        # Usage
        self.format_usage(ctx, formatter)

        # Help text
        self.format_help_text(ctx, formatter)

        # Arguments section
        with formatter.section("Arguments"):
            formatter.write_dl([("INPUT_PATH", "File or directory to convert")])

        # Options (not format_options which may include epilog)
        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)
        if opts:
            with formatter.section("Options"):
                formatter.write_dl(opts)

        # Commands
        commands = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            commands.append((name, cmd.get_short_help_str(limit=formatter.width)))
        if commands:
            with formatter.section("Commands"):
                formatter.write_dl(commands)


# =============================================================================
# Utility functions
# =============================================================================


class LoggingContext:
    """Context manager for temporarily disabling/re-enabling console logging.

    This provides a clean way to manage loguru console handler lifecycle,
    especially useful for batch processing with Rich progress bars.

    Usage:
        logging_ctx = LoggingContext(console_handler_id, verbose)
        with logging_ctx.suspend_console():
            # Rich progress bar here - no console log conflicts
            ...
        # Console logging automatically restored
    """

    def __init__(self, console_handler_id: int | None, verbose: bool = False) -> None:
        self.original_handler_id = console_handler_id
        self.verbose = verbose
        self._current_handler_id: int | None = console_handler_id
        self._suspended = False

    @property
    def current_handler_id(self) -> int | None:
        """Get the current console handler ID."""
        return self._current_handler_id

    def suspend_console(self) -> LoggingContext:
        """Return self as context manager for suspend/resume."""
        return self

    def __enter__(self) -> LoggingContext:
        """Suspend console logging."""
        if self._current_handler_id is not None and not self._suspended:
            try:
                logger.remove(self._current_handler_id)
                self._suspended = True
            except ValueError:
                pass  # Handler already removed
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Resume console logging."""
        if self._suspended:
            console_level = "DEBUG" if self.verbose else "INFO"
            self._current_handler_id = logger.add(
                sys.stderr,
                level=console_level,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            )
            self._suspended = False


def setup_logging(
    verbose: bool,
    log_dir: str | None = None,
    log_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> tuple[int | None, Path | None]:
    """Configure logging based on configuration.

    Args:
        verbose: Enable DEBUG level for console output.
        log_dir: Directory for log files. Supports ~ expansion.
                 Can be overridden by MARKIT_LOG_DIR env var.
        log_level: Log level for file output.
        rotation: Log file rotation size.
        retention: Log file retention period.

    Returns:
        Tuple of (console_handler_id, log_file_path).
        Console handler ID can be used to temporarily disable console logging.
        Log file path is None if file logging is disabled.
    """
    from datetime import datetime

    logger.remove()

    # Console output level
    console_level = "DEBUG" if verbose else "INFO"
    console_handler_id = logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )

    # Check environment variable override
    env_log_dir = os.environ.get("MARKIT_LOG_DIR")
    if env_log_dir:
        log_dir = env_log_dir

    # Add file logging (independent handler, not affected by console disable)
    log_file_path: Path | None = None
    if log_dir:
        log_path = Path(log_dir).expanduser()
        log_path.mkdir(parents=True, exist_ok=True)
        # Generate log filename with current timestamp (matching loguru's format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = log_path / f"markit_{timestamp}.log"
        logger.add(
            log_file_path,
            level=log_level,
            rotation=rotation,
            retention=retention,
            serialize=True,
        )

    return console_handler_id, log_file_path


def print_version(ctx: Context, param: Any, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    console.print(f"markit {__version__}")
    ctx.exit(0)


# =============================================================================
# Main CLI app
# =============================================================================


@click.group(
    cls=MarkitGroup,
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
    help="Enable/disable alt text generation for images.",
)
@click.option(
    "--desc/--no-desc",
    default=None,
    help="Enable/disable JSON description file for images.",
)
@click.option(
    "--ocr/--no-ocr",
    default=None,
    help="Enable/disable OCR for scanned documents.",
)
@click.option(
    "--screenshot/--no-screenshot",
    default=None,
    help="Enable/disable page screenshots for PDF/PPTX.",
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
    "--llm-concurrency",
    type=int,
    default=None,
    help="Number of concurrent LLM requests (default from config).",
)
@click.option(
    "--batch-concurrency",
    "-j",
    type=int,
    default=None,
    help="Number of concurrent batch tasks (default from config).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
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
    batch_concurrency: int | None,
    llm_concurrency: int | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Markit - Document to Markdown converter with LLM enhancement.

    Convert various document formats to Markdown with optional LLM-powered
    enhancement for format optimization and image analysis.

    \b
    Presets:
        rich     - LLM + alt + desc + screenshot (complex documents)
        standard - LLM + alt + desc (normal documents)
        minimal  - No enhancement (just convert)

    \b
    Examples:
        markit document.docx                      # Convert single file
        markit document.pdf --preset rich         # Use rich preset
        markit document.pdf --preset rich --ocr   # Rich + OCR for scans
        markit document.pdf --preset rich --no-desc  # Rich without desc
        markit ./docs/ -o ./output/ --resume      # Batch conversion
        markit config list                        # Show configuration
    """
    # If subcommand is invoked, let it handle
    if ctx.invoked_subcommand is not None:
        return

    # Get input path from context (set by MarkitGroup.parse_args)
    ctx.ensure_object(dict)
    input_path_str = ctx.obj.get("_input_path")

    if not input_path_str:
        click.echo(ctx.get_help())
        ctx.exit(0)

    input_path = Path(input_path_str)
    if not input_path.exists():
        console.print(f"[red]Error: Path '{input_path}' does not exist.[/red]")
        ctx.exit(1)

    # Load configuration first
    config_manager = ConfigManager()
    cfg = config_manager.load(config_path=config_path)

    # Setup logging with configuration
    console_handler_id, log_file_path = setup_logging(
        verbose=verbose,
        log_dir=cfg.log.dir,
        log_level=cfg.log.level,
        rotation=cfg.log.rotation,
        retention=cfg.log.retention,
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
    from markit.config import get_preset

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
    if batch_concurrency is not None:
        cfg.batch.concurrency = batch_concurrency
    if llm_concurrency is not None:
        cfg.llm.concurrency = llm_concurrency

    logger.debug(f"Processing: {input_path.resolve()}")
    logger.debug(f"Output directory: {output.resolve()}")

    async def run_workflow() -> None:
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
            )
            return

        # Single file mode
        await process_single_file(input_path, output, cfg, dry_run, log_file_path)

    asyncio.run(run_workflow())


# =============================================================================
# Config subcommands
# =============================================================================


@app.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("list")
def config_list() -> None:
    """Show current effective configuration."""
    manager = ConfigManager()
    cfg = manager.load()

    config_dict = cfg.model_dump(mode="json")
    config_json = json.dumps(config_dict, indent=2, ensure_ascii=False)

    syntax = Syntax(config_json, "json", theme="monokai", line_numbers=False)
    console.print(syntax)


@config.command("path")
def config_path_cmd() -> None:
    """Show configuration file paths."""
    manager = ConfigManager()
    manager.load()

    console.print("[bold]Configuration file search order:[/bold]")
    console.print("  1. --config CLI argument")
    console.print("  2. MARKIT_CONFIG environment variable")
    console.print("  3. ./markit.json (current directory)")
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
        output_path = output_path / "markit.json"

    # Check if file exists (not directory)
    if output_path.exists() and output_path.is_file():
        if not click.confirm(f"{output_path} already exists. Overwrite?"):
            raise click.Abort()

    # Save default config
    saved_path = manager.save(output_path)
    console.print(f"[green]Configuration file created:[/green] {saved_path}")
    console.print("\nEdit this file to customize your settings.")
    console.print("Run 'markit config list' to see the current configuration.")


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

    try:
        manager.set(key, parsed_value)
        manager.save()
        console.print(f"[green]Set {key} = {parsed_value}[/green]")

    except Exception as e:
        console.print(f"[red]Error setting value:[/red] {e}")
        raise SystemExit(1)


# =============================================================================
# Processing functions
# =============================================================================


async def process_single_file(
    input_path: Path,
    output_dir: Path,
    cfg: MarkitConfig,
    dry_run: bool,
    log_file_path: Path | None = None,
) -> None:
    """Process a single file."""
    # Validate file size to prevent DoS
    try:
        validate_file_size(input_path, MAX_DOCUMENT_SIZE)
    except ValueError as e:
        console.print(Panel(f"[red]{e}[/red]", title="Error"))
        raise SystemExit(1)

    # Detect file format
    fmt = detect_format(input_path)
    if fmt == FileFormat.UNKNOWN:
        console.print(
            Panel(
                f"[red]Unsupported file format: {input_path.suffix}[/red]",
                title="Error",
            )
        )
        raise SystemExit(1)

    # Get converter
    converter = get_converter(input_path, config=cfg)
    if converter is None:
        console.print(
            Panel(
                f"[red]No converter available for format: {fmt.value}[/red]",
                title="Error",
            )
        )
        raise SystemExit(1)

    if dry_run:
        console.print(
            Panel(
                f"[yellow]Would convert:[/yellow] {input_path}\n"
                f"[yellow]Format:[/yellow] {fmt.value.upper()}\n"
                f"[yellow]Output:[/yellow] {output_dir / (input_path.name + '.md')}",
                title="Dry Run",
            )
        )
        raise SystemExit(0)

    # Create output directory (symlink safety)
    from markit.security import check_symlink_safety

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    started_at = datetime.now()
    total_llm_cost = 0.0
    error_msg = None

    try:
        # Convert document
        logger.info(f"Converting {input_path.name}...")
        result = await asyncio.to_thread(
            converter.convert,
            input_path,
            output_dir=output_dir,
        )

        # Generate output filename with conflict resolution
        base_output_file = output_dir / f"{input_path.name}.md"
        output_file = resolve_output_path(base_output_file, cfg.output.on_conflict)

        if output_file is None:
            # Skip strategy: file exists, skip processing
            logger.info(f"[SKIP] Output exists: {base_output_file}")
            console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # Process images from markdown (extract base64 if any)
        image_processor = ImageProcessor(config=cfg.image)
        base64_images = image_processor.extract_base64_images(result.markdown)

        # Count screenshots (page/slide images for OCR/LLM)
        page_images = result.metadata.get("page_images", [])
        screenshots_count = len(page_images)

        # Count embedded images (extracted from document content)
        embedded_images_count = len(base64_images)

        if base64_images:
            logger.info(f"Processing {len(base64_images)} embedded images...")
            image_result = image_processor.process_and_save(
                base64_images,
                output_dir=output_dir,
                base_name=input_path.name,
            )
            result.markdown = image_processor.replace_base64_with_paths(
                result.markdown,
                image_result.saved_images,
            )
            embedded_images_count = len(image_result.saved_images)

        # Write output markdown with basic frontmatter
        base_md_content = _add_basic_frontmatter(result.markdown, input_path.name)
        atomic_write_text(output_file, base_md_content)
        logger.info(f"Written output: {output_file}")

        # LLM processing
        llm_usage: dict[str, dict[str, Any]] = {}
        llm_cost = 0.0
        img_analysis: ImageAnalysisResult | None = None

        # Check if OCR+LLM mode
        ocr_llm_mode = result.metadata.get("ocr_llm_mode", False)
        pptx_llm_mode = result.metadata.get("pptx_llm_mode", False)

        if (ocr_llm_mode or pptx_llm_mode) and cfg.llm.enabled:
            mode_name = "PPTX+LLM" if pptx_llm_mode else "OCR+LLM"
            logger.info(f"[LLM] {input_path.name}: Starting {mode_name} processing")
            # Use result.markdown which already has base64 images replaced with assets/ paths
            # Do NOT use extracted_text from metadata as it contains raw base64 data
            extracted_text = result.markdown
            page_images = result.metadata.get("page_images", [])

            # Enhanced document (clean + frontmatter in one flow)
            (
                cleaned_content,
                frontmatter,
                enhance_cost,
                enhance_usage,
            ) = await enhance_document_with_vision(
                extracted_text, page_images, cfg, source=input_path.name
            )
            llm_cost += enhance_cost
            _merge_llm_usage(llm_usage, enhance_usage)

            # Build final content with page image comments
            result.markdown = cleaned_content
            if page_images:
                commented_images = [
                    f"<!-- ![Page {img['page']}](screenshots/{img['name']}) -->"
                    for img in sorted(page_images, key=lambda x: x.get("page", 0))
                ]
                result.markdown += (
                    "\n\n<!-- Page images for reference -->\n"
                    + "\n".join(commented_images)
                )

            # Write LLM version directly (no extra process_with_llm call needed)
            from markit.llm import LLMProcessor

            # Validate image references - remove any LLM-hallucinated non-existent images
            assets_dir = output_dir / "assets"
            if assets_dir.exists():
                result.markdown = ImageProcessor.remove_nonexistent_images(
                    result.markdown, assets_dir
                )

            processor = LLMProcessor(cfg.llm, cfg.prompts)
            llm_output = output_file.with_suffix(".llm.md")
            llm_content = processor.format_llm_output(result.markdown, frontmatter)
            atomic_write_text(llm_output, llm_content)
            logger.info(f"Written LLM version: {llm_output}")

            # Analyze embedded images (only images from this file)
            if (cfg.image.alt_enabled or cfg.image.desc_enabled) and base64_images:
                assets_dir = output_dir / "assets"
                # Only match images from this specific file
                escaped_name = escape_glob_pattern(input_path.name)
                saved_images = (
                    list(assets_dir.glob(f"{escaped_name}*"))
                    if assets_dir.exists()
                    else []
                )
                saved_images = [
                    p
                    for p in saved_images
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                ]
                if saved_images:
                    (
                        result.markdown,
                        image_cost,
                        image_usage,
                        img_analysis,
                    ) = await analyze_images_with_llm(
                        saved_images,
                        result.markdown,
                        output_file,
                        cfg,
                        input_path,
                        concurrency_limit=cfg.llm.concurrency,
                    )
                    llm_cost += image_cost
                    _merge_llm_usage(llm_usage, image_usage)

        elif cfg.llm.enabled:
            # Check if this is a standalone image file
            is_standalone_image = input_path.suffix.lower() in (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            )

            # Get saved images from assets directory (only images from this file)
            assets_dir = output_dir / "assets"
            escaped_name = escape_glob_pattern(input_path.name)
            saved_images = (
                list(assets_dir.glob(f"{escaped_name}*")) if assets_dir.exists() else []
            )
            saved_images = [
                p
                for p in saved_images
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            ]

            if is_standalone_image and saved_images:
                # Standalone image: only run image analysis (single LLM call)
                # analyze_images_with_llm generates frontmatter via _format_standalone_image_markdown
                logger.info(f"[LLM] {input_path.name}: Processing standalone image")
                (
                    _,
                    image_cost,
                    image_usage,
                    img_analysis,
                ) = await analyze_images_with_llm(
                    saved_images,
                    result.markdown,
                    output_file,
                    cfg,
                    input_path,
                    concurrency_limit=cfg.llm.concurrency,
                )
                llm_cost += image_cost
                _merge_llm_usage(llm_usage, image_usage)

                # Write base .md with basic frontmatter
                base_md_content = _add_basic_frontmatter(
                    result.markdown, input_path.name
                )
                atomic_write_text(output_file, base_md_content)
            else:
                # Save original markdown BEFORE LLM processing for base .md file
                # Base .md should not have LLM-generated alt text
                original_markdown = result.markdown

                result.markdown, doc_cost, doc_usage = await process_with_llm(
                    result.markdown, input_path.name, cfg, output_file
                )
                llm_cost += doc_cost
                _merge_llm_usage(llm_usage, doc_usage)

                # Image analysis (if enabled)
                # NOTE: This updates .llm.md file with LLM-generated alt text
                # The base .md file keeps original alt text
                if (cfg.image.alt_enabled or cfg.image.desc_enabled) and saved_images:
                    (
                        _,  # Don't use updated markdown for base .md
                        image_cost,
                        image_usage,
                        img_analysis,
                    ) = await analyze_images_with_llm(
                        saved_images,
                        result.markdown,
                        output_file,
                        cfg,
                        input_path,
                        concurrency_limit=cfg.llm.concurrency,
                    )
                    llm_cost += image_cost
                    _merge_llm_usage(llm_usage, image_usage)

                # Update base .md with basic frontmatter using ORIGINAL markdown
                # This ensures .md keeps original alt text, not LLM-generated
                base_md_content = _add_basic_frontmatter(
                    original_markdown, input_path.name
                )
                atomic_write_text(output_file, base_md_content)

        # Write image descriptions (single file)
        if img_analysis and cfg.image.desc_enabled:
            write_assets_desc_json(output_dir, [img_analysis])

        # Generate report with token usage
        finished_at = datetime.now()
        duration_seconds = (finished_at - started_at).total_seconds()

        total_llm_cost = llm_cost
        total_llm_usage: dict[str, dict[str, Any]] = llm_usage
        total_input_tokens = sum(
            u.get("input_tokens", 0) for u in total_llm_usage.values()
        )
        total_output_tokens = sum(
            u.get("output_tokens", 0) for u in total_llm_usage.values()
        )
        total_requests = sum(u.get("requests", 0) for u in total_llm_usage.values())

        token_status = "estimated" if total_llm_usage else "unknown"

        # Build report with structure consistent with batch mode
        # files is a dict with relative paths as keys
        report = {
            "version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "log_file": str(log_file_path) if log_file_path else None,
            "summary": {
                "total": 1,
                "completed": 1,
                "failed": 0,
                "duration_seconds": duration_seconds,
            },
            "llm_usage": {
                "models": total_llm_usage,
                "total_requests": total_requests,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost_usd": total_llm_cost,
                "token_status": token_status,
            },
            "files": {
                input_path.name: {
                    "status": "completed",
                    "error": None,
                    "output": str(
                        output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled
                        else output_file
                    ),
                    "images_extracted": embedded_images_count,
                    "screenshots": screenshots_count,
                    "duration_seconds": duration_seconds,
                    "llm_usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "cost_usd": total_llm_cost,
                        "token_status": token_status,
                    },
                }
            },
        }

        # Generate report file path with hash-based naming
        # Format: reports/markit.<hash>.report.json
        task_options = {
            "llm_enabled": cfg.llm.enabled,
            "ocr_enabled": cfg.ocr.enabled,
            "screenshot_enabled": cfg.screenshot.enabled,
            "image_alt_enabled": cfg.image.alt_enabled,
            "image_desc_enabled": cfg.image.desc_enabled,
        }
        task_hash = compute_task_hash(input_path, output_dir, task_options)
        report_path = get_report_file_path(
            output_dir, task_hash, cfg.output.on_conflict
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_json(report_path, report)
        logger.info(f"Report saved: {report_path}")

    except Exception as e:
        error_msg = str(e)
        console.print(Panel(f"[red]{error_msg}[/red]", title="Error"))
        raise SystemExit(1) from e

    finally:
        if error_msg:
            logger.warning(f"Failed to process {input_path.name}: {error_msg}")


async def process_with_llm(
    markdown: str,
    source: str,
    cfg: MarkitConfig,
    output_file: Path,
    page_images: list[dict] | None = None,
    processor: LLMProcessor | None = None,
    original_markdown: str | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """Process markdown with LLM and write enhanced version to .llm.md file.

    The LLM-enhanced content is written to output_file with .llm.md suffix.
    Returns the original markdown unchanged for use in base .md file.

    Args:
        markdown: Markdown content to process
        source: Source file name (used as LLM context identifier)
        cfg: Configuration with LLM and prompt settings
        output_file: Base output file path (.llm.md suffix added automatically)
        page_images: Optional page image info for adding commented references
        processor: Optional shared LLMProcessor (created if not provided)
        original_markdown: Original markdown for preserving slide comments
            that may be lost during vision enhancement

    Returns:
        Tuple of (original_markdown, cost_usd, llm_usage):
        - original_markdown: Input markdown unchanged (for base .md file)
        - cost_usd: LLM API cost for this file
        - llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}

    Side Effects:
        Writes LLM-enhanced content to {output_file}.llm.md
    """
    from markit.llm import LLMProcessor

    try:
        if processor is None:
            processor = LLMProcessor(cfg.llm, cfg.prompts)

        # Extract protected content from original if provided
        original_protected = None
        if original_markdown:
            original_protected = processor.extract_protected_content(original_markdown)

        cleaned, frontmatter = await processor.process_document(markdown, source)

        # Restore any content from original that was lost
        # This is a fallback - process_document already uses placeholder protection
        if original_protected:
            cleaned = processor.restore_protected_content(cleaned, original_protected)

        # Validate image references - remove any LLM-hallucinated non-existent images
        assets_dir = output_file.parent / "assets"
        if assets_dir.exists():
            cleaned = ImageProcessor.remove_nonexistent_images(cleaned, assets_dir)

        # Write LLM version
        llm_output = output_file.with_suffix(".llm.md")
        llm_content = processor.format_llm_output(cleaned, frontmatter)

        # Check if page_images comments already exist in content
        # process_document's placeholder protection should preserve them
        # Append missing page image comments
        if page_images:
            page_header = "<!-- Page images for reference -->"
            has_page_images_header = page_header in llm_content

            # Build the complete page images section
            commented_images = [
                f"<!-- ![Page {img['page']}](screenshots/{img['name']}) -->"
                for img in sorted(page_images, key=lambda x: x.get("page", 0))
            ]

            if not has_page_images_header:
                # No header exists, add complete section
                llm_content += "\n\n" + page_header + "\n" + "\n".join(commented_images)
            else:
                # Header exists, check for missing page comments
                import re

                for comment in commented_images:
                    # Check if this specific page is already referenced
                    page_match = re.search(r"!\[Page\s+(\d+)\]", comment)
                    if page_match:
                        page_num = page_match.group(1)
                        # Look for this page number in any form (commented or not)
                        if not re.search(rf"!\[Page\s+{page_num}\]", llm_content):
                            # Append missing page comment
                            llm_content = llm_content.rstrip() + "\n" + comment

        atomic_write_text(llm_output, llm_content)
        logger.info(f"Written LLM version: {llm_output}")

        # Get usage for THIS file only, not global cumulative usage
        cost = processor.get_context_cost(source)
        usage = processor.get_context_usage(source)
        return markdown, cost, usage  # Return original for base .md file

    except Exception as e:
        logger.warning(f"LLM processing failed: {e}")
        return markdown, 0.0, {}


def _format_standalone_image_markdown(
    input_path: Path,
    analysis: ImageAnalysis,
    image_ref_path: str,
    include_frontmatter: bool = False,
) -> str:
    """Format analysis results for a standalone image file.

    Creates a rich markdown document with:
    - Optional frontmatter (for .llm.md files)
    - Title (image filename)
    - Image preview
    - Image description section
    - Extracted text section (if any text was found)

    Args:
        input_path: Original image file path
        analysis: ImageAnalysis result with caption, description, extracted_text
        image_ref_path: Relative path for image reference
        include_frontmatter: Whether to include YAML frontmatter

    Returns:
        Formatted markdown string
    """
    sections = []

    # Frontmatter (for .llm.md files)
    if include_frontmatter:
        from datetime import datetime

        timestamp = datetime.now().astimezone().isoformat()
        frontmatter_lines = [
            "---",
            f"title: {input_path.stem}",
            f"description: {analysis.caption}",
            f"source: {input_path.name}",
            "tags:",
            "- image",
            "- analysis",
            f"markit_processed: {timestamp}",
            "---",
            "",
        ]
        sections.append("\n".join(frontmatter_lines))

    # Title
    sections.append(f"# {input_path.stem}\n")

    # Image preview with alt text
    sections.append(f"![{analysis.caption}]({image_ref_path})\n")

    # Image description section
    if analysis.description:
        desc = analysis.description.strip()
        # Only add section header if description doesn't already start with a header
        if not desc.startswith("#"):
            sections.append("## Image Description\n")
        sections.append(f"{desc}\n")

    # Extracted text section (only if text was found)
    if analysis.extracted_text and analysis.extracted_text.strip():
        sections.append("## Extracted Text\n")
        sections.append(f"```\n{analysis.extracted_text}\n```\n")

    return "\n".join(sections)


async def analyze_images_with_llm(
    image_paths: list[Path],
    markdown: str,
    output_file: Path,
    cfg: MarkitConfig,
    input_path: Path | None = None,
    concurrency_limit: int | None = None,
    processor: LLMProcessor | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]], ImageAnalysisResult | None]:
    """Analyze images with LLM Vision using batch processing.

    Uses batch analysis to reduce LLM calls (10 images per call instead of 1).

    Behavior controlled by config:
    - alt_enabled: Update alt text in markdown
    - desc_enabled: Collect asset descriptions (caller writes JSON)

    Args:
        image_paths: List of image file paths
        markdown: Original markdown content
        output_file: Output markdown file path
        cfg: Configuration
        input_path: Source input file path (for absolute path in JSON)
        concurrency_limit: Max concurrent LLM requests (unused, kept for API compat)
        processor: Optional shared LLMProcessor (created if not provided)

    Returns:
        Tuple of (updated_markdown, cost_usd, llm_usage, image_analysis_result):
        - updated_markdown: Markdown with updated alt text (if alt_enabled)
        - cost_usd: LLM API cost for image analysis
        - llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}
        - image_analysis_result: Analysis data for JSON output (None if desc_enabled=False)
    """
    import re
    from datetime import datetime

    from markit.llm import LLMProcessor

    alt_enabled = cfg.image.alt_enabled
    desc_enabled = cfg.image.desc_enabled

    try:
        if processor is None:
            processor = LLMProcessor(cfg.llm, cfg.prompts)

        # Use unique context for image analysis to track usage separately from doc processing
        # Format: "full_path:images" ensures isolation even for files with same name in different dirs
        # This prevents usage from concurrent files being mixed together
        source_path = (
            str(input_path.resolve()) if input_path else str(output_file.resolve())
        )
        context = f"{source_path}:images"

        # Detect document language from markdown content
        language = _detect_language(markdown)

        # Use batch analysis
        logger.info(f"Analyzing {len(image_paths)} images in batches...")
        analyses = await processor.analyze_images_batch(
            image_paths,
            language=language,
            max_images_per_batch=DEFAULT_MAX_IMAGES_PER_BATCH,
            context=context,
        )

        timestamp = datetime.now().astimezone().isoformat()

        # Collect asset descriptions for JSON output
        asset_descriptions: list[dict[str, Any]] = []

        # Check if this is a standalone image file
        is_standalone_image = (
            input_path is not None
            and input_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            and len(image_paths) == 1
        )

        # Process results (analyses is in same order as image_paths)
        results: list[tuple[Path, ImageAnalysis | None, str]] = []
        for image_path, analysis in zip(image_paths, analyses):
            results.append((image_path, analysis, timestamp))

            # Collect for JSON output (if desc_enabled)
            if desc_enabled:
                asset_descriptions.append(
                    {
                        "asset": str(image_path.resolve()),
                        "alt": analysis.caption,
                        "desc": analysis.description,
                        "text": analysis.extracted_text or "",
                        "llm_usage": analysis.llm_usage or {},
                        "created": timestamp,
                    }
                )

            # Update alt text in markdown (if alt_enabled)
            if alt_enabled and not is_standalone_image:
                old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                markdown = re.sub(old_pattern, new_ref, markdown)

        # Update .llm.md file
        llm_output = output_file.with_suffix(".llm.md")
        if is_standalone_image and results and results[0][1] is not None:
            # For standalone images, write the rich formatted content with frontmatter
            assert input_path is not None
            _, analysis, _ = results[0]
            if analysis:
                rich_content = _format_standalone_image_markdown(
                    input_path,
                    analysis,
                    f"assets/{input_path.name}",
                    include_frontmatter=True,
                )
                # Normalize whitespace (ensure headers have blank lines before/after)
                rich_content = LLMProcessor._normalize_whitespace(rich_content)
                atomic_write_text(llm_output, rich_content)
        elif alt_enabled and llm_output.exists():
            # For other files, just update alt text
            llm_content = llm_output.read_text(encoding="utf-8")
            for image_path, analysis, _ in results:
                if analysis is None:
                    continue
                old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                llm_content = re.sub(old_pattern, new_ref, llm_content)
            atomic_write_text(llm_output, llm_content)

        # Build analysis result for caller to aggregate
        analysis_result: ImageAnalysisResult | None = None
        if desc_enabled and asset_descriptions:
            source_path = str(input_path.resolve()) if input_path else output_file.stem
            analysis_result = ImageAnalysisResult(
                source_file=source_path,
                assets=asset_descriptions,
            )

        # Get usage for THIS file only using context-based tracking
        # This is concurrency-safe: only includes LLM calls tagged with this context
        incremental_usage = processor.get_context_usage(context)
        incremental_cost = processor.get_context_cost(context)

        return (
            markdown,
            incremental_cost,
            incremental_usage,
            analysis_result,
        )

    except Exception as e:
        logger.warning(f"Image analysis failed: {e}")
        return markdown, 0.0, {}, None


async def enhance_document_with_vision(
    extracted_text: str,
    page_images: list[dict],
    cfg: MarkitConfig,
    source: str = "document",
    processor: LLMProcessor | None = None,
) -> tuple[str, str, float, dict[str, dict[str, Any]]]:
    """Enhance document by combining extracted text with page images.

    This is used for OCR+LLM mode where we have:
    1. Text extracted programmatically (pymupdf4llm/markitdown) - accurate content
    2. Page images - visual reference for layout/structure

    The LLM uses both to produce optimized markdown + frontmatter.

    Args:
        extracted_text: Text extracted by pymupdf4llm/markitdown
        page_images: List of page image info dicts with 'path' key
        cfg: Configuration
        source: Source file name for logging context
        processor: Optional shared LLMProcessor (created if not provided)

    Returns:
        Tuple of (cleaned_markdown, frontmatter_yaml, cost_usd, llm_usage)
    """
    from markit.llm import LLMProcessor

    try:
        if processor is None:
            processor = LLMProcessor(cfg.llm, cfg.prompts)

        # Sort images by page number
        def get_page_num(img_info: dict) -> int:
            return img_info.get("page", 0)

        sorted_images = sorted(page_images, key=get_page_num)

        # Convert to Path list
        image_paths = [Path(img["path"]) for img in sorted_images]

        logger.info(
            f"[START] {source}: Enhancing with {len(image_paths)} page images..."
        )

        # Call the combined enhancement method (clean + frontmatter)
        cleaned_content, frontmatter = await processor.enhance_document_complete(
            extracted_text, image_paths, source=source
        )

        # Get usage for THIS file only, not global cumulative usage
        return (
            cleaned_content,
            frontmatter,
            processor.get_context_cost(source),
            processor.get_context_usage(source),
        )

    except Exception as e:
        logger.warning(f"Document enhancement failed: {e}")
        # Return original text with basic frontmatter as fallback
        basic_frontmatter = f"title: {source}\nsource: {source}"
        return extracted_text, basic_frontmatter, 0.0, {}


async def process_batch(
    input_dir: Path,
    output_dir: Path,
    cfg: MarkitConfig,
    resume: bool,
    dry_run: bool,
    verbose: bool = False,
    console_handler_id: int | None = None,
    log_file_path: Path | None = None,
) -> None:
    """Process directory in batch mode."""
    from markit.batch import BatchProcessor, ProcessResult

    # Supported extensions
    extensions = set(EXTENSION_MAP.keys())

    # Build task options for report (before BatchProcessor init for hash calculation)
    # Note: input_dir and output_dir will be converted to absolute paths by init_state()
    task_options: dict[str, Any] = {
        "concurrency": cfg.batch.concurrency,
        "llm_enabled": cfg.llm.enabled,
        "ocr_enabled": cfg.ocr.enabled,
        "screenshot_enabled": cfg.screenshot.enabled,
        "image_alt_enabled": cfg.image.alt_enabled,
        "image_desc_enabled": cfg.image.desc_enabled,
    }
    if cfg.llm.enabled and cfg.llm.model_list:
        task_options["llm_models"] = [
            m.litellm_params.model for m in cfg.llm.model_list
        ]

    batch = BatchProcessor(
        cfg.batch,
        output_dir,
        input_path=input_dir,
        log_file=log_file_path,
        on_conflict=cfg.output.on_conflict,
        task_options=task_options,
    )
    files = batch.discover_files(input_dir, extensions)

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        raise SystemExit(0)

    from markit.security import check_symlink_safety

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        console.print(
            Panel(
                f"[yellow]Would process {len(files)} files[/yellow]\n"
                f"[yellow]Input:[/yellow] {input_dir}\n"
                f"[yellow]Output:[/yellow] {output_dir}",
                title="Dry Run",
            )
        )
        for f in files[:10]:
            console.print(f"  - {f.name}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more")
        raise SystemExit(0)

    # Create shared LLM runtime and processor for batch mode
    shared_processor = None
    if cfg.llm.enabled:
        from markit.llm import LLMProcessor, LLMRuntime

        runtime = LLMRuntime(concurrency=cfg.llm.concurrency)
        shared_processor = LLMProcessor(cfg.llm, cfg.prompts, runtime=runtime)
        logger.info(
            f"Created shared LLMProcessor with concurrency={cfg.llm.concurrency}"
        )

    async def process_file(file_path: Path) -> ProcessResult:
        """Process a single file."""
        import time

        start_time = time.perf_counter()
        logger.info(f"[START] {file_path.name}")

        try:
            # Validate file size
            validate_file_size(file_path, MAX_DOCUMENT_SIZE)

            converter = get_converter(file_path, config=cfg)
            if converter is None:
                logger.warning(
                    f"[SKIP] {file_path.name}: No converter for {file_path.suffix}"
                )
                return ProcessResult(
                    success=False,
                    error=f"No converter for {file_path.suffix}",
                )

            # Calculate relative path to preserve directory structure
            try:
                rel_path = file_path.parent.relative_to(input_dir)
                file_output_dir = output_dir / rel_path
            except ValueError:
                # file_path is not under input_dir (shouldn't happen in batch mode)
                file_output_dir = output_dir

            # Create output subdirectory if needed
            file_output_dir.mkdir(parents=True, exist_ok=True)

            convert_start = time.perf_counter()
            result = await asyncio.to_thread(
                converter.convert,
                file_path,
                output_dir=file_output_dir,
            )
            convert_time = time.perf_counter() - convert_start
            logger.info(f"[CONVERT] {file_path.name}: {convert_time:.2f}s")

            # Generate output filename with conflict resolution
            base_output_file = file_output_dir / f"{file_path.name}.md"
            output_file = resolve_output_path(base_output_file, cfg.output.on_conflict)

            if output_file is None:
                # Skip strategy: file exists, skip processing
                logger.info(f"[SKIP] Output exists: {base_output_file}")
                return ProcessResult(
                    success=True,
                    output_path=str(base_output_file),
                    error="skipped (exists)",
                )

            # Process images
            image_processor = ImageProcessor(config=cfg.image)
            base64_images = image_processor.extract_base64_images(result.markdown)

            # Count screenshots (page/slide images for OCR/LLM)
            page_images = result.metadata.get("page_images", [])
            screenshots_count = len(page_images)

            # Count embedded images (extracted from document content)
            embedded_images_count = len(base64_images)
            image_result = None  # Initialize for later use in PPTX+LLM mode

            if base64_images:
                image_result = image_processor.process_and_save(
                    base64_images,
                    output_dir=file_output_dir,
                    base_name=file_path.name,
                )
                result.markdown = image_processor.replace_base64_with_paths(
                    result.markdown,
                    image_result.saved_images,
                )
                # Also update extracted_text in metadata for PPTX+LLM mode
                if "extracted_text" in result.metadata:
                    result.metadata["extracted_text"] = (
                        image_processor.replace_base64_with_paths(
                            result.metadata["extracted_text"],
                            image_result.saved_images,
                        )
                    )
                embedded_images_count = len(image_result.saved_images)

            # Write base output with basic frontmatter
            base_md_content = _add_basic_frontmatter(result.markdown, file_path.name)
            atomic_write_text(output_file, base_md_content)

            # Image/page processing with LLM Vision
            llm_cost = 0.0
            llm_usage: dict[str, dict[str, Any]] = {}
            img_analysis: ImageAnalysisResult | None = None  # For JSON aggregation
            ocr_llm_mode = result.metadata.get("ocr_llm_mode", False)
            pptx_llm_mode = result.metadata.get("pptx_llm_mode", False)

            # Get saved images from assets directory
            assets_dir = file_output_dir / "assets"
            saved_images: list[Path] = []
            if assets_dir.exists():
                # Match pattern: {input_name}-*.{ext} or {input_name}.*.{ext}
                # Escape special glob characters in filename to prevent injection
                escaped_name = escape_glob_pattern(file_path.name)
                saved_images = list(assets_dir.glob(f"{escaped_name}*"))
                saved_images = [
                    p
                    for p in saved_images
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                ]

            if (ocr_llm_mode or pptx_llm_mode) and cfg.llm.enabled:
                # Enhanced mode: Use extracted text + page images for LLM processing
                mode_name = "PPTX+LLM" if pptx_llm_mode else "OCR+LLM"
                logger.info(f"[LLM] {file_path.name}: Starting {mode_name} processing")
                # Use result.markdown which has base64 images replaced with assets/ paths
                # Do NOT use metadata["extracted_text"] as it may contain raw base64 data
                extracted_text = result.markdown
                page_images = result.metadata.get("page_images", [])
                logger.debug(
                    f"[DEBUG] page_images count={len(page_images)}, "
                    f"image_result={image_result is not None}"
                )

                # Build commented image links
                commented_images_str = ""
                if page_images:
                    commented_images = [
                        f"<!-- ![Page {img['page']}](screenshots/{img['name']}) -->"
                        for img in sorted(page_images, key=lambda x: x.get("page", 0))
                    ]
                    commented_images_str = (
                        "\n\n<!-- Page images for reference -->\n"
                        + "\n".join(commented_images)
                    )

                # Step 1: Write .md file with original extracted text + basic frontmatter
                base_md_content = _add_basic_frontmatter(extracted_text, file_path.name)
                base_md_content += commented_images_str
                atomic_write_text(output_file, base_md_content)
                logger.info(f"Written base: {output_file}")

                # Step 2: LLM enhancement (clean + frontmatter in one flow)
                if page_images:
                    enhance_start = time.perf_counter()
                    (
                        cleaned_content,
                        frontmatter,
                        enhance_cost,
                        enhance_usage,
                    ) = await enhance_document_with_vision(
                        extracted_text,
                        page_images,
                        cfg,
                        source=file_path.name,
                        processor=shared_processor,
                    )
                    enhance_time = time.perf_counter() - enhance_start
                    logger.info(
                        f"[LLM] {file_path.name}: Vision enhancement {enhance_time:.2f}s, ${enhance_cost:.4f}"
                    )
                    llm_cost += enhance_cost
                    _merge_llm_usage(llm_usage, enhance_usage)

                    # LLM may introduce base64 images - strip them
                    # Use saved image path if available, otherwise remove
                    has_base64_before = "data:image" in cleaned_content
                    if image_result is not None and image_result.saved_images:
                        first_image = image_result.saved_images[0]
                        logger.debug(
                            f"[DEBUG] Stripping base64, has_before={has_base64_before}, "
                            f"image_path={first_image.path.name}"
                        )
                        cleaned_content = image_processor.strip_base64_images(
                            cleaned_content,
                            replacement_path=f"assets/{first_image.path.name}",
                        )
                    else:
                        logger.debug(
                            f"[DEBUG] Stripping base64 (no saved images), "
                            f"has_before={has_base64_before}, image_result={image_result}"
                        )
                        cleaned_content = image_processor.strip_base64_images(
                            cleaned_content
                        )
                    has_base64_after = "data:image" in cleaned_content
                    logger.debug(f"[DEBUG] After strip: has_base64={has_base64_after}")

                    # Build final content
                    result.markdown = cleaned_content + commented_images_str

                    # Validate image references - remove any LLM-hallucinated non-existent images
                    if assets_dir.exists():
                        result.markdown = ImageProcessor.remove_nonexistent_images(
                            result.markdown, assets_dir
                        )

                    # Write LLM version directly (no extra process_with_llm call needed)
                    llm_output = output_file.with_suffix(".llm.md")
                    assert (
                        shared_processor is not None
                    )  # guaranteed by cfg.llm.enabled check
                    llm_content = shared_processor.format_llm_output(
                        result.markdown, frontmatter
                    )
                    atomic_write_text(llm_output, llm_content)
                    logger.info(f"Written LLM version: {llm_output}")

                # Step 4: Analyze embedded images (not page screenshots)
                if (cfg.image.alt_enabled or cfg.image.desc_enabled) and saved_images:
                    # Filter out page/slide screenshots, only analyze embedded images
                    import re

                    page_pattern = re.compile(
                        r"\.page\d+\.|\.slide\d+\.", re.IGNORECASE
                    )
                    embedded_images = [
                        p for p in saved_images if not page_pattern.search(p.name)
                    ]

                    if embedded_images:
                        img_start = time.perf_counter()
                        (
                            result.markdown,
                            image_cost,
                            image_usage,
                            img_analysis,
                        ) = await analyze_images_with_llm(
                            embedded_images,
                            result.markdown,
                            output_file,
                            cfg,
                            file_path,
                            concurrency_limit=cfg.llm.concurrency,
                            processor=shared_processor,
                        )
                        img_time = time.perf_counter() - img_start
                        logger.info(
                            f"[LLM] {file_path.name}: Embedded image analysis {img_time:.2f}s ({len(embedded_images)} images), ${image_cost:.4f}"
                        )
                        llm_cost += image_cost
                        _merge_llm_usage(llm_usage, image_usage)

            elif cfg.llm.enabled:
                # Check if this is a standalone image file
                is_standalone_image = file_path.suffix.lower() in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                )

                if is_standalone_image and saved_images:
                    # Standalone image: only run image analysis (single LLM call)
                    # analyze_images_with_llm generates frontmatter via _format_standalone_image_markdown
                    logger.info(f"[LLM] {file_path.name}: Processing standalone image")
                    img_start = time.perf_counter()
                    (
                        _,
                        image_cost,
                        image_usage,
                        img_analysis,
                    ) = await analyze_images_with_llm(
                        saved_images,
                        result.markdown,
                        output_file,
                        cfg,
                        file_path,
                        concurrency_limit=cfg.llm.concurrency,
                        processor=shared_processor,
                    )
                    img_time = time.perf_counter() - img_start
                    logger.info(
                        f"[LLM] {file_path.name}: Image analysis {img_time:.2f}s, ${image_cost:.4f}"
                    )
                    llm_cost += image_cost
                    _merge_llm_usage(llm_usage, image_usage)

                    # Write base .md with basic frontmatter
                    base_md_content = _add_basic_frontmatter(
                        result.markdown, file_path.name
                    )
                    atomic_write_text(output_file, base_md_content)
                else:
                    # Normal LLM processing (clean + frontmatter)
                    # Save original markdown BEFORE LLM processing for base .md file
                    # Base .md should not have LLM-generated alt text
                    original_markdown = result.markdown

                    logger.info(
                        f"[LLM] {file_path.name}: Starting standard LLM processing"
                    )
                    llm_start = time.perf_counter()
                    result.markdown, doc_cost, doc_usage = await process_with_llm(
                        result.markdown,
                        file_path.name,
                        cfg,
                        output_file,
                        processor=shared_processor,
                    )
                    llm_time = time.perf_counter() - llm_start
                    logger.info(
                        f"[LLM] {file_path.name}: Standard processing {llm_time:.2f}s, ${doc_cost:.4f}"
                    )
                    llm_cost += doc_cost
                    _merge_llm_usage(llm_usage, doc_usage)

                    # Image analysis mode (if enabled)
                    # NOTE: This updates .llm.md file with LLM-generated alt text
                    # The base .md file keeps original alt text
                    if (
                        cfg.image.alt_enabled or cfg.image.desc_enabled
                    ) and saved_images:
                        img_start = time.perf_counter()
                        (
                            _,  # Don't use updated markdown for base .md
                            image_cost,
                            image_usage,
                            img_analysis,
                        ) = await analyze_images_with_llm(
                            saved_images,
                            result.markdown,
                            output_file,
                            cfg,
                            file_path,
                            concurrency_limit=cfg.llm.concurrency,
                            processor=shared_processor,
                        )
                        img_time = time.perf_counter() - img_start
                        logger.info(
                            f"[LLM] {file_path.name}: Image analysis {img_time:.2f}s ({len(saved_images)} images), ${image_cost:.4f}"
                        )
                        llm_cost += image_cost
                        _merge_llm_usage(llm_usage, image_usage)

                    # Update base .md with basic frontmatter using ORIGINAL markdown
                    # This ensures .md keeps original alt text, not LLM-generated
                    base_md_content = _add_basic_frontmatter(
                        original_markdown, file_path.name
                    )
                    atomic_write_text(output_file, base_md_content)

            total_time = time.perf_counter() - start_time
            logger.info(
                f"[DONE] {file_path.name}: {total_time:.2f}s "
                f"(images={embedded_images_count}, screenshots={screenshots_count}, cost=${llm_cost:.4f})"
            )

            return ProcessResult(
                success=True,
                output_path=str(
                    output_file.with_suffix(".llm.md")
                    if cfg.llm.enabled
                    else output_file
                ),
                images_extracted=embedded_images_count,
                screenshots=screenshots_count,
                llm_cost_usd=llm_cost,
                llm_usage=llm_usage,
                image_analysis_result=img_analysis,
            )

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"[FAIL] {file_path.name}: {e} ({total_time:.2f}s)")
            return ProcessResult(success=False, error=str(e))

    # Run batch processing
    logger.info(
        f"Processing {len(files)} files with concurrency {cfg.batch.concurrency}"
    )

    state = await batch.process_batch(
        files,
        process_file,
        resume=resume,
        options=task_options,
        verbose=verbose,
        console_handler_id=console_handler_id,
    )

    # Print summary
    batch.print_summary()

    # Write aggregated image analysis JSON (if any)
    if batch.image_analysis_results and cfg.image.desc_enabled:
        write_assets_desc_json(output_dir, batch.image_analysis_results)

    # Save report (logging is done inside save_report)
    batch.save_report()

    # Exit with appropriate code
    if state.failed_count > 0:
        raise SystemExit(10)  # PARTIAL_FAILURE


if __name__ == "__main__":
    app()
