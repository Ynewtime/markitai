"""Command-line interface for Markitai."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from markitai.fetch import FetchCache, FetchStrategy
    from markitai.llm import ImageAnalysis, LLMProcessor

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
from rich.panel import Panel
from rich.syntax import Syntax

from markitai import __version__
from markitai.config import ConfigManager, MarkitaiConfig
from markitai.constants import (
    DEFAULT_MAX_IMAGES_PER_BATCH,
    IMAGE_EXTENSIONS,
    MAX_DOCUMENT_SIZE,
)
from markitai.converter import FileFormat, detect_format
from markitai.converter.base import EXTENSION_MAP
from markitai.image import ImageProcessor
from markitai.json_order import order_report
from markitai.security import (
    atomic_write_json,
    atomic_write_text,
    validate_file_size,
)
from markitai.utils.output import resolve_output_path
from markitai.utils.paths import ensure_dir, ensure_screenshots_dir
from markitai.workflow.helpers import (
    add_basic_frontmatter as _add_basic_frontmatter,
)
from markitai.workflow.helpers import (
    create_llm_processor,
    write_images_json,
)
from markitai.workflow.helpers import (
    detect_language as _detect_language,
)
from markitai.workflow.helpers import (
    merge_llm_usage as _merge_llm_usage,
)
from markitai.workflow.single import ImageAnalysisResult

console = Console()
# Separate stderr console for status/progress (doesn't mix with stdout output)
stderr_console = Console(stderr=True)


class ProgressReporter:
    """Progress reporter for single file/URL conversion.

    In non-verbose mode, shows:
    1. Spinner during conversion/processing stages
    2. Completion messages after each stage
    3. Clears all output before final result

    In verbose mode, does nothing (logging handles feedback).
    """

    def __init__(self, enabled: bool = True):
        """Initialize progress reporter.

        Args:
            enabled: Whether to show progress (False in verbose mode)
        """
        self.enabled = enabled
        self._status = None
        self._messages: list[str] = []

    def start_spinner(self, message: str) -> None:
        """Start showing a spinner with message."""
        if not self.enabled:
            return
        self.stop_spinner()  # Stop any existing spinner
        self._status = stderr_console.status(f"[cyan]{message}[/cyan]", spinner="dots")
        self._status.start()

    def stop_spinner(self) -> None:
        """Stop the current spinner."""
        if self._status is not None:
            self._status.stop()
            self._status = None

    def log(self, message: str) -> None:
        """Print a progress message."""
        if not self.enabled:
            return
        self.stop_spinner()
        self._messages.append(message)
        stderr_console.print(f"[dim]{message}[/dim]")

    def clear_and_finish(self) -> None:
        """Clear all progress output before printing final result.

        Uses ANSI escape codes to move cursor up and clear lines.
        """
        if not self.enabled:
            return
        self.stop_spinner()

        # Clear previous messages by moving cursor up and clearing lines
        if self._messages:
            # Move cursor up N lines and clear each line
            for _ in self._messages:
                # Move up one line and clear it
                stderr_console.file.write("\033[A\033[2K")
            stderr_console.file.flush()
            self._messages.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_spinner()
        return False


# URL pattern for detecting URLs
_URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def is_url(s: str) -> bool:
    """Check if string is a URL (http:// or https://)."""
    return bool(_URL_PATTERN.match(s))


def url_to_filename(url: str) -> str:
    """Generate a safe filename from URL.

    Examples:
        https://example.com/page.html -> page.html.md
        https://example.com/path/to/doc -> doc.md
        https://example.com/ -> example_com.md
        https://youtube.com/watch?v=abc -> youtube_com_watch.md
    """
    parsed = urlparse(url)

    # Try to get filename from path
    path = parsed.path.rstrip("/")
    if path:
        # Get last segment of path
        filename = path.split("/")[-1]
        if filename:
            # Sanitize for cross-platform compatibility
            filename = _sanitize_filename(filename)
            return f"{filename}.md"

    # Fallback: use domain name
    domain = parsed.netloc.replace(".", "_").replace(":", "_")
    path_part = parsed.path.strip("/").replace("/", "_")[:50]  # limit length
    if path_part:
        return f"{_sanitize_filename(domain)}_{_sanitize_filename(path_part)}.md"
    return f"{_sanitize_filename(domain)}.md"


def _sanitize_filename(name: str) -> str:
    """Sanitize filename for cross-platform compatibility.

    Removes or replaces characters that are invalid on Windows/Linux/macOS.
    """
    # Characters invalid on Windows: \ / : * ? " < > |
    # Also replace other problematic characters
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    # Remove leading/trailing spaces and dots (Windows issue)
    name = name.strip(". ")
    # Limit length (255 is common max, but leave room for .md extension)
    if len(name) > 200:
        name = name[:200]
    return name or "unnamed"


# Import shared ThreadPoolExecutor shutdown function from utils.executor
# This module provides a global executor shared across all conversion operations
from markitai.utils.executor import shutdown_converter_executor


def compute_task_hash(
    input_path: Path,
    output_dir: Path,
    options: dict[str, Any] | None = None,
) -> str:
    """Compute hash from task input parameters.

    Hash is based on:
    - input_path (resolved)
    - output_dir (resolved)
    - key task options (llm, ocr, etc.)

    This ensures different parameter combinations produce different hashes.

    Args:
        input_path: Input file or directory path
        output_dir: Output directory path
        options: Task options dict (llm, ocr, etc.)

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
                "llm",
                "ocr",
                "screenshot",
                "alt",
                "desc",
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

    Format: reports/markitai.<hash>.report.json
    Respects on_conflict strategy for rename.

    Args:
        output_dir: Output directory
        task_hash: Task hash string
        on_conflict: Conflict resolution strategy

    Returns:
        Path to the report file
    """
    reports_dir = output_dir / "reports"
    base_path = reports_dir / f"markitai.{task_hash}.report.json"

    if not base_path.exists():
        return base_path

    if on_conflict == "skip":
        return base_path  # Will be handled by caller
    elif on_conflict == "overwrite":
        return base_path
    else:  # rename
        seq = 2
        while True:
            new_path = reports_dir / f"markitai.{task_hash}.v{seq}.report.json"
            if not new_path.exists():
                return new_path
            seq += 1


# =============================================================================
# Custom CLI Group
# =============================================================================


class MarkitaiGroup(click.Group):
    """Custom Group that supports main command with arguments and subcommands.

    This allows:
        markitai document.docx --llm          # Convert file (main command)
        markitai urls.urls -o out             # URL list batch (.urls auto-detected)
        markitai config list                  # Subcommand
    """

    # Options that take a path argument (so we skip their values when looking for INPUT)
    _PATH_OPTIONS = {"-o", "--output", "-c", "--config"}

    def parse_args(self, ctx: Context, args: list[str]) -> list[str]:
        """Parse arguments, detecting if first arg is a subcommand or file path."""
        # Find INPUT: first positional arg that's not:
        # - An option flag (starts with -)
        # - A subcommand
        # - A value for a path option
        ctx.ensure_object(dict)
        skip_next = False
        input_idx = None

        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue

            # Check if this is an option that takes a value
            if arg in self._PATH_OPTIONS or arg.startswith(
                tuple(f"{opt}=" for opt in self._PATH_OPTIONS)
            ):
                if "=" not in arg:
                    skip_next = True  # Next arg is the option's value
                continue

            if arg.startswith("-"):
                # Other options (flags or with values)
                # For simplicity, assume they don't need skipping unless it's a known path option
                continue

            # First positional argument
            if arg in self.commands:
                # It's a subcommand - stop looking
                break
            else:
                # It's a file path - store for later use
                ctx.obj["_input_path"] = arg
                input_idx = i
                break

        # Remove INPUT from args so Group doesn't treat it as subcommand
        if input_idx is not None:
            args = args[:input_idx] + args[input_idx + 1 :]

        return super().parse_args(ctx, args)

    def format_usage(
        self,
        ctx: Context,
        formatter: click.HelpFormatter,
    ) -> None:
        """Custom usage line to show INPUT argument."""
        formatter.write_usage(
            ctx.command_path,
            "[OPTIONS] INPUT [COMMAND]",
        )

    def format_help(self, ctx: Context, formatter: click.HelpFormatter) -> None:
        """Custom help formatting to show INPUT argument."""
        # Usage
        self.format_usage(ctx, formatter)

        # Help text
        self.format_help_text(ctx, formatter)

        # Arguments section
        with formatter.section("Arguments"):
            formatter.write_dl(
                [
                    (
                        "INPUT",
                        "File, directory, URL, or .urls file to convert",
                    )
                ]
            )

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


class InterceptHandler(logging.Handler):
    """Intercept standard logging and forward to loguru.

    This allows capturing logs from dependencies (litellm, instructor, etc.)
    into our unified logging system.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    verbose: bool,
    log_dir: str | None = None,
    log_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "7 days",
    quiet: bool = False,
) -> tuple[int | None, Path | None]:
    """Configure logging based on configuration.

    Args:
        verbose: Enable DEBUG level for console output.
        log_dir: Directory for log files. Supports ~ expansion.
                 Can be overridden by MARKITAI_LOG_DIR env var.
        log_level: Log level for file output.
        rotation: Log file rotation size.
        retention: Log file retention period.
        quiet: If True, disable console logging entirely (for single file mode).
               Logs will still be written to file if log_dir is configured.

    Returns:
        Tuple of (console_handler_id, log_file_path).
        Console handler ID can be used to temporarily disable console logging.
        Log file path is None if file logging is disabled.
    """
    from datetime import datetime

    logger.remove()

    # Console logging: disabled in quiet mode, otherwise based on verbose flag
    console_handler_id: int | None = None
    if not quiet:
        console_level = "DEBUG" if verbose else "INFO"
        console_handler_id = logger.add(
            sys.stderr,
            level=console_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        )

    # Check environment variable override
    env_log_dir = os.environ.get("MARKITAI_LOG_DIR")
    if env_log_dir:
        log_dir = env_log_dir

    # Add file logging (independent handler, not affected by console disable)
    log_file_path: Path | None = None
    if log_dir:
        log_path = Path(log_dir).expanduser()
        log_path.mkdir(parents=True, exist_ok=True)
        # Generate log filename with current timestamp (matching loguru's format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = log_path / f"markitai_{timestamp}.log"
        logger.add(
            log_file_path,
            level=log_level,
            rotation=rotation,
            retention=retention,
            serialize=True,
        )

    # Intercept standard logging from dependencies (litellm, instructor, etc.)
    # and route to loguru for unified log handling
    intercept_handler = InterceptHandler()
    for logger_name in ["LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx"]:
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger.handlers.clear()  # Remove existing handlers (e.g., StreamHandler)
        stdlib_logger.addHandler(intercept_handler)
        stdlib_logger.propagate = False  # Don't propagate to root logger

    return console_handler_id, log_file_path


def print_version(ctx: Context, param: Any, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    console.print(f"markitai {__version__}")
    ctx.exit(0)


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
    "--url-concurrency",
    type=int,
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
    quiet_console = is_single_mode and not verbose

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
def config_list() -> None:
    """Show current effective configuration."""
    manager = ConfigManager()
    cfg = manager.load()

    config_dict = cfg.model_dump(mode="json", exclude_none=True)
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
@click.option(
    "--scope",
    type=click.Choice(["project", "global", "all"]),
    default="all",
    help="Cache scope to display (default: all).",
)
def cache_stats(as_json: bool, verbose: bool, limit: int, scope: str) -> None:
    """Show cache statistics."""
    from rich.table import Table

    from markitai.constants import (
        DEFAULT_CACHE_DB_FILENAME,
        DEFAULT_PROJECT_CACHE_DIR,
    )
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
        cache: SQLiteCache, cache_name: str, limit: int, as_json: bool
    ) -> dict[str, Any]:
        """Collect and optionally print verbose cache details."""
        by_model = cache.stats_by_model()
        entries = cache.list_entries(limit)

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
        "project": None,
        "global": None,
        "enabled": cfg.cache.enabled,
    }

    # Check project cache (current directory)
    project_cache: SQLiteCache | None = None
    if scope in ("project", "all"):
        project_cache_path = (
            Path.cwd() / DEFAULT_PROJECT_CACHE_DIR / DEFAULT_CACHE_DB_FILENAME
        )
        if project_cache_path.exists():
            try:
                project_cache = SQLiteCache(
                    project_cache_path, cfg.cache.max_size_bytes
                )
                stats_data["project"] = project_cache.stats()
            except Exception as e:
                stats_data["project"] = {"error": str(e)}

    # Check global cache
    global_cache: SQLiteCache | None = None
    if scope in ("global", "all"):
        global_cache_path = (
            Path(cfg.cache.global_dir).expanduser() / DEFAULT_CACHE_DB_FILENAME
        )
        if global_cache_path.exists():
            try:
                global_cache = SQLiteCache(global_cache_path, cfg.cache.max_size_bytes)
                stats_data["global"] = global_cache.stats()
            except Exception as e:
                stats_data["global"] = {"error": str(e)}

    # Collect verbose data if needed
    if verbose:
        if (
            project_cache
            and stats_data["project"]
            and "error" not in stats_data["project"]
        ):
            verbose_data = print_verbose_details(
                project_cache, "Project Cache", limit, as_json
            )
            stats_data["project"]["by_model"] = verbose_data["by_model"]
            stats_data["project"]["entries"] = verbose_data["entries"]

        if (
            global_cache
            and stats_data["global"]
            and "error" not in stats_data["global"]
        ):
            verbose_data = print_verbose_details(
                global_cache, "Global Cache", limit, as_json
            )
            stats_data["global"]["by_model"] = verbose_data["by_model"]
            stats_data["global"]["entries"] = verbose_data["entries"]

    if as_json:
        # Use soft_wrap=True to prevent rich from breaking long lines
        console.print(
            json.dumps(stats_data, indent=2, ensure_ascii=False), soft_wrap=True
        )
    else:
        console.print("[bold]Cache Statistics[/bold]")
        console.print(f"Enabled: {cfg.cache.enabled}")
        console.print()

        if scope in ("project", "all"):
            if stats_data["project"]:
                p = stats_data["project"]
                if "error" in p:
                    console.print(f"[red]Project cache error:[/red] {p['error']}")
                else:
                    console.print("[bold]Project Cache[/bold]")
                    console.print(f"  Path: {p['db_path']}")
                    console.print(f"  Entries: {p['count']}")
                    console.print(f"  Size: {p['size_mb']} MB / {p['max_size_mb']} MB")
                    console.print()
            else:
                console.print("[dim]No project cache found in current directory[/dim]")
                console.print()

        if scope in ("global", "all"):
            if stats_data["global"]:
                g = stats_data["global"]
                if "error" in g:
                    console.print(f"[red]Global cache error:[/red] {g['error']}")
                else:
                    console.print("[bold]Global Cache[/bold]")
                    console.print(f"  Path: {g['db_path']}")
                    console.print(f"  Entries: {g['count']}")
                    console.print(f"  Size: {g['size_mb']} MB / {g['max_size_mb']} MB")
                    console.print()
            else:
                console.print("[dim]No global cache found[/dim]")


@cache.command("clear")
@click.option(
    "--scope",
    type=click.Choice(["project", "global", "all"]),
    default="project",
    help="Which cache to clear (default: project).",
)
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
def cache_clear(scope: str, include_spa_domains: bool, yes: bool) -> None:
    """Clear cache entries."""
    from markitai.constants import (
        DEFAULT_CACHE_DB_FILENAME,
        DEFAULT_PROJECT_CACHE_DIR,
    )
    from markitai.llm import SQLiteCache

    manager = ConfigManager()
    cfg = manager.load()

    # Confirm if not --yes
    if not yes:
        scope_desc = {
            "project": "project cache (current directory)",
            "global": "global cache (~/.markitai)",
            "all": "ALL caches (project + global)",
        }
        desc = scope_desc[scope]
        if include_spa_domains:
            desc += " + learned SPA domains"
        if not click.confirm(f"Clear {desc}?"):
            console.print("[yellow]Aborted[/yellow]")
            return

    result = {"project": 0, "global": 0, "spa_domains": 0}

    # Clear project cache
    if scope in ("project", "all"):
        project_cache_path = (
            Path.cwd() / DEFAULT_PROJECT_CACHE_DIR / DEFAULT_CACHE_DB_FILENAME
        )
        if project_cache_path.exists():
            try:
                project_cache = SQLiteCache(
                    project_cache_path, cfg.cache.max_size_bytes
                )
                result["project"] = project_cache.clear()
            except Exception as e:
                console.print(f"[red]Failed to clear project cache:[/red] {e}")

    # Clear global cache
    if scope in ("global", "all"):
        global_cache_path = (
            Path(cfg.cache.global_dir).expanduser() / DEFAULT_CACHE_DB_FILENAME
        )
        if global_cache_path.exists():
            try:
                global_cache = SQLiteCache(global_cache_path, cfg.cache.max_size_bytes)
                result["global"] = global_cache.clear()
            except Exception as e:
                console.print(f"[red]Failed to clear global cache:[/red] {e}")

    # Clear SPA domains if requested
    if include_spa_domains:
        from markitai.fetch import get_spa_domain_cache

        try:
            spa_cache = get_spa_domain_cache()
            result["spa_domains"] = spa_cache.clear()
        except Exception as e:
            console.print(f"[red]Failed to clear SPA domains:[/red] {e}")

    # Report results
    total = result["project"] + result["global"]
    if total > 0 or result["spa_domains"] > 0:
        console.print(f"[green]Cleared {total} cache entries[/green]")
        if result["project"] > 0:
            console.print(f"  Project: {result['project']}")
        if result["global"] > 0:
            console.print(f"  Global: {result['global']}")
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


# =============================================================================
# Check dependencies command
# =============================================================================


@app.command("check-deps")
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
    - Tesseract OCR (for scanned document processing)
    - LLM API configuration (for content enhancement)
    """
    import json
    import shutil
    import subprocess

    from rich.panel import Panel
    from rich.table import Table

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
        "install_hint": "npm install -g agent-browser && npx playwright install chromium",
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

    # 3. Check Tesseract OCR
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        try:
            proc = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = (
                proc.stdout.strip().split("\n")[0]
                if proc.returncode == 0
                else "unknown"
            )
            results["tesseract"] = {
                "name": "Tesseract OCR",
                "description": "OCR for scanned documents",
                "status": "ok",
                "message": f"Found at {tesseract_path} ({version})",
                "install_hint": "",
            }
        except Exception as e:
            results["tesseract"] = {
                "name": "Tesseract OCR",
                "description": "OCR for scanned documents",
                "status": "error",
                "message": f"Found but failed to run: {e}",
                "install_hint": "Reinstall tesseract",
            }
    else:
        results["tesseract"] = {
            "name": "Tesseract OCR",
            "description": "OCR for scanned documents",
            "status": "missing",
            "message": "tesseract command not found",
            "install_hint": "apt install tesseract-ocr (Linux) / brew install tesseract (macOS)",
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

    # 5. Check vision model configuration (models with supports_vision=true)
    vision_models = [
        m for m in configured_models if m.model_info and m.model_info.supports_vision
    ]
    if vision_models:
        vision_model_names = [m.litellm_params.model for m in vision_models]
        results["vision-model"] = {
            "name": "Vision Model",
            "description": "Image analysis (alt text, descriptions)",
            "status": "ok",
            "message": f"Configured: {', '.join(vision_model_names[:2])}{'...' if len(vision_model_names) > 2 else ''}",
            "install_hint": "",
        }
    else:
        results["vision-model"] = {
            "name": "Vision Model",
            "description": "Image analysis (alt text, descriptions)",
            "status": "warning",
            "message": "No vision model configured (set model_info.supports_vision=true)",
            "install_hint": "Add supports_vision: true to model_info in model_list",
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


# =============================================================================
# Processing functions
# =============================================================================


async def process_single_file(
    input_path: Path,
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    log_file_path: Path | None = None,
    verbose: bool = False,
) -> None:
    """Process a single file using workflow/core pipeline.

    After conversion completes, outputs the final markdown to stdout.
    If LLM is enabled, outputs .llm.md content; otherwise outputs .md content.
    """
    from datetime import datetime

    from markitai.workflow.core import (
        ConversionContext,
        convert_document_core,
    )

    # Validate file size to prevent DoS
    try:
        validate_file_size(input_path, MAX_DOCUMENT_SIZE)
    except ValueError as e:
        console.print(Panel(f"[red]{e}[/red]", title="Error"))
        raise SystemExit(1)

    # Detect file format for dry-run display
    fmt = detect_format(input_path)
    if fmt == FileFormat.UNKNOWN:
        console.print(
            Panel(
                f"[red]Unsupported file format: {input_path.suffix}[/red]",
                title="Error",
            )
        )
        raise SystemExit(1)

    # Handle dry-run
    if dry_run:
        cache_status = "enabled" if cfg.cache.enabled else "disabled"
        dry_run_msg = (
            f"[yellow]Would convert:[/yellow] {input_path}\n"
            f"[yellow]Format:[/yellow] {fmt.value.upper()}\n"
            f"[yellow]Output:[/yellow] {output_dir / (input_path.name + '.md')}\n"
            f"[yellow]Cache:[/yellow] {cache_status}"
        )
        console.print(Panel(dry_run_msg, title="Dry Run"))
        if cfg.cache.enabled:
            console.print(
                "[dim]Tip: Use 'markitai cache stats -v' to view cached entries[/dim]"
            )
        raise SystemExit(0)

    # Progress reporter for non-verbose mode feedback
    progress = ProgressReporter(enabled=not verbose)
    started_at = datetime.now()
    error_msg = None

    try:
        progress.start_spinner(f"Converting {input_path.name}...")

        # Create conversion context
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=output_dir,
            config=cfg,
            project_dir=output_dir.parent,
        )

        # Run core conversion pipeline
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        if not result.success:
            if result.error:
                raise RuntimeError(result.error)
            raise RuntimeError("Unknown conversion error")

        if result.skip_reason == "exists":
            progress.stop_spinner()
            base_output_file = output_dir / f"{input_path.name}.md"
            console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # Show conversion complete message
        progress.log(f"Converted: {input_path.name}")

        # Write image descriptions (single file)
        if ctx.image_analysis and cfg.image.desc_enabled:
            write_images_json(output_dir, [ctx.image_analysis])

        # Generate report
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()

        input_tokens = sum(u.get("input_tokens", 0) for u in ctx.llm_usage.values())
        output_tokens = sum(u.get("output_tokens", 0) for u in ctx.llm_usage.values())
        requests = sum(u.get("requests", 0) for u in ctx.llm_usage.values())

        report = {
            "version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "log_file": str(log_file_path) if log_file_path else None,
            "summary": {
                "total_documents": 1,
                "completed_documents": 1,
                "failed_documents": 0,
                "duration": duration,
            },
            "llm_usage": {
                "models": ctx.llm_usage,
                "requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": ctx.llm_cost,
            },
            "documents": {
                input_path.name: {
                    "status": "completed",
                    "error": None,
                    "output": str(
                        ctx.output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled and ctx.output_file
                        else ctx.output_file
                    ),
                    "images": ctx.embedded_images_count,
                    "screenshots": ctx.screenshots_count,
                    "duration": duration,
                    "llm_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": ctx.llm_cost,
                    },
                }
            },
        }

        # Generate report file path
        task_options = {
            "llm": cfg.llm.enabled,
            "ocr": cfg.ocr.enabled,
            "screenshot": cfg.screenshot.enabled,
            "alt": cfg.image.alt_enabled,
            "desc": cfg.image.desc_enabled,
        }
        task_hash = compute_task_hash(input_path, output_dir, task_options)
        report_path = get_report_file_path(
            output_dir, task_hash, cfg.output.on_conflict
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_json(report_path, report, order_func=order_report)
        logger.info(f"Report saved: {report_path}")

        # Clear progress output before printing final result
        progress.clear_and_finish()

        # Output final markdown to stdout
        if ctx.output_file:
            final_output_file = (
                ctx.output_file.with_suffix(".llm.md")
                if cfg.llm.enabled
                else ctx.output_file
            )
            if final_output_file.exists():
                final_content = final_output_file.read_text(encoding="utf-8")
                print(final_content)

    except Exception as e:
        error_msg = str(e)
        console.print(Panel(f"[red]{error_msg}[/red]", title="Error"))
        sys.exit(1)

    finally:
        if error_msg:
            logger.warning(f"Failed to process {input_path.name}: {error_msg}")


async def process_url(
    url: str,
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    verbose: bool,
    log_file_path: Path | None = None,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
) -> None:
    """Process a URL and convert to Markdown.

    Supports multiple fetch strategies:
    - auto: Detect JS-required pages and fallback automatically
    - static: Direct HTTP request via markitdown (fastest)
    - browser: Headless browser via agent-browser (for JS-rendered pages)
    - jina: Jina Reader API (cloud-based, no local dependencies)

    Also supports:
    - LLM enhancement via --llm flag for document cleaning and frontmatter
    - Image downloading and analysis via --alt/--desc flags

    Note: --screenshot and --ocr are not supported for URLs.

    Args:
        url: URL to convert (http:// or https://)
        output_dir: Output directory for the markdown file
        cfg: Configuration
        dry_run: If True, only show what would be done
        verbose: If True, print logs before output
        log_file_path: Path to log file (for report)
        fetch_strategy: Strategy to use for fetching URL content
        explicit_fetch_strategy: If True, strategy was explicitly set via CLI flag
    """
    from markitai.fetch import (
        AgentBrowserNotFoundError,
        FetchError,
        FetchStrategy,
        JinaRateLimitError,
        fetch_url,
    )
    from markitai.image import download_url_images

    # Default to auto strategy if not specified
    if fetch_strategy is None:
        fetch_strategy = FetchStrategy(cfg.fetch.strategy)
    # At this point fetch_strategy is guaranteed to be non-None
    assert fetch_strategy is not None  # for type checker

    # Warn about unsupported/ignored options for URL mode
    # Note: --alt and --desc are now supported (images will be downloaded)
    # --screenshot is now supported for URLs (captures full-page screenshot via browser)
    # --ocr is not applicable for URLs
    if cfg.ocr.enabled:
        logger.warning("[URL] --ocr is not supported for URL conversion, ignored")

    # Generate output filename from URL
    filename = url_to_filename(url)

    if dry_run:
        llm_status = "enabled" if cfg.llm.enabled else "disabled"
        cache_status = "enabled" if cfg.cache.enabled else "disabled"
        fetch_strategy_str = fetch_strategy.value if fetch_strategy else "auto"
        dry_run_msg = (
            f"[yellow]Would convert URL:[/yellow] {url}\n"
            f"[yellow]Output:[/yellow] {output_dir / filename}\n"
            f"[yellow]Fetch strategy:[/yellow] {fetch_strategy_str}\n"
            f"[yellow]LLM:[/yellow] {llm_status}\n"
            f"[yellow]Cache:[/yellow] {cache_status}"
        )
        console.print(Panel(dry_run_msg, title="Dry Run"))
        if cfg.cache.enabled:
            console.print(
                "[dim]Tip: Use 'markitai cache stats -v' to view cached entries[/dim]"
            )
        raise SystemExit(0)

    # Create output directory
    from markitai.security import check_symlink_safety

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

    from datetime import datetime

    started_at = datetime.now()
    llm_cost = 0.0
    llm_usage: dict[str, dict[str, Any]] = {}

    # Progress reporter for non-verbose mode feedback
    progress = ProgressReporter(enabled=not verbose)

    # Track cache hit for reporting
    fetch_cache_hit = False

    # Initialize fetch cache if caching is enabled
    fetch_cache: FetchCache | None = None
    if cfg.cache.enabled:
        from markitai.fetch import get_fetch_cache

        cache_dir = output_dir.parent / ".markitai"
        fetch_cache = get_fetch_cache(cache_dir, cfg.cache.max_size_bytes)

    try:
        logger.info(f"Fetching URL: {url} (strategy: {fetch_strategy.value})")
        progress.start_spinner(f"Fetching {url}...")

        # Fetch URL using the configured strategy
        # Prepare screenshot options if enabled
        screenshot_dir = (
            ensure_screenshots_dir(output_dir) if cfg.screenshot.enabled else None
        )

        try:
            fetch_result = await fetch_url(
                url,
                fetch_strategy,
                cfg.fetch,
                explicit_strategy=explicit_fetch_strategy,
                cache=fetch_cache,
                skip_read_cache=cfg.cache.no_cache,
                screenshot=cfg.screenshot.enabled,
                screenshot_dir=screenshot_dir,
                screenshot_config=cfg.screenshot if cfg.screenshot.enabled else None,
            )
            fetch_cache_hit = fetch_result.cache_hit
            used_strategy = fetch_result.strategy_used
            original_markdown = fetch_result.content
            screenshot_path = fetch_result.screenshot_path
            logger.info(f"Fetched via {used_strategy}: {url}")
        except AgentBrowserNotFoundError:
            console.print(
                Panel(
                    "[red]agent-browser is not installed.[/red]\n\n"
                    "Install with:\n"
                    "  npm install -g agent-browser\n"
                    "  agent-browser install\n\n"
                    "[dim]Or use --jina for cloud-based rendering.[/dim]",
                    title="Error",
                )
            )
            raise SystemExit(1)
        except JinaRateLimitError:
            console.print(
                Panel(
                    "[red]Jina Reader rate limit exceeded (free tier: 20 RPM).[/red]\n\n"
                    "[dim]Try again later or use --agent-browser for local rendering.[/dim]",
                    title="Error",
                )
            )
            raise SystemExit(1)
        except FetchError as e:
            console.print(Panel(f"[red]{e}[/red]", title="Error"))
            raise SystemExit(1)

        if not original_markdown.strip():
            console.print(
                Panel(
                    f"[red]No content extracted from URL: {url}[/red]\n"
                    "[dim]The page may be empty, require JavaScript, or use an unsupported format.[/dim]",
                    title="Error",
                )
            )
            raise SystemExit(1)

        # Generate output path with conflict resolution
        base_output_file = output_dir / filename
        output_file = resolve_output_path(base_output_file, cfg.output.on_conflict)

        if output_file is None:
            logger.info(f"[SKIP] Output exists: {base_output_file}")
            console.print(f"[yellow]Skipped (exists):[/yellow] {base_output_file}")
            return

        # original_markdown was already set from fetch_result.content above
        markdown_for_llm = original_markdown
        progress.log(f"Fetched via {used_strategy}: {url}")

        # Download images from URLs if --alt or --desc is enabled
        # Only update markdown_for_llm, keep original_markdown unchanged
        downloaded_images: list[Path] = []
        images_count = 0
        screenshots_count = 1 if screenshot_path and screenshot_path.exists() else 0
        img_analysis: ImageAnalysisResult | None = None

        # Log screenshot capture if successful
        if screenshot_path and screenshot_path.exists():
            progress.log(f"Screenshot captured: {screenshot_path.name}")
            logger.info(f"Screenshot saved: {screenshot_path}")

        if cfg.image.alt_enabled or cfg.image.desc_enabled:
            progress.start_spinner("Downloading images...")
            download_result = await download_url_images(
                markdown=original_markdown,
                output_dir=output_dir,
                base_url=url,
                config=cfg.image,
                source_name=url_to_filename(url).replace(".md", ""),
                concurrency=5,
                timeout=30,
            )
            markdown_for_llm = download_result.updated_markdown
            downloaded_images = download_result.downloaded_paths
            images_count = len(downloaded_images)

            if download_result.failed_urls:
                for failed_url in download_result.failed_urls:
                    logger.warning(f"Failed to download image: {failed_url}")

            if downloaded_images:
                progress.log(f"Downloaded {len(downloaded_images)} images")
            else:
                progress.log("No images to download")

        # Write base .md file with original content (no image link replacement)
        base_content = _add_basic_frontmatter(
            original_markdown,
            url,
            fetch_strategy=used_strategy,
            screenshot_path=screenshot_path,
            output_dir=output_dir,
        )
        atomic_write_text(output_file, base_content)
        logger.info(f"Written output: {output_file}")

        # LLM processing (if enabled) uses markdown with local image paths
        final_content = base_content
        if cfg.llm.enabled:
            logger.info(f"[LLM] Processing URL content: {url}")

            # Check if image analysis should run
            should_analyze_images = (
                cfg.image.alt_enabled or cfg.image.desc_enabled
            ) and downloaded_images

            # Check for multi-source content (static + browser + screenshot)
            has_multi_source = (
                fetch_result.static_content is not None
                or fetch_result.browser_content is not None
            )
            has_screenshot = screenshot_path and screenshot_path.exists()
            use_vision_enhancement = has_multi_source and has_screenshot

            if use_vision_enhancement and screenshot_path:
                # Multi-source URL with screenshot: use vision LLM
                progress.start_spinner("Processing with Vision LLM (multi-source)...")
                multi_source_content = _build_multi_source_content(
                    fetch_result.static_content,
                    fetch_result.browser_content,
                    markdown_for_llm,
                )
                logger.info(
                    f"[URL] Using vision enhancement for multi-source URL: {url}"
                )

                _, doc_cost, doc_usage = await _process_url_with_vision(
                    multi_source_content,
                    screenshot_path,
                    url,
                    cfg,
                    output_file,
                    project_dir=output_dir.parent,
                )
                llm_cost += doc_cost
                _merge_llm_usage(llm_usage, doc_usage)

                # Run image analysis if needed
                if should_analyze_images:
                    (
                        _,
                        image_cost,
                        image_usage,
                        img_analysis,
                    ) = await analyze_images_with_llm(
                        downloaded_images,
                        multi_source_content,
                        output_file,
                        cfg,
                        Path(url),
                        concurrency_limit=cfg.llm.concurrency,
                        project_dir=output_dir.parent,
                    )
                    llm_cost += image_cost
                    _merge_llm_usage(llm_usage, image_usage)
                progress.log("LLM processing complete (vision enhanced)")
            elif should_analyze_images:
                # Standard processing with image analysis
                progress.start_spinner("Processing document and images with LLM...")

                # Create parallel tasks
                doc_task = process_with_llm(
                    markdown_for_llm,
                    url,  # Use URL as source identifier
                    cfg,
                    output_file,
                    project_dir=output_dir.parent,
                )
                img_task = analyze_images_with_llm(
                    downloaded_images,
                    markdown_for_llm,
                    output_file,
                    cfg,
                    Path(url),  # Use URL as source path
                    concurrency_limit=cfg.llm.concurrency,
                    project_dir=output_dir.parent,
                )

                # Execute in parallel
                doc_result, img_result = await asyncio.gather(doc_task, img_task)

                # Unpack results
                _, doc_cost, doc_usage = doc_result
                _, image_cost, image_usage, img_analysis = img_result

                llm_cost += doc_cost + image_cost
                _merge_llm_usage(llm_usage, doc_usage)
                _merge_llm_usage(llm_usage, image_usage)
                progress.log("LLM processing complete (document + images)")
            else:
                # Only document processing, no images to analyze
                progress.start_spinner("Processing with LLM...")
                _, doc_cost, doc_usage = await process_with_llm(
                    markdown_for_llm,
                    url,  # Use URL as source identifier
                    cfg,
                    output_file,
                    project_dir=output_dir.parent,
                )
                llm_cost += doc_cost
                _merge_llm_usage(llm_usage, doc_usage)
                progress.log("LLM processing complete")

            # Read the LLM-processed content for stdout output
            llm_output_file = output_file.with_suffix(".llm.md")
            if llm_output_file.exists():
                final_content = llm_output_file.read_text(encoding="utf-8")

        # Write image descriptions (if enabled and images were analyzed)
        if img_analysis and cfg.image.desc_enabled:
            write_images_json(output_dir, [img_analysis])

        # Generate report before final output
        finished_at = datetime.now()
        duration = (finished_at - started_at).total_seconds()

        input_tokens = sum(u.get("input_tokens", 0) for u in llm_usage.values())
        output_tokens = sum(u.get("output_tokens", 0) for u in llm_usage.values())
        requests = sum(u.get("requests", 0) for u in llm_usage.values())

        task_options = {
            "llm": cfg.llm.enabled,
            "url": url,
        }
        task_hash = compute_task_hash(output_dir, output_dir, task_options)
        report_path = get_report_file_path(
            output_dir, task_hash, cfg.output.on_conflict
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine cache hit status (LLM was enabled but no tokens used)
        llm_cache_hit = cfg.llm.enabled and requests == 0

        report = {
            "version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "log_file": str(log_file_path) if log_file_path else None,
            "options": {
                "llm": cfg.llm.enabled,
                "cache": cfg.cache.enabled,
                "fetch_strategy": used_strategy,
                "alt": cfg.image.alt_enabled,
                "desc": cfg.image.desc_enabled,
            },
            "summary": {
                "total_documents": 0,
                "completed_documents": 0,
                "failed_documents": 0,
                "total_urls": 1,
                "completed_urls": 1,
                "failed_urls": 0,
                "duration": duration,
            },
            "llm_usage": {
                "models": llm_usage,
                "requests": requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": llm_cost,
            },
            "urls": {
                url: {
                    "status": "completed",
                    "source_file": "cli",
                    "error": None,
                    "output": str(
                        output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled
                        else output_file
                    ),
                    "fetch_strategy": used_strategy,
                    "fetch_cache_hit": fetch_cache_hit,
                    "llm_cache_hit": llm_cache_hit,
                    "images": images_count,
                    "screenshots": screenshots_count,
                    "duration": duration,
                    "llm_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": llm_cost,
                    },
                }
            },
        }

        atomic_write_json(report_path, report, order_func=order_report)
        logger.info(f"Report saved: {report_path}")

        # Clear progress output before printing final result
        progress.clear_and_finish()

        # Output to stdout (single URL mode behavior, same as single file)
        print(final_content)

    except SystemExit:
        raise
    except Exception as e:
        console.print(Panel(f"[red]{e}[/red]", title="Error"))
        raise SystemExit(1)


async def process_url_batch(
    url_entries: list,  # list[UrlEntry] but imported dynamically
    output_dir: Path,
    cfg: MarkitaiConfig,
    dry_run: bool,
    verbose: bool,
    log_file_path: Path | None = None,
    concurrency: int = 3,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
) -> None:
    """Batch process multiple URLs from a URL list file.

    Shows progress bar similar to file batch processing.
    Each URL is processed concurrently up to the concurrency limit.

    Args:
        url_entries: List of UrlEntry objects from parse_url_list()
        output_dir: Output directory for all markdown files
        cfg: Configuration
        dry_run: If True, only show what would be done
        verbose: If True, enable verbose logging
        log_file_path: Path to log file (for report)
        concurrency: Max concurrent URL processing (default 3)
        fetch_strategy: Strategy to use for fetching URL content
        explicit_fetch_strategy: If True, strategy was explicitly set via CLI flag
    """
    from datetime import datetime

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from markitai.fetch import (
        AgentBrowserNotFoundError,
        FetchError,
        FetchStrategy,
        JinaRateLimitError,
        fetch_url,
        get_fetch_cache,
    )
    from markitai.image import download_url_images
    from markitai.security import check_symlink_safety

    # Default to auto strategy if not specified
    if fetch_strategy is None:
        fetch_strategy = FetchStrategy(cfg.fetch.strategy)
    assert fetch_strategy is not None  # for type checker

    # Dry run: just show what would be done
    if dry_run:
        console.print(
            Panel(
                f"[yellow]Would process {len(url_entries)} URLs[/yellow]\n"
                f"[yellow]Output directory:[/yellow] {output_dir}",
                title="Dry Run - URL Batch",
            )
        )
        for entry in url_entries[:10]:
            filename = entry.output_name or url_to_filename(entry.url).replace(
                ".md", ""
            )
            console.print(f"  - {entry.url} -> {filename}.md")
        if len(url_entries) > 10:
            console.print(f"  ... and {len(url_entries) - 10} more")
        raise SystemExit(0)

    # Create output directory
    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

    # Initialize fetch cache if caching is enabled
    fetch_cache = None
    if cfg.cache.enabled:
        cache_dir = output_dir.parent / ".markitai"
        fetch_cache = get_fetch_cache(cache_dir, cfg.cache.max_size_bytes)

    started_at = datetime.now()
    total_llm_cost = 0.0
    total_llm_usage: dict[str, dict[str, Any]] = {}
    completed = 0
    failed = 0
    results: dict[str, dict] = {}

    semaphore = asyncio.Semaphore(concurrency)

    async def process_single_url(entry, progress_task, progress_obj) -> None:
        """Process a single URL."""
        nonlocal completed, failed, total_llm_cost

        url = entry.url
        custom_name = entry.output_name
        url_fetch_strategy = "unknown"

        async with semaphore:
            try:
                # Generate filename
                if custom_name:
                    filename = f"{custom_name}.md"
                else:
                    filename = url_to_filename(url)

                logger.info(f"Processing URL: {url} (strategy: {fetch_strategy.value})")
                progress_obj.update(progress_task, description=f"[cyan]{url[:50]}...")

                # Fetch URL using the configured strategy
                try:
                    fetch_result = await fetch_url(
                        url,
                        fetch_strategy,
                        cfg.fetch,
                        explicit_strategy=explicit_fetch_strategy,
                        cache=fetch_cache,
                        skip_read_cache=cfg.cache.no_cache,
                    )
                    url_fetch_strategy = fetch_result.strategy_used
                    markdown_content = fetch_result.content
                    cache_status = " [cache]" if fetch_result.cache_hit else ""
                    logger.info(
                        f"Fetched via {url_fetch_strategy}{cache_status}: {url}"
                    )
                except AgentBrowserNotFoundError:
                    logger.error(f"agent-browser not installed for: {url}")
                    results[url] = {
                        "status": "failed",
                        "error": "agent-browser not installed",
                    }
                    failed += 1
                    return
                except JinaRateLimitError:
                    logger.error(f"Jina Reader rate limit exceeded for: {url}")
                    results[url] = {
                        "status": "failed",
                        "error": "Jina Reader rate limit exceeded (20 RPM)",
                    }
                    failed += 1
                    return
                except FetchError as e:
                    logger.error(f"Failed to fetch {url}: {e}")
                    results[url] = {"status": "failed", "error": str(e)}
                    failed += 1
                    return

                if not markdown_content.strip():
                    logger.warning(f"No content extracted from URL: {url}")
                    results[url] = {
                        "status": "failed",
                        "error": "No content extracted",
                    }
                    failed += 1
                    return

                # Download images if --alt or --desc is enabled
                images_count = 0
                if cfg.image.alt_enabled or cfg.image.desc_enabled:
                    download_result = await download_url_images(
                        markdown=markdown_content,
                        output_dir=output_dir,
                        base_url=url,
                        config=cfg.image,
                        source_name=filename.replace(".md", ""),
                        concurrency=5,
                        timeout=30,
                    )
                    markdown_content = download_result.updated_markdown
                    images_count = len(download_result.downloaded_paths)

                # Generate output path with conflict resolution
                base_output_file = output_dir / filename
                output_file = resolve_output_path(
                    base_output_file, cfg.output.on_conflict
                )

                if output_file is None:
                    logger.info(f"[SKIP] Output exists: {base_output_file}")
                    results[url] = {"status": "skipped", "error": "Output exists"}
                    return

                # Write base .md file with frontmatter
                base_content = _add_basic_frontmatter(
                    markdown_content,
                    url,
                    fetch_strategy=url_fetch_strategy,
                    output_dir=output_dir,
                )
                atomic_write_text(output_file, base_content)

                llm_cost = 0.0
                llm_usage: dict[str, dict[str, Any]] = {}

                # LLM processing (if enabled)
                if cfg.llm.enabled:
                    _, doc_cost, doc_usage = await process_with_llm(
                        markdown_content,
                        url,
                        cfg,
                        output_file,
                        project_dir=output_dir.parent,
                    )
                    llm_cost += doc_cost
                    _merge_llm_usage(llm_usage, doc_usage)

                total_llm_cost += llm_cost
                _merge_llm_usage(total_llm_usage, llm_usage)

                results[url] = {
                    "status": "completed",
                    "error": None,
                    "output": str(
                        output_file.with_suffix(".llm.md")
                        if cfg.llm.enabled
                        else output_file
                    ),
                    "fetch_strategy": url_fetch_strategy,
                    "images": images_count,
                }
                completed += 1
                logger.info(f"Completed via {url_fetch_strategy}: {url}")

            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                results[url] = {"status": "failed", "error": str(e)}
                failed += 1

            finally:
                progress_obj.advance(progress_task)

    # Process all URLs with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing URLs...", total=len(url_entries))

        tasks = [process_single_url(entry, task, progress) for entry in url_entries]
        await asyncio.gather(*tasks)

    # Generate report
    finished_at = datetime.now()
    duration = (finished_at - started_at).total_seconds()

    input_tokens = sum(u.get("input_tokens", 0) for u in total_llm_usage.values())
    output_tokens = sum(u.get("output_tokens", 0) for u in total_llm_usage.values())
    requests = sum(u.get("requests", 0) for u in total_llm_usage.values())

    task_options = {
        "llm": cfg.llm.enabled,
        "alt": cfg.image.alt_enabled,
        "desc": cfg.image.desc_enabled,
    }
    task_hash = compute_task_hash(output_dir, output_dir, task_options)
    report_path = get_report_file_path(output_dir, task_hash, cfg.output.on_conflict)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "log_file": str(log_file_path) if log_file_path else None,
        "summary": {
            "total_documents": 0,
            "completed_documents": 0,
            "failed_documents": 0,
            "total_urls": len(url_entries),
            "completed_urls": completed,
            "failed_urls": failed,
            "duration": duration,
        },
        "llm_usage": {
            "models": total_llm_usage,
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": total_llm_cost,
        },
        "urls": results,
    }

    atomic_write_json(report_path, report, order_func=order_report)
    logger.info(f"Report saved: {report_path}")

    # Print summary
    console.print()
    console.print(
        Panel(
            f"[green]Completed:[/green] {completed}\n"
            f"[red]Failed:[/red] {failed}\n"
            f"[dim]Duration:[/dim] {duration:.1f}s\n"
            f"[dim]Report:[/dim] {report_path}",
            title="URL Batch Complete",
        )
    )


def _build_multi_source_content(
    static_content: str | None,
    browser_content: str | None,
    fallback_content: str,
) -> str:
    """Build content from URL fetch result (single-source strategy).

    With the static-first + browser-fallback strategy, we only have one
    valid source at a time. This function simply returns the primary content
    without adding any source labels (which would leak into the final output).

    Args:
        static_content: Content from static/jina fetch (may be None)
        browser_content: Content from browser fetch (may be None)
        fallback_content: Primary content from FetchResult.content

    Returns:
        Single-source content without labels
    """
    # With single-source strategy, fallback_content is already the best source
    # No need to merge or add labels - just return the primary content
    return fallback_content.strip() if fallback_content else ""


async def _process_url_with_vision(
    content: str,
    screenshot_path: Path,
    url: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    processor: LLMProcessor | None = None,
    project_dir: Path | None = None,
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """Process URL content with vision enhancement using screenshot.

    This provides similar functionality to PDF/PPTX vision enhancement,
    using the page screenshot as visual reference for content extraction.

    Args:
        content: Markdown content (may be multi-source combined)
        screenshot_path: Path to the URL screenshot
        url: Original URL (used as source identifier)
        cfg: Configuration
        output_file: Output file path
        processor: Optional shared LLMProcessor
        project_dir: Project directory for cache

    Returns:
        Tuple of (original_content, cost_usd, llm_usage)
    """
    from markitai.workflow.helpers import create_llm_processor

    try:
        if processor is None:
            processor = create_llm_processor(cfg, project_dir=project_dir)

        # Use URL-specific vision enhancement (no slide/page marker protection)
        cleaned_content, frontmatter = await processor.enhance_url_with_vision(
            content, screenshot_path, context=url
        )

        # Format and write LLM output
        llm_output = output_file.with_suffix(".llm.md")
        llm_content = processor.format_llm_output(cleaned_content, frontmatter)

        # Add screenshot reference as comment
        screenshot_comment = (
            f"\n\n<!-- Screenshot for reference -->\n"
            f"<!-- ![Screenshot](screenshots/{screenshot_path.name}) -->"
        )
        llm_content += screenshot_comment

        atomic_write_text(llm_output, llm_content)
        logger.info(f"Written LLM version with vision: {llm_output}")

        # Get usage for this URL
        cost = processor.get_context_cost(url)
        usage = processor.get_context_usage(url)
        return content, cost, usage

    except Exception as e:
        logger.warning(
            f"Vision enhancement failed for {url}: {e}, falling back to standard processing"
        )
        # Fallback to standard processing
        return await process_with_llm(
            content,
            url,
            cfg,
            output_file,
            processor=processor,
            project_dir=project_dir,
        )


async def process_with_llm(
    markdown: str,
    source: str,
    cfg: MarkitaiConfig,
    output_file: Path,
    page_images: list[dict] | None = None,
    processor: LLMProcessor | None = None,
    original_markdown: str | None = None,
    project_dir: Path | None = None,
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
        original_markdown: Original markdown for detecting hallucinated images
        project_dir: Project directory for cache isolation

    Returns:
        Tuple of (original_markdown, cost_usd, llm_usage):
        - original_markdown: Input markdown unchanged (for base .md file)
        - cost_usd: LLM API cost for this file
        - llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}

    Side Effects:
        Writes LLM-enhanced content to {output_file}.llm.md
    """
    try:
        if processor is None:
            processor = create_llm_processor(cfg, project_dir=project_dir)

        cleaned, frontmatter = await processor.process_document(markdown, source)

        # Remove hallucinated image URLs (URLs that don't exist in original)
        original_for_comparison = original_markdown if original_markdown else markdown
        cleaned = ImageProcessor.remove_hallucinated_images(
            cleaned, original_for_comparison
        )

        # Validate local image references - remove non-existent assets
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

    This is a wrapper that delegates to workflow/helpers.format_standalone_image_markdown.

    Args:
        input_path: Original image file path
        analysis: ImageAnalysis result with caption, description, extracted_text
        image_ref_path: Relative path for image reference
        include_frontmatter: Whether to include YAML frontmatter

    Returns:
        Formatted markdown string
    """
    from markitai.workflow.helpers import format_standalone_image_markdown

    return format_standalone_image_markdown(
        input_path, analysis, image_ref_path, include_frontmatter
    )


async def analyze_images_with_llm(
    image_paths: list[Path],
    markdown: str,
    output_file: Path,
    cfg: MarkitaiConfig,
    input_path: Path | None = None,
    concurrency_limit: int | None = None,
    processor: LLMProcessor | None = None,
    project_dir: Path | None = None,
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
        project_dir: Project directory for persistent cache scope

    Returns:
        Tuple of (updated_markdown, cost_usd, llm_usage, image_analysis_result):
        - updated_markdown: Markdown with updated alt text (if alt_enabled)
        - cost_usd: LLM API cost for image analysis
        - llm_usage: Per-model usage {model: {requests, input_tokens, output_tokens, cost_usd}}
        - image_analysis_result: Analysis data for JSON output (None if desc_enabled=False)
    """
    import re
    from datetime import datetime

    alt_enabled = cfg.image.alt_enabled
    desc_enabled = cfg.image.desc_enabled

    try:
        if processor is None:
            processor = create_llm_processor(cfg, project_dir=project_dir)

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
            and input_path.suffix.lower() in IMAGE_EXTENSIONS
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
                from markitai.utils.text import normalize_markdown_whitespace

                rich_content = normalize_markdown_whitespace(rich_content)
                atomic_write_text(llm_output, rich_content)
        elif alt_enabled:
            # For other files, update alt text in .llm.md
            # Wait for .llm.md file to exist (it's written by parallel doc processing)
            max_wait_seconds = 120  # Max wait time
            poll_interval = 0.5  # Check every 0.5 seconds
            waited = 0.0
            while not llm_output.exists() and waited < max_wait_seconds:
                await asyncio.sleep(poll_interval)
                waited += poll_interval

            if llm_output.exists():
                llm_content = llm_output.read_text(encoding="utf-8")
                for image_path, analysis, _ in results:
                    if analysis is None:
                        continue
                    old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
                    new_ref = f"![{analysis.caption}](assets/{image_path.name})"
                    llm_content = re.sub(old_pattern, new_ref, llm_content)
                atomic_write_text(llm_output, llm_content)
            else:
                logger.warning(
                    f"Skipped alt text update: {llm_output} not created within {max_wait_seconds}s"
                )

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
    cfg: MarkitaiConfig,
    source: str = "document",
    processor: LLMProcessor | None = None,
    project_dir: Path | None = None,
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
        project_dir: Project directory for persistent cache scope

    Returns:
        Tuple of (cleaned_markdown, frontmatter_yaml, cost_usd, llm_usage)
    """
    try:
        if processor is None:
            processor = create_llm_processor(cfg, project_dir=project_dir)

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


def _check_vision_model_config(cfg: Any, console: Any, verbose: bool = False) -> None:
    """Check vision model configuration when image analysis is enabled.

    Args:
        cfg: Configuration object
        console: Rich console for output
        verbose: Whether to show extra details
    """
    # Only check if image analysis is enabled
    if not (cfg.image.alt_enabled or cfg.image.desc_enabled):
        return

    # Check if LLM is enabled
    if not cfg.llm.enabled:
        from rich.panel import Panel

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

    def is_vision_model(model_config: Any) -> bool:
        """Check if model supports vision (config override or auto-detect)."""
        if (
            model_config.model_info
            and model_config.model_info.supports_vision is not None
        ):
            return model_config.model_info.supports_vision
        info = get_model_info_cached(model_config.litellm_params.model)
        return info.get("supports_vision", False)

    vision_models = [m for m in cfg.llm.model_list if is_vision_model(m)]

    if not vision_models and cfg.llm.model_list:
        from rich.panel import Panel

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


def _check_agent_browser_for_urls(cfg: Any, console: Any) -> None:
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
        from rich.panel import Panel

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


def _warn_case_sensitivity_mismatches(
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


def _create_process_file(
    cfg: MarkitaiConfig,
    input_dir: Path,
    output_dir: Path,
    preconverted_map: dict[Path, Path],
    shared_processor: LLMProcessor | None,
):
    """Create a process_file function using workflow/core pipeline.

    This factory function creates a closure that captures the batch processing
    context for conversion.

    Args:
        cfg: Markitai configuration
        input_dir: Input directory for relative path calculation
        output_dir: Output directory
        preconverted_map: Map of pre-converted legacy Office files
        shared_processor: Shared LLM processor for batch mode

    Returns:
        An async function that processes a single file and returns ProcessResult
    """
    from markitai.batch import ProcessResult
    from markitai.workflow.core import ConversionContext, convert_document_core

    async def process_file(file_path: Path) -> ProcessResult:
        """Process a single file using workflow/core pipeline."""
        import time

        start_time = time.perf_counter()
        logger.info(f"[START] {file_path.name}")

        try:
            # Calculate relative path to preserve directory structure
            try:
                rel_path = file_path.parent.relative_to(input_dir)
                file_output_dir = output_dir / rel_path
            except ValueError:
                file_output_dir = output_dir

            # Create conversion context
            ctx = ConversionContext(
                input_path=file_path,
                output_dir=file_output_dir,
                config=cfg,
                actual_file=preconverted_map.get(file_path),
                shared_processor=shared_processor,
                project_dir=output_dir.parent,
                use_multiprocess_images=True,
                input_base_dir=input_dir,
            )

            # Run core conversion pipeline
            result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

            total_time = time.perf_counter() - start_time

            if not result.success:
                logger.error(
                    f"[FAIL] {file_path.name}: {result.error} ({total_time:.2f}s)"
                )
                return ProcessResult(success=False, error=result.error)

            if result.skip_reason == "exists":
                logger.info(
                    f"[SKIP] Output exists: {file_output_dir / f'{file_path.name}.md'}"
                )
                return ProcessResult(
                    success=True,
                    output_path=str(file_output_dir / f"{file_path.name}.md"),
                    error="skipped (exists)",
                )

            # Determine cache hit
            cache_hit = cfg.llm.enabled and not ctx.llm_usage

            logger.info(
                f"[DONE] {file_path.name}: {total_time:.2f}s "
                f"(images={ctx.embedded_images_count}, screenshots={ctx.screenshots_count}, cost=${ctx.llm_cost:.4f})"
                + (" [cache]" if cache_hit else "")
            )

            return ProcessResult(
                success=True,
                output_path=str(
                    ctx.output_file.with_suffix(".llm.md")
                    if cfg.llm.enabled and ctx.output_file
                    else ctx.output_file
                ),
                images=ctx.embedded_images_count,
                screenshots=ctx.screenshots_count,
                cost_usd=ctx.llm_cost,
                llm_usage=ctx.llm_usage,
                image_analysis_result=ctx.image_analysis,
                cache_hit=cache_hit,
            )

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"[FAIL] {file_path.name}: {e} ({total_time:.2f}s)")
            return ProcessResult(success=False, error=str(e))

    return process_file


def _create_url_processor(
    cfg: MarkitaiConfig,
    output_dir: Path,
    fetch_strategy: FetchStrategy | None,
    explicit_fetch_strategy: bool,
    shared_processor: LLMProcessor | None = None,
) -> Callable:
    """Create a URL processing function for batch processing.

    Args:
        cfg: Configuration
        output_dir: Output directory
        fetch_strategy: Fetch strategy to use
        explicit_fetch_strategy: Whether strategy was explicitly specified
        shared_processor: Optional shared LLMProcessor

    Returns:
        Async function that processes a single URL and returns ProcessResult
    """
    from markitai.batch import ProcessResult
    from markitai.fetch import (
        AgentBrowserNotFoundError,
        FetchError,
        FetchStrategy,
        JinaRateLimitError,
        fetch_url,
        get_fetch_cache,
    )
    from markitai.image import download_url_images

    # Determine fetch strategy (use config default if not specified)
    _fetch_strategy = fetch_strategy
    if _fetch_strategy is None:
        _fetch_strategy = FetchStrategy(cfg.fetch.strategy)

    # Initialize fetch cache for URL processing
    url_fetch_cache = None
    if cfg.cache.enabled:
        url_cache_dir = output_dir.parent / ".markitai"
        url_fetch_cache = get_fetch_cache(url_cache_dir, cfg.cache.max_size_bytes)

    # Prepare screenshot directory if enabled
    url_screenshot_dir = (
        ensure_screenshots_dir(output_dir) if cfg.screenshot.enabled else None
    )

    async def process_url(
        url: str,
        source_file: Path,
        custom_name: str | None = None,
    ) -> tuple[ProcessResult, dict[str, Any]]:
        """Process a single URL.

        Args:
            url: URL to process
            source_file: Path to the .urls file containing this URL
            custom_name: Optional custom output name

        Returns:
            Tuple of (ProcessResult, extra_info dict with fetch_strategy)
        """
        import time

        start_time = time.perf_counter()
        extra_info: dict[str, Any] = {
            "fetch_strategy": "unknown",
        }

        try:
            # Generate filename
            if custom_name:
                filename = f"{custom_name}.md"
            else:
                filename = url_to_filename(url)

            logger.info(f"[URL] Processing: {url} (strategy: {_fetch_strategy.value})")

            # Fetch URL using the configured strategy
            try:
                fetch_result = await fetch_url(
                    url,
                    _fetch_strategy,
                    cfg.fetch,
                    explicit_strategy=explicit_fetch_strategy,
                    cache=url_fetch_cache,
                    skip_read_cache=cfg.cache.no_cache,
                    screenshot=cfg.screenshot.enabled,
                    screenshot_dir=url_screenshot_dir,
                    screenshot_config=cfg.screenshot
                    if cfg.screenshot.enabled
                    else None,
                )
                extra_info["fetch_strategy"] = fetch_result.strategy_used
                original_markdown = fetch_result.content
                screenshot_path = fetch_result.screenshot_path
                cache_status = " [cache]" if fetch_result.cache_hit else ""
                logger.debug(
                    f"[URL] Fetched via {fetch_result.strategy_used}{cache_status}: {url}"
                )
            except AgentBrowserNotFoundError:
                logger.error(f"[URL] agent-browser not installed for: {url}")
                return ProcessResult(
                    success=False,
                    error="agent-browser not installed",
                ), extra_info
            except JinaRateLimitError:
                logger.error(f"[URL] Jina rate limit exceeded for: {url}")
                return ProcessResult(
                    success=False,
                    error="Jina Reader rate limit exceeded (20 RPM)",
                ), extra_info
            except FetchError as e:
                logger.error(f"[URL] Fetch failed {url}: {e}")
                return ProcessResult(success=False, error=str(e)), extra_info

            if not original_markdown.strip():
                logger.warning(f"[URL] No content: {url}")
                return ProcessResult(
                    success=False,
                    error="No content extracted",
                ), extra_info

            markdown_for_llm = original_markdown

            # Check for multi-source content (static + browser + screenshot)
            has_multi_source = (
                fetch_result.static_content is not None
                or fetch_result.browser_content is not None
            )
            has_screenshot = screenshot_path and screenshot_path.exists()

            logger.debug(
                f"[URL] Multi-source check: static={fetch_result.static_content is not None}, "
                f"browser={fetch_result.browser_content is not None}, "
                f"has_multi_source={has_multi_source}, has_screenshot={has_screenshot}"
            )

            # Download images if --alt or --desc is enabled
            images_count = 0
            screenshots_count = 1 if has_screenshot else 0
            downloaded_images: list[Path] = []

            if has_screenshot and screenshot_path:
                logger.debug(f"[URL] Screenshot captured: {screenshot_path.name}")
            if cfg.image.alt_enabled or cfg.image.desc_enabled:
                download_result = await download_url_images(
                    markdown=original_markdown,
                    output_dir=output_dir,
                    base_url=url,
                    config=cfg.image,
                    source_name=filename.replace(".md", ""),
                    concurrency=5,
                    timeout=30,
                )
                markdown_for_llm = download_result.updated_markdown
                downloaded_images = download_result.downloaded_paths
                images_count = len(downloaded_images)

            # Generate output path
            base_output_file = output_dir / filename
            output_file = resolve_output_path(base_output_file, cfg.output.on_conflict)

            if output_file is None:
                logger.info(f"[URL] Skipped (exists): {base_output_file}")
                return ProcessResult(
                    success=True,
                    output_path=str(base_output_file),
                    error="skipped (exists)",
                ), extra_info

            # Write base .md file with original content
            base_content = _add_basic_frontmatter(
                original_markdown,
                url,
                fetch_strategy=fetch_result.strategy_used if fetch_result else None,
                screenshot_path=screenshot_path,
                output_dir=output_dir,
            )
            atomic_write_text(output_file, base_content)

            # LLM processing uses markdown with local image paths
            url_llm_usage: dict[str, dict[str, Any]] = {}
            llm_cost = 0.0
            img_analysis = None

            if cfg.llm.enabled:
                # Check if image analysis should run
                should_analyze_images = (
                    cfg.image.alt_enabled or cfg.image.desc_enabled
                ) and downloaded_images

                # Check if we should use vision enhancement (multi-source + screenshot)
                use_vision_enhancement = (
                    has_multi_source and has_screenshot and screenshot_path
                )

                if use_vision_enhancement:
                    # Multi-source URL with screenshot: use vision LLM for better content extraction
                    # Build multi-source markdown content for LLM
                    multi_source_content = _build_multi_source_content(
                        fetch_result.static_content,
                        fetch_result.browser_content,
                        markdown_for_llm,  # Fallback primary content
                    )

                    logger.info(
                        f"[URL] Using vision enhancement for multi-source URL: {url}"
                    )

                    # Use vision enhancement with screenshot
                    assert (
                        screenshot_path is not None
                    )  # Guaranteed by use_vision_enhancement check
                    _, cost, url_llm_usage = await _process_url_with_vision(
                        multi_source_content,
                        screenshot_path,
                        url,
                        cfg,
                        output_file,
                        processor=shared_processor,
                        project_dir=output_dir.parent,
                    )
                    llm_cost = cost

                    # Run image analysis in parallel if needed
                    if should_analyze_images:
                        (
                            _,
                            image_cost,
                            image_usage,
                            img_analysis,
                        ) = await analyze_images_with_llm(
                            downloaded_images,
                            multi_source_content,
                            output_file,
                            cfg,
                            Path(url),
                            concurrency_limit=cfg.llm.concurrency,
                            processor=shared_processor,
                            project_dir=output_dir.parent,
                        )
                        _merge_llm_usage(url_llm_usage, image_usage)
                        llm_cost += image_cost
                elif should_analyze_images:
                    # Standard processing with image analysis
                    doc_task = process_with_llm(
                        markdown_for_llm,
                        url,
                        cfg,
                        output_file,
                        processor=shared_processor,
                        project_dir=output_dir.parent,
                    )
                    img_task = analyze_images_with_llm(
                        downloaded_images,
                        markdown_for_llm,
                        output_file,
                        cfg,
                        Path(url),  # Use URL as source path
                        concurrency_limit=cfg.llm.concurrency,
                        processor=shared_processor,
                        project_dir=output_dir.parent,
                    )

                    # Execute in parallel
                    doc_result, img_result = await asyncio.gather(doc_task, img_task)

                    # Unpack results
                    _, cost, url_llm_usage = doc_result
                    _, image_cost, image_usage, img_analysis = img_result

                    _merge_llm_usage(url_llm_usage, image_usage)
                    llm_cost = cost + image_cost
                else:
                    # Only document processing
                    _, cost, url_llm_usage = await process_with_llm(
                        markdown_for_llm,
                        url,
                        cfg,
                        output_file,
                        processor=shared_processor,
                        project_dir=output_dir.parent,
                    )
                    llm_cost = cost

            # Track cache hit: LLM enabled but no usage means cache hit
            is_cache_hit = cfg.llm.enabled and not url_llm_usage

            total_time = time.perf_counter() - start_time
            logger.info(
                f"[URL] Completed via {extra_info['fetch_strategy']}: {url} "
                f"({total_time:.2f}s)" + (" [cache]" if is_cache_hit else "")
            )

            return ProcessResult(
                success=True,
                output_path=str(
                    output_file.with_suffix(".llm.md")
                    if cfg.llm.enabled
                    else output_file
                ),
                images=images_count,
                screenshots=screenshots_count,
                cost_usd=llm_cost,
                llm_usage=url_llm_usage,
                image_analysis_result=img_analysis,
                cache_hit=is_cache_hit,
            ), extra_info

        except Exception as e:
            total_time = time.perf_counter() - start_time
            logger.error(f"[URL] Failed {url}: {e} ({total_time:.2f}s)")
            return ProcessResult(success=False, error=str(e)), extra_info

    return process_url


async def process_batch(
    input_dir: Path,
    output_dir: Path,
    cfg: MarkitaiConfig,
    resume: bool,
    dry_run: bool,
    verbose: bool = False,
    console_handler_id: int | None = None,
    log_file_path: Path | None = None,
    fetch_strategy: FetchStrategy | None = None,
    explicit_fetch_strategy: bool = False,
) -> None:
    """Process directory in batch mode."""
    from markitai.batch import BatchProcessor

    # Supported extensions
    extensions = set(EXTENSION_MAP.keys())

    # Build task options for report (before BatchProcessor init for hash calculation)
    # Note: input_dir and output_dir will be converted to absolute paths by init_state()
    task_options: dict[str, Any] = {
        "concurrency": cfg.batch.concurrency,
        "llm": cfg.llm.enabled,
        "ocr": cfg.ocr.enabled,
        "screenshot": cfg.screenshot.enabled,
        "alt": cfg.image.alt_enabled,
        "desc": cfg.image.desc_enabled,
    }
    if cfg.llm.enabled and cfg.llm.model_list:
        task_options["models"] = [m.litellm_params.model for m in cfg.llm.model_list]

    batch = BatchProcessor(
        cfg.batch,
        output_dir,
        input_path=input_dir,
        log_file=log_file_path,
        on_conflict=cfg.output.on_conflict,
        task_options=task_options,
    )
    files = batch.discover_files(input_dir, extensions)

    # Discover .urls files for URL batch processing
    from markitai.urls import find_url_list_files, parse_url_list

    url_list_files = find_url_list_files(input_dir)
    url_entries_from_files: list = []  # List of (source_file, UrlEntry)

    for url_file in url_list_files:
        try:
            entries = parse_url_list(url_file)
            for entry in entries:
                url_entries_from_files.append((url_file, entry))
            if entries:
                logger.info(f"Found {len(entries)} URLs in {url_file.name}")
        except Exception as e:
            logger.warning(f"Failed to parse URL list {url_file}: {e}")

    # Check agent-browser availability if URLs will be processed
    if url_entries_from_files:
        _check_agent_browser_for_urls(cfg, console)

    if not files and not url_entries_from_files:
        console.print("[yellow]No supported files or URL lists found.[/yellow]")
        raise SystemExit(0)

    # Warn about potential case-sensitivity mismatches in --no-cache-for patterns
    if cfg.cache.no_cache_patterns:
        _warn_case_sensitivity_mismatches(files, input_dir, cfg.cache.no_cache_patterns)

    from markitai.security import check_symlink_safety

    check_symlink_safety(output_dir, allow_symlinks=cfg.output.allow_symlinks)
    ensure_dir(output_dir)

    if dry_run:
        # Build dry run message
        cache_status = "enabled" if cfg.cache.enabled else "disabled"
        dry_run_msg = f"[yellow]Would process {len(files)} files[/yellow]"
        if url_entries_from_files:
            dry_run_msg += f"\n[yellow]Would process {len(url_entries_from_files)} URLs from {len(url_list_files)} .urls files[/yellow]"
        dry_run_msg += f"\n[yellow]Input:[/yellow] {input_dir}\n[yellow]Output:[/yellow] {output_dir}"
        dry_run_msg += f"\n[yellow]Cache:[/yellow] {cache_status}"

        console.print(Panel(dry_run_msg, title="Dry Run"))
        for f in files[:10]:
            console.print(f"  - {f.name}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more files")
        if url_entries_from_files:
            console.print("[dim]URL list files:[/dim]")
            for url_file in url_list_files[:5]:
                console.print(f"  - {url_file.name}")
            if len(url_list_files) > 5:
                console.print(f"  ... and {len(url_list_files) - 5} more .urls files")
        if cfg.cache.enabled:
            console.print(
                "[dim]Tip: Use 'markitai cache stats -v' to view cached entries[/dim]"
            )
        raise SystemExit(0)

    # Record batch start time before any processing (including pre-conversion)
    from datetime import datetime

    batch_started_at = datetime.now().astimezone().isoformat()

    # Start Live display early to capture all logs (including URL processing)
    # This ensures all INFO+ logs go to the panel instead of console
    batch.start_live_display(
        verbose=verbose,
        console_handler_id=console_handler_id,
        total_files=len(files),
        total_urls=len(url_entries_from_files),
    )

    # Pre-convert legacy Office files using batch COM (Windows only)
    # This reduces overhead by starting each Office app only once
    legacy_suffixes = {".doc", ".ppt", ".xls"}
    legacy_files = [f for f in files if f.suffix.lower() in legacy_suffixes]
    preconverted_map: dict[Path, Path] = {}
    preconvert_temp_dir: tempfile.TemporaryDirectory | None = None

    if legacy_files:
        import platform

        if platform.system() == "Windows":
            from markitai.converter.legacy import batch_convert_legacy_files

            # Create temp directory for pre-converted files
            preconvert_temp_dir = tempfile.TemporaryDirectory(
                prefix="markitai_preconv_"
            )
            preconvert_path = Path(preconvert_temp_dir.name)

            logger.info(f"Pre-converting {len(legacy_files)} legacy files...")
            preconverted_map = batch_convert_legacy_files(legacy_files, preconvert_path)
            if preconverted_map:
                logger.info(
                    f"Pre-converted {len(preconverted_map)}/{len(legacy_files)} files with MS Office COM"
                )

    # Create shared LLM runtime and processor for batch mode
    shared_processor = None
    if cfg.llm.enabled:
        from markitai.llm import LLMRuntime

        runtime = LLMRuntime(concurrency=cfg.llm.concurrency)
        # Use output directory's parent as project dir for project-level cache
        project_dir = output_dir.parent if output_dir else Path.cwd()
        shared_processor = create_llm_processor(
            cfg, project_dir=project_dir, runtime=runtime
        )
        logger.info(
            f"Created shared LLMProcessor with concurrency={cfg.llm.concurrency}"
        )

    # Create process_file using workflow/core implementation
    process_file = _create_process_file(
        cfg=cfg,
        input_dir=input_dir,
        output_dir=output_dir,
        preconverted_map=preconverted_map,
        shared_processor=shared_processor,
    )
    logger.debug("Using workflow/core implementation for batch processing")

    # Initialize state for URL tracking
    from markitai.batch import FileStatus, UrlState

    # Group URL entries by source file and collect source file list
    url_sources_set: set[str] = set()
    if url_entries_from_files:
        for source_file, _entry in url_entries_from_files:
            url_sources_set.add(str(source_file))

    # Initialize batch state with files
    if files or url_entries_from_files:
        batch.state = batch.init_state(
            input_dir=input_dir,
            files=files,
            options=task_options,
            started_at=batch_started_at,
        )
        # Add URL source files to state
        batch.state.url_sources = list(url_sources_set)

        # Initialize URL states in batch state
        for source_file, entry in url_entries_from_files:
            batch.state.urls[entry.url] = UrlState(
                url=entry.url,
                source_file=str(source_file),
                status=FileStatus.PENDING,
            )

    # Create URL processor function
    url_processor = None
    if url_entries_from_files:
        url_processor = _create_url_processor(
            cfg=cfg,
            output_dir=output_dir,
            fetch_strategy=fetch_strategy,
            explicit_fetch_strategy=explicit_fetch_strategy,
            shared_processor=shared_processor,
        )

    # Create separate semaphores for file and URL processing
    # This allows file processing and URL fetching to run at their own concurrency levels
    file_semaphore = asyncio.Semaphore(cfg.batch.concurrency)
    url_semaphore = asyncio.Semaphore(cfg.batch.url_concurrency)

    async def process_url_with_state(
        url: str,
        source_file: Path,
        custom_name: str | None,
    ) -> None:
        """Process a URL and update batch state."""
        assert batch.state is not None
        assert url_processor is not None

        url_state = batch.state.urls.get(url)
        if url_state is None:
            return

        # Update state to in_progress
        url_state.status = FileStatus.IN_PROGRESS
        url_state.started_at = datetime.now().astimezone().isoformat()

        start_time = asyncio.get_event_loop().time()

        try:
            async with url_semaphore:
                result, extra_info = await url_processor(url, source_file, custom_name)

            if result.success:
                url_state.status = FileStatus.COMPLETED
                url_state.output = result.output_path
                url_state.fetch_strategy = extra_info.get("fetch_strategy")
                url_state.images = result.images
                url_state.cost_usd = result.cost_usd
                url_state.llm_usage = result.llm_usage
                url_state.cache_hit = result.cache_hit
                # Collect image analysis for JSON output
                if result.image_analysis_result is not None:
                    batch.image_analysis_results.append(result.image_analysis_result)
            else:
                url_state.status = FileStatus.FAILED
                url_state.error = result.error

        except Exception as e:
            url_state.status = FileStatus.FAILED
            url_state.error = str(e)
            logger.error(f"[URL] Failed {url}: {e}")

        finally:
            end_time = asyncio.get_event_loop().time()
            url_state.completed_at = datetime.now().astimezone().isoformat()
            url_state.duration = end_time - start_time

            # Update progress
            batch.update_url_status(url, completed=True)

        # Save state (non-blocking, throttled)
        await asyncio.to_thread(batch.save_state)

    async def process_file_with_state(file_path: Path) -> None:
        """Process a file and update batch state."""
        assert batch.state is not None

        file_key = str(file_path)
        file_state = batch.state.files.get(file_key)

        if file_state is None:
            return

        # Update state to in_progress
        file_state.status = FileStatus.IN_PROGRESS
        file_state.started_at = datetime.now().astimezone().isoformat()

        start_time = asyncio.get_event_loop().time()

        try:
            async with file_semaphore:
                result = await process_file(file_path)

            if result.success:
                file_state.status = FileStatus.COMPLETED
                file_state.output = result.output_path
                file_state.images = result.images
                file_state.screenshots = result.screenshots
                file_state.cost_usd = result.cost_usd
                file_state.llm_usage = result.llm_usage
                file_state.cache_hit = result.cache_hit
                # Collect image analysis for JSON output
                if result.image_analysis_result is not None:
                    batch.image_analysis_results.append(result.image_analysis_result)
            else:
                file_state.status = FileStatus.FAILED
                file_state.error = result.error

        except Exception as e:
            file_state.status = FileStatus.FAILED
            file_state.error = str(e)
            logger.error(f"[FAIL] {file_path.name}: {e}")

        finally:
            end_time = asyncio.get_event_loop().time()
            file_state.completed_at = datetime.now().astimezone().isoformat()
            file_state.duration = end_time - start_time

            # Update progress
            batch.advance_progress()

        # Save state (non-blocking, throttled)
        await asyncio.to_thread(batch.save_state)

    # Run all tasks in parallel (URLs + files)
    state = batch.state
    try:
        if files or url_entries_from_files:
            # Build task list
            all_tasks = []

            # Add URL tasks
            for source_file, entry in url_entries_from_files:
                all_tasks.append(
                    process_url_with_state(entry.url, source_file, entry.output_name)
                )

            # Add file tasks
            for file_path in files:
                all_tasks.append(process_file_with_state(file_path))

            if all_tasks:
                logger.info(
                    f"Processing {len(files)} files and {len(url_entries_from_files)} URLs "
                    f"with concurrency {cfg.batch.concurrency}"
                )

                # Run all tasks in parallel
                await asyncio.gather(*all_tasks, return_exceptions=True)

    finally:
        # Stop Live display and restore console handler
        # This must be done before printing summary
        batch.stop_live_display()

        # Clean up pre-conversion temp directory
        if preconvert_temp_dir is not None:
            preconvert_temp_dir.cleanup()

    if state:
        # Update state timestamp
        state.updated_at = datetime.now().astimezone().isoformat()
        batch.save_state(force=True)

        # Print summary (uses state for URL stats)
        batch.print_summary(
            url_completed=state.completed_urls_count,
            url_failed=state.failed_urls_count,
            url_cache_hits=sum(
                1
                for u in state.urls.values()
                if u.status == FileStatus.COMPLETED and u.cache_hit
            ),
            url_sources=len(state.url_sources),
        )

        # Write aggregated image analysis JSON (if any)
        if batch.image_analysis_results and cfg.image.desc_enabled:
            write_images_json(output_dir, batch.image_analysis_results)

        # Save report (logging is done inside save_report)
        batch.save_report()

    # Exit with appropriate code
    total_failed = (state.failed_count if state else 0) + (
        state.failed_urls_count if state else 0
    )
    if total_failed > 0:
        raise SystemExit(10)  # PARTIAL_FAILURE


if __name__ == "__main__":
    app()
