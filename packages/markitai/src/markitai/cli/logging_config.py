"""Logging configuration for Markitai CLI.

This module contains logging setup utilities and context managers
for managing log output during CLI operations.

Key features:
- Unified loguru-based logging with consistent formatting
- Intercepts third-party library logs (litellm, httpx, instructor, etc.)
- Suppresses noisy dependency warnings by default
- Clean console output with level-based formatting
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from click import Context
from loguru import logger

if TYPE_CHECKING:
    from rich.console import Console

# Import version for print_version
from markitai import __version__


def _get_console() -> Console:
    """Lazy import to avoid circular dependency."""
    from markitai.cli.console import get_console

    return get_console()


# Third-party loggers to intercept and route to loguru
INTERCEPTED_LOGGERS = [
    # LiteLLM and its components
    "LiteLLM",
    "LiteLLM Router",
    "LiteLLM Proxy",
    "litellm",  # Lowercase variant
    # HTTP clients
    "httpx",
    "httpcore",
    "urllib3",
    # Instructor (structured output)
    "instructor",
    # Playwright (including sub-loggers)
    "playwright",
    "playwright.sync_api",
    "playwright.async_api",
    # Document processing
    "markitdown",
    "markitdown._markitdown",
    "pymupdf",
    "fitz",
    # OCR
    "rapidocr",
    "rapidocr_onnxruntime",
    "onnxruntime",
    # Audio (pydub uses logging)
    "pydub",
    # OpenAI client
    "openai",
    # Other
    "PIL",
    "PIL.Image",
    "charset_normalizer",
    # Async/concurrent
    "asyncio",
    "concurrent.futures",
]


def _suppress_onnx_runtime_logs() -> None:
    """Suppress ONNX Runtime C++ logs via environment variables.

    ONNX Runtime logs directly to stderr in C++, bypassing Python logging.
    Must be called before any ONNX Runtime imports.
    """
    # Suppress ONNX Runtime session logging
    os.environ.setdefault("ORT_LOGGING_LEVEL", "3")  # WARNING level
    os.environ.setdefault("ORT_CPP_LOG_SEVERITY_LEVEL", "3")


# Warning messages to suppress (regex patterns)
SUPPRESSED_WARNINGS = [
    # LiteLLM async cleanup
    r"coroutine 'close_litellm_async_clients' was never awaited",
    # Pydub ffmpeg warning
    r"Couldn't find ffmpeg or avconv",
    r"ffmpeg not found",
    # Pydantic field warnings
    r"Field .* has conflict with protected namespace",
    # PIL/Pillow deprecation
    r"ANTIALIAS is deprecated",
    # httpx warnings
    r"Async methods should be used with an async client",
]


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
            self._current_handler_id = logger.add(
                sys.stderr,
                level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
                filter=lambda record: _should_show_log(record, self.verbose),
            )
            self._suspended = False


class InterceptHandler(logging.Handler):
    """Intercept standard logging and forward to loguru.

    This allows capturing logs from dependencies (litellm, instructor, etc.)
    into our unified logging system.

    Instead of trying to trace call frames (which is fragile),
    we use the record's built-in location info.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Use record's built-in location info instead of frame tracing
        # This is more reliable for intercepted logs
        logger.bind(
            name=record.name,
            module=record.module,
            function=record.funcName,
            line=record.lineno,
        ).opt(exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    verbose: bool,
    log_dir: str | None = None,
    log_level: str = "DEBUG",
    log_format: str = "text",
    rotation: str = "10 MB",
    retention: str = "7 days",
    quiet: bool = False,
) -> tuple[int | None, Path | None]:
    """Configure logging based on configuration.

    This function sets up a unified logging system that:
    - Intercepts logs from all third-party dependencies
    - Suppresses noisy warnings that don't affect functionality
    - Provides consistent formatting across all log sources
    - Routes everything through loguru for unified output

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

    # Suppress ONNX Runtime C++ logs (must be before any imports)
    _suppress_onnx_runtime_logs()

    # Suppress noisy warnings from dependencies
    _setup_warning_filters()

    logger.remove()

    # Console logging: disabled in quiet mode
    # DEBUG goes to file only; console shows INFO+ with filter
    console_handler_id: int | None = None
    if not quiet:
        console_handler_id = logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            filter=lambda record: _should_show_log(record, verbose),
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
        if log_format == "json":
            # Compact JSON format
            logger.add(
                log_file_path,
                level=log_level,
                rotation=rotation,
                retention=retention,
                format='{{"ts":"{time:YYYY-MM-DDTHH:mm:ss}","lvl":"{level.name}","src":"{module}:{line}","msg":"{message}"}}',
            )
        else:
            # Human-readable format (default)
            logger.add(
                log_file_path,
                level=log_level,
                rotation=rotation,
                retention=retention,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {module}:{line: <3} | {message}",
            )

    # Intercept standard logging from all third-party dependencies
    # and route to loguru for unified log handling
    _setup_log_interception()

    return console_handler_id, log_file_path


def _setup_warning_filters() -> None:
    """Configure warning filters to suppress noisy dependency warnings."""
    # Suppress common noisy warnings
    for pattern in SUPPRESSED_WARNINGS:
        warnings.filterwarnings("ignore", message=pattern)

    # Suppress pydantic field warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    # Suppress RuntimeWarning from async cleanup
    warnings.filterwarnings(
        "ignore",
        message=r"coroutine .* was never awaited",
        category=RuntimeWarning,
    )


def _setup_log_interception() -> None:
    """Intercept third-party library logs and route to loguru.

    Intercepted loggers only capture WARNING+ to reduce noise.
    """
    intercept_handler = InterceptHandler()

    for logger_name in INTERCEPTED_LOGGERS:
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger.handlers.clear()  # Remove existing handlers
        stdlib_logger.addHandler(intercept_handler)
        stdlib_logger.propagate = False  # Don't propagate to root logger
        # Only capture WARNING+ from third-party libs to reduce noise
        stdlib_logger.setLevel(logging.WARNING)


def _is_third_party_log(name: str, module: str) -> bool:
    """Check if a log comes from a third-party library.

    Uses exact prefix matching instead of substring matching
    to avoid false positives (e.g., "PIL" matching "compiler").

    Args:
        name: Logger name (e.g., "httpx.client")
        module: Module name (e.g., "client")

    Returns:
        True if the log is from a known third-party library
    """
    name_lower = name.lower()
    module_lower = module.lower()

    for intercepted in INTERCEPTED_LOGGERS:
        intercepted_lower = intercepted.lower()
        # Exact match or prefix match with dot separator
        # e.g., "httpx" matches "httpx" and "httpx.client"
        if name_lower == intercepted_lower or name_lower.startswith(
            f"{intercepted_lower}."
        ):
            return True
        # Module exact match
        if module_lower == intercepted_lower:
            return True

    return False


def _should_show_log(record: Any, verbose: bool) -> bool:
    """Filter function for console logging.

    Console only shows INFO+ (DEBUG goes to file only).
    verbose mode shows more detail in INFO messages, not DEBUG.

    Args:
        record: Loguru Record object
        verbose: Whether verbose mode is enabled

    Returns:
        True if the log should be shown
    """
    level = record["level"].name

    # DEBUG never goes to console (file only)
    if level == "DEBUG":
        return False

    # Always show warnings and above
    if level in ("WARNING", "ERROR", "CRITICAL"):
        return True

    # Get the module/name from the record (bound by InterceptHandler)
    name = record.get("extra", {}).get("name", "")
    module = record.get("extra", {}).get("module", "")

    # Filter out third-party INFO
    if level == "INFO" and _is_third_party_log(name, module):
        return False

    # In non-verbose mode, filter out most INFO logs
    # Only show key milestones: file writes, completions
    if not verbose and level == "INFO":
        msg = record.get("message", "")
        # Only show file write and completion messages
        if not any(
            kw in msg for kw in ["Written", "Saved", "Complete", "finished", "Report"]
        ):
            return False

    return True


def print_version(ctx: Context, param: Any, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    _get_console().print(f"markitai {__version__}")
    ctx.exit(0)
