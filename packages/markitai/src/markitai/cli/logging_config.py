"""Logging configuration for Markitai CLI.

This module contains logging setup utilities and context managers
for managing log output during CLI operations.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from click import Context
from loguru import logger
from rich.console import Console

if TYPE_CHECKING:
    pass

# Import version for print_version
from markitai import __version__

console = Console()


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
