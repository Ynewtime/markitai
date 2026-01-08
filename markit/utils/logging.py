"""Logging configuration using structlog."""

import logging
import re
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from structlog.typing import EventDict, WrappedLogger

import structlog
from rich.console import Console

# Global console instance for coordinated output
_console: Console | None = None
_log_output: TextIO = sys.stderr

# Pattern for detecting base64 image data
_BASE64_PATTERN = re.compile(
    r"(data:image/[^;]+;base64,)[A-Za-z0-9+/=]{100,}|"  # data URI
    r"[A-Za-z0-9+/=]{500,}"  # plain base64 (long strings)
)

# Noisy third-party loggers to suppress at DEBUG level
_NOISY_LOGGERS = [
    "httpcore",
    "httpx",
    "urllib3",
    "PIL",
    "asyncio",
]


def get_console() -> Console:
    """Get the global Rich console for coordinated output."""
    global _console
    if _console is None:
        _console = Console(stderr=True)
    return _console


def set_log_output(output: TextIO) -> None:
    """Set the log output stream (for Progress console coordination)."""
    global _log_output
    _log_output = output


def _truncate_base64(
    _logger: "WrappedLogger", _method_name: str, event_dict: "EventDict"
) -> "EventDict":
    """Truncate base64 data in log messages to prevent log explosion."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str) and len(value) > 200 and _BASE64_PATTERN.search(value):
            # Check for base64 image data patterns and truncate
            truncated = _BASE64_PATTERN.sub(
                lambda m: f"{m.group(1) if m.group(1) else ''}[BASE64:{len(m.group(0))} chars]",
                value,
            )
            event_dict[key] = truncated
    return event_dict


def _filter_event_dict(
    _logger: "WrappedLogger", _method_name: str, event_dict: "EventDict"
) -> "EventDict":
    """Filter out excessively long values from event dict."""
    max_value_length = 500
    for key, value in list(event_dict.items()):
        if isinstance(value, str) and len(value) > max_value_length:
            event_dict[key] = value[:max_value_length] + f"... [{len(value)} chars total]"
        elif isinstance(value, (bytes, bytearray)) and len(value) > max_value_length:
            event_dict[key] = f"[BINARY DATA: {len(value)} bytes]"
    return event_dict


# Keys that are handled specially by ConsoleRenderer (not user context)
_INTERNAL_KEYS = {"event", "level", "timestamp", "_record", "_from_structlog"}


def _add_separator(
    _logger: "WrappedLogger", _method_name: str, event_dict: "EventDict"
) -> "EventDict":
    """Add a visual separator between event message and context variables."""
    # Check if there are any user-provided context keys (not internal ones)
    has_context = any(k not in _INTERNAL_KEYS for k in event_dict)

    if has_context and "event" in event_dict:
        # Append separator to event message
        event_dict["event"] = f"{event_dict['event']} |"

    return event_dict


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False,
    console: Console | None = None,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output (logs to both console and file)
                  Supports date-based rotation with 7-day retention.
        json_format: If True, output JSON format (for production)
        console: Optional Rich Console for coordinated output with Progress
    """
    global _console, _log_output

    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Use provided console or create new one
    if console is not None:
        _console = console

    # Clear existing handlers on root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Suppress noisy third-party loggers (set them to WARNING unless we're at DEBUG)
    for logger_name in _NOISY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        # Only show WARNING+ from these loggers unless explicitly debugging
        third_party_logger.setLevel(max(log_level, logging.WARNING))

    # Shared processors for structlog (includes base64 truncation)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _truncate_base64,  # Truncate base64 data before logging
        _filter_event_dict,  # Filter overly long values
        _add_separator,  # Add visual separator between message and context
    ]

    # Choose renderer based on format
    if json_format:
        final_processor: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        final_processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
            pad_event_to=0,  # Don't pad event string (removes ugly spaces)
            pad_level=False,  # Don't pad level (removes [info     ] -> [info])
        )

    # Create formatter for standard logging handlers
    # This wraps structlog's processors for use with stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_processor,
        ],
    )

    # Console handler - always output to stderr
    console_handler = logging.StreamHandler(_log_output)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file formatter (consistent with console but no colors)
        if json_format:
            file_formatter = structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        else:
            # For file, use a non-colored console format (same structure as terminal)
            file_formatter = structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(
                        colors=False,  # No colors in file
                        exception_formatter=structlog.dev.plain_traceback,
                        pad_event_to=0,  # Don't pad event string
                        pad_level=False,  # Don't pad level
                    ),
                ],
            )

        # Use TimedRotatingFileHandler for daily log rotation
        # Logs are rotated at midnight, keeping 7 days of history
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=7,  # Keep 7 days of logs
            encoding="utf-8",
        )
        # Set suffix to include date for rotated files
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure structlog to use stdlib logging
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # Allow reconfiguration
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a logger instance.

    Args:
        name: Optional logger name

    Returns:
        A structlog bound logger
    """
    return structlog.get_logger(name)
