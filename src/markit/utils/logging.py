"""Logging configuration using structlog."""

import logging
import re
import sys
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import structlog
from rich.console import Console

if TYPE_CHECKING:
    from structlog.typing import EventDict, WrappedLogger

# Re-export BoundLogger for type hints in other modules
BoundLogger = structlog.stdlib.BoundLogger


# =============================================================================
# Request Context Infrastructure
# =============================================================================

# Context variables for request tracking (thread-safe via contextvars)
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_file_context_var: ContextVar[str | None] = ContextVar("file_context", default=None)
_provider_context_var: ContextVar[str | None] = ContextVar("provider_context", default=None)
_model_context_var: ContextVar[str | None] = ContextVar("model_context", default=None)


def generate_request_id() -> str:
    """Generate a unique 8-character request ID for tracing.

    Returns:
        A short UUID string (8 characters) for request correlation.

    Example:
        >>> request_id = generate_request_id()
        >>> print(request_id)  # e.g., "a1b2c3d4"
    """
    return str(uuid.uuid4())[:8]


def set_request_context(
    request_id: str | None = None,
    file_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Set request context variables for logging.

    These context variables are automatically injected into all log messages
    via the _inject_request_context processor.

    Args:
        request_id: Unique identifier for this request/operation
        file_path: File being processed
        provider: LLM provider being used
        model: LLM model being used
    """
    if request_id is not None:
        _request_id_var.set(request_id)
    if file_path is not None:
        _file_context_var.set(file_path)
    if provider is not None:
        _provider_context_var.set(provider)
    if model is not None:
        _model_context_var.set(model)


def clear_request_context() -> None:
    """Clear all request context variables."""
    _request_id_var.set(None)
    _file_context_var.set(None)
    _provider_context_var.set(None)
    _model_context_var.set(None)


def get_request_id() -> str | None:
    """Get the current request ID from context.

    Returns:
        Current request ID or None if not set.
    """
    return _request_id_var.get()


@contextmanager
def request_context(
    request_id: str | None = None,
    file_path: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Generator[str, None, None]:
    """Context manager for request tracing.

    Automatically sets and clears request context. If no request_id is provided,
    generates a new one.

    Args:
        request_id: Optional request ID (generated if not provided)
        file_path: Optional file path being processed
        provider: Optional LLM provider
        model: Optional LLM model

    Yields:
        The request ID being used

    Example:
        >>> with request_context(file_path="/path/to/doc.pdf") as req_id:
        ...     log.info("Processing started")  # Auto-includes file=/path/to/doc.pdf
    """
    # Save current context
    old_request_id = _request_id_var.get()
    old_file = _file_context_var.get()
    old_provider = _provider_context_var.get()
    old_model = _model_context_var.get()

    # Set new context
    new_request_id = request_id or generate_request_id()
    set_request_context(
        request_id=new_request_id,
        file_path=file_path,
        provider=provider,
        model=model,
    )

    try:
        yield new_request_id
    finally:
        # Restore previous context
        _request_id_var.set(old_request_id)
        _file_context_var.set(old_file)
        _provider_context_var.set(old_provider)
        _model_context_var.set(old_model)


def _inject_request_context(
    _logger: "WrappedLogger", _method_name: str, event_dict: "EventDict"
) -> "EventDict":
    """Inject request context variables into log messages.

    This processor automatically adds request_id, file, provider, and model
    to log messages if they are set in the context and not already present
    in the event dict.

    Args:
        _logger: The wrapped logger (unused)
        _method_name: The log method name (unused)
        event_dict: The event dictionary to modify

    Returns:
        Modified event dictionary with context variables injected
    """
    # Inject request_id if set and not already in event
    request_id = _request_id_var.get()
    if request_id and "request_id" not in event_dict:
        event_dict["request_id"] = request_id

    # Inject file context if set and not already in event
    file_ctx = _file_context_var.get()
    if file_ctx and "file" not in event_dict:
        event_dict["file"] = file_ctx

    # Inject provider context if set and not already in event
    provider_ctx = _provider_context_var.get()
    if provider_ctx and "provider" not in event_dict:
        event_dict["provider"] = provider_ctx

    # Inject model context if set and not already in event
    model_ctx = _model_context_var.get()
    if model_ctx and "model" not in event_dict:
        event_dict["model"] = model_ctx

    return event_dict


# =============================================================================
# Logging Configuration
# =============================================================================


class SafeStreamHandler(logging.StreamHandler):
    """A StreamHandler that handles encoding errors gracefully.

    On Windows, the console may use CP1252 encoding which cannot display
    CJK characters. This handler catches encoding errors and replaces
    problematic characters.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record, handling encoding errors gracefully."""
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Replace characters that can't be encoded
                safe_msg = msg.encode(stream.encoding or "utf-8", errors="replace").decode(
                    stream.encoding or "utf-8", errors="replace"
                )
                stream.write(safe_msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


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
    # Google Gemini SDK loggers (suppress "AFC is enabled" spam)
    "google.genai",
    "google_genai",  # Actual logger name used in models.py
    "google.ai",
    "google.api_core",
    "google.auth",
    "grpc",
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
    console_level: str | None = None,
    file_level: str | None = None,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Root Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output (logs to both console and file)
                  Supports date-based rotation with 7-day retention.
        json_format: If True, output JSON format (for production)
        console: Optional Rich Console for coordinated output with Progress
        console_level: Optional override for console handler level
        file_level: Optional override for file handler level
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

    # Shared processors for structlog (includes base64 truncation and context injection)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _inject_request_context,  # Inject request_id, file, provider, model from context
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
            sort_keys=False,  # Preserve insertion order of log fields
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
    # Use SafeStreamHandler to handle encoding errors gracefully on Windows
    console_handler = SafeStreamHandler(_log_output)

    # Determine console level
    c_level = getattr(logging, console_level.upper(), log_level) if console_level else log_level
    console_handler.setLevel(c_level)

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
                        sort_keys=False,  # Preserve insertion order of log fields
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

        # Determine file level
        f_level = getattr(logging, file_level.upper(), log_level) if file_level else log_level
        file_handler.setLevel(f_level)

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


def create_task_log_path(log_dir: str | Path, prefix: str = "task") -> tuple[str, Path]:
    """Create a unique task log file path with timestamp and UUID.

    This function generates a unique log file path for task-level logging.
    Each task (convert/batch) gets its own log file for easier debugging
    and traceability.

    Args:
        log_dir: Directory to store log files
        prefix: Prefix for the log file name (e.g., "task", "convert", "batch")

    Returns:
        Tuple of (task_id, log_file_path)

    Example:
        >>> task_id, log_path = create_task_log_path(".logs", "convert")
        >>> print(log_path)  # .logs/convert_20260109_143052_a1b2c3d4.log
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    task_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"{prefix}_{timestamp}_{task_id}.log"

    return task_id, log_file


def setup_task_logging(
    log_dir: str | Path,
    prefix: str = "task",
    verbose: bool = False,
) -> tuple[str, Path]:
    """Setup logging for a task with unified behavior.

    This is a convenience function that:
    1. Creates a unique task log file
    2. Configures logging with appropriate levels for console and file
    3. Returns task_id and log_path for tracking

    Console behavior:
    - verbose=False: Only WARNING and above (keeps output clean for spinners/progress)
    - verbose=True: DEBUG and above (full logging for debugging)

    File behavior:
    - Always logs DEBUG level for complete task history

    Args:
        log_dir: Directory to store log files
        prefix: Prefix for the log file name
        verbose: Enable verbose console output

    Returns:
        Tuple of (task_id, log_file_path)
    """
    task_id, log_path = create_task_log_path(log_dir, prefix)

    # Console level based on verbose flag
    console_level = "DEBUG" if verbose else "WARNING"

    setup_logging(
        level="DEBUG",  # Root level allows all logs to flow
        log_file=str(log_path),
        console_level=console_level,
        file_level="DEBUG",  # Always capture full details in file
    )

    return task_id, log_path
