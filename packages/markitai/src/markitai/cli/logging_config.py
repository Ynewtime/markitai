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

import json
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


def _suppress_mupdf_logs() -> None:
    """Suppress MuPDF C-level logs that bypass Python logging.

    MuPDF (via PyMuPDF) logs directly to stderr, which can clutter CLI output
    with format warnings (e.g., "No common ancestor in structure tree").
    """
    try:
        # PyMuPDF might not be installed in all environments
        import fitz

        if hasattr(fitz, "TOOLS") and hasattr(fitz.TOOLS, "mupdf_display_errors"):
            fitz.TOOLS.mupdf_display_errors(False)
    except ImportError:
        pass


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


def _console_sink(message: Any) -> None:
    """Write a rendered console log line via the shared rich stderr Console.

    Routing through the Console keeps log lines coordinated with an active
    Live display (ui.StageList): rich prints them above the Live region
    instead of tearing through the spinner line and leaving stale frames
    behind. The sink receives colorized output (colorize=True); Text.from_ansi
    preserves the styling and rich strips it when stderr is not a terminal.
    """
    from rich.text import Text

    from markitai.cli.console import get_stderr_console

    get_stderr_console().print(
        Text.from_ansi(str(message).rstrip("\n")), soft_wrap=True
    )


def _should_show_quiet_log(record: Any) -> bool:
    """Filter for the quiet-mode console sink (ERROR+ only).

    Even in quiet mode, third-party retry-loop noise duplicates markitai's
    own [LLM:...] failure summaries and must stay off the console (see
    _is_third_party_retry_noise). This is the sink active in single-URL
    stdout mode, where unfiltered instructor errors previously leaked.
    """
    return not _is_third_party_retry_noise(record, _get_record_field(record, "name"))


def _format_json_log(record: Any) -> str:
    """Loguru format callable producing one JSON object per line.

    json.dumps handles escaping (quotes, backslashes, newlines) in the
    message.  The serialized payload is stashed in record["extra"] because
    the returned string is still treated as a loguru format template.
    """
    record["extra"]["_json"] = json.dumps(
        {
            "ts": record["time"].strftime("%Y-%m-%dT%H:%M:%S"),
            "lvl": record["level"].name,
            "src": f"{record['module']}:{record['line']}",
            "msg": record["message"],
        },
        ensure_ascii=False,
    )
    return "{extra[_json]}\n"


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
        log_format: Log format ("text" or "json").
                    Can be overridden by MARKITAI_LOG_FORMAT env var.
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

    # Suppress MuPDF C-level logs (directly to stderr)
    _suppress_mupdf_logs()

    # Suppress noisy warnings from dependencies
    _setup_warning_filters()

    logger.remove()

    # Console logging: quiet mode only shows ERROR+, normal mode shows INFO+
    # DEBUG goes to file only; console shows INFO+ (or ERROR+ in quiet) with filter
    console_handler_id: int | None = None
    if quiet:
        # In quiet mode, still surface errors so LLM failures aren't invisible
        console_handler_id = logger.add(
            _console_sink,
            level="ERROR",
            format="<level>{message}</level>",
            colorize=True,
            filter=_should_show_quiet_log,
        )
    else:
        console_handler_id = logger.add(
            _console_sink,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            colorize=True,
            filter=lambda record: _should_show_log(record, verbose),
        )

    # Check environment variable overrides
    env_log_dir = os.environ.get("MARKITAI_LOG_DIR")
    if env_log_dir:
        log_dir = env_log_dir

    env_log_format = os.environ.get("MARKITAI_LOG_FORMAT")
    if env_log_format and env_log_format in ("text", "json"):
        log_format = env_log_format

    # Add file logging (independent handler, not affected by console disable)
    log_file_path: Path | None = None
    if log_dir:
        log_path = Path(log_dir).expanduser()
        log_path.mkdir(parents=True, exist_ok=True)
        # Generate log filename with current timestamp (matching loguru's format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file_path = log_path / f"markitai_{timestamp}.log"
        if log_format == "json":
            # Compact JSON format (one JSON object per line)
            logger.add(
                log_file_path,
                level=log_level,
                rotation=rotation,
                retention=retention,
                format=_format_json_log,
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


def _get_record_field(record: Any, key: str) -> str:
    """Read a field from a loguru record, preferring bound extras.

    InterceptHandler.emit() calls logger.log() from inside this module, so
    the record's *native* name/module always resolve to
    "markitai.cli.logging_config" for every intercepted third-party log —
    the actual origin only survives in the bound extras (record.name /
    record.module from the original stdlib LogRecord). Native loguru calls
    from markitai's own code never populate extras, so they fall through to
    the native field, which is correct for them.
    """
    extra_value = record.get("extra", {}).get(key, "")
    if isinstance(extra_value, str) and extra_value:
        return extra_value
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _is_internal_info_noise(record: Any) -> bool:
    """Return whether an INFO log is diagnostic noise for terminal output."""
    message = record.get("message", "")
    name = _get_record_field(record, "name").lower()
    module = _get_record_field(record, "module").lower()

    # Per-call timing summaries (e.g. "document_process: claude-agent/haiku
    # tokens=1574+6729 time=73558ms cost=$0.046399") are exactly what -v
    # users need to see why a stage is taking long — the spinner collapses
    # the whole LLM stage into one static line, and these are otherwise the
    # only signal of per-call latency/retries/fallback. Let them through
    # even though their module would otherwise be filtered as noise below.
    if " time=" in message:
        return False

    noisy_prefixes = (
        "[Router]",
        "[HybridRouter]",
        "[LLM] ",
        "[Fetch] This run's remote extraction services may receive URLs ",
        "Fetching URL:",
        "Fetched via ",
        "Processing URL:",
        "Completed via ",
        "Screenshot saved:",
        "Written output:",
        "Written LLM version",
        "[Core] Written pure Vision output:",
        "Created fallback LLM file with screenshot:",
        "Analyzing ",
    )
    if any(message.startswith(prefix) for prefix in noisy_prefixes):
        return True

    noisy_modules = {"core", "document", "vision"}
    if module in noisy_modules:
        return True

    noisy_names = (
        "markitai.llm.processor",
        "markitai.llm.document",
        "markitai.llm.vision",
        "markitai.workflow.core",
    )
    return any(name.endswith(candidate) for candidate in noisy_names)


# instructor's own internal retry loop (instructor/v2/core/retry.py) logs
# every attempt and the final exhaustion at ERROR level via stdlib logging,
# regardless of whether the failure is retryable or already reported
# elsewhere. markitai's own [LLM:...] failure summaries (processor.py,
# vision.py, document.py) already surface the same failure with call-level
# context, so these are pure duplication on the console. File logs are
# unaffected — this only trims the console filter.
_THIRD_PARTY_RETRY_NOISE_PREFIXES = (
    "API call failed on attempt ",
    "Max retries exceeded. Total attempts:",
)


def _is_third_party_retry_noise(record: Any, name: str) -> bool:
    """Return whether a WARNING/ERROR log is third-party retry-loop noise."""
    if not name.lower().startswith("instructor"):
        return False
    message = record.get("message", "")
    return any(
        message.startswith(prefix) for prefix in _THIRD_PARTY_RETRY_NOISE_PREFIXES
    )


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

    # Get the module/name from the record (bound by InterceptHandler for stdlib logs,
    # or from native loguru record fields for first-party logs).
    name = _get_record_field(record, "name")
    module = _get_record_field(record, "module")

    # Always show warnings and above, except known third-party retry-loop
    # noise that duplicates markitai's own failure summaries (see
    # _is_third_party_retry_noise).
    if level in ("WARNING", "ERROR", "CRITICAL"):
        return not _is_third_party_retry_noise(record, name)

    # Filter out third-party INFO
    if level == "INFO" and _is_third_party_log(name, module):
        return False

    # In non-verbose mode, filter out ALL INFO logs from console
    # User-facing output uses ui.* functions (ui.success, ui.summary, etc.)
    # which write directly to console, not through logger
    if not verbose and level == "INFO":
        return False

    if level == "INFO" and _is_internal_info_noise(record):
        return False

    return True


def print_version(ctx: Context, param: Any, value: bool) -> None:
    """Print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    _get_console().print(f"markitai {__version__}")
    ctx.exit(0)
