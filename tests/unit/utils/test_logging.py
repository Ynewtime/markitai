"""Tests for logging utilities module."""

import io
import logging
from pathlib import Path

import pytest

from markit.utils.logging import (
    SafeStreamHandler,
    _add_separator,
    _filter_event_dict,
    _inject_request_context,
    _truncate_base64,
    clear_request_context,
    create_task_log_path,
    generate_request_id,
    get_console,
    get_logger,
    get_request_id,
    request_context,
    set_log_output,
    set_request_context,
    setup_logging,
    setup_task_logging,
)


class TestGenerateRequestId:
    """Tests for generate_request_id function."""

    def test_returns_string(self):
        """Test that request ID is a string."""
        result = generate_request_id()
        assert isinstance(result, str)

    def test_returns_8_characters(self):
        """Test that request ID is 8 characters long."""
        result = generate_request_id()
        assert len(result) == 8

    def test_generates_unique_ids(self):
        """Test that each call generates a unique ID."""
        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestRequestContext:
    """Tests for request context functions."""

    def test_set_and_get_request_id(self):
        """Test setting and getting request ID."""
        clear_request_context()
        assert get_request_id() is None

        set_request_context(request_id="test123")
        assert get_request_id() == "test123"

        clear_request_context()
        assert get_request_id() is None

    def test_set_request_context_partial(self):
        """Test setting partial context."""
        clear_request_context()
        set_request_context(file_path="/path/to/file")
        # Only file_path should be set, not request_id
        assert get_request_id() is None

    def test_clear_request_context(self):
        """Test clearing all context."""
        set_request_context(
            request_id="test",
            file_path="/path",
            provider="openai",
            model="gpt-4",
        )
        clear_request_context()
        assert get_request_id() is None


class TestRequestContextManager:
    """Tests for request_context context manager."""

    def test_yields_request_id(self):
        """Test that context manager yields request ID."""
        with request_context(request_id="custom_id") as req_id:
            assert req_id == "custom_id"
            assert get_request_id() == "custom_id"

    def test_generates_request_id_if_not_provided(self):
        """Test that request ID is generated if not provided."""
        with request_context() as req_id:
            assert req_id is not None
            assert len(req_id) == 8
            assert get_request_id() == req_id

    def test_restores_previous_context(self):
        """Test that previous context is restored after exit."""
        set_request_context(request_id="outer")

        with request_context(request_id="inner") as inner_id:
            assert inner_id == "inner"
            assert get_request_id() == "inner"

        assert get_request_id() == "outer"
        clear_request_context()

    def test_restores_context_on_exception(self):
        """Test that context is restored even if exception raised."""
        set_request_context(request_id="outer")

        with pytest.raises(ValueError), request_context(request_id="inner"):
            assert get_request_id() == "inner"
            raise ValueError("test error")

        assert get_request_id() == "outer"
        clear_request_context()

    def test_sets_all_context_variables(self):
        """Test setting all context variables."""
        clear_request_context()

        with request_context(
            request_id="req123",
            file_path="/path/to/file.pdf",
            provider="anthropic",
            model="claude-sonnet-4",
        ):
            assert get_request_id() == "req123"
            # Other context vars are internal, but we can verify via injection

        clear_request_context()


class TestInjectRequestContext:
    """Tests for _inject_request_context processor."""

    def test_injects_request_id(self):
        """Test injecting request ID into event dict."""
        set_request_context(request_id="test123")
        event_dict = {"event": "test message"}

        result = _inject_request_context(None, "info", event_dict)

        assert result["request_id"] == "test123"
        clear_request_context()

    def test_injects_file_context(self):
        """Test injecting file context."""
        set_request_context(file_path="/path/to/file.pdf")
        event_dict = {"event": "test"}

        result = _inject_request_context(None, "info", event_dict)

        assert result["file"] == "/path/to/file.pdf"
        clear_request_context()

    def test_injects_provider_context(self):
        """Test injecting provider context."""
        set_request_context(provider="openai")
        event_dict = {"event": "test"}

        result = _inject_request_context(None, "info", event_dict)

        assert result["provider"] == "openai"
        clear_request_context()

    def test_injects_model_context(self):
        """Test injecting model context."""
        set_request_context(model="gpt-4o")
        event_dict = {"event": "test"}

        result = _inject_request_context(None, "info", event_dict)

        assert result["model"] == "gpt-4o"
        clear_request_context()

    def test_does_not_overwrite_existing_keys(self):
        """Test that existing keys are not overwritten."""
        set_request_context(request_id="context_id")
        event_dict = {"event": "test", "request_id": "explicit_id"}

        result = _inject_request_context(None, "info", event_dict)

        assert result["request_id"] == "explicit_id"
        clear_request_context()

    def test_handles_none_context(self):
        """Test handling when context is None."""
        clear_request_context()
        event_dict = {"event": "test"}

        result = _inject_request_context(None, "info", event_dict)

        assert "request_id" not in result
        assert "file" not in result


class TestSafeStreamHandler:
    """Tests for SafeStreamHandler class."""

    def test_emit_normal_message(self):
        """Test emitting a normal message."""
        stream = io.StringIO()
        handler = SafeStreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Hello World",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        assert "Hello World" in stream.getvalue()

    def test_emit_unicode_message(self):
        """Test emitting message with unicode characters."""
        stream = io.StringIO()
        handler = SafeStreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="你好世界 🌍",
            args=(),
            exc_info=None,
        )

        handler.emit(record)
        assert "你好世界" in stream.getvalue()


class TestTruncateBase64:
    """Tests for _truncate_base64 processor."""

    def test_truncates_data_uri(self):
        """Test truncating base64 data URI."""
        long_base64 = "A" * 200
        event_dict = {"event": "test", "image": f"data:image/png;base64,{long_base64}"}

        result = _truncate_base64(None, "info", event_dict)

        assert "[BASE64:" in result["image"]
        assert long_base64 not in result["image"]

    def test_truncates_plain_base64(self):
        """Test truncating plain base64 string."""
        long_base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" * 20
        event_dict = {"event": "test", "data": long_base64}

        result = _truncate_base64(None, "info", event_dict)

        assert "[BASE64:" in result["data"]

    def test_does_not_truncate_short_strings(self):
        """Test that short strings are not modified."""
        event_dict = {"event": "test", "data": "short string"}

        result = _truncate_base64(None, "info", event_dict)

        assert result["data"] == "short string"

    def test_handles_non_string_values(self):
        """Test handling non-string values."""
        event_dict = {"event": "test", "count": 42, "flag": True}

        result = _truncate_base64(None, "info", event_dict)

        assert result["count"] == 42
        assert result["flag"] is True


class TestFilterEventDict:
    """Tests for _filter_event_dict processor."""

    def test_truncates_long_string(self):
        """Test truncating long string values."""
        long_string = "x" * 1000
        event_dict = {"event": "test", "data": long_string}

        result = _filter_event_dict(None, "info", event_dict)

        assert len(result["data"]) < len(long_string)
        assert "chars total" in result["data"]

    def test_handles_binary_data(self):
        """Test handling binary data."""
        binary_data = b"x" * 1000
        event_dict = {"event": "test", "data": binary_data}

        result = _filter_event_dict(None, "info", event_dict)

        assert "[BINARY DATA:" in result["data"]
        assert "bytes" in result["data"]

    def test_handles_bytearray(self):
        """Test handling bytearray."""
        binary_data = bytearray(b"x" * 1000)
        event_dict = {"event": "test", "data": binary_data}

        result = _filter_event_dict(None, "info", event_dict)

        assert "[BINARY DATA:" in result["data"]

    def test_does_not_truncate_short_string(self):
        """Test that short strings are not modified."""
        event_dict = {"event": "test", "data": "short"}

        result = _filter_event_dict(None, "info", event_dict)

        assert result["data"] == "short"


class TestAddSeparator:
    """Tests for _add_separator processor."""

    def test_adds_separator_when_context_present(self):
        """Test adding separator when context keys present."""
        event_dict = {"event": "Processing", "file": "/path/to/file"}

        result = _add_separator(None, "info", event_dict)

        assert result["event"] == "Processing |"

    def test_no_separator_without_context(self):
        """Test no separator when no context keys."""
        event_dict = {"event": "Processing", "level": "info", "timestamp": "2025-01-01"}

        result = _add_separator(None, "info", event_dict)

        # Internal keys only, no separator added
        assert result["event"] == "Processing"

    def test_handles_missing_event(self):
        """Test handling event dict without event key."""
        event_dict = {"file": "/path"}

        result = _add_separator(None, "info", event_dict)

        # Should not raise, just return unchanged
        assert result == event_dict


class TestGetConsole:
    """Tests for get_console function."""

    def test_returns_console(self):
        """Test that get_console returns a Console."""
        from rich.console import Console

        console = get_console()
        assert isinstance(console, Console)

    def test_returns_same_instance(self):
        """Test that get_console returns the same instance."""
        console1 = get_console()
        console2 = get_console()
        assert console1 is console2


class TestSetLogOutput:
    """Tests for set_log_output function."""

    def test_sets_output_stream(self):
        """Test setting log output stream."""
        stream = io.StringIO()
        set_log_output(stream)
        # Just verify it doesn't raise
        # The actual effect is tested in integration tests


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_configures_root_logger(self):
        """Test that root logger is configured."""
        setup_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_supports_json_format(self):
        """Test JSON format configuration."""
        # Just verify it doesn't raise
        setup_logging(level="INFO", json_format=True)

    def test_supports_file_logging(self, tmp_path):
        """Test file logging configuration."""
        log_file = tmp_path / "test.log"

        setup_logging(level="INFO", log_file=str(log_file))

        # Log something
        logger = logging.getLogger("test")
        logger.info("Test message")

        # Verify file was created
        assert log_file.exists()

    def test_creates_log_directory(self, tmp_path):
        """Test that log directory is created."""
        log_file = tmp_path / "subdir" / "test.log"

        setup_logging(level="INFO", log_file=str(log_file))

        assert log_file.parent.exists()

    def test_console_level_override(self):
        """Test console level override."""
        setup_logging(level="DEBUG", console_level="WARNING")

        root = logging.getLogger()
        # Root level should be DEBUG
        assert root.level == logging.DEBUG
        # But console handler should be WARNING (tested via handler)

    def test_file_level_override(self, tmp_path):
        """Test file level override."""
        log_file = tmp_path / "test.log"

        setup_logging(level="INFO", log_file=str(log_file), file_level="DEBUG")

        # Just verify it doesn't raise

    def test_suppresses_noisy_loggers(self):
        """Test that noisy third-party loggers are suppressed."""
        setup_logging(level="INFO")

        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level >= logging.WARNING


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_with_methods(self):
        """Test that get_logger returns a logger with expected methods."""
        setup_logging(level="INFO")
        logger = get_logger("test")
        # Check it has standard logging methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_with_name(self):
        """Test getting logger with specific name."""
        setup_logging(level="INFO")
        logger = get_logger("mymodule")
        assert logger is not None

    def test_without_name(self):
        """Test getting logger without name."""
        setup_logging(level="INFO")
        logger = get_logger()
        assert logger is not None


class TestCreateTaskLogPath:
    """Tests for create_task_log_path function."""

    def test_creates_log_directory(self, tmp_path):
        """Test that log directory is created."""
        log_dir = tmp_path / "logs"
        task_id, log_path = create_task_log_path(log_dir, "test")

        assert log_dir.exists()

    def test_returns_tuple(self, tmp_path):
        """Test that function returns tuple."""
        task_id, log_path = create_task_log_path(tmp_path, "test")

        assert isinstance(task_id, str)
        assert isinstance(log_path, Path)

    def test_task_id_is_8_chars(self, tmp_path):
        """Test that task ID is 8 characters."""
        task_id, _ = create_task_log_path(tmp_path, "test")
        assert len(task_id) == 8

    def test_log_path_has_prefix(self, tmp_path):
        """Test that log path contains prefix."""
        _, log_path = create_task_log_path(tmp_path, "convert")
        assert "convert_" in log_path.name

    def test_log_path_has_extension(self, tmp_path):
        """Test that log path has .log extension."""
        _, log_path = create_task_log_path(tmp_path, "test")
        assert log_path.suffix == ".log"


class TestSetupTaskLogging:
    """Tests for setup_task_logging function."""

    def test_creates_log_file(self, tmp_path):
        """Test that log file is created."""
        task_id, log_path = setup_task_logging(tmp_path, "test")

        # Log something to create the file
        logger = get_logger("test")
        logger.info("Test message")

        assert log_path.exists()

    def test_returns_task_id_and_path(self, tmp_path):
        """Test that function returns task ID and path."""
        task_id, log_path = setup_task_logging(tmp_path, "batch")

        assert isinstance(task_id, str)
        assert len(task_id) == 8
        assert isinstance(log_path, Path)
        assert "batch_" in log_path.name

    def test_verbose_mode(self, tmp_path):
        """Test verbose mode configuration."""
        task_id, log_path = setup_task_logging(tmp_path, "test", verbose=True)

        # Just verify it doesn't raise
        assert task_id is not None

    def test_quiet_mode(self, tmp_path):
        """Test quiet mode configuration."""
        task_id, log_path = setup_task_logging(tmp_path, "test", verbose=False)

        # Just verify it doesn't raise
        assert task_id is not None
