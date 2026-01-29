"""Tests for markitai.utils.text module."""

from __future__ import annotations

from markitai.utils.text import clean_control_characters, format_error_message


class TestCleanControlCharacters:
    """Tests for clean_control_characters function."""

    def test_clean_null_character(self):
        """Test that null character (0x00) is removed."""
        text = "hello\x00world"
        result = clean_control_characters(text)
        assert result == "helloworld"

    def test_clean_bell_character(self):
        """Test that bell character (0x07) is removed."""
        text = "hello\x07world"
        result = clean_control_characters(text)
        assert result == "helloworld"

    def test_preserve_newline(self):
        """Test that newline is preserved by default."""
        text = "line1\nline2"
        result = clean_control_characters(text)
        assert result == "line1\nline2"

    def test_preserve_tab(self):
        """Test that tab is preserved by default."""
        text = "col1\tcol2"
        result = clean_control_characters(text)
        assert result == "col1\tcol2"

    def test_preserve_carriage_return(self):
        """Test that carriage return is preserved by default."""
        text = "line1\r\nline2"
        result = clean_control_characters(text)
        assert result == "line1\r\nline2"

    def test_remove_whitespace_when_disabled(self):
        """Test that whitespace is removed when preserve_whitespace=False."""
        text = "line1\nline2\ttab"
        result = clean_control_characters(text, preserve_whitespace=False)
        assert result == "line1line2tab"

    def test_clean_multiple_control_chars(self):
        """Test cleaning multiple different control characters."""
        # Contains: null (0x00), bell (0x07), backspace (0x08), form feed (0x0C)
        text = "a\x00b\x07c\x08d\x0ce"
        result = clean_control_characters(text)
        assert result == "abcde"

    def test_clean_escape_sequence(self):
        """Test that escape character (0x1B) is removed."""
        text = "hello\x1b[31mworld"  # ANSI escape sequence
        result = clean_control_characters(text)
        assert result == "hello[31mworld"

    def test_empty_string(self):
        """Test with empty string."""
        result = clean_control_characters("")
        assert result == ""

    def test_no_control_chars(self):
        """Test string without control characters."""
        text = "Hello, World! 你好世界"
        result = clean_control_characters(text)
        assert result == text

    def test_json_problematic_chars(self):
        """Test characters that commonly cause JSON parsing errors."""
        # These are characters that LLMs sometimes output and break JSON
        text = '{"key": "value\x00with\x1fnull"}'
        result = clean_control_characters(text)
        assert result == '{"key": "valuewithnull"}'
        # Should be valid JSON now
        import json

        parsed = json.loads(result)
        assert parsed["key"] == "valuewithnull"


class TestFormatErrorMessage:
    """Tests for format_error_message function."""

    def test_simple_exception(self):
        """Test formatting a simple exception."""
        try:
            raise ValueError("test error message")
        except Exception as e:
            result = format_error_message(e)
            assert "ValueError" in result
            assert "test error message" in result
            assert "Traceback" not in result

    def test_exception_with_traceback_in_message(self):
        """Test that traceback embedded in message is removed."""
        try:
            raise RuntimeError(
                "Error message\nTraceback (most recent call last):\n"
                '  File "test.py", line 1\n    code here'
            )
        except Exception as e:
            result = format_error_message(e)
            assert "Traceback" not in result
            assert "Error message" in result

    def test_chained_exception_with_cause(self):
        """Test formatting chained exception with __cause__."""
        try:
            try:
                raise ConnectionError("network down")
            except ConnectionError as e1:
                raise RuntimeError("request failed") from e1
        except Exception as e:
            result = format_error_message(e)
            # Should extract the root cause message
            assert "network down" in result or "request failed" in result
            assert "Traceback" not in result

    def test_chained_exception_with_context(self):
        """Test formatting chained exception with __context__."""
        try:
            try:
                raise KeyError("missing key")
            except KeyError:
                raise ValueError("validation failed")
        except Exception as e:
            result = format_error_message(e)
            assert "Traceback" not in result

    def test_long_message_truncation(self):
        """Test that long messages are truncated."""
        try:
            raise ValueError("a" * 300)
        except Exception as e:
            result = format_error_message(e, max_length=50)
            assert len(result) < 100
            assert "..." in result

    def test_empty_exception_args(self):
        """Test exception with no args."""
        try:
            raise RuntimeError()
        except Exception as e:
            result = format_error_message(e)
            assert "RuntimeError" in result

    def test_none_input(self):
        """Test with None input."""
        result = format_error_message(None)
        assert result == "Unknown error"

    def test_non_exception_input(self):
        """Test with non-exception input."""
        result = format_error_message("just a string")
        assert result == "just a string"

    def test_removes_exception_prefix(self):
        """Test that duplicate exception type prefix is removed."""
        try:
            raise ValueError("ValueError: doubled prefix")
        except Exception as e:
            result = format_error_message(e)
            assert result.count("ValueError") == 1

    def test_preserves_unicode(self):
        """Test that Unicode messages are preserved."""
        try:
            raise ValueError("请求超时 (120 秒)")
        except Exception as e:
            result = format_error_message(e)
            assert "请求超时" in result
            assert "120 秒" in result

    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        try:
            raise ValueError("x" * 100)
        except Exception as e:
            result = format_error_message(e, max_length=20)
            # Should be around 30 chars: "ValueError: " + 20 chars + "..."
            assert "..." in result
            assert len(result) < 50

    def test_multiline_message_collapsed(self):
        """Test that multiline messages are collapsed to single line."""
        try:
            raise ValueError(
                "Environment variable not found: API_KEY\n\n"
                "  Purpose: API authentication\n\n"
                "  To set it:\n"
                "    export API_KEY=your_value\n\n"
                "  Or add to .env file:\n"
                "    API_KEY=your_value"
            )
        except Exception as e:
            result = format_error_message(e)
            # Should not contain newlines - collapsed to single line
            assert "\n" not in result
            # Key info should still be present
            assert "Environment variable not found" in result
            assert "API_KEY" in result

    def test_multiline_with_multiple_spaces(self):
        """Test that multiple spaces are collapsed after newline removal."""
        try:
            raise RuntimeError("Error:   \n\n   too many   \n   spaces")
        except Exception as e:
            result = format_error_message(e)
            # Should not have multiple consecutive spaces
            assert "  " not in result
            assert "\n" not in result


class TestFormatErrorMessageIntegration:
    """Integration tests simulating real LLM errors."""

    def test_litellm_style_error(self):
        """Test error similar to LiteLLM APIConnectionError."""

        class MockAPIConnectionError(Exception):
            def __init__(self, message: str):
                super().__init__(message)

        try:
            raise MockAPIConnectionError(
                "Copilot 请求超时 (120 秒)。请检查网络连接或增加超时时间。"
            )
        except Exception as e:
            result = format_error_message(e)
            assert "请求超时" in result
            assert "120 秒" in result
            assert "Traceback" not in result

    def test_timeout_error_chain(self):
        """Test chained timeout error like in copilot.py."""
        try:
            try:
                raise TimeoutError("Timeout after 60.0s waiting for session.idle")
            except TimeoutError as e1:
                raise RuntimeError("Copilot 请求超时 (120 秒)") from e1
        except Exception as e:
            result = format_error_message(e)
            assert "Traceback" not in result
            # Should get a useful message
            assert "超时" in result or "Timeout" in result
