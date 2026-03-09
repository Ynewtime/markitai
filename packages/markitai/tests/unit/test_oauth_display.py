"""Tests for OAuth display utilities."""

from __future__ import annotations

import io
import sys

import pytest
from rich.console import Console

from markitai.providers.oauth_display import (
    DeviceCodeInterceptor,
    parse_chatgpt_device_code,
    show_device_code,
    show_oauth_start,
    show_oauth_success,
    suppress_stdout,
)


def _make_test_console() -> tuple[Console, io.StringIO]:
    """Create a test console that writes to a StringIO buffer (no ANSI)."""
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, width=120, highlight=False)
    return console, buf


class TestSuppressStdout:
    """Tests for suppress_stdout context manager."""

    def test_captures_stdout(self) -> None:
        """stdout writes inside the context are captured."""
        with suppress_stdout() as captured:
            print("hello from stdout")
        assert "hello from stdout" in captured.getvalue()

    def test_restores_stdout_after(self) -> None:
        """sys.stdout is restored after exiting context."""
        original = sys.stdout
        with suppress_stdout():
            pass
        assert sys.stdout is original

    def test_restores_stdout_on_exception(self) -> None:
        """sys.stdout is restored even if an exception occurs."""
        original = sys.stdout
        with pytest.raises(ValueError), suppress_stdout():
            raise ValueError("boom")
        assert sys.stdout is original


class TestShowOAuthStart:
    """Tests for show_oauth_start."""

    def test_gemini_start_message(self) -> None:
        """Gemini start message includes provider label."""
        console, buf = _make_test_console()
        show_oauth_start("gemini-cli", console=console)
        output = buf.getvalue()
        assert "Gemini" in output
        assert "Authentication" in output
        assert "browser" in output.lower()

    def test_chatgpt_start_message(self) -> None:
        """ChatGPT uses correct provider label."""
        console, buf = _make_test_console()
        show_oauth_start("chatgpt", console=console)
        assert "ChatGPT" in buf.getvalue()


class TestShowDeviceCode:
    """Tests for show_device_code."""

    def test_displays_url_and_code(self) -> None:
        """Device code display includes URL and code."""
        console, buf = _make_test_console()
        show_device_code(
            "https://chatgpt.com/auth/device",
            "ABCD-1234",
            console=console,
        )
        output = buf.getvalue()
        assert "chatgpt.com/auth/device" in output
        assert "ABCD-1234" in output

    def test_includes_phishing_warning(self) -> None:
        """Device code display includes phishing warning."""
        console, buf = _make_test_console()
        show_device_code("https://example.com", "CODE", console=console)
        assert "phishing" in buf.getvalue().lower()


class TestShowOAuthSuccess:
    """Tests for show_oauth_success."""

    def test_success_with_user(self) -> None:
        """Success message includes user info when provided."""
        console, buf = _make_test_console()
        show_oauth_success("gemini-cli", user="me@example.com", console=console)
        output = buf.getvalue()
        assert "Gemini" in output
        assert "authenticated" in output
        assert "me@example.com" in output

    def test_success_without_user(self) -> None:
        """Success message works without user info."""
        console, buf = _make_test_console()
        show_oauth_success("chatgpt", console=console)
        output = buf.getvalue()
        assert "ChatGPT" in output
        assert "authenticated" in output

    def test_success_with_detail(self) -> None:
        """Success message includes detail when provided."""
        console, buf = _make_test_console()
        show_oauth_success(
            "gemini-cli",
            detail="Saved to ~/.gemini/oauth_creds.json",
            console=console,
        )
        assert "oauth_creds.json" in buf.getvalue()


class TestParseChatGPTDeviceCode:
    """Tests for parse_chatgpt_device_code."""

    def test_parses_litellm_output(self) -> None:
        """Parses device code from LiteLLM authenticator output."""
        output = (
            "Sign in with ChatGPT using device code:\n"
            "1) Visit https://chatgpt.com/auth/device\n"
            "2) Enter code: ABCD-1234\n"
            "Device codes are a common phishing target.\n"
        )
        result = parse_chatgpt_device_code(output)
        assert result is not None
        url, code = result
        assert url == "https://chatgpt.com/auth/device"
        assert code == "ABCD-1234"

    def test_returns_none_for_unrelated_output(self) -> None:
        """Returns None when output doesn't contain device code info."""
        assert parse_chatgpt_device_code("some random output") is None

    def test_returns_none_for_partial_output(self) -> None:
        """Returns None when only URL or only code is present."""
        assert parse_chatgpt_device_code("Visit https://example.com") is None
        assert parse_chatgpt_device_code("Enter code: ABC") is None


class TestDeviceCodeInterceptor:
    """Tests for DeviceCodeInterceptor."""

    def test_intercepts_and_displays_device_code(self) -> None:
        """Interceptor captures device code and displays Rich output."""
        console, buf = _make_test_console()
        interceptor = DeviceCodeInterceptor(console=console)

        # Simulate LiteLLM's print() calls
        interceptor.write("Sign in with ChatGPT using device code:\n")
        interceptor.write("1) Visit https://chatgpt.com/auth/device\n")
        interceptor.write("2) Enter code: TEST-CODE\n")

        output = buf.getvalue()
        assert interceptor.displayed
        assert "chatgpt.com/auth/device" in output
        assert "TEST-CODE" in output

    def test_suppresses_output_after_display(self) -> None:
        """After displaying, subsequent writes are silently consumed."""
        console, buf = _make_test_console()
        interceptor = DeviceCodeInterceptor(console=console)

        # Trigger display
        interceptor.write(
            "1) Visit https://chatgpt.com/auth/device\n2) Enter code: CODE\n"
        )
        output_after_display = buf.getvalue()

        # Further writes should not add output
        interceptor.write("More noise\n")
        assert buf.getvalue() == output_after_display

    def test_returns_write_length(self) -> None:
        """write() returns the length of the input string."""
        interceptor = DeviceCodeInterceptor(console=_make_test_console()[0])
        assert interceptor.write("hello") == 5

    def test_not_displayed_for_unrelated_output(self) -> None:
        """Interceptor does not display for non-device-code output."""
        console, buf = _make_test_console()
        interceptor = DeviceCodeInterceptor(console=console)
        interceptor.write("some unrelated text\n")
        assert not interceptor.displayed
        assert buf.getvalue() == ""
