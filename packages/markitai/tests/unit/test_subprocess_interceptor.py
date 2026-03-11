"""Unit tests for SubprocessInterceptor.

Tests cover output pattern matching, formatting, and subprocess mechanics
for intercepting stdout/stderr from CLI login subprocesses (copilot, claude-agent).
"""

from __future__ import annotations

import sys

import pytest

from markitai.providers.subprocess_interceptor import (
    _CODE_PATTERN,
    _ERROR_PATTERN,
    _SUCCESS_PATTERNS,
    _URL_PATTERN,
    _WAITING_PATTERN,
    SubprocessInterceptor,
)

# ---------------------------------------------------------------------------
# Mock OutputTarget
# ---------------------------------------------------------------------------


class MockOutputManager:
    """Mock output target for testing SubprocessInterceptor."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.line_count: int = 0
        self._spinner_msg: str | None = None

    def print(self, text: str, *, style: str | None = None) -> None:
        self.lines.append(text)
        self.line_count += text.count("\n") + 1

    def start_spinner(self, message: str) -> None:
        self._spinner_msg = message

    def stop_spinner(self) -> None:
        self._spinner_msg = None


# ===========================================================================
# Pattern matching tests
# ===========================================================================


class TestURLPattern:
    """Tests for URL detection regex."""

    def test_matches_https_url(self) -> None:
        m = _URL_PATTERN.search("Visit https://github.com/login/device")
        assert m is not None
        assert m.group(1) == "https://github.com/login/device"

    def test_matches_http_url(self) -> None:
        m = _URL_PATTERN.search("Open http://example.com/auth")
        assert m is not None
        assert m.group(1) == "http://example.com/auth"

    def test_no_match_without_url(self) -> None:
        assert _URL_PATTERN.search("No URL here") is None

    def test_matches_url_with_query_params(self) -> None:
        m = _URL_PATTERN.search("Go to https://example.com/auth?code=abc&state=123")
        assert m is not None
        assert m.group(1) == "https://example.com/auth?code=abc&state=123"


class TestCodePattern:
    """Tests for device code extraction regex."""

    def test_matches_enter_code_format(self) -> None:
        m = _CODE_PATTERN.search("Enter code: ABCD-1234")
        assert m is not None
        assert m.group(1) == "ABCD-1234"

    def test_matches_code_colon_format(self) -> None:
        m = _CODE_PATTERN.search("code: WXYZ-5678")
        assert m is not None
        assert m.group(1) == "WXYZ-5678"

    def test_case_insensitive(self) -> None:
        m = _CODE_PATTERN.search("ENTER CODE: ABCD-1234")
        assert m is not None
        assert m.group(1) == "ABCD-1234"

    def test_no_match_for_non_code(self) -> None:
        assert _CODE_PATTERN.search("random text") is None

    def test_matches_lowercase_code(self) -> None:
        """Device codes may appear in mixed case from some providers."""
        m = _CODE_PATTERN.search("code: abcd-1234")
        assert m is not None
        assert m.group(1) == "abcd-1234"


class TestSuccessPatterns:
    """Tests for success message detection."""

    def test_signed_in_successfully_as(self) -> None:
        for pattern in _SUCCESS_PATTERNS:
            m = pattern.search("Signed in successfully as user@example.com")
            if m:
                assert m.group(1).strip() == "user@example.com"
                return
        pytest.fail("No success pattern matched")

    def test_logged_in_as(self) -> None:
        for pattern in _SUCCESS_PATTERNS:
            m = pattern.search("Logged in as octocat")
            if m:
                assert m.group(1).strip() == "octocat"
                return
        pytest.fail("No success pattern matched")

    def test_authenticated_alone(self) -> None:
        matched = any(p.search("Authenticated") for p in _SUCCESS_PATTERNS)
        assert matched

    def test_authenticated_as_user(self) -> None:
        for pattern in _SUCCESS_PATTERNS:
            m = pattern.search("Authenticated as john")
            if m:
                assert m.group(1).strip() == "john"
                return
        pytest.fail("No success pattern matched")

    def test_no_match_for_random_text(self) -> None:
        matched = any(p.search("Hello world") for p in _SUCCESS_PATTERNS)
        assert not matched


class TestWaitingPattern:
    """Tests for waiting message detection."""

    def test_matches_waiting_for_authorization(self) -> None:
        assert _WAITING_PATTERN.search("Waiting for authorization...") is not None

    def test_matches_waiting_for_login(self) -> None:
        assert _WAITING_PATTERN.search("Waiting for login...") is not None

    def test_case_insensitive(self) -> None:
        assert _WAITING_PATTERN.search("WAITING FOR auth") is not None

    def test_no_match_for_unrelated(self) -> None:
        assert _WAITING_PATTERN.search("Processing files") is None


class TestErrorPattern:
    """Tests for error message detection."""

    def test_matches_error(self) -> None:
        assert _ERROR_PATTERN.search("Error: something went wrong") is not None

    def test_matches_failed(self) -> None:
        assert _ERROR_PATTERN.search("Authentication failed") is not None

    def test_matches_denied(self) -> None:
        assert _ERROR_PATTERN.search("Access denied") is not None

    def test_case_insensitive(self) -> None:
        assert _ERROR_PATTERN.search("ERROR: bad request") is not None

    def test_no_match_for_success(self) -> None:
        assert _ERROR_PATTERN.search("Signed in successfully") is None


# ===========================================================================
# Line formatting tests
# ===========================================================================


class TestFormatLine:
    """Tests for _format_line output formatting."""

    def setup_method(self) -> None:
        self.output = MockOutputManager()
        self.interceptor = SubprocessInterceptor(self.output)

    def test_url_line_formatted_as_link(self) -> None:
        result = self.interceptor._format_line(
            "Visit https://github.com/login/device", "copilot"
        )
        assert result is not None
        assert "[link=" in result
        assert "https://github.com/login/device" in result

    def test_device_code_highlighted_cyan(self) -> None:
        result = self.interceptor._format_line("Enter code: ABCD-1234", "copilot")
        assert result is not None
        assert "[bold cyan]" in result
        assert "ABCD-1234" in result

    def test_waiting_line_starts_spinner(self) -> None:
        result = self.interceptor._format_line(
            "Waiting for authorization...", "copilot"
        )
        assert result is None  # handled via spinner, not print
        assert self.output._spinner_msg is not None
        assert "Copilot" in self.output._spinner_msg

    def test_success_line_shows_green_checkmark(self) -> None:
        result = self.interceptor._format_line(
            "Signed in successfully as octocat", "copilot"
        )
        assert result is not None
        assert "[green]\u2713[/]" in result
        assert "Copilot" in result
        assert "octocat" in result

    def test_claude_agent_success_shows_claude_label(self) -> None:
        result = self.interceptor._format_line(
            "Authenticated as user@example.com", "claude-agent"
        )
        assert result is not None
        assert "[green]\u2713[/]" in result
        assert "Claude" in result
        assert "user@example.com" in result

    def test_error_line_formatted_red(self) -> None:
        result = self.interceptor._format_line(
            "Error: authentication failed", "copilot"
        )
        assert result is not None
        assert "[red]\u2717[/]" in result

    def test_unknown_line_formatted_dim(self) -> None:
        result = self.interceptor._format_line("Some unknown output", "copilot")
        assert result is not None
        assert "[dim]" in result
        assert "Some unknown output" in result

    def test_empty_line_returns_none(self) -> None:
        result = self.interceptor._format_line("", "copilot")
        assert result is None

    def test_whitespace_only_line_returns_none(self) -> None:
        result = self.interceptor._format_line("   ", "copilot")
        assert result is None

    def test_all_formatted_lines_have_indentation(self) -> None:
        """All non-None formatted lines should start with 4-space indent."""
        lines = [
            "Visit https://github.com/login/device",
            "Enter code: ABCD-1234",
            "Signed in successfully as octocat",
            "Error: something",
            "Some random text",
        ]
        for line in lines:
            result = self.interceptor._format_line(line, "copilot")
            if result is not None:
                assert result.startswith("    "), (
                    f"Line should start with 4-space indent: {result!r}"
                )

    def test_success_without_user_shows_authenticated(self) -> None:
        """'Authenticated' alone should still show success."""
        result = self.interceptor._format_line("Authenticated", "claude-agent")
        assert result is not None
        assert "[green]\u2713[/]" in result
        assert "Claude" in result

    def test_spinner_stopped_before_non_waiting_line(self) -> None:
        """After a waiting line starts spinner, next non-waiting line stops it."""
        # First, start spinner
        self.interceptor._format_line("Waiting for authorization...", "copilot")
        assert self.output._spinner_msg is not None

        # Next line should stop spinner
        self.interceptor._format_line("Signed in successfully as octocat", "copilot")
        assert self.output._spinner_msg is None

    def test_logged_in_as_user_copilot(self) -> None:
        result = self.interceptor._format_line("Logged in as octocat", "copilot")
        assert result is not None
        assert "[green]\u2713[/]" in result
        assert "octocat" in result

    def test_url_with_code_on_same_line(self) -> None:
        """Both URL and code are shown when they appear on the same line."""
        result = self.interceptor._format_line(
            "Visit https://github.com/login/device and enter code: ABCD-1234",
            "copilot",
        )
        assert result is not None
        # Both URL and code must be present
        assert "https://github.com/login/device" in result
        assert "ABCD-1234" in result
        # URL comes before code (visit first, then enter code)
        url_pos = result.index("github.com/login/device")
        code_pos = result.index("ABCD-1234")
        assert url_pos < code_pos

    def test_copilot_combined_auth_line(self) -> None:
        """Real copilot PTY output: URL + code on single line."""
        result = self.interceptor._format_line(
            "To authenticate, visit https://github.com/login/device and enter code 73FF-BEDB.",
            "copilot",
        )
        assert result is not None
        assert "https://github.com/login/device" in result
        assert "73FF-BEDB" in result


# ===========================================================================
# Subprocess mechanics tests
# ===========================================================================


class TestSubprocessMechanics:
    """Tests for subprocess execution via PIPE."""

    async def test_returns_exit_code_zero(self) -> None:
        """Exit code 0 is returned for successful subprocess."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)
        exit_code = await interceptor.run(
            [sys.executable, "-c", "pass"], provider="copilot"
        )
        assert exit_code == 0

    async def test_returns_nonzero_exit_code(self) -> None:
        """Nonzero exit code is propagated."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)
        exit_code = await interceptor.run(
            [sys.executable, "-c", "import sys; sys.exit(1)"], provider="copilot"
        )
        assert exit_code == 1

    async def test_empty_output_works(self) -> None:
        """Subprocess with no output doesn't crash."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)
        exit_code = await interceptor.run(
            [sys.executable, "-c", "pass"], provider="copilot"
        )
        assert exit_code == 0
        assert output.lines == []

    async def test_output_manager_receives_formatted_lines(self) -> None:
        """OutputManager.print() is called for each non-empty output line."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        script = (
            'print("Visit https://github.com/login/device");'
            'print("Enter code: ABCD-1234");'
            'print("Signed in successfully as octocat")'
        )
        exit_code = await interceptor.run(
            [sys.executable, "-c", script], provider="copilot"
        )

        assert exit_code == 0
        assert len(output.lines) == 3
        assert "https://github.com/login/device" in output.lines[0]
        assert "ABCD-1234" in output.lines[1]
        assert "octocat" in output.lines[2]

    async def test_waiting_line_uses_spinner_not_print(self) -> None:
        """Waiting messages use spinner instead of print."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        script = (
            'print("Waiting for authorization...");'
            'print("Signed in successfully as octocat")'
        )
        await interceptor.run([sys.executable, "-c", script], provider="copilot")

        # Only the success line should be in print output
        assert len(output.lines) == 1
        assert "octocat" in output.lines[0]

    async def test_spinner_stopped_on_success_after_waiting(self) -> None:
        """Spinner started by waiting line is stopped when success line arrives."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        script = (
            'print("Waiting for authorization...");'
            'print("Signed in successfully as octocat")'
        )
        await interceptor.run([sys.executable, "-c", script], provider="copilot")
        assert output._spinner_msg is None

    async def test_echo(self) -> None:
        """Run a real subprocess that echoes text."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        exit_code = await interceptor.run(
            [sys.executable, "-c", 'print("Hello from subprocess")'],
            provider="copilot",
        )

        assert exit_code == 0
        assert len(output.lines) == 1
        assert "Hello from subprocess" in output.lines[0]

    async def test_stderr_merged(self) -> None:
        """stderr is merged into stdout via STDOUT redirect."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        exit_code = await interceptor.run(
            [
                sys.executable,
                "-c",
                'import sys; sys.stderr.write("stderr msg\\n")',
            ],
            provider="copilot",
        )

        assert exit_code == 0
        assert len(output.lines) == 1
        assert "stderr msg" in output.lines[0]


# ===========================================================================
# Copilot login end-to-end formatting
# ===========================================================================


class TestCopilotLoginFormatting:
    """End-to-end formatting tests simulating copilot login output."""

    async def test_full_copilot_login_flow(self) -> None:
        """Simulate a full copilot login flow with all message types."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        script = (
            'print("Visit https://github.com/login/device");'
            'print("Enter code: ABCD-1234");'
            'print("Waiting for authorization...");'
            'print("Logged in as octocat")'
        )
        exit_code = await interceptor.run(
            [sys.executable, "-c", script], provider="copilot"
        )

        assert exit_code == 0
        # 3 printed lines (waiting goes to spinner)
        assert len(output.lines) == 3

        # URL line
        assert "    " in output.lines[0]  # indented
        assert "[link=" in output.lines[0]

        # Code line
        assert "[bold cyan]" in output.lines[1]
        assert "ABCD-1234" in output.lines[1]

        # Success line
        assert "[green]\u2713[/]" in output.lines[2]
        assert "Copilot" in output.lines[2]
        assert "octocat" in output.lines[2]


# ===========================================================================
# Claude auth login end-to-end formatting
# ===========================================================================


class TestRawLineCapture:
    """Tests for raw line capture and logging."""

    async def test_raw_lines_captured(self) -> None:
        """Interceptor stores raw lines for post-mortem analysis."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        script = (
            'print("Visit https://github.com/login/device");'
            'print("Enter code: ABCD-1234");'
            'print("Some error occurred");'
            "import sys; sys.exit(1)"
        )
        await interceptor.run([sys.executable, "-c", script], provider="copilot")

        assert len(interceptor.raw_lines) == 3
        assert "https://github.com/login/device" in interceptor.raw_lines[0]
        assert "ABCD-1234" in interceptor.raw_lines[1]
        assert "Some error occurred" in interceptor.raw_lines[2]

    async def test_raw_lines_empty_for_no_output(self) -> None:
        """raw_lines is empty when subprocess produces no output."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)
        await interceptor.run([sys.executable, "-c", "pass"], provider="copilot")
        assert interceptor.raw_lines == []


class TestSubprocessEdgeCases:
    """Tests for subprocess edge cases."""

    async def test_subprocess_captures_output(self) -> None:
        """Output is captured and formatted."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        exit_code = await interceptor.run(
            [sys.executable, "-c", 'print("Hello from subprocess")'],
            provider="copilot",
        )

        assert exit_code == 0
        assert len(output.lines) == 1
        assert "Hello from subprocess" in output.lines[0]

    async def test_strips_ansi_for_pattern_matching(self) -> None:
        """ANSI escape codes are stripped before pattern matching."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        exit_code = await interceptor.run(
            [
                sys.executable,
                "-c",
                r'print("\033[1mEnter code: ABCD-1234\033[0m")',
            ],
            provider="copilot",
        )

        assert exit_code == 0
        assert any("ABCD-1234" in line for line in output.lines)
        # Should match the code pattern and format with bold cyan
        assert any("[bold cyan]" in line for line in output.lines)

    async def test_exit_code_propagated(self) -> None:
        """Non-zero exit codes are propagated."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        exit_code = await interceptor.run(
            [sys.executable, "-c", "import sys; sys.exit(42)"],
            provider="copilot",
        )

        assert exit_code == 42

    async def test_raw_lines_captured(self) -> None:
        """Raw lines are captured for post-mortem analysis."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        await interceptor.run(
            [sys.executable, "-c", 'print("line one"); print("line two")'],
            provider="copilot",
        )

        assert len(interceptor.raw_lines) == 2


class TestClaudeAuthLoginFormatting:
    """End-to-end formatting tests simulating claude auth login output."""

    async def test_authenticated_message(self) -> None:
        """Claude 'Authenticated' message should show green success."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        exit_code = await interceptor.run(
            [sys.executable, "-c", 'print("Authenticated")'],
            provider="claude-agent",
        )

        assert exit_code == 0
        assert len(output.lines) == 1
        assert "[green]\u2713[/]" in output.lines[0]
        assert "Claude" in output.lines[0]

    async def test_claude_error_formatted_red(self) -> None:
        """Error output from claude auth should be red."""
        output = MockOutputManager()
        interceptor = SubprocessInterceptor(output)

        await interceptor.run(
            [sys.executable, "-c", 'print("Error: authentication failed")'],
            provider="claude-agent",
        )

        assert len(output.lines) == 1
        assert "[red]\u2717[/]" in output.lines[0]
