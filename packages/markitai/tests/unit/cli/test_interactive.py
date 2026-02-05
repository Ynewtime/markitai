"""Tests for interactive CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from markitai.cli.interactive import (
    InteractiveSession,
    detect_llm_provider,
)


class TestProviderDetection:
    """Tests for LLM provider auto-detection."""

    def test_detect_claude_cli_authenticated(self) -> None:
        """Should detect authenticated Claude CLI."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch(
                "markitai.cli.interactive._check_claude_auth",
                return_value=True,
            ),
        ):
            result = detect_llm_provider()
            assert result is not None
            assert result.provider == "claude-agent"
            assert result.model == "claude-agent/sonnet"
            assert result.authenticated is True

    def test_detect_copilot_cli_authenticated(self) -> None:
        """Should detect authenticated Copilot CLI when Claude not available."""
        with (
            patch(
                "shutil.which",
                side_effect=lambda x: "/usr/bin/copilot" if x == "copilot" else None,
            ),
            patch(
                "markitai.cli.interactive._check_copilot_auth",
                return_value=True,
            ),
        ):
            result = detect_llm_provider()
            assert result is not None
            assert result.provider == "copilot"
            assert result.model == "copilot/gpt-4o"

    def test_detect_anthropic_api_key(self) -> None:
        """Should detect ANTHROPIC_API_KEY environment variable."""
        with (
            patch("shutil.which", return_value=None),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=True),
        ):
            result = detect_llm_provider()
            assert result is not None
            assert result.provider == "anthropic"
            assert result.model == "anthropic/claude-sonnet-4-20250514"

    def test_detect_openai_api_key(self) -> None:
        """Should detect OPENAI_API_KEY when no other provider available."""
        with (
            patch("shutil.which", return_value=None),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True),
        ):
            result = detect_llm_provider()
            assert result is not None
            assert result.provider == "openai"
            assert result.model == "openai/gpt-4o"

    def test_detect_gemini_api_key(self) -> None:
        """Should detect GEMINI_API_KEY when no other provider available."""
        with (
            patch("shutil.which", return_value=None),
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        ):
            result = detect_llm_provider()
            assert result is not None
            assert result.provider == "gemini"
            assert result.model == "gemini/gemini-2.0-flash"

    def test_detect_no_provider(self) -> None:
        """Should return None when no provider detected."""
        with (
            patch("shutil.which", return_value=None),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = detect_llm_provider()
            assert result is None


class TestInteractiveSession:
    """Tests for InteractiveSession class."""

    def test_session_initialization(self) -> None:
        """Should initialize with default values."""
        session = InteractiveSession()
        assert session.input_path is None
        assert session.enable_llm is False
        assert session.provider_result is None


class TestInteractivePrompts:
    """Tests for interactive prompt flow."""

    @patch("questionary.select")
    def test_prompt_input_type(self, mock_select: MagicMock) -> None:
        """Should prompt for input type."""
        mock_select.return_value.ask.return_value = "directory"
        session = InteractiveSession()

        from markitai.cli.interactive import prompt_input_type

        result = prompt_input_type(session)

        assert result == "directory"
        mock_select.assert_called_once()

    @patch("questionary.path")
    def test_prompt_input_path(self, mock_path: MagicMock) -> None:
        """Should prompt for input path."""
        mock_path.return_value.ask.return_value = "./docs"
        session = InteractiveSession()
        session.input_type = "directory"

        from markitai.cli.interactive import prompt_input_path

        result = prompt_input_path(session)

        assert result == Path("./docs")

    @patch("questionary.confirm")
    def test_prompt_enable_llm(self, mock_confirm: MagicMock) -> None:
        """Should prompt for LLM enablement."""
        mock_confirm.return_value.ask.return_value = True
        session = InteractiveSession()

        from markitai.cli.interactive import prompt_enable_llm

        result = prompt_enable_llm(session)

        assert result is True


class TestRunInteractive:
    """Tests for the main run_interactive function."""

    @patch("markitai.cli.interactive.prompt_input_type")
    @patch("markitai.cli.interactive.prompt_input_path")
    @patch("markitai.cli.interactive.prompt_output_dir")
    @patch("markitai.cli.interactive.prompt_enable_llm")
    @patch("markitai.cli.interactive.prompt_llm_options")
    @patch("markitai.cli.interactive.prompt_configure_provider")
    def test_run_interactive_basic_flow(
        self,
        mock_configure: MagicMock,
        mock_options: MagicMock,
        mock_llm: MagicMock,
        mock_output: MagicMock,
        mock_path: MagicMock,
        mock_type: MagicMock,
    ) -> None:
        """Should run through all prompts in order."""
        mock_type.return_value = "file"
        mock_path.return_value = Path("test.pdf")
        mock_output.return_value = Path("./output")
        mock_llm.return_value = False

        from markitai.cli.interactive import run_interactive

        session = run_interactive()

        assert session.input_type == "file"
        mock_type.assert_called_once()
        mock_path.assert_called_once()
        mock_output.assert_called_once()
        mock_llm.assert_called_once()
